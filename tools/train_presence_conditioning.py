"""Train a Stage 03/04/05 SegFormer-B3 arm with the shared fixed protocol."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dino.embedding import select_device
from src.dino.download import sha256_file
from src.segmentation.data import SegmentationDataset, load_aligned_presence
from src.segmentation.dino_conditioning import (
    DINO_DIM,
    DirectDinoEmbeddingStore,
    DirectDinoSegformer,
    DirectDinoSegmentationDataset,
)
from src.segmentation.dino_moe import DinoSoftMoESegformer, RoutingAccumulator
from src.segmentation.experiment import evaluate, resolve_pretrained, save_json, seed_everything, segmentation_loss
from src.segmentation.model import PresenceSegformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("control", "conditioned", "direct_dino", "dino_soft_moe"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--presence-dir", type=Path, default=Path("outputs/class_presence/linear_518"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps"))
    parser.add_argument(
        "--augmented-dino-dir", type=Path, default=Path("outputs/dinov2_518/augmented_train")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="CUDA float16 autocast; use identically for both arms")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume the same run from its full-state last.pt checkpoint",
    )
    args = parser.parse_args()
    for name in ("max_steps", "eval_every", "batch_size", "grad_accum"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid worker count, learning rate or weight decay")
    if args.resume_from is None and args.output_dir.resolve().exists():
        parser.error("Output directory already exists; use a fresh run directory")
    if args.resume_from is not None:
        if not args.output_dir.resolve().is_dir():
            parser.error("Resume requires the existing run output directory")
        if not args.resume_from.resolve().is_file():
            parser.error(f"Resume checkpoint does not exist: {args.resume_from}")
        if args.resume_from.resolve().parent != args.output_dir.resolve():
            parser.error("Resume checkpoint must belong to --output-dir")
        if args.num_workers != 0:
            parser.error("Exact Stage 03/04/05 resume requires --num-workers 0")
    if args.data_dir.resolve() in args.output_dir.resolve().parents:
        parser.error("Output directory cannot be under read-only data/")
    return args


def numpy_rng_state() -> dict[str, object]:
    """Store NumPy RNG state using only checkpoint-safe Python values."""
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def restore_numpy_rng_state(state: dict[str, object]) -> None:
    np.random.set_state((
        str(state["name"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


def save_resume_checkpoint(path: Path, state: dict[str, object]) -> None:
    """Atomically replace the full-state checkpoint used for interruption recovery."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def validate_resume_config(args: argparse.Namespace, config: dict[str, object]) -> None:
    """Reject protocol changes when continuing an existing Stage 03/04/05 run."""
    keys = (
        "arm", "data_dir", "pretrained_dir", "cache_dir", "hf_endpoint",
        "device", "seed", "max_steps", "eval_every", "batch_size", "grad_accum",
        "num_workers", "learning_rate", "weight_decay", "amp",
    )
    keys += (
        ("embedding_dir", "augmented_dino_dir")
        if args.arm in {"direct_dino", "dino_soft_moe"}
        else ("presence_dir",)
    )
    for key in keys:
        actual = getattr(args, key)
        if isinstance(actual, Path):
            actual = str(actual)
        if config.get(key) != actual:
            raise ValueError(
                f"Resume protocol differs at {key}: existing={config.get(key)!r}, requested={actual!r}"
            )


@torch.inference_mode()
def evaluate_embedding_store_routing(
    model: DinoSoftMoESegformer,
    store: DirectDinoEmbeddingStore,
    split: str,
    device: torch.device,
    amp: bool,
    batch_size: int = 1024,
) -> dict[str, object]:
    """Measure router utilization directly over cached embeddings at the best checkpoint."""
    model.eval()
    accumulator = RoutingAccumulator()
    if split == "train":
        embeddings = store.augmented.reshape(-1, DINO_DIM)
        count = embeddings.shape[0]

        def batch_at(start: int) -> np.ndarray:
            return np.asarray(embeddings[start : start + batch_size]).copy()
    else:
        ids = store.split_ids[split]
        count = len(ids)

        def batch_at(start: int) -> np.ndarray:
            group = ids[start : start + batch_size]
            return np.stack([store.embedding(image_id, split, "r0") for image_id in group])

    for start in range(0, count, batch_size):
        batch = torch.from_numpy(batch_at(start)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            weights = model.routing_weights(batch)
        accumulator.update(weights)
    summary = accumulator.summary()
    summary["embedding_distribution"] = (
        "all 8 canonical train states per image" if split == "train" else "original r0"
    )
    return summary


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently supports CUDA only")
    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    seed_everything(args.seed)
    dino_arm = args.arm in {"direct_dino", "dino_soft_moe"}
    moe_arm = args.arm == "dino_soft_moe"
    if dino_arm:
        dino_store = DirectDinoEmbeddingStore(
            args.data_dir, args.embedding_dir, args.augmented_dino_dir, require_full_augmented=True
        )
        datasets = {
            split: DirectDinoSegmentationDataset(
                args.data_dir, split, dino_store, augment=split == "train"
            )
            for split in ("train", "val_stratified", "val_domain")
        }
        split_counts = {split: len(dataset) for split, dataset in datasets.items()}
        model = (
            DinoSoftMoESegformer(str(pretrained_dir))
            if moe_arm else DirectDinoSegformer(str(pretrained_dir))
        ).to(device)
    else:
        aligned = load_aligned_presence(args.data_dir, args.presence_dir)
        datasets = {
            split: SegmentationDataset(args.data_dir, aligned[split], augment=split == "train")
            for split in ("train", "val_stratified", "val_domain")
        }
        split_counts = {split: len(rows) for split, rows in aligned.items()}
        model = PresenceSegformer(str(pretrained_dir), args.arm == "conditioned").to(device)
    model.encoder.gradient_checkpointing_enable()
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size, shuffle=True, generator=loader_generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        datasets["val_stratified"],
        batch_size=1, num_workers=args.num_workers,
    )
    domain_loader = DataLoader(
        datasets["val_domain"],
        batch_size=1, num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    config_path = args.output_dir / "config.json"
    if args.resume_from is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        serialized_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key != "resume_from"
        }
        if dino_arm:
            serialized_args.pop("presence_dir")
        else:
            serialized_args.pop("embedding_dir")
            serialized_args.pop("augmented_dino_dir")
        run_metadata: dict[str, object] = {
            **serialized_args,
            "pretrained_path": str(pretrained_dir), "pretrained_model": "nvidia/mit-b3",
            "pretrained_revision": pretrained_dir.name if pretrained_dir.parent.name == "snapshots" else None,
            "pretrained_sha256": sha256_file(next(
                pretrained_dir / name for name in ("model.safetensors", "pytorch_model.bin")
                if (pretrained_dir / name).is_file()
            )),
            "input_size": [1024, 1024], "num_classes": 8, "ignore_index": 255,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "loss": "multiclass Cross Entropy", "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR", "gradient_checkpointing": True,
            "conditioning_type": args.arm,
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "conditioning_parameter_count": sum(
                parameter.numel() for parameter in model.film.parameters()
            ) if model.film is not None else 0,
            "split_counts": split_counts,
            "split_sha256": {
                split: sha256_file(args.data_dir / "splits" / f"{split}.txt") for split in split_counts
            },
        }
        if dino_arm:
            run_metadata.update({
                "dino_source": "Stage 01 accepted Combined CLS || MeanPatch embeddings",
                "dino_model": "facebook/dinov2-base",
                "dino_revision": dino_store.base_metadata["resolved_revision"],
                "dino_input_dimension": 1536,
                "conditioning_architecture": "LayerNorm -> Linear 1536->256 -> GELU -> Linear 256->1536",
                "film_channels": 768,
                "augmentation_alignment": "same image ID and canonical D4 transform state",
                "base_embedding_metadata_sha256": sha256_file(args.embedding_dir / "metadata.json"),
                "augmented_embedding_metadata_sha256": sha256_file(args.augmented_dino_dir / "metadata.json"),
            })
            if moe_arm:
                run_metadata.update({
                    "initialization_strategy": "nvidia/mit-b3 start; Stage 04 best.pt is not loaded",
                    "moe_type": "DINO-conditioned dense soft routing",
                    "num_experts": 4,
                    "router_architecture": "LayerNorm -> Linear 1536->256 -> GELU -> Linear 256->4 -> Softmax",
                    "expert_architecture": "Conv1x1 768->192 -> GELU -> Conv1x1 192->768",
                    "expert_output_initialization": "normal std=1e-5; zero bias",
                    "moe_residual": "F_moe = F_film + sum_i(w_i * E_i(F_film))",
                    "routing": "all four experts execute; no Top-k or hard routing",
                    "router_regularization": None,
                    "moe_parameter_count": sum(
                        parameter.numel() for parameter in model.moe.parameters()
                    ),
                    "router_parameter_count": sum(
                        parameter.numel() for parameter in model.moe.router.parameters()
                    ),
                    "expert_parameter_count_each": sum(
                        parameter.numel() for parameter in model.moe.experts[0].parameters()
                    ),
                })
        else:
            run_metadata.update({
                "presence_source": "Stage 02 predicted probabilities; train is in-sample prediction",
                "presence_columns": ["Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"],
                "presence_csv_sha256": {
                    split: sha256_file(args.presence_dir / f"predictions_{split}.csv") for split in split_counts
                },
            })
        save_json(config_path, run_metadata)
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_config(args, config)
        current_sha256 = sha256_file(next(
            pretrained_dir / name for name in ("model.safetensors", "pytorch_model.bin")
            if (pretrained_dir / name).is_file()
        ))
        if config.get("pretrained_sha256") != current_sha256:
            raise ValueError("Resume pretrained checkpoint SHA256 differs from the original run")
    best_miou = -1.0
    best_step = 0
    history: list[dict[str, float | int | None]] = []
    routing_history: list[dict[str, object]] = []
    train_routing = RoutingAccumulator() if moe_arm else None
    start_step = 0
    iterator_start_generator_state = loader_generator.get_state()
    iterator = iter(train_loader)
    batches_in_iterator = 0
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        required = {
            "schema_version", "arm", "step", "model", "optimizer", "scheduler", "scaler",
            "best_miou", "best_step", "history", "python_rng_state", "numpy_rng_state",
            "torch_rng_state", "loader_generator_state", "iterator_start_generator_state",
            "batches_in_iterator",
        }
        if moe_arm:
            required.add("routing_history")
        missing = required - checkpoint.keys()
        if missing or checkpoint["schema_version"] != 1 or checkpoint["arm"] != args.arm:
            raise ValueError(f"Invalid Stage 03/04/05 resume checkpoint; missing={sorted(missing)}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_miou = float(checkpoint["best_miou"])
        best_step = int(checkpoint["best_step"])
        history = list(checkpoint["history"])
        if moe_arm:
            routing_history = list(checkpoint["routing_history"])
        start_step = int(checkpoint["step"])
        if start_step >= args.max_steps:
            raise ValueError(f"Run already reached max_steps={args.max_steps}")

        iterator_start_generator_state = checkpoint["iterator_start_generator_state"]
        batches_in_iterator = int(checkpoint["batches_in_iterator"])
        loader_generator.set_state(iterator_start_generator_state)
        iterator = iter(train_loader)
        for _ in range(batches_in_iterator):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError("Saved train iterator position exceeds the loader length") from error
        loader_generator.set_state(checkpoint["loader_generator_state"])
        random.setstate(checkpoint["python_rng_state"])
        restore_numpy_rng_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"Resumed {args.arm} from step {start_step}", flush=True)

    for step in tqdm(range(start_step + 1, args.max_steps + 1), desc=f"Training {args.arm}"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator_start_generator_state = loader_generator.get_state()
                iterator = iter(train_loader)
                batches_in_iterator = 0
                batch = next(iterator)
            batches_in_iterator += 1
            pixels, target, conditioning = batch[:3]
            pixels, target, conditioning = pixels.to(device), target.to(device), conditioning.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                if moe_arm:
                    logits, routing_weights = model(pixels, conditioning, return_routing=True)
                else:
                    logits = model(pixels, conditioning)
                loss = segmentation_loss(logits, target) / args.grad_accum
            if train_routing is not None:
                train_routing.update(routing_weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            val = evaluate(model, val_loader, device, args.amp, collect_routing=moe_arm)
            score = val["miou"]
            if score is None:
                raise ValueError("Validation has no valid semantic pixels")
            history.append({"step": step, "train_loss": loss_total, "val_stratified_miou": score})
            save_json(args.output_dir / "history.json", history)
            if train_routing is not None:
                routing_history.append({
                    "step": step,
                    "train_observed_batches": train_routing.summary(),
                    "val_stratified": val["routing"],
                })
                save_json(args.output_dir / "routing_statistics.json", {
                    "history": routing_history,
                    "final_best": None,
                })
                train_routing = RoutingAccumulator()
            print(f"step={step} train_loss={loss_total:.4f} val_stratified_mIoU={score:.4f}", flush=True)
            if score > best_miou:
                best_miou, best_step = score, step
                torch.save({"model": model.state_dict(), "step": step, "arm": args.arm}, args.output_dir / "best.pt")
            save_resume_checkpoint(args.output_dir / "last.pt", {
                "schema_version": 1,
                "arm": args.arm,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_miou": best_miou,
                "best_step": best_step,
                "history": history,
                "routing_history": routing_history if moe_arm else [],
                "python_rng_state": random.getstate(),
                "numpy_rng_state": numpy_rng_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "loader_generator_state": loader_generator.get_state(),
                "iterator_start_generator_state": iterator_start_generator_state,
                "batches_in_iterator": batches_in_iterator,
            })
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    val_metrics = evaluate(model, val_loader, device, args.amp, collect_routing=moe_arm)
    domain_metrics = evaluate(model, domain_loader, device, args.amp, collect_routing=moe_arm)
    metrics = {
        "arm": args.arm, "best_step": best_step,
        "checkpoint_selection": "maximum val_stratified mIoU only",
        "val_stratified": val_metrics,
        "val_domain": domain_metrics,
    }
    save_json(args.output_dir / "metrics.json", metrics)
    if moe_arm:
        final_routing = {
            "checkpoint_step": best_step,
            "train": evaluate_embedding_store_routing(
                model, dino_store, "train", device, args.amp
            ),
            "val_stratified": val_metrics["routing"],
            "val_domain": domain_metrics["routing"],
        }
        save_json(args.output_dir / "routing_statistics.json", {
            "history": routing_history,
            "final_best": final_routing,
        })
    print(f"Saved {args.output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
