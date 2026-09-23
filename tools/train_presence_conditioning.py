"""Train one Stage 03 SegFormer-B3 arm with fixed splits and predicted presence."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dino.embedding import select_device
from src.dino.download import sha256_file
from src.segmentation.data import SegmentationDataset, load_aligned_presence
from src.segmentation.experiment import evaluate, resolve_pretrained, save_json, seed_everything, segmentation_loss
from src.segmentation.model import PresenceSegformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "conditioned"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--presence-dir", type=Path, default=Path("outputs/class_presence/linear_518"))
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
    args = parser.parse_args()
    for name in ("max_steps", "eval_every", "batch_size", "grad_accum"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Invalid worker count, learning rate or weight decay")
    if args.output_dir.resolve().exists():
        parser.error("Output directory already exists; use a fresh run directory")
    if args.data_dir.resolve() in args.output_dir.resolve().parents:
        parser.error("Output directory cannot be under read-only data/")
    return args


def main() -> int:
    args = parse_args()
    aligned = load_aligned_presence(args.data_dir, args.presence_dir)
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently supports CUDA only")
    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    seed_everything(args.seed)
    model = PresenceSegformer(str(pretrained_dir), args.arm == "conditioned").to(device)
    model.encoder.gradient_checkpointing_enable()
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SegmentationDataset(args.data_dir, aligned["train"], augment=True),
        batch_size=args.batch_size, shuffle=True, generator=loader_generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        SegmentationDataset(args.data_dir, aligned["val_stratified"], augment=False),
        batch_size=1, num_workers=args.num_workers,
    )
    domain_loader = DataLoader(
        SegmentationDataset(args.data_dir, aligned["val_domain"], augment=False),
        batch_size=1, num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pretrained_path": str(pretrained_dir), "pretrained_model": "nvidia/mit-b3",
        "pretrained_revision": pretrained_dir.name if pretrained_dir.parent.name == "snapshots" else None,
        "pretrained_sha256": sha256_file(next(
            pretrained_dir / name for name in ("model.safetensors", "pytorch_model.bin")
            if (pretrained_dir / name).is_file()
        )),
        "input_size": [1024, 1024], "num_classes": 8, "ignore_index": 255,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "presence_source": "Stage 02 predicted probabilities; train is in-sample prediction",
        "presence_columns": ["Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"],
        "split_counts": {key: len(rows) for key, rows in aligned.items()},
        "split_sha256": {
            split: sha256_file(args.data_dir / "splits" / f"{split}.txt") for split in aligned
        },
        "presence_csv_sha256": {
            split: sha256_file(args.presence_dir / f"predictions_{split}.csv") for split in aligned
        },
    })
    best_miou = -1.0
    best_step = 0
    iterator = iter(train_loader)
    history: list[dict[str, float | int | None]] = []
    for step in tqdm(range(1, args.max_steps + 1), desc=f"Training {args.arm}"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.grad_accum):
            try:
                pixels, target, probability, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                pixels, target, probability, _ = next(iterator)
            pixels, target, probability = pixels.to(device), target.to(device), probability.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(pixels, probability)
                loss = segmentation_loss(logits, target) / args.grad_accum
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            loss_total += float(loss.detach())
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            val = evaluate(model, val_loader, device, args.amp)
            score = val["miou"]
            if score is None:
                raise ValueError("Validation has no valid semantic pixels")
            history.append({"step": step, "train_loss": loss_total, "val_stratified_miou": score})
            save_json(args.output_dir / "history.json", history)
            print(f"step={step} train_loss={loss_total:.4f} val_stratified_mIoU={score:.4f}", flush=True)
            if score > best_miou:
                best_miou, best_step = score, step
                torch.save({"model": model.state_dict(), "step": step, "arm": args.arm}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    metrics = {
        "arm": args.arm, "best_step": best_step,
        "checkpoint_selection": "maximum val_stratified mIoU only",
        "val_stratified": evaluate(model, val_loader, device, args.amp),
        "val_domain": evaluate(model, domain_loader, device, args.amp),
    }
    save_json(args.output_dir / "metrics.json", metrics)
    print(f"Saved {args.output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
