"""One full-resolution Stage 03 forward/backward for each experiment arm."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dino.embedding import select_device
from src.segmentation.data import SegmentationDataset, load_aligned_presence
from src.segmentation.experiment import evaluate, resolve_pretrained, save_json, seed_everything, segmentation_loss
from src.segmentation.model import PresenceSegformer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--presence-dir", type=Path, default=Path("outputs/class_presence/linear_518"))
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/presence_conditioning/smoke"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output_dir.resolve().exists():
        parser.error("Output directory already exists; use a fresh smoke directory")
    aligned = load_aligned_presence(args.data_dir, args.presence_dir)
    device = select_device(args.device)
    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    pixels, target, probability, image_id = SegmentationDataset(
        args.data_dir, aligned["train"][:1], augment=False
    )[0]
    assert pixels.shape == (3, 1024, 1024) and target.shape == (1024, 1024)
    assert probability.shape == (7,) and torch.isfinite(probability).all()
    assert image_id == aligned["train"][0][0]
    # A changed ignore pixel cannot affect CE at any valid pixel.
    synthetic_logits = torch.randn(1, 8, 2, 2)
    synthetic_target = torch.tensor([[[255, 1], [2, 3]]])
    ignore_loss = segmentation_loss(synthetic_logits, synthetic_target)
    synthetic_logits[:, :, 0, 0] += 1000
    assert torch.allclose(ignore_loss, segmentation_loss(synthetic_logits, synthetic_target))
    results: dict[str, object] = {
        "device": str(device), "image_id": image_id,
        "image_shape": list(pixels.shape), "probability_shape": list(probability.shape),
        "aligned_split_counts": {split: len(rows) for split, rows in aligned.items()},
        "ignore_index_excluded": True,
        "presence_source": "Stage 02 predicted probabilities; train is in-sample prediction",
    }
    for arm in ("control", "conditioned"):
        seed_everything(42)
        model = PresenceSegformer(str(pretrained_dir), arm == "conditioned").to(device)
        model.encoder.gradient_checkpointing_enable()
        captured: list[tuple[int, ...]] = []
        if model.film is not None:
            model.film.mlp.register_forward_hook(lambda _module, _input, output: captured.append(tuple(output.shape)))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
        initial_classifier = model.decode_head.classifier.weight.detach().clone()
        logits = model(pixels.unsqueeze(0).to(device), probability.unsqueeze(0).to(device))
        assert logits.shape == (1, 8, 1024, 1024)
        loss = segmentation_loss(logits, target.unsqueeze(0).to(device))
        assert torch.isfinite(loss)
        loss.backward()
        assert model.decode_head.classifier.weight.grad is not None
        if model.film is not None:
            assert captured and captured[-1] == (1, 2 * model.film.channels)
            assert model.film.mlp[-1].weight.grad is not None
        optimizer.step()
        assert not torch.equal(initial_classifier, model.decode_head.classifier.weight)
        val_loader = DataLoader(SegmentationDataset(args.data_dir, aligned["val_stratified"][:1], augment=False))
        val_metrics = evaluate(model, val_loader, device, amp=False)
        assert val_metrics["miou"] is not None
        results[arm] = {
            "logits_shape": list(logits.shape), "loss": float(loss.detach().cpu()),
            "backward": True, "optimizer_step": True,
            "film_output_shape": list(captured[-1]) if captured else None,
            "one_image_val_stratified_miou": val_metrics["miou"],
        }
        print(f"{arm}: full-resolution forward/backward passed, loss={float(loss.detach()):.4f}", flush=True)
        del model, optimizer, logits, loss, initial_classifier
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    save_json(args.output_dir / "smoke.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
