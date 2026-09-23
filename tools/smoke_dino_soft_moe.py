"""Run the required Stage 05 full-resolution DINO Soft MoE smoke test."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.dino.embedding import select_device
from src.segmentation.data import encode_image_mask, load_image_mask
from src.segmentation.dino_conditioning import (
    DirectDinoEmbeddingStore,
    DirectDinoSegmentationDataset,
)
from src.segmentation.dino_moe import DinoSoftMoESegformer, RoutingAccumulator
from src.segmentation.experiment import resolve_pretrained, segmentation_loss
from src.segmentation.geometry import apply_stage03_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps"))
    parser.add_argument(
        "--augmented-dino-dir", type=Path,
        default=Path("outputs/dinov2_518/augmented_train_smoke"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dino_moe/smoke"))
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def gradient_norm(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().float().norm().cpu())
        for parameter in module.parameters() if parameter.grad is not None
    )


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    store = DirectDinoEmbeddingStore(
        args.data_dir, args.embedding_dir, args.augmented_dino_dir,
        require_full_augmented=False,
    )
    image_id = next(iter(store.augmented_rows))
    train = DirectDinoSegmentationDataset(
        args.data_dir, "train", store, augment=True, image_ids=[image_id]
    )
    random.seed(42)
    pixels, target, embedding, returned_id, state = train[0]
    random.seed(42)
    original_image, original_mask = load_image_mask(args.data_dir, image_id)
    expected_image, expected_mask, expected_state = apply_stage03_geometry(
        original_image, original_mask, random
    )
    expected_pixels, expected_target = encode_image_mask(expected_image, expected_mask)
    if (
        returned_id != image_id
        or state != expected_state
        or not torch.equal(pixels, expected_pixels)
        or not torch.equal(target, expected_target)
        or not np.array_equal(embedding.numpy(), store.embedding(image_id, "train", state))
    ):
        raise AssertionError("Stage 05 image, state and DINO embedding are not strictly aligned")

    for split in ("val_stratified", "val_domain"):
        validation = DirectDinoSegmentationDataset(
            args.data_dir, split, store, augment=False, image_ids=[store.split_ids[split][0]]
        )
        _, _, val_embedding, val_id, val_state = validation[0]
        if val_state != "r0" or not np.array_equal(
            val_embedding.numpy(), store.embedding(val_id, split, "r0")
        ):
            raise AssertionError(f"{split} must use the original r0 embedding")

    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    model = DinoSoftMoESegformer(str(pretrained_dir)).to(device)
    model.encoder.gradient_checkpointing_enable()
    model.train()
    batch_pixels = pixels.unsqueeze(0).to(device)
    batch_target = target.unsqueeze(0).to(device)
    batch_embedding = embedding.unsqueeze(0).to(device)
    if batch_pixels.shape != (1, 3, 1024, 1024) or batch_embedding.shape != (1, 1536):
        raise AssertionError("Stage 05 smoke must use full 1024 segmentation input and 1536 DINO input")

    feature = model.decode_feature(batch_pixels)
    film_feature = model.film(feature, batch_embedding)
    if film_feature.shape[1] != 768:
        raise AssertionError("Direct DINO FiLM feature must have 768 channels")
    moe_feature, router_logits, weights = model.moe(film_feature, batch_embedding)
    if router_logits.shape != (1, 4) or weights.shape != (1, 4):
        raise AssertionError("Router logits and weights must have shape [B, 4]")
    if not torch.allclose(weights.sum(dim=1), torch.ones(1, device=device), atol=1e-5):
        raise AssertionError("Soft routing weights do not sum to one")
    if not torch.all(weights > 0) or torch.any(weights == 1):
        raise AssertionError("Stage 05 must execute dense soft routing without hard/Top-k weights")

    expert_shapes = []
    with torch.no_grad():
        manual_delta = torch.zeros_like(film_feature)
        for index, expert in enumerate(model.moe.experts):
            expert_output = expert(film_feature)
            expert_shapes.append(list(expert_output.shape))
            if expert_output.shape != film_feature.shape:
                raise AssertionError(f"Expert {index} output does not match F_film")
            manual_delta = manual_delta + weights[:, index, None, None, None] * expert_output
    if not torch.allclose(moe_feature, film_feature + manual_delta, atol=1e-6, rtol=1e-5):
        raise AssertionError("Weighted residual MoE fusion is incorrect")
    initial_delta_mean_abs = float(manual_delta.detach().float().abs().mean().cpu())
    if initial_delta_mean_abs >= 0.01:
        raise AssertionError("Small Expert initialization does not preserve the initial v3 feature")

    logits = model.classify_feature(moe_feature, batch_pixels.shape[-2:])
    if logits.shape != (1, 8, 1024, 1024):
        raise AssertionError(f"Classifier output must be [B,8,H,W], got {tuple(logits.shape)}")
    loss = segmentation_loss(logits, batch_target)
    if not torch.isfinite(loss):
        raise AssertionError("Stage 05 loss is non-finite")
    ignore_probe = batch_target.clone()
    ignore_probe[:, 0, 0] = 255
    per_pixel = F.cross_entropy(logits.detach(), ignore_probe, ignore_index=255, reduction="none")
    if not torch.all(per_pixel[ignore_probe == 255] == 0):
        raise AssertionError("Ignore index 255 contributes to Cross Entropy")
    loss.backward()
    router_gradient_norm = gradient_norm(model.moe.router)
    expert_gradient_norms = [gradient_norm(expert) for expert in model.moe.experts]
    if router_gradient_norm <= 0 or any(value <= 0 for value in expert_gradient_norms):
        raise AssertionError("Router and every Expert must receive non-zero gradients")
    if any("dinov2" in type(module).__name__.lower() for module in model.modules()):
        raise AssertionError("Frozen DINOv2 was included in the segmentation backward graph")
    if any("presence" in name.lower() for name, _ in model.named_parameters()):
        raise AssertionError("Stage 05 unexpectedly contains Presence parameters")

    routing_accumulator = RoutingAccumulator()
    routing_accumulator.update(weights)
    routing_summary = routing_accumulator.summary()
    router_parameters = sum(parameter.numel() for parameter in model.moe.router.parameters())
    expert_parameters = [sum(parameter.numel() for parameter in expert.parameters()) for expert in model.moe.experts]
    moe_parameters = sum(parameter.numel() for parameter in model.moe.parameters())
    if router_parameters != 397572 or expert_parameters != [295872] * 4 or moe_parameters != 1581060:
        raise AssertionError("Stage 05 Router or Expert parameter count differs from the fixed design")

    report = {
        "status": "pass",
        "device": str(device),
        "checks": {
            "segmentation_input_shape": list(batch_pixels.shape),
            "dino_input_shape": list(batch_embedding.shape),
            "augmentation_embedding_alignment": True,
            "sample_image_id": image_id,
            "sample_transform_state": state,
            "validation_embedding_state": "r0",
            "film_feature_shape": list(film_feature.shape),
            "router_logits_shape": list(router_logits.shape),
            "routing_weights_shape": list(weights.shape),
            "routing_weights": weights.detach().float().cpu().tolist()[0],
            "routing_weights_sum": float(weights.detach().float().sum().cpu()),
            "expert_output_shapes": expert_shapes,
            "weighted_fusion": True,
            "residual_fusion": True,
            "initial_delta_mean_abs": initial_delta_mean_abs,
            "logits_shape": list(logits.shape),
            "ignore_255_excluded": True,
            "loss": float(loss.detach().cpu()),
            "loss_finite": True,
            "backward": True,
            "router_gradient_norm": router_gradient_norm,
            "expert_gradient_norms": expert_gradient_norms,
            "dinov2_in_backward_graph": False,
            "reads_gt_presence": False,
            "hard_or_topk_routing": False,
        },
        "routing_summary": routing_summary,
        "parameter_counts": {
            "router": router_parameters,
            "expert_each": expert_parameters[0],
            "experts_total": sum(expert_parameters),
            "moe_total": moe_parameters,
            "direct_dino_film": sum(parameter.numel() for parameter in model.film.parameters()),
            "trainable_total": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
