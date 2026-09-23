"""Run the required Stage 04 alignment and full-resolution forward/backward smoke."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.dino.embedding import select_device
from src.segmentation.data import encode_image_mask, load_image_mask
from src.segmentation.dino_conditioning import (
    DINO_DIM,
    DirectDinoEmbeddingStore,
    DirectDinoSegformer,
    DirectDinoSegmentationDataset,
)
from src.segmentation.experiment import resolve_pretrained, segmentation_loss
from src.segmentation.geometry import (
    TRANSFORM_STATES,
    apply_stage03_geometry,
    apply_transform,
    canonical_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps"))
    parser.add_argument(
        "--augmented-dino-dir", type=Path,
        default=Path("outputs/dinov2_518/augmented_train_smoke"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/direct_dino_conditioning/smoke"),
    )
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def legacy_geometry(
    image: np.ndarray, mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if rng.random() < 0.5:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    turns = rng.randrange(4)
    return np.rot90(image, turns), np.rot90(mask, turns)


def check_geometry() -> dict[str, int]:
    asymmetric = np.arange(5 * 7).reshape(5, 7)
    counts = {state: 0 for state in TRANSFORM_STATES}
    for horizontal, vertical, turns in itertools.product((False, True), (False, True), range(4)):
        direct = asymmetric
        if horizontal:
            direct = np.flip(direct, 1)
        if vertical:
            direct = np.flip(direct, 0)
        direct = np.rot90(direct, turns)
        state = canonical_state(horizontal, vertical, turns)
        if not np.array_equal(direct, apply_transform(asymmetric, state)):
            raise AssertionError(f"Canonical geometry mismatch: {horizontal}/{vertical}/{turns}/{state}")
        counts[state] += 1
    if set(counts.values()) != {2}:
        raise AssertionError(f"Expected two of 16 source combinations per D4 state: {counts}")

    image = np.arange(9 * 11 * 3, dtype=np.int64).reshape(9, 11, 3)
    mask = np.arange(9 * 11, dtype=np.int64).reshape(9, 11)
    for seed in range(100):
        actual_image, actual_mask, _ = apply_stage03_geometry(
            image, mask, random.Random(seed)
        )
        expected_image, expected_mask = legacy_geometry(image, mask, random.Random(seed))
        if not np.array_equal(actual_image, expected_image) or not np.array_equal(actual_mask, expected_mask):
            raise AssertionError(f"Stage 03 augmentation changed at seed {seed}")
    return counts


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    geometry_counts = check_geometry()
    store = DirectDinoEmbeddingStore(
        args.data_dir, args.embedding_dir, args.augmented_dino_dir,
        require_full_augmented=False,
    )
    smoke_id = next(iter(store.augmented_rows))
    train = DirectDinoSegmentationDataset(
        args.data_dir, "train", store, augment=True, image_ids=[smoke_id]
    )
    random.seed(42)
    pixels, target, embedding, image_id, state = train[0]
    random.seed(42)
    original_image, original_mask = load_image_mask(args.data_dir, smoke_id)
    expected_image, expected_mask, expected_state = apply_stage03_geometry(
        original_image, original_mask, random
    )
    expected_pixels, expected_target = encode_image_mask(expected_image, expected_mask)
    if image_id != smoke_id or state != expected_state:
        raise AssertionError("Dataset image ID or transform state does not match the sampled geometry")
    if not torch.equal(pixels, expected_pixels) or not torch.equal(target, expected_target):
        raise AssertionError("Dataset image/mask augmentation differs from its recorded state")
    if not np.array_equal(
        embedding.numpy(), store.embedding(image_id, "train", state)
    ):
        raise AssertionError("Dataset selected an embedding from the wrong image/state pair")
    if pixels.shape != (3, 1024, 1024) or target.shape != (1024, 1024):
        raise AssertionError("Segmentation smoke is not using the full 1024x1024 input")
    if embedding.shape != (DINO_DIM,):
        raise AssertionError("Direct DINO sample must have 1536 features")

    validation_states: dict[str, str] = {}
    for split in ("val_stratified", "val_domain"):
        validation = DirectDinoSegmentationDataset(
            args.data_dir, split, store, augment=False, image_ids=[store.split_ids[split][0]]
        )
        _, _, val_embedding, val_id, val_state = validation[0]
        if val_state != "r0" or not np.array_equal(
            val_embedding.numpy(), store.embedding(val_id, split, "r0")
        ):
            raise AssertionError(f"{split} did not use its original r0 embedding")
        validation_states[split] = val_state

    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    model = DirectDinoSegformer(str(pretrained_dir)).to(device)
    model.encoder.gradient_checkpointing_enable()
    model.train()
    batch_pixels = pixels.unsqueeze(0).to(device)
    batch_target = target.unsqueeze(0).to(device)
    batch_embedding = embedding.unsqueeze(0).to(device)
    gamma, beta = model.film.modulation(batch_embedding)
    adapter_output_shape = [batch_embedding.shape[0], gamma.shape[1] + beta.shape[1]]
    if gamma.shape != (1, 768) or beta.shape != (1, 768) or adapter_output_shape != [1, 1536]:
        raise AssertionError("Direct DINO adapter output or gamma/beta shape is wrong")
    probe_feature = torch.randn((1, 768, 2, 3), device=device)
    if model.film(probe_feature, batch_embedding).shape != probe_feature.shape:
        raise AssertionError("FiLM spatial broadcasting failed")

    logits = model(batch_pixels, batch_embedding)
    if logits.shape != (1, 8, 1024, 1024):
        raise AssertionError(f"Unexpected segmentation logits shape: {tuple(logits.shape)}")
    loss = segmentation_loss(logits, batch_target)
    if not torch.isfinite(loss):
        raise AssertionError("Segmentation loss is non-finite")
    ignore_probe_target = batch_target.clone()
    ignore_probe_target[:, 0, 0] = 255
    per_pixel = F.cross_entropy(logits.detach(), ignore_probe_target, ignore_index=255, reduction="none")
    ignored = ignore_probe_target == 255
    if not ignored.any() or not torch.all(per_pixel[ignored] == 0):
        raise AssertionError("Ignore label 255 contributes to Cross Entropy")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise AssertionError("Backward produced no trainable gradients")
    if any("presence" in name.lower() for name, _ in model.named_parameters()):
        raise AssertionError("Direct DINO model unexpectedly contains Presence parameters")
    if any("dinov2" in type(module).__name__.lower() for module in model.modules()):
        raise AssertionError("Frozen DINOv2 was included in the segmentation backward graph")

    conditioning_parameters = sum(parameter.numel() for parameter in model.film.parameters())
    if conditioning_parameters != 791296:
        raise AssertionError(f"Unexpected Direct DINO adapter parameter count: {conditioning_parameters}")
    report = {
        "status": "pass",
        "device": str(device),
        "checks": {
            "segmentation_input_shape": list(batch_pixels.shape),
            "augmentation_matches_recorded_state": True,
            "sample_image_id": image_id,
            "sample_transform_state": state,
            "image_embedding_pair_exact": True,
            "dino_input_shape": list(batch_embedding.shape),
            "conditioning_output_shape": adapter_output_shape,
            "gamma_shape": list(gamma.shape),
            "beta_shape": list(beta.shape),
            "film_broadcast": True,
            "logits_shape": list(logits.shape),
            "ignore_255_excluded": True,
            "loss": float(loss.detach().cpu()),
            "loss_finite": True,
            "backward": True,
            "dinov2_in_backward_graph": False,
            "validation_states": validation_states,
            "reads_gt_presence": False,
            "loads_presence_probability": False,
        },
        "geometry": {
            "legacy_stage03_equivalent_for_100_seeds": True,
            "source_combinations_per_canonical_state": geometry_counts,
            "canonical_probability": {state_name: count / 16 for state_name, count in geometry_counts.items()},
        },
        "conditioning_parameter_count": conditioning_parameters,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "smoke.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
