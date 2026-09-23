"""Fixed-split image/mask loading with strictly aligned predicted presence."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.presence.labels import CLASS_NAMES, SPLITS
from src.segmentation.geometry import apply_stage03_geometry


FOREGROUND_NAMES = CLASS_NAMES[1:]
IMAGE_SIZE = (1024, 1024)
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def load_image_mask(data_dir: Path, image_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one immutable full-resolution image/mask pair and validate it."""
    with Image.open(data_dir / "train" / "images" / f"{image_id}.png") as file:
        image = np.asarray(file.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(data_dir / "train" / "masks" / f"{image_id}.png") as file:
        mask = np.asarray(file, dtype=np.uint8).copy()
    if image.shape != (*IMAGE_SIZE, 3) or mask.shape != IMAGE_SIZE:
        raise ValueError(f"{image_id}: expected 1024x1024 image and mask")
    if np.any(mask > 8):
        raise ValueError(f"{image_id}: mask contains invalid class ID")
    return image, mask


def encode_image_mask(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Stage 03 normalization and convert raw labels to train targets."""
    pixels = (image.astype(np.float32) / 255.0 - MEAN) / STD
    target = np.where(mask == 0, 255, mask.astype(np.int64) - 1)
    return (
        torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
        torch.from_numpy(np.ascontiguousarray(target)),
    )


def load_aligned_presence(data_dir: Path, presence_dir: Path) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Check count, order, IDs, duplicates and values against every fixed split."""
    aligned: dict[str, list[tuple[str, np.ndarray]]] = {}
    seen_global: set[str] = set()
    expected_columns = ["image_id", "split", *CLASS_NAMES]
    for split in SPLITS:
        ids = (data_dir / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{split}: empty split or duplicate image IDs")
        if seen_global.intersection(ids):
            raise ValueError(f"{split}: image IDs overlap another split")
        seen_global.update(ids)
        csv_path = presence_dir / f"predictions_{split}.csv"
        rows: list[tuple[str, np.ndarray]] = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise ValueError(f"{csv_path}: expected columns {expected_columns}, got {reader.fieldnames}")
            seen_rows: set[str] = set()
            for row in reader:
                image_id = row["image_id"]
                if image_id in seen_rows:
                    raise ValueError(f"{csv_path}: duplicate ID {image_id}")
                seen_rows.add(image_id)
                if row["split"] != split:
                    raise ValueError(f"{csv_path}: wrong split for {image_id}")
                try:
                    probabilities = np.asarray([float(row[name]) for name in FOREGROUND_NAMES], dtype=np.float32)
                    background = float(row["Background"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{csv_path}: invalid probability for {image_id}") from error
                if probabilities.shape != (7,) or not np.isfinite(probabilities).all():
                    raise ValueError(f"{csv_path}: non-finite or wrongly shaped probability for {image_id}")
                if not np.isfinite(background) or not 0 <= background <= 1 or np.any(probabilities < 0) or np.any(probabilities > 1):
                    raise ValueError(f"{csv_path}: probability outside [0, 1] for {image_id}")
                rows.append((image_id, probabilities))
        if len(rows) != len(ids) or [image_id for image_id, _ in rows] != ids:
            raise ValueError(f"{csv_path}: count or ID order differs from {split}.txt")
        aligned[split] = rows
    return aligned


class SegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    """Load full 1024-pixel images; never derive conditioning from masks."""

    def __init__(self, data_dir: Path, rows: list[tuple[str, np.ndarray]], augment: bool) -> None:
        self.data_dir = data_dir
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_id, probabilities = self.rows[index]
        image, mask = load_image_mask(self.data_dir, image_id)
        if self.augment:
            image, mask, _ = apply_stage03_geometry(image, mask, random)
        pixels, target = encode_image_mask(image, mask)
        return (
            pixels,
            target,
            torch.from_numpy(probabilities.copy()),
            image_id,
        )
