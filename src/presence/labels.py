"""Align eight-class presence labels with frozen DINOv2 embeddings."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


CLASS_NAMES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
)
SPLITS = ("train", "val_stratified", "val_domain")


@dataclass(frozen=True)
class Sample:
    row: int
    image_id: str
    split: str


def load_samples(
    data_dir: Path, embedding_dir: Path, limit_per_split: int | None
) -> tuple[list[Sample], dict[str, object]]:
    """Verify exact fixed-split ordering before selecting any smoke subset."""
    metadata = json.loads((embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("scope") != "full" or metadata.get("count") != 6996:
        raise ValueError("Stage 02 requires the accepted full 518 embedding")
    if metadata.get("preprocessing", {}).get("resize_size") != [518, 518]:
        raise ValueError("Embedding was not generated with 518x518 input")
    with (embedding_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    expected: list[tuple[str, str]] = []
    for split in SPLITS:
        path = data_dir / "splits" / f"{split}.txt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["split_sha256"][split]:
            raise ValueError(f"Split changed since embedding extraction: {path}")
        expected.extend((image_id, split) for image_id in path.read_text().splitlines())
    if len(manifest) != len(expected) or len(expected) != 6996:
        raise ValueError("Embedding manifest and fixed splits have different lengths")
    samples: list[Sample] = []
    selected_counts = dict.fromkeys(SPLITS, 0)
    for row, (entry, (image_id, split)) in enumerate(zip(manifest, expected)):
        if (
            int(entry["row"]) != row
            or entry["image_id"] != image_id
            or entry["split"] != split
            or entry["image_path"] != f"data/train/images/{image_id}.png"
        ):
            raise ValueError(f"Embedding manifest mismatch at row {row}")
        if limit_per_split is None or selected_counts[split] < limit_per_split:
            samples.append(Sample(row, image_id, split))
            selected_counts[split] += 1
    return samples, metadata


def build_labels(data_dir: Path, samples: list[Sample]) -> np.ndarray:
    """Use raw mask IDs 1..8; Ignore (0) contributes to no presence bit."""
    labels = np.zeros((len(samples), len(CLASS_NAMES)), dtype=np.uint8)
    for index, sample in enumerate(tqdm(samples, desc="Presence labels", unit="mask")):
        path = data_dir / "train" / "masks" / f"{sample.image_id}.png"
        with Image.open(path) as mask:
            mask.load()
            if mask.size != (1024, 1024) or mask.mode != "L":
                raise ValueError(f"Unexpected mask size/mode: {path}")
            histogram = mask.histogram()
        if any(histogram[9:]):
            raise ValueError(f"Mask has illegal class ID: {path}")
        labels[index] = np.asarray(histogram[1:9], dtype=np.int64) > 0
    return labels


def split_indices(samples: list[Sample]) -> dict[str, np.ndarray]:
    return {
        split: np.asarray([i for i, sample in enumerate(samples) if sample.split == split])
        for split in SPLITS
    }
