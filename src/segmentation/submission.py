"""AIC test-image loading and strictly validated segmentation submission output."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.dino.embedding import ProcessorConfig, preprocess_rgb
from src.presence.labels import CLASS_NAMES
from src.segmentation.data import IMAGE_SIZE, MEAN, STD


EXPECTED_TEST_COUNT = 1300
OFFICIAL_LABEL_NAMES = {
    0: "Ignore",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren",
    6: "Vegetation",
    7: "Agricultural",
    8: "Vehicle",
}
TRAIN_TO_OFFICIAL = np.arange(1, 9, dtype=np.uint8)


def audit_label_definition(label_file: Path) -> list[dict[str, object]]:
    """Require Label.txt and the model's eight output channels to agree exactly."""
    matches = re.findall(
        r'(\d+)\s*:\s*["\']([^"\']+)["\']',
        label_file.read_text(encoding="utf-8-sig"),
    )
    found = {int(class_id): name for class_id, name in matches}
    if len(matches) != len(OFFICIAL_LABEL_NAMES) or found != OFFICIAL_LABEL_NAMES:
        raise ValueError(
            f"Label definition mismatch: found {found}, expected {OFFICIAL_LABEL_NAMES}"
        )
    expected_train_names = tuple(OFFICIAL_LABEL_NAMES[index] for index in range(1, 9))
    if tuple(CLASS_NAMES) != expected_train_names:
        raise ValueError(
            f"Internal model class order differs from Label.txt: {CLASS_NAMES} != {expected_train_names}"
        )
    return [
        {
            "model_channel": channel,
            "official_id": int(TRAIN_TO_OFFICIAL[channel]),
            "class_name": OFFICIAL_LABEL_NAMES[int(TRAIN_TO_OFFICIAL[channel])],
        }
        for channel in range(len(TRAIN_TO_OFFICIAL))
    ]


def collect_test_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    """Return unique test PNGs in stable filename order after a full input audit."""
    paths = sorted(image_dir.glob("*.png"), key=lambda path: path.name)
    if len(paths) != EXPECTED_TEST_COUNT or len({path.name for path in paths}) != len(paths):
        raise ValueError(
            f"Expected {EXPECTED_TEST_COUNT} unique test PNGs in {image_dir}, got {len(paths)}"
        )
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != IMAGE_SIZE:
                raise ValueError(
                    f"Expected 1024x1024 RGB PNG, got {image.format}/{image.mode}/{image.size}: {path}"
                )
    return paths if limit is None else paths[:limit]


class V3TestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Prepare the same full image for SegFormer and frozen DINOv2 r0 inference."""

    def __init__(self, paths: list[Path], processor: ProcessorConfig) -> None:
        if not paths:
            raise ValueError("Test dataset is empty")
        self.paths = paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            rgb = np.asarray(image, dtype=np.uint8).copy()
        pixels = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        segmentation = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        dino = preprocess_rgb(rgb, self.processor)
        return segmentation, dino, path.name


def save_official_prediction(prediction: np.ndarray, target: Path) -> Counter[int]:
    """Map model channels 0..7 explicitly to official IDs 1..8."""
    if prediction.shape != IMAGE_SIZE:
        raise ValueError(f"Prediction must be 1024x1024, got {prediction.shape}")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError(f"Prediction must contain integer class IDs, got {prediction.dtype}")
    if int(prediction.min()) < 0 or int(prediction.max()) > 7:
        raise ValueError("Internal prediction IDs must be within 0..7")
    official = TRAIN_TO_OFFICIAL[prediction]
    Image.fromarray(official, mode="L").save(target, format="PNG", optimize=False)
    values, counts = np.unique(official, return_counts=True)
    return Counter(dict(zip(values.tolist(), counts.tolist())))


def audit_prediction_directory(
    inputs: list[Path], prediction_dir: Path
) -> dict[str, object]:
    """Verify exact filenames, PNG encoding, size and official label range."""
    outputs = sorted(prediction_dir.glob("*.png"), key=lambda path: path.name)
    input_names = [path.name for path in inputs]
    output_names = [path.name for path in outputs]
    if output_names != input_names:
        missing = sorted(set(input_names) - set(output_names))[:10]
        extra = sorted(set(output_names) - set(input_names))[:10]
        raise ValueError(f"Prediction filenames differ from test inputs; missing={missing}, extra={extra}")
    unexpected = [
        path.name for path in prediction_dir.iterdir()
        if not path.is_file() or path.suffix.lower() != ".png"
    ]
    if unexpected:
        raise ValueError(f"Prediction directory contains unexpected entries: {unexpected[:10]}")

    histogram: Counter[int] = Counter()
    for path in outputs:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.palette is not None:
                raise ValueError(f"Submission mask must be non-palette grayscale PNG: {path}")
            if image.size != IMAGE_SIZE:
                raise ValueError(f"Submission mask must be 1024x1024: {path}")
            array = np.asarray(image, dtype=np.uint8)
        values, counts = np.unique(array, return_counts=True)
        if int(values.min()) < 1 or int(values.max()) > 8:
            raise ValueError(f"Submission mask IDs must be within 1..8: {path}")
        histogram.update(dict(zip(values.tolist(), counts.tolist())))
    return {
        "file_count": len(outputs),
        "filenames_exact_match": True,
        "format": "PNG",
        "mode": "L (single-channel grayscale, no palette)",
        "size": [1024, 1024],
        "allowed_ids": [1, 8],
        "observed_ids": sorted(histogram),
        "pixel_histogram": {str(index): int(histogram.get(index, 0)) for index in range(1, 9)},
    }


def build_submission_zip(prediction_dir: Path, target: Path) -> None:
    """Put every prediction PNG directly at ZIP root and verify CRC."""
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(prediction_dir.glob("*.png"), key=lambda item: item.name):
            archive.write(path, arcname=path.name)
    with ZipFile(target) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        if any(PurePosixPath(info.filename).parent != PurePosixPath(".") for info in entries):
            raise ValueError("Submission ZIP contains a nested directory")
        if any(PurePosixPath(info.filename).suffix.lower() != ".png" for info in entries):
            raise ValueError("Submission ZIP contains a non-PNG file")
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP CRC validation failed")
