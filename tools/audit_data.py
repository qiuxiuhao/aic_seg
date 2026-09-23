"""Read-only audit of the fixed UAV segmentation dataset and splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = (
    "Ignore",
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
EXPECTED_SIZE = (1024, 1024)


class PairInfo(NamedTuple):
    counts: list[int]
    image_size: tuple[int, int]
    mask_size: tuple[int, int]
    image_mode: str
    mask_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_audit"))
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Decode only the first N IDs per split for a smoke check.",
    )
    args = parser.parse_args()
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("--limit-per-split must be positive")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if data_dir == output_dir or data_dir in output_dir.parents:
        parser.error("--output-dir must be outside the read-only data directory")
    return args


def read_label_names(path: Path, errors: list[str]) -> None:
    """Compare the on-disk label definition with the project contract."""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        errors.append(f"Cannot read {path}: {exc}")
        return
    found = {int(key): name for key, name in re.findall(r'(\d+)\s*:\s*["\']([^"\']+)["\']', content)}
    expected = dict(enumerate(CLASS_NAMES))
    if found != expected:
        errors.append(f"Label definition mismatch: found {found}, expected {expected}")


def collect_files(directory: Path, errors: list[str]) -> dict[str, Path]:
    """Index PNGs by stem and record files with unexpected names or types."""
    indexed: dict[str, Path] = {}
    try:
        files = sorted(directory.iterdir())
    except OSError as exc:
        errors.append(f"Cannot list {directory}: {exc}")
        return indexed
    for path in files:
        if not path.is_file():
            errors.append(f"Unexpected non-file entry: {path}")
        elif path.suffix.lower() != ".png":
            errors.append(f"Unexpected non-PNG file: {path}")
        elif path.stem in indexed:
            errors.append(f"Duplicate image ID in {directory}: {path.stem}")
        else:
            indexed[path.stem] = path
    return indexed


def read_splits(data_dir: Path, errors: list[str]) -> dict[str, list[str]]:
    """Read the existing split files without changing their contents."""
    splits: dict[str, list[str]] = {}
    for name in SPLITS:
        path = data_dir / "splits" / f"{name}.txt"
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            errors.append(f"Cannot read {path}: {exc}")
            splits[name] = []
            continue
        ids = [line.strip() for line in lines]
        if any(not item for item in ids):
            errors.append(f"Blank ID in {path}")
        ids = [item for item in ids if item]
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            errors.append(f"Duplicate IDs in {path}: {duplicates}")
        splits[name] = ids
    return splits


def check_index(
    images: dict[str, Path],
    masks: dict[str, Path],
    splits: dict[str, list[str]],
    errors: list[str],
) -> dict[str, object]:
    """Check file pairing, split disjointness, and total coverage."""
    image_ids, mask_ids = set(images), set(masks)
    for label, missing in (
        ("Masks missing for images", image_ids - mask_ids),
        ("Images missing for masks", mask_ids - image_ids),
    ):
        if missing:
            errors.append(f"{label}: {sorted(missing)}")
    split_sets = {name: set(ids) for name, ids in splits.items()}
    for index, name in enumerate(SPLITS):
        for other in SPLITS[index + 1 :]:
            overlap = split_sets[name] & split_sets[other]
            if overlap:
                errors.append(f"Split overlap {name}/{other}: {sorted(overlap)}")
    union = set().union(*split_sets.values())
    for label, missing in (
        ("Split IDs without images", union - image_ids),
        ("Split IDs without masks", union - mask_ids),
        ("Images absent from splits", image_ids - union),
        ("Masks absent from splits", mask_ids - union),
    ):
        if missing:
            errors.append(f"{label}: {sorted(missing)}")
    return {
        "image_count": len(image_ids),
        "mask_count": len(mask_ids),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_union_count": len(union),
    }


def inspect_pair(image_path: Path, mask_path: Path, errors: list[str]) -> PairInfo | None:
    """Decode one image/mask pair and count every raw mask class ID."""
    try:
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
            image_mode = image.mode
        with Image.open(mask_path) as mask:
            mask.load()
            mask_size = mask.size
            mask_mode = mask.mode
            if mask_mode not in ("L", "P"):
                errors.append(f"Unexpected mask mode {mask_mode}: {mask_path}")
                return None
            histogram = mask.histogram()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        errors.append(f"Cannot decode {image_path} / {mask_path}: {exc}")
        return None
    if image_size != EXPECTED_SIZE or mask_size != EXPECTED_SIZE or image_size != mask_size:
        errors.append(f"Size mismatch {image_path.stem}: image={image_size}, mask={mask_size}")
    if image_mode != "RGB":
        errors.append(f"Unexpected image mode {image_mode}: {image_path}")
    invalid = {index: count for index, count in enumerate(histogram) if index > 8 and count}
    if invalid:
        errors.append(f"Invalid mask IDs {invalid}: {mask_path}")
    if sum(histogram) != mask_size[0] * mask_size[1]:
        errors.append(f"Mask histogram pixel count mismatch: {mask_path}")
    return PairInfo(histogram[: len(CLASS_NAMES)], image_size, mask_size, image_mode, mask_mode)


def write_reports(
    output_dir: Path,
    summary: dict[str, object],
    pixels: dict[str, list[int]],
    presence: dict[str, list[int]],
) -> None:
    """Write one JSON audit and one per-class CSV outside data/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "class_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "class_id", "class_name", "pixel_count", "image_count"))
        for split in SPLITS:
            for class_id, name in enumerate(CLASS_NAMES):
                writer.writerow((split, class_id, name, pixels[split][class_id], presence[split][class_id]))


def main() -> int:
    args = parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    errors: list[str] = []
    read_label_names(data_dir / "Label.txt", errors)
    images = collect_files(data_dir / "train" / "images", errors)
    masks = collect_files(data_dir / "train" / "masks", errors)
    splits = read_splits(data_dir, errors)
    index_summary = check_index(images, masks, splits, errors)

    pixels = {name: [0] * len(CLASS_NAMES) for name in SPLITS}
    presence = {name: [0] * len(CLASS_NAMES) for name in SPLITS}
    inspected = {name: 0 for name in SPLITS}
    image_sizes: Counter[str] = Counter()
    mask_sizes: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()
    mask_modes: Counter[str] = Counter()
    for split in SPLITS:
        selected = splits[split]
        if args.limit_per_split is not None:
            selected = selected[: args.limit_per_split]
        for image_id in selected:
            if image_id not in images or image_id not in masks:
                continue
            pair = inspect_pair(images[image_id], masks[image_id], errors)
            if pair is None:
                continue
            inspected[split] += 1
            image_sizes[f"{pair.image_size[0]}x{pair.image_size[1]}"] += 1
            mask_sizes[f"{pair.mask_size[0]}x{pair.mask_size[1]}"] += 1
            image_modes[pair.image_mode] += 1
            mask_modes[pair.mask_mode] += 1
            for class_id, count in enumerate(pair.counts):
                pixels[split][class_id] += count
                presence[split][class_id] += int(count > 0)

    summary: dict[str, object] = {
        "status": "pass" if not errors else "fail",
        "scope": "smoke" if args.limit_per_split is not None else "full",
        "expected_size": list(EXPECTED_SIZE),
        "label_names": dict(enumerate(CLASS_NAMES)),
        **index_summary,
        "inspected_pairs": inspected,
        "observed_image_sizes": dict(image_sizes),
        "observed_mask_sizes": dict(mask_sizes),
        "observed_image_modes": dict(image_modes),
        "observed_mask_modes": dict(mask_modes),
        "observed_mask_ids": [
            class_id
            for class_id in range(len(CLASS_NAMES))
            if any(pixels[split][class_id] for split in SPLITS)
        ],
        "errors": errors,
    }
    write_reports(output_dir, summary, pixels, presence)
    print(json.dumps({k: summary[k] for k in ("status", "scope", "image_count", "mask_count", "split_counts", "inspected_pairs", "errors")}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
