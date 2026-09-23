"""Cache Stage 03-aligned D4 DINO Combined embeddings for train images."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.dino.download import sha256_file
from src.dino.embedding import extract_batch, load_frozen_model, preprocess_rgb, select_device
from src.segmentation.geometry import (
    STATE_TO_INDEX,
    TRANSFORM_DEFINITIONS,
    TRANSFORM_STATES,
    apply_transform,
)


EXPECTED_MODEL = "facebook/dinov2-base"
EXPECTED_TRAIN_COUNT = 5597
EXPECTED_DIM = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--base-embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov2_518/augmented_train"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Smoke only: cache the first N train IDs")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be positive")
    if args.output_dir.resolve().exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.data_dir.resolve() == args.output_dir.resolve() or args.data_dir.resolve() in args.output_dir.resolve().parents:
        parser.error("Output directory must be outside read-only data/")
    return args


def load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not ids or not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate image ID in {path}")
    return ids


def load_base(base_dir: Path, data_dir: Path) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
    metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_preprocess = {
        "source_size": [1024, 1024],
        "resize_size": [518, 518],
        "resize_method": "PIL bicubic, entire image, no crop",
    }
    if metadata.get("scope") != "full" or metadata.get("count") != 6996:
        raise ValueError("Stage 01 base cache must be the accepted full 6996-image cache")
    if metadata.get("model_id") != EXPECTED_MODEL or metadata.get("features", {}).get("combined") != EXPECTED_DIM:
        raise ValueError("Stage 01 cache has an unexpected DINO model or feature dimension")
    for key, value in expected_preprocess.items():
        if metadata.get("preprocessing", {}).get(key) != value:
            raise ValueError(f"Stage 01 preprocessing differs at {key}")
    for split in ("train", "val_stratified", "val_domain"):
        actual = sha256_file(data_dir / "splits" / f"{split}.txt")
        if metadata.get("split_sha256", {}).get(split) != actual:
            raise ValueError(f"Stage 01 cache split hash differs for {split}")

    mapping: dict[str, int] = {}
    with (base_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row", "image_id", "split", "image_path"]:
            raise ValueError("Unexpected Stage 01 manifest columns")
        for expected_row, row in enumerate(reader):
            if int(row["row"]) != expected_row or row["image_id"] in mapping:
                raise ValueError("Stage 01 manifest has a duplicate ID or non-contiguous row")
            mapping[row["image_id"]] = expected_row
    combined = np.load(base_dir / "combined.npy", mmap_mode="r")
    if combined.shape != (6996, EXPECTED_DIM) or combined.dtype != np.float32:
        raise ValueError(f"Unexpected Stage 01 Combined array: {combined.shape} {combined.dtype}")
    if len(mapping) != combined.shape[0]:
        raise ValueError("Stage 01 manifest count differs from Combined array")
    return combined, mapping, metadata


def resolve_snapshot(args: argparse.Namespace, metadata: dict[str, object]) -> Path:
    revision = metadata.get("resolved_revision")
    if not isinstance(revision, str):
        raise ValueError("Stage 01 metadata has no resolved DINO revision")
    snapshot = args.snapshot_dir or args.cache_dir / "snapshots" / revision
    snapshot = snapshot.resolve()
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    if not all((snapshot / name).is_file() for name in required):
        raise FileNotFoundError(f"Incomplete frozen DINO snapshot: {snapshot}")
    if snapshot.name != revision:
        raise ValueError(f"DINO snapshot revision {snapshot.name} differs from Stage 01 {revision}")
    return snapshot


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    base_dir = args.base_embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    train_ids_all = load_ids(data_dir / "splits" / "train.txt")
    if len(train_ids_all) != EXPECTED_TRAIN_COUNT:
        raise ValueError(f"Expected {EXPECTED_TRAIN_COUNT} train IDs, got {len(train_ids_all)}")
    train_ids = train_ids_all if args.limit is None else train_ids_all[: args.limit]
    base_combined, base_rows, base_metadata = load_base(base_dir, data_dir)
    missing = [image_id for image_id in train_ids if image_id not in base_rows]
    if missing:
        raise ValueError(f"Stage 01 cache is missing train IDs: {missing[:5]}")

    snapshot = resolve_snapshot(args, base_metadata)
    device = select_device(args.device)
    model, processor = load_frozen_model(snapshot, device)
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined.npy"
    cached = np.lib.format.open_memmap(
        combined_path, mode="w+", dtype=np.float32,
        shape=(len(train_ids), len(TRANSFORM_STATES), EXPECTED_DIM),
    )
    for row, image_id in enumerate(train_ids):
        cached[row, STATE_TO_INDEX["r0"]] = base_combined[base_rows[image_id]]

    started = time.perf_counter()
    non_identity = [state for state in TRANSFORM_STATES if state != "r0"]
    with tqdm(total=len(train_ids) * len(non_identity), desc="D4 DINO cache", unit="embedding") as progress:
        for state in non_identity:
            for start in range(0, len(train_ids), args.batch_size):
                group = train_ids[start : start + args.batch_size]
                tensors = []
                for image_id in group:
                    image_path = data_dir / "train" / "images" / f"{image_id}.png"
                    with Image.open(image_path) as file:
                        if file.size != (1024, 1024) or file.mode != "RGB":
                            raise ValueError(f"Expected 1024x1024 RGB image: {image_path}")
                        image = np.asarray(file, dtype=np.uint8).copy()
                    tensors.append(preprocess_rgb(apply_transform(image, state), processor))
                _, _, embeddings = extract_batch(model, torch.stack(tensors), device)
                cached[start : start + len(group), STATE_TO_INDEX[state]] = embeddings
                progress.update(len(group))
    cached.flush()
    elapsed = time.perf_counter() - started

    verified = np.load(combined_path, mmap_mode="r")
    expected_shape = (len(train_ids), 8, EXPECTED_DIM)
    if verified.shape != expected_shape or verified.dtype != np.float32:
        raise ValueError(f"Unexpected augmented cache: {verified.shape} {verified.dtype}")
    if not np.isfinite(verified).all():
        raise ValueError("Augmented DINO cache contains non-finite values")
    for row, image_id in enumerate(train_ids):
        if not np.array_equal(verified[row, 0], base_combined[base_rows[image_id]]):
            raise ValueError(f"r0 reuse mismatch for {image_id}")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("array_row", "image_id", "state_index", "transform_state", "source"))
        pairs: set[tuple[str, str]] = set()
        for row, image_id in enumerate(train_ids):
            for state_index, state in enumerate(TRANSFORM_STATES):
                pair = (image_id, state)
                if pair in pairs:
                    raise ValueError(f"Duplicate augmented cache pair: {pair}")
                pairs.add(pair)
                writer.writerow((row, image_id, state_index, state, "stage01_reuse" if state == "r0" else "offline_dino"))
    if len(pairs) != len(train_ids) * 8:
        raise ValueError("Augmented manifest pair count is incomplete")

    metadata = {
        "scope": "full" if args.limit is None else "smoke",
        "image_count": len(train_ids),
        "expected_full_image_count": EXPECTED_TRAIN_COUNT,
        "state_count": len(TRANSFORM_STATES),
        "pair_count": len(pairs),
        "array_shape": list(expected_shape),
        "dtype": "float32",
        "all_finite": True,
        "unique_image_state_pairs": True,
        "model_id": EXPECTED_MODEL,
        "resolved_revision": base_metadata["resolved_revision"],
        "snapshot": str(snapshot),
        "features": {"combined": EXPECTED_DIM, "definition": "CLS 768 || MeanPatch 768"},
        "preprocessing": base_metadata["preprocessing"],
        "transforms": {
            "stage03_sampling": "horizontal flip p=0.5, then vertical flip p=0.5, then uniform randrange(4) CCW rotation",
            "canonical_states": list(TRANSFORM_STATES),
            "canonical_probability": {state: 0.125 for state in TRANSFORM_STATES},
            "definitions": TRANSFORM_DEFINITIONS,
            "r0_reused_from_stage01": True,
        },
        "train_split_sha256": sha256_file(data_dir / "splits" / "train.txt"),
        "stage01_metadata_sha256": sha256_file(base_dir / "metadata.json"),
        "stage01_manifest_sha256": sha256_file(base_dir / "manifest.csv"),
        "stage01_combined_sha256": sha256_file(base_dir / "combined.npy"),
        "combined_sha256": sha256_file(combined_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "embedding_seconds_excluding_r0": elapsed,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved and verified {len(train_ids)} images x 8 states x 1536 to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
