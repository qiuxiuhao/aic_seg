"""Extract frozen 518-pixel DINOv2 embeddings for the fixed dataset splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path


SPLITS = ("train", "val_stratified", "val_domain")
DEFAULT_ENDPOINT = "https://hf-mirror.com"
MODEL_ID = "facebook/dinov2-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--audit-file", type=Path, default=Path("outputs/data_audit/audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov2_518/full"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--revision", default="main", help="HF revision; resolved commit is recorded")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke: first N IDs per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit_per_split is not None and args.limit_per_split <= 0):
        parser.error("Batch size and limit must be positive")
    if not args.hf_endpoint.startswith("https://"):
        parser.error("--hf-endpoint must be an HTTPS URL")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    if data_dir == output_dir or data_dir in output_dir.parents:
        parser.error("Output directory must be outside the read-only data directory")
    if data_dir == cache_dir or data_dir in cache_dir.parents:
        parser.error("Cache directory must be outside the read-only data directory")
    if output_dir.exists():
        parser.error(f"Output directory already exists; choose a new run directory: {output_dir}")
    return args


def load_manifest(data_dir: Path, limit_per_split: int | None) -> list[tuple[str, str, Path]]:
    """Preserve the three existing split orders and validate their image paths."""
    rows: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for split in SPLITS:
        split_path = data_dir / "splits" / f"{split}.txt"
        ids = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines()]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Blank or duplicate ID in {split_path}")
        selected = ids if limit_per_split is None else ids[:limit_per_split]
        for image_id in selected:
            if image_id in seen:
                raise ValueError(f"Duplicate ID across splits: {image_id}")
            seen.add(image_id)
            image_path = data_dir / "train" / "images" / f"{image_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            rows.append((image_id, split, image_path))
    return rows


def check_audit(path: Path) -> None:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("scope") != "full":
        raise ValueError(f"Stage 00 full audit has not passed: {path}")
    if audit.get("split_union_count") != 6996:
        raise ValueError(f"Unexpected audited dataset size: {path}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    check_audit(args.audit_file)
    rows = load_manifest(data_dir, args.limit_per_split)
    if args.limit_per_split is None and len(rows) != 6996:
        raise ValueError(f"Expected 6996 full-run IDs, got {len(rows)}")

    # The Hub reads environment variables at import time. Keep downloads inside outputs/.
    os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    import numpy as np
    import torch
    import transformers
    from tqdm.auto import tqdm

    from src.dino.download import download_snapshot
    from src.dino.embedding import extract_batch, load_frozen_model, preprocess_image, select_device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    print(f"Downloading or reusing {MODEL_ID} from {args.hf_endpoint}; device={device}", flush=True)
    snapshot, download_info = download_snapshot(args.hf_endpoint, args.revision, cache_dir)
    model, processor = load_frozen_model(snapshot, device)

    cls_batches: list[np.ndarray] = []
    mean_batches: list[np.ndarray] = []
    combined_batches: list[np.ndarray] = []
    embedding_start = time.perf_counter()
    with tqdm(total=len(rows), desc="Embedding", unit="img", mininterval=1.0) as progress:
        for start in range(0, len(rows), args.batch_size):
            group = rows[start : start + args.batch_size]
            batch = torch.stack([preprocess_image(path, processor) for _, _, path in group])
            cls, mean_patch, combined = extract_batch(model, batch, device)
            cls_batches.append(cls)
            mean_batches.append(mean_patch)
            combined_batches.append(combined)
            progress.update(len(group))
    embedding_seconds = time.perf_counter() - embedding_start

    cls_array = np.concatenate(cls_batches, axis=0)
    mean_array = np.concatenate(mean_batches, axis=0)
    combined_array = np.concatenate(combined_batches, axis=0)
    if (cls_array.shape, mean_array.shape, combined_array.shape) != (
        (len(rows), 768), (len(rows), 768), (len(rows), 1536)
    ):
        raise ValueError("Unexpected embedding array shapes")
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "cls.npy", cls_array)
    np.save(output_dir / "mean_patch.npy", mean_array)
    np.save(output_dir / "combined.npy", combined_array)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row", "image_id", "split", "image_path"))
        for row_number, (image_id, split, _) in enumerate(rows):
            writer.writerow((row_number, image_id, split, f"data/train/images/{image_id}.png"))
    metadata = {
        "scope": "smoke" if args.limit_per_split is not None else "full",
        "count": len(rows),
        "split_counts": {name: sum(split == name for _, split, _ in rows) for name in SPLITS},
        "model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_revision": snapshot.name,
        "download_files": download_info["files"],
        "hf_endpoint": args.hf_endpoint,
        "preprocessing": {
            "source_size": [1024, 1024],
            "resize_size": [518, 518],
            "resize_method": "PIL bicubic, entire image, no crop",
            "rescale_factor": float(processor.rescale_factor),
            "image_mean": list(processor.image_mean),
            "image_std": list(processor.image_std),
        },
        "features": {"cls": 768, "mean_patch": 768, "combined": 1536},
        "dtype": "float32",
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "embedding_seconds": embedding_seconds,
        "images_per_second": len(rows) / embedding_seconds,
        "split_sha256": {
            name: file_sha256(data_dir / "splits" / f"{name}.txt") for name in SPLITS
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(rows)} embeddings to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
