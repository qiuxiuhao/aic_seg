"""Run v3 Direct DINO inference on AIC test images and build submission.zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dino.download import sha256_file
from src.dino.embedding import extract_batch, load_frozen_model, select_device
from src.segmentation.dino_conditioning import DirectDinoSegformer
from src.segmentation.experiment import resolve_pretrained
from src.segmentation.submission import (
    EXPECTED_TEST_COUNT,
    V3TestDataset,
    audit_label_definition,
    audit_prediction_directory,
    build_submission_zip,
    collect_test_images,
    save_official_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-image-dir", type=Path, default=Path("data/test/images"))
    parser.add_argument("--label-file", type=Path, default=Path("data/Label.txt"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path("outputs/direct_dino_conditioning/direct_dino_b3_bs4"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--dino-snapshot-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 for SegFormer inference")
    parser.add_argument("--limit", type=int, default=None, help="Smoke only: infer the first N images")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Batch size/limit must be positive and workers must be non-negative")
    if args.output_dir.resolve().exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.amp and args.device in {"cpu", "mps"}:
        parser.error("--amp is only supported with CUDA")
    return args


def names_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_dino_snapshot(
    snapshot: Path, revision: str, stage01_embedding_dir: Path
) -> dict[str, str]:
    """Verify online test DINO weights are exactly those used for Stage 01 embeddings."""
    manifest = json.loads((snapshot / "download_manifest.json").read_text(encoding="utf-8"))
    stage01 = json.loads((stage01_embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    if (
        manifest.get("model_id") != "facebook/dinov2-base"
        or manifest.get("revision") != revision
        or stage01.get("model_id") != "facebook/dinov2-base"
        or stage01.get("resolved_revision") != revision
        or stage01.get("download_files") != manifest.get("files")
    ):
        raise ValueError("DINO snapshot does not match the Stage 01 embeddings used by v3")
    expected_files = manifest.get("files")
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    if not isinstance(expected_files, dict) or set(expected_files) != set(required):
        raise ValueError("Unexpected DINO download manifest contents")
    hashes: dict[str, str] = {}
    for name in required:
        path = snapshot / name
        expected = expected_files[name]
        actual_sha256 = sha256_file(path)
        if path.stat().st_size != expected.get("bytes") or actual_sha256 != expected.get("sha256"):
            raise ValueError(f"DINO snapshot file differs from its accepted manifest: {path}")
        hashes[name] = actual_sha256
    return hashes


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        config.get("arm") != "direct_dino"
        or config.get("conditioning_type") != "direct_dino"
        or config.get("dino_model") != "facebook/dinov2-base"
        or config.get("dino_input_dimension") != 1536
        or config.get("input_size") != [1024, 1024]
        or config.get("num_classes") != 8
        or metrics.get("arm") != "direct_dino"
    ):
        raise ValueError("Run directory is not the accepted v3 Direct DINO experiment")

    label_mapping = audit_label_definition(args.label_file.resolve())
    test_paths = collect_test_images(args.test_image_dir.resolve(), args.limit)
    revision = config.get("dino_revision")
    if not isinstance(revision, str):
        raise ValueError("v3 config has no frozen DINO revision")
    dino_snapshot = args.dino_snapshot_dir or args.cache_dir / "snapshots" / revision
    dino_snapshot = dino_snapshot.resolve()
    if dino_snapshot.name != revision:
        raise ValueError(f"DINO snapshot revision differs from v3: {dino_snapshot.name} != {revision}")
    embedding_dir = config.get("embedding_dir")
    if not isinstance(embedding_dir, str):
        raise ValueError("v3 config has no Stage 01 embedding directory")
    dino_hashes = audit_dino_snapshot(dino_snapshot, revision, Path(embedding_dir).resolve())
    dino_model, processor = load_frozen_model(dino_snapshot, device)
    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    pretrained_weight = next(
        pretrained_dir / name for name in ("model.safetensors", "pytorch_model.bin")
        if (pretrained_dir / name).is_file()
    )
    if sha256_file(pretrained_weight) != config.get("pretrained_sha256"):
        raise ValueError("SegFormer pretrained weight differs from the v3 training configuration")

    model = DirectDinoSegformer(str(pretrained_dir)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("arm") != "direct_dino"
        or checkpoint.get("step") != metrics.get("best_step")
    ):
        raise ValueError("best.pt arm/step differs from v3 metrics.json")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    loader = DataLoader(
        V3TestDataset(test_paths, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir()
    with torch.inference_mode():
        for segmentation_pixels, dino_pixels, names in tqdm(loader, desc="v3 test inference"):
            _, _, combined = extract_batch(dino_model, dino_pixels, device)
            embeddings = torch.from_numpy(combined).to(device)
            segmentation_pixels = segmentation_pixels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(segmentation_pixels, embeddings)
            predictions = logits.argmax(dim=1).cpu().numpy()
            for prediction, name in zip(predictions, names):
                save_official_prediction(prediction, prediction_dir / name)

    audit = audit_prediction_directory(test_paths, prediction_dir)
    submission_path = args.output_dir / "submission.zip"
    build_submission_zip(prediction_dir, submission_path)
    scope = "full" if args.limit is None else "smoke"
    if scope == "full" and audit["file_count"] != EXPECTED_TEST_COUNT:
        raise ValueError("Full submission does not contain exactly 1300 masks")
    report = {
        "schema_version": 1,
        "scope": scope,
        "model": "v3 SegFormer-B3 + Direct DINO FiLM",
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "dino_model": "facebook/dinov2-base",
        "dino_revision": revision,
        "dino_snapshot_sha256": dino_hashes,
        "dino_feature": "Combined = CLS 768 || MeanPatch 768",
        "dino_preprocessing": "complete original test image -> bicubic Resize 518x518; no crop",
        "segmentation_preprocessing": "complete original 1024x1024 RGB image; Stage 03/04 normalization",
        "inference": "single pass; no TTA; no post-processing",
        "device": str(device),
        "amp": args.amp,
        "batch_size": args.batch_size,
        "test_image_count": len(test_paths),
        "test_filename_sha256": names_sha256(test_paths),
        "ignore_label": {"official_id": 0, "class_name": "Ignore", "predicted": False},
        "label_mapping": label_mapping,
        "audit": audit,
        "submission_zip": submission_path.name,
        "submission_zip_sha256": sha256_file(submission_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "submission_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"READY: {submission_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
