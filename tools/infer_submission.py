"""Run v1-v4 AIC test inference and build a strictly audited submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dino.download import sha256_file
from src.dino.embedding import extract_batch, load_frozen_model, select_device
from src.presence.labels import CLASS_NAMES
from src.segmentation.dino_conditioning import DINO_DIM, DirectDinoSegformer
from src.segmentation.dino_moe import DinoSoftMoESegformer, RoutingAccumulator
from src.segmentation.experiment import resolve_pretrained
from src.segmentation.model import PresenceSegformer
from src.segmentation.submission import (
    EXPECTED_TEST_COUNT,
    DinoTestDataset,
    SegmentationTestDataset,
    audit_label_definition,
    audit_prediction_directory,
    build_submission_zip,
    collect_test_images,
    save_official_prediction,
)


@dataclass(frozen=True)
class VersionSpec:
    arm: str
    model_name: str
    default_run_dir: Path
    uses_dino: bool
    uses_presence: bool
    uses_moe: bool


VERSIONS = {
    "v1": VersionSpec(
        "control", "v1 SegFormer-B3",
        Path("outputs/presence_conditioning/control_b3_bs4"), False, False, False,
    ),
    "v2": VersionSpec(
        "conditioned", "v2 SegFormer-B3 + Presence Conditioning",
        Path("outputs/presence_conditioning/conditioned_b3_bs4"), True, True, False,
    ),
    "v3": VersionSpec(
        "direct_dino", "v3 SegFormer-B3 + Direct DINO FiLM",
        Path("outputs/direct_dino_conditioning/direct_dino_b3_bs4"), True, False, False,
    ),
    "v4": VersionSpec(
        "dino_soft_moe", "v4 SegFormer-B3 + Direct DINO FiLM + DINO Soft MoE",
        Path("outputs/dino_moe/dino_soft_moe_b3_bs4"), True, False, True,
    ),
}


def parse_args(default_version: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", choices=tuple(VERSIONS), default=default_version,
        required=default_version is None,
    )
    parser.add_argument("--test-image-dir", type=Path, default=Path("data/test/images"))
    parser.add_argument("--label-file", type=Path, default=Path("data/Label.txt"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--presence-dir", type=Path, default=Path("outputs/class_presence/linear_518")
    )
    parser.add_argument(
        "--embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, default=None)
    parser.add_argument("--dino-snapshot-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/hf_cache"))
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA FP16 for segmentation")
    parser.add_argument("--limit", type=int, default=None, help="Smoke only: first N images")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Batch size/limit must be positive and workers must be non-negative")
    if args.output_dir.resolve().exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    if args.amp and args.device in {"cpu", "mps"}:
        parser.error("--amp is only supported with CUDA")
    return args


def resolve_run_dir(requested: Path | None, spec: VersionSpec) -> Path:
    if requested is not None:
        return requested.resolve()
    canonical = spec.default_run_dir
    nested_sync = Path("outputs") / canonical
    for candidate in (canonical, nested_sync):
        if (candidate / "best.pt").is_file():
            return candidate.resolve()
    return canonical.resolve()


def names_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_dino_snapshot(
    snapshot: Path, revision: str, stage01_embedding_dir: Path
) -> tuple[dict[str, str], dict[str, object]]:
    """Verify online DINO files against the accepted Stage 01 metadata."""
    manifest = json.loads((snapshot / "download_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((stage01_embedding_dir / "metadata.json").read_text(encoding="utf-8"))
    if (
        manifest.get("model_id") != "facebook/dinov2-base"
        or manifest.get("revision") != revision
        or metadata.get("model_id") != "facebook/dinov2-base"
        or metadata.get("resolved_revision") != revision
        or metadata.get("download_files") != manifest.get("files")
        or metadata.get("preprocessing", {}).get("resize_size") != [518, 518]
        or metadata.get("preprocessing", {}).get("resize_method")
        != "PIL bicubic, entire image, no crop"
        or metadata.get("features", {}).get("combined") != DINO_DIM
    ):
        raise ValueError("DINO snapshot does not match the accepted Stage 01 Combined embedding")
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
    return hashes, metadata


def load_presence_predictor(
    presence_dir: Path, revision: str, device: torch.device
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, Path]:
    """Load the Stage 02 selected Combined linear probe and train-only scaler."""
    selection = json.loads((presence_dir / "feature_selection.json").read_text(encoding="utf-8"))
    run_config = json.loads((presence_dir / "run_config.json").read_text(encoding="utf-8"))
    checkpoint_path = presence_dir / "features" / "combined" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        selection.get("selected_feature") != "combined"
        or run_config.get("selected_feature") != "combined"
        or run_config.get("label_order") != list(CLASS_NAMES)
        or run_config.get("embedding_revision") != revision
        or checkpoint.get("model") != "linear"
        or checkpoint.get("feature") != "combined"
        or checkpoint.get("embedding_revision") != revision
    ):
        raise ValueError("Stage 02 Presence Head does not match v2 Combined DINO conditioning")
    scaler_mean = checkpoint["scaler_mean"].float()
    scaler_std = checkpoint["scaler_std"].float()
    if (
        scaler_mean.shape != (DINO_DIM,)
        or scaler_std.shape != (DINO_DIM,)
        or not torch.isfinite(scaler_mean).all()
        or not torch.isfinite(scaler_std).all()
        or not torch.all(scaler_std > 0)
    ):
        raise ValueError("Invalid Stage 02 Presence Head scaler")
    model = nn.Linear(DINO_DIM, 8).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, scaler_mean.to(device), scaler_std.to(device), checkpoint_path


def validate_experiment(
    config: dict[str, object], metrics: dict[str, object], spec: VersionSpec
) -> None:
    if (
        config.get("arm") != spec.arm
        or metrics.get("arm") != spec.arm
        or config.get("input_size") != [1024, 1024]
        or config.get("num_classes") != 8
        or config.get("ignore_index") != 255
    ):
        raise ValueError(f"Run directory is not the accepted {spec.arm} experiment")
    if spec.uses_dino and spec.arm in {"direct_dino", "dino_soft_moe"} and (
        config.get("dino_model") != "facebook/dinov2-base"
        or config.get("dino_input_dimension") != DINO_DIM
    ):
        raise ValueError("DINO experiment configuration differs from the accepted protocol")
    if spec.uses_moe and (
        config.get("conditioning_type") != "dino_soft_moe"
        or config.get("num_experts") != 4
        or config.get("routing") != "all four experts execute; no Top-k or hard routing"
    ):
        raise ValueError("v4 is not the accepted four-expert dense Soft MoE")


def create_model(version: str, pretrained_dir: Path) -> nn.Module:
    if version == "v1":
        return PresenceSegformer(str(pretrained_dir), conditioning=False)
    if version == "v2":
        return PresenceSegformer(str(pretrained_dir), conditioning=True)
    if version == "v3":
        return DirectDinoSegformer(str(pretrained_dir))
    if version == "v4":
        return DinoSoftMoESegformer(str(pretrained_dir))
    raise ValueError(f"Unsupported version: {version}")


def main(default_version: str | None = None) -> int:
    args = parse_args(default_version)
    started = time.perf_counter()
    spec = VERSIONS[args.version]
    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires a CUDA device")

    run_dir = resolve_run_dir(args.run_dir, spec)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_experiment(config, metrics, spec)
    label_mapping = audit_label_definition(args.label_file.resolve())
    test_paths = collect_test_images(args.test_image_dir.resolve(), args.limit)

    dino_model: nn.Module | None = None
    processor = None
    dino_hashes: dict[str, str] | None = None
    revision: str | None = None
    presence_model: nn.Linear | None = None
    scaler_mean: torch.Tensor | None = None
    scaler_std: torch.Tensor | None = None
    presence_checkpoint: Path | None = None
    embedding_dir = args.embedding_dir.resolve()
    if spec.uses_dino:
        configured_revision = config.get("dino_revision") if args.version in {"v3", "v4"} else None
        if configured_revision is None:
            presence_config = json.loads(
                (args.presence_dir / "run_config.json").read_text(encoding="utf-8")
            )
            configured_revision = presence_config.get("embedding_revision")
        if not isinstance(configured_revision, str):
            raise ValueError(f"{args.version} configuration has no frozen DINO revision")
        revision = configured_revision
        dino_snapshot = args.dino_snapshot_dir or args.cache_dir / "snapshots" / revision
        dino_snapshot = dino_snapshot.resolve()
        if dino_snapshot.name != revision:
            raise ValueError(f"DINO snapshot revision differs: {dino_snapshot.name} != {revision}")
        dino_hashes, _ = audit_dino_snapshot(dino_snapshot, revision, embedding_dir)
        dino_model, processor = load_frozen_model(dino_snapshot, device)
        if spec.uses_presence:
            presence_model, scaler_mean, scaler_std, presence_checkpoint = load_presence_predictor(
                args.presence_dir.resolve(), revision, device
            )

    pretrained_dir = resolve_pretrained(args.pretrained_dir, args.cache_dir, args.hf_endpoint)
    pretrained_weight = next(
        pretrained_dir / name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (pretrained_dir / name).is_file()
    )
    if sha256_file(pretrained_weight) != config.get("pretrained_sha256"):
        raise ValueError("SegFormer pretrained weight differs from the training configuration")
    model = create_model(args.version, pretrained_dir).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("arm") != spec.arm or checkpoint.get("step") != metrics.get("best_step"):
        raise ValueError("best.pt arm/step differs from metrics.json")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    if spec.uses_dino:
        if processor is None:
            raise RuntimeError("DINO processor was not loaded")
        dataset = DinoTestDataset(test_paths, processor)
    else:
        dataset = SegmentationTestDataset(test_paths)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir()
    routing = RoutingAccumulator() if spec.uses_moe else None

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.version} test inference"):
            if spec.uses_dino:
                segmentation_pixels, dino_pixels, names = batch
                if dino_model is None:
                    raise RuntimeError("DINO model was not loaded")
                _, _, combined = extract_batch(dino_model, dino_pixels, device)
                embedding = torch.from_numpy(combined).to(device)
            else:
                segmentation_pixels, names = batch
                embedding = None
            segmentation_pixels = segmentation_pixels.to(device)

            if args.version == "v1":
                conditioning = torch.zeros(
                    (segmentation_pixels.shape[0], 7), dtype=torch.float32, device=device
                )
            elif args.version == "v2":
                if presence_model is None or scaler_mean is None or scaler_std is None or embedding is None:
                    raise RuntimeError("Presence predictor was not loaded")
                probabilities = torch.sigmoid(presence_model((embedding - scaler_mean) / scaler_std))
                conditioning = probabilities[:, 1:]
            else:
                if embedding is None:
                    raise RuntimeError("DINO embedding was not extracted")
                conditioning = embedding

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                if routing is not None:
                    logits, weights = model(segmentation_pixels, conditioning, return_routing=True)
                    routing.update(weights)
                else:
                    logits = model(segmentation_pixels, conditioning)
            predictions = logits.argmax(dim=1).cpu().numpy()
            for prediction, name in zip(predictions, names):
                save_official_prediction(prediction, prediction_dir / name)

    audit = audit_prediction_directory(test_paths, prediction_dir)
    submission_path = args.output_dir / "submission.zip"
    build_submission_zip(prediction_dir, submission_path)
    scope = "full" if args.limit is None else "smoke"
    if scope == "full" and audit["file_count"] != EXPECTED_TEST_COUNT:
        raise ValueError("Full submission does not contain exactly 1300 masks")
    report: dict[str, object] = {
        "schema_version": 2,
        "scope": scope,
        "version": args.version,
        "model": spec.model_name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "segmentation_preprocessing": "complete original 1024x1024 RGB image; training normalization",
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
    if spec.uses_dino:
        report.update(
            {
                "dino_model": "facebook/dinov2-base",
                "dino_revision": revision,
                "dino_snapshot_sha256": dino_hashes,
                "dino_feature": "Combined = CLS 768 || MeanPatch 768",
                "dino_preprocessing": "complete original test image -> bicubic Resize 518x518; no crop",
            }
        )
    if presence_checkpoint is not None:
        report["presence_head"] = {
            "checkpoint": str(presence_checkpoint),
            "checkpoint_sha256": sha256_file(presence_checkpoint),
            "output": "sigmoid probabilities; foreground columns 1..7",
        }
    if routing is not None:
        report["test_routing"] = routing.summary()
    (args.output_dir / "submission_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"READY: {submission_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
