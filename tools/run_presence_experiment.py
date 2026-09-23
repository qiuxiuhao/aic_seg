"""Validate class presence from frozen DINOv2 features on the fixed splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.dino.embedding import select_device
from src.presence.labels import CLASS_NAMES, SPLITS, Sample, build_labels, load_samples, split_indices
from src.presence.metrics import best_f1_threshold, evaluate
from src.presence.probe import ProbeResult, predict_probabilities, train_linear_probe


FEATURES = ("cls", "mean_patch", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("outputs/dinov2_518/full_mps"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/class_presence/linear_518"))
    parser.add_argument("--features", choices=("all", *FEATURES), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke: first N IDs from each fixed split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    for name in ("batch_size", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.limit_per_split is not None and args.limit_per_split <= 0:
        parser.error("--limit-per-split must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative")
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if data_dir == output_dir or data_dir in output_dir.parents:
        parser.error("Output directory must be outside the read-only data directory")
    if output_dir.exists():
        parser.error(f"Output directory already exists; choose a new run directory: {output_dir}")
    return args


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_labels(output_dir: Path, samples: list[Sample], labels: np.ndarray) -> None:
    np.save(output_dir / "labels.npy", labels)
    with (output_dir / "labels_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("local_row", "embedding_row", "image_id", "split"))
        for local_row, sample in enumerate(samples):
            writer.writerow((local_row, sample.row, sample.image_id, sample.split))
    with (output_dir / "label_prevalence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "class_id", "class_name", "positive_images", "images"))
        indices = split_indices(samples)
        for split in SPLITS:
            for class_index, name in enumerate(CLASS_NAMES):
                writer.writerow(
                    (split, class_index + 1, name, int(labels[indices[split], class_index].sum()), len(indices[split]))
                )


def save_predictions(
    path: Path, samples: list[Sample], indices: np.ndarray, probabilities: np.ndarray
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "split", *CLASS_NAMES))
        for sample_index, scores in zip(indices, probabilities):
            sample = samples[int(sample_index)]
            writer.writerow((sample.image_id, sample.split, *(f"{score:.9g}" for score in scores)))


def save_domain_errors(
    path: Path,
    samples: list[Sample],
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "class_name", "error", "probability", "threshold"))
        for local_row, sample_index in enumerate(indices):
            for class_index, name in enumerate(CLASS_NAMES):
                truth = bool(labels[int(sample_index), class_index])
                predicted = bool(probabilities[local_row, class_index] >= thresholds[class_index])
                if truth != predicted:
                    writer.writerow(
                        (
                            samples[int(sample_index)].image_id,
                            name,
                            "FP" if predicted else "FN",
                            f"{probabilities[local_row, class_index]:.9g}",
                            f"{thresholds[class_index]:.9g}",
                        )
                    )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    embedding_dir = args.embedding_dir.resolve()
    output_dir = args.output_dir.resolve()
    samples, embedding_metadata = load_samples(data_dir, embedding_dir, args.limit_per_split)
    indices = split_indices(samples)
    labels = build_labels(data_dir, samples)
    if labels.shape != (len(samples), 8) or labels.dtype != np.uint8:
        raise ValueError("Presence labels must be uint8 [N, 8]")
    device = select_device(args.device)
    print(f"Presence samples={len(samples)}, device={device}", flush=True)
    chosen_features = FEATURES if args.features == "all" else (args.features,)
    embedding_rows = np.asarray([sample.row for sample in samples])
    output_dir.mkdir(parents=True, exist_ok=False)
    save_labels(output_dir, samples, labels)
    feature_dir = output_dir / "features"
    feature_dir.mkdir()
    results: dict[str, tuple[ProbeResult, np.ndarray, dict[str, object]]] = {}

    for feature in chosen_features:
        array = np.load(embedding_dir / f"{feature}.npy", mmap_mode="r")
        if array.shape != (6996, embedding_metadata["features"][feature]) or array.dtype != np.float32:
            raise ValueError(f"Unexpected embedding shape or dtype for {feature}")
        selected = np.asarray(array[embedding_rows], dtype=np.float32)
        train, val = indices["train"], indices["val_stratified"]
        print(f"Training linear probe on {feature} ({selected.shape[1]} features)", flush=True)
        result = train_linear_probe(
            selected[train], labels[train], selected[val], labels[val], device,
            seed=args.seed, batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        model = nn.Linear(selected.shape[1], 8).to(device)
        model.load_state_dict(result.state_dict)
        val_probabilities = predict_probabilities(
            model, selected[val], result.scaler_mean, result.scaler_std, device
        )
        val_report = evaluate(labels[val], val_probabilities, np.full(8, 0.5))
        if abs(float(val_report["foreground_macro_ap"]) - result.best_val_ap) > 1e-5:
            raise ValueError(f"Best checkpoint cannot reproduce validation AP for {feature}")
        run_dir = feature_dir / feature
        run_dir.mkdir()
        torch.save(
            {
                "model": "linear",
                "feature": feature,
                "state_dict": result.state_dict,
                "scaler_mean": torch.from_numpy(result.scaler_mean),
                "scaler_std": torch.from_numpy(result.scaler_std),
                "best_epoch": result.best_epoch,
                "val_foreground_macro_ap": result.best_val_ap,
                "embedding_revision": embedding_metadata["resolved_revision"],
                "seed": args.seed,
            },
            run_dir / "best.pt",
        )
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_foreground_macro_ap"))
            writer.writeheader()
            writer.writerows(result.history)
        save_json(run_dir / "val_metrics_at_0.5.json", val_report)
        results[feature] = (result, val_probabilities, val_report)
        print(f"{feature}: best epoch={result.best_epoch}, val foreground macro AP={result.best_val_ap:.4f}", flush=True)

    selected_feature = max(chosen_features, key=lambda feature: results[feature][0].best_val_ap)
    result, val_probabilities, _ = results[selected_feature]
    save_json(
        output_dir / "feature_selection.json",
        {
            "criterion": "val_stratified foreground macro AP",
            "scores": {feature: results[feature][0].best_val_ap for feature in chosen_features},
            "selected_feature": selected_feature,
        },
    )
    thresholds = np.asarray(
        [best_f1_threshold(labels[indices["val_stratified"], i], val_probabilities[:, i]) for i in range(8)]
    )
    save_json(output_dir / "val_thresholds.json", dict(zip(CLASS_NAMES, map(float, thresholds))))
    val_report = evaluate(labels[indices["val_stratified"]], val_probabilities, thresholds)
    save_json(output_dir / "val_stratified_metrics.json", val_report)

    selected_array = np.load(embedding_dir / f"{selected_feature}.npy", mmap_mode="r")
    model = nn.Linear(selected_array.shape[1], 8).to(device)
    model.load_state_dict(result.state_dict)
    for split in SPLITS:
        split_rows = indices[split]
        features = np.asarray(selected_array[embedding_rows[split_rows]], dtype=np.float32)
        probabilities = predict_probabilities(
            model, features, result.scaler_mean, result.scaler_std, device
        )
        if split == "val_stratified" and not np.allclose(probabilities, val_probabilities, atol=1e-6):
            raise ValueError("Validation probabilities changed after feature selection")
        save_predictions(output_dir / f"predictions_{split}.csv", samples, split_rows, probabilities)
        if split == "val_domain":
            domain_report = evaluate(labels[split_rows], probabilities, thresholds)
            save_json(output_dir / "val_domain_metrics.json", domain_report)
            save_domain_errors(
                output_dir / "val_domain_errors.csv", samples, split_rows, labels, probabilities, thresholds
            )

    save_json(
        output_dir / "run_config.json",
        {
            "scope": "smoke" if args.limit_per_split is not None else "full",
            "count": len(samples),
            "split_counts": {split: len(indices[split]) for split in SPLITS},
            "label_order": list(CLASS_NAMES),
            "ignore_id": 0,
            "presence_min_pixels": 1,
            "embedding_dir": str(embedding_dir),
            "embedding_revision": embedding_metadata["resolved_revision"],
            "embedding_preprocessing": embedding_metadata["preprocessing"],
            "feature_selection": "val_stratified foreground macro AP",
            "threshold_selection": "val_stratified maximum F1 per class",
            "selected_feature": selected_feature,
            "device": str(device),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    )
    print(
        f"Selected {selected_feature}; val_stratified foreground AP={val_report['foreground_macro_ap']:.4f}; "
        f"val_domain foreground AP={domain_report['foreground_macro_ap']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
