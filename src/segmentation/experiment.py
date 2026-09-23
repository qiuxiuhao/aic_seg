"""Shared Stage 03 checkpoint, training and evaluation helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.presence.labels import CLASS_NAMES
from src.segmentation.model import PresenceSegformer


MODEL_ID = "nvidia/mit-b3"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_pretrained(pretrained_dir: Path | None, cache_dir: Path, endpoint: str) -> Path:
    """Fetch the original ImageNet B3 encoder through the selected HF endpoint."""
    if pretrained_dir is not None:
        path = pretrained_dir.resolve()
    else:
        path = Path(snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"],
            cache_dir=cache_dir,
            endpoint=endpoint,
        ))
    if not (path / "config.json").is_file() or not any(
        (path / name).is_file() for name in ("pytorch_model.bin", "model.safetensors")
    ):
        raise FileNotFoundError(f"Incomplete {MODEL_ID} checkpoint: {path}")
    return path


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, ignore_index=255)


def evaluate(model: PresenceSegformer, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, object]:
    """Count valid pixels only; report all eight IoUs and their mean."""
    model.eval()
    confusion = torch.zeros((8, 8), dtype=torch.int64)
    with torch.inference_mode():
        for pixels, target, probability, _ in tqdm(loader, desc="Evaluating", leave=False):
            pixels, probability = pixels.to(device), probability.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(pixels, probability)
            predicted = logits.argmax(dim=1).cpu()
            valid = target != 255
            counts = torch.bincount((target[valid] * 8 + predicted[valid]).view(-1), minlength=64)
            confusion += counts.reshape(8, 8)
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = [float(intersection[i] / union[i]) if union[i] else None for i in range(8)]
    valid_iou = [value for value in iou if value is not None]
    return {
        "miou": float(sum(valid_iou) / len(valid_iou)) if valid_iou else None,
        "class_iou": dict(zip(CLASS_NAMES, iou)),
        "valid_pixels": int(confusion.sum()),
        "confusion": confusion.tolist(),
    }


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
