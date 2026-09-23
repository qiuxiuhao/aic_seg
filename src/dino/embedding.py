"""Frozen DINOv2-B/14 inference on complete 1024-pixel UAV images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


MODEL_ID = "facebook/dinov2-base"
SOURCE_SIZE = (1024, 1024)
INPUT_SIZE = (518, 518)
PATCH_SIZE = 14
HIDDEN_SIZE = 768
PATCH_COUNT = (INPUT_SIZE[0] // PATCH_SIZE) * (INPUT_SIZE[1] // PATCH_SIZE)


@dataclass(frozen=True)
class ProcessorConfig:
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    rescale_factor: float
    resample: int


def select_device(requested: str) -> torch.device:
    """Select CUDA, MPS, or CPU; reject unavailable explicit choices."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device mps/cpu or install a compatible PyTorch build")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu or an Apple Silicon PyTorch build")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def load_frozen_model(
    snapshot: Path, device: torch.device
) -> tuple[torch.nn.Module, ProcessorConfig]:
    """Load the cached official HF snapshot without another network request."""
    config_path = snapshot / "preprocessor_config.json"
    image_config = json.loads(config_path.read_text(encoding="utf-8"))
    processor = ProcessorConfig(
        image_mean=tuple(image_config["image_mean"]),
        image_std=tuple(image_config["image_std"]),
        rescale_factor=float(image_config["rescale_factor"]),
        resample=int(image_config["resample"]),
    )
    model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, use_safetensors=True
    )
    config = model.config
    if (config.model_type, config.patch_size, config.hidden_size) != (
        "dinov2", PATCH_SIZE, HIDDEN_SIZE
    ):
        raise ValueError("Snapshot is not the expected DINOv2 ViT-B/14 model")
    if int(processor.resample) != int(Image.Resampling.BICUBIC):
        raise ValueError("Unexpected processor interpolation; expected bicubic")
    if len(processor.image_mean) != 3 or len(processor.image_std) != 3:
        raise ValueError("Expected three-channel normalization parameters")
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, processor


def preprocess_rgb(rgb: np.ndarray, processor: ProcessorConfig) -> torch.Tensor:
    """Resize one complete RGB array to 518 and apply official normalization."""
    if rgb.shape != (*SOURCE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB array with shape (1024, 1024, 3), got {rgb.shape} {rgb.dtype}")
    resized = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").resize(
        INPUT_SIZE, resample=Image.Resampling.BICUBIC
    )
    rgb = np.asarray(resized, dtype=np.float32)
    rgb *= float(processor.rescale_factor)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def preprocess_image(path: Path, processor: ProcessorConfig) -> torch.Tensor:
    """Resize the whole image directly to 518; apply official channel normalization."""
    with Image.open(path) as image:
        if image.size != SOURCE_SIZE or image.mode != "RGB":
            raise ValueError(f"Expected 1024x1024 RGB image: {path}")
        rgb = np.asarray(image, dtype=np.uint8).copy()
    return preprocess_rgb(rgb, processor)


@torch.inference_mode()
def extract_batch(
    model: torch.nn.Module, batch: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CLS, mean patch, and concatenated float32 global features."""
    if tuple(batch.shape[1:]) != (3, *INPUT_SIZE):
        raise ValueError(f"Expected batch shape [N, 3, 518, 518], got {tuple(batch.shape)}")
    outputs = model(pixel_values=batch.to(device))
    tokens = outputs.last_hidden_state
    if tuple(tokens.shape[1:]) != (PATCH_COUNT + 1, HIDDEN_SIZE):
        raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
    cls = tokens[:, 0].float().cpu().numpy().copy()
    mean_patch = tokens[:, 1:].mean(dim=1).float().cpu().numpy().copy()
    combined = np.concatenate((cls, mean_patch), axis=1)
    for name, array in (("cls", cls), ("mean_patch", mean_patch), ("combined", combined)):
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {name} embedding")
    return cls, mean_patch, combined
