"""Direct DINO cache alignment, dataset, and single-point FiLM model."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import SegformerForImageClassification
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

from src.dino.download import sha256_file
from src.segmentation.data import encode_image_mask, load_image_mask
from src.segmentation.geometry import STATE_TO_INDEX, TRANSFORM_STATES, apply_stage03_geometry


DINO_DIM = 1536
DECODER_DIM = 768
EXPECTED_SPLIT_COUNTS = {"train": 5597, "val_stratified": 700, "val_domain": 699}


def _split_ids(data_dir: Path, split: str) -> list[str]:
    ids = [line.strip() for line in (data_dir / "splits" / f"{split}.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines()]
    if len(ids) != EXPECTED_SPLIT_COUNTS[split] or len(ids) != len(set(ids)) or not all(ids):
        raise ValueError(f"Unexpected or duplicate IDs in {split}.txt")
    return ids


class DirectDinoEmbeddingStore:
    """Strict ID/state index over Stage 01 r0 and Stage 04 train D4 arrays."""

    def __init__(
        self,
        data_dir: Path,
        base_embedding_dir: Path,
        augmented_train_dir: Path,
        require_full_augmented: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.base_embedding_dir = base_embedding_dir
        self.augmented_train_dir = augmented_train_dir
        self.split_ids = {split: _split_ids(data_dir, split) for split in EXPECTED_SPLIT_COUNTS}
        self.base, self.base_rows, self.base_metadata = self._load_base()
        self.augmented, self.augmented_rows, self.augmented_metadata = self._load_augmented(
            require_full_augmented
        )

    def _load_base(self) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
        directory = self.base_embedding_dir
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata.get("scope") != "full"
            or metadata.get("count") != 6996
            or metadata.get("model_id") != "facebook/dinov2-base"
            or metadata.get("features", {}).get("combined") != DINO_DIM
            or metadata.get("preprocessing", {}).get("resize_size") != [518, 518]
            or metadata.get("preprocessing", {}).get("resize_method") != "PIL bicubic, entire image, no crop"
        ):
            raise ValueError("Stage 01 cache metadata does not match accepted Combined 1536 full-image features")
        for split in EXPECTED_SPLIT_COUNTS:
            if metadata.get("split_sha256", {}).get(split) != sha256_file(
                self.data_dir / "splits" / f"{split}.txt"
            ):
                raise ValueError(f"Stage 01 cache split hash differs for {split}")
        mapping: dict[str, int] = {}
        with (directory / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["row", "image_id", "split", "image_path"]:
                raise ValueError("Unexpected Stage 01 manifest columns")
            for expected_row, row in enumerate(reader):
                image_id = row["image_id"]
                if int(row["row"]) != expected_row or image_id in mapping:
                    raise ValueError("Stage 01 manifest row order or ID uniqueness failed")
                mapping[image_id] = expected_row
        array = np.load(directory / "combined.npy", mmap_mode="r")
        if array.shape != (6996, DINO_DIM) or array.dtype != np.float32 or len(mapping) != 6996:
            raise ValueError("Unexpected Stage 01 Combined array")
        expected_order = [image_id for split in EXPECTED_SPLIT_COUNTS for image_id in self.split_ids[split]]
        if list(mapping) != expected_order:
            raise ValueError("Stage 01 manifest does not match fixed split order")
        return array, mapping, metadata

    def _load_augmented(
        self, require_full: bool
    ) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
        directory = self.augmented_train_dir
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        expected_scope = "full" if require_full else metadata.get("scope")
        if (
            metadata.get("scope") != expected_scope
            or metadata.get("model_id") != "facebook/dinov2-base"
            or metadata.get("resolved_revision") != self.base_metadata.get("resolved_revision")
            or metadata.get("features", {}).get("combined") != DINO_DIM
            or metadata.get("transforms", {}).get("canonical_states") != list(TRANSFORM_STATES)
            or metadata.get("train_split_sha256") != sha256_file(self.data_dir / "splits" / "train.txt")
            or metadata.get("stage01_metadata_sha256") != sha256_file(self.base_embedding_dir / "metadata.json")
            or metadata.get("stage01_manifest_sha256") != sha256_file(self.base_embedding_dir / "manifest.csv")
            or metadata.get("stage01_combined_sha256") != sha256_file(self.base_embedding_dir / "combined.npy")
        ):
            raise ValueError("Augmented DINO cache metadata does not match Stage 01 or the fixed D4 protocol")
        image_count = int(metadata.get("image_count", -1))
        if require_full and image_count != EXPECTED_SPLIT_COUNTS["train"]:
            raise ValueError("Formal training requires all 5597 train IDs in the augmented cache")
        array = np.load(directory / "combined.npy", mmap_mode="r")
        if array.shape != (image_count, 8, DINO_DIM) or array.dtype != np.float32:
            raise ValueError(f"Unexpected augmented Combined array: {array.shape} {array.dtype}")
        if metadata.get("combined_sha256") != sha256_file(directory / "combined.npy"):
            raise ValueError("Augmented Combined SHA256 differs from metadata")
        if not np.isfinite(array).all():
            raise ValueError("Augmented Combined array contains non-finite values")

        mapping: dict[str, int] = {}
        pair_count = 0
        with (directory / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = ["array_row", "image_id", "state_index", "transform_state", "source"]
            if reader.fieldnames != expected_fields:
                raise ValueError("Unexpected augmented manifest columns")
            seen_pairs: set[tuple[str, str]] = set()
            for row in reader:
                image_id, state = row["image_id"], row["transform_state"]
                array_row, state_index = int(row["array_row"]), int(row["state_index"])
                if state not in STATE_TO_INDEX or state_index != STATE_TO_INDEX[state]:
                    raise ValueError("Augmented manifest has an invalid state mapping")
                if array_row >= image_count or (image_id, state) in seen_pairs:
                    raise ValueError("Augmented manifest has an invalid row or duplicate pair")
                if image_id in mapping and mapping[image_id] != array_row:
                    raise ValueError("One image ID maps to multiple augmented rows")
                mapping[image_id] = array_row
                seen_pairs.add((image_id, state))
                pair_count += 1
        expected_ids = self.split_ids["train"][:image_count]
        if list(mapping) != expected_ids or pair_count != image_count * 8:
            raise ValueError("Augmented manifest is incomplete or not in train split order")
        return array, mapping, metadata

    def embedding(self, image_id: str, split: str, state: str) -> np.ndarray:
        if split == "train" and state != "r0":
            if image_id not in self.augmented_rows:
                raise KeyError(f"No augmented DINO embedding for {image_id}")
            return np.asarray(self.augmented[self.augmented_rows[image_id], STATE_TO_INDEX[state]])
        if state != "r0":
            raise ValueError(f"Validation split {split} must use r0, got {state}")
        return np.asarray(self.base[self.base_rows[image_id]])


class DirectDinoSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]]):
    """Use the exact Stage 03 image transform and select the matching DINO state."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        store: DirectDinoEmbeddingStore,
        augment: bool,
        image_ids: list[str] | None = None,
    ) -> None:
        if split not in EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"Unknown split: {split}")
        if augment != (split == "train"):
            raise ValueError("Only train may use Stage 03 random geometry augmentation")
        self.data_dir = data_dir
        self.split = split
        self.store = store
        self.augment = augment
        self.image_ids = store.split_ids[split] if image_ids is None else image_ids
        if not self.image_ids or any(image_id not in store.split_ids[split] for image_id in self.image_ids):
            raise ValueError(f"Dataset IDs are not a non-empty subset of {split}")
        if augment and any(image_id not in store.augmented_rows for image_id in self.image_ids):
            raise ValueError("Train dataset contains an ID missing from augmented DINO cache")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        image_id = self.image_ids[index]
        image, mask = load_image_mask(self.data_dir, image_id)
        state = "r0"
        if self.augment:
            image, mask, state = apply_stage03_geometry(image, mask, random)
        pixels, target = encode_image_mask(image, mask)
        embedding = self.store.embedding(image_id, self.split, state)
        if embedding.shape != (DINO_DIM,) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid DINO embedding for {image_id}/{state}")
        return pixels, target, torch.from_numpy(embedding.copy()), image_id, state


class DirectDinoFiLM(nn.Module):
    """Map Combined DINO features to one 768-channel FiLM modulation."""

    def __init__(self, channels: int = DECODER_DIM, hidden: int = 256) -> None:
        super().__init__()
        if channels != DECODER_DIM or hidden != 256:
            raise ValueError("Stage 04 fixes channels=768 and hidden=256")
        self.norm = nn.LayerNorm(DINO_DIM)
        self.adapter = nn.Sequential(
            nn.Linear(DINO_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.channels = channels

    def modulation(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[1] != DINO_DIM:
            raise ValueError("DINO Combined embedding must have shape [B, 1536]")
        output = self.adapter(self.norm(embedding))
        if output.shape != (embedding.shape[0], 2 * self.channels):
            raise ValueError("Direct DINO adapter must output [B, 1536]")
        return output.chunk(2, dim=1)

    def forward(self, feature: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError("FiLM feature must have shape [B, 768, H, W]")
        gamma, beta = self.modulation(embedding.to(dtype=feature.dtype))
        return (1 + gamma[:, :, None, None]) * feature + beta[:, :, None, None]


class DirectDinoSegformer(nn.Module):
    """Stage 03 SegFormer-B3 with Direct DINO FiLM at the same decoder point."""

    def __init__(self, pretrained_dir: str) -> None:
        super().__init__()
        pretrained = SegformerForImageClassification.from_pretrained(pretrained_dir)
        config = pretrained.config
        config.num_labels = 8
        if config.decoder_hidden_size != DECODER_DIM:
            raise ValueError(f"Expected 768 decoder channels, got {config.decoder_hidden_size}")
        self.encoder = pretrained.segformer
        self.decode_head = SegformerDecodeHead(config)
        self.film = DirectDinoFiLM(config.decoder_hidden_size, hidden=256)

    def decode_feature(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return the same 768-channel fused decoder feature used by Stage 04."""
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Pixels must have shape [B, 3, H, W]")
        states = self.encoder(pixels, output_hidden_states=True).hidden_states
        fused = []
        output_size = states[0].shape[-2:]
        for state, projection in zip(states, self.decode_head.linear_projections):
            height, width = state.shape[-2:]
            feature = projection(state).transpose(1, 2).reshape(pixels.shape[0], -1, height, width)
            fused.append(F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False))
        feature = self.decode_head.linear_fuse(torch.cat(fused[::-1], dim=1))
        feature = self.decode_head.activation(self.decode_head.batch_norm(feature))
        return self.decode_head.dropout(feature)

    def classify_feature(self, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """Apply the shared eight-class head and restore segmentation resolution."""
        logits = self.decode_head.classifier(feature)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, pixels: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.shape != (pixels.shape[0], DINO_DIM):
            raise ValueError("DINO Combined embedding must have shape [B, 1536]")
        feature = self.decode_feature(pixels)
        feature = self.film(feature, embedding)
        return self.classify_feature(feature, pixels.shape[-2:])
