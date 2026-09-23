"""Clean SegFormer-B3 with one optional FiLM injection before its classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerForImageClassification
from transformers.models.segformer.modeling_segformer import SegformerDecodeHead


class PresenceFiLM(nn.Module):
    """Apply continuous seven-class presence to a fused decoder feature."""

    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.channels = channels

    def forward(self, feature: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        if probability.ndim != 2 or probability.shape != (feature.shape[0], 7):
            raise ValueError("Presence must have shape [B, 7]")
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError("FiLM feature must have shape [B, C, H, W]")
        gamma, beta = self.mlp(probability.to(dtype=feature.dtype)).chunk(2, dim=1)
        return (1 + gamma[:, :, None, None]) * feature + beta[:, :, None, None]


class PresenceSegformer(nn.Module):
    """Same encoder/decoder for OFF and ON; only ON instantiates FiLM."""

    def __init__(self, pretrained_dir: str, conditioning: bool) -> None:
        super().__init__()
        pretrained = SegformerForImageClassification.from_pretrained(pretrained_dir)
        config = pretrained.config
        config.num_labels = 8
        self.encoder = pretrained.segformer
        self.decode_head = SegformerDecodeHead(config)
        self.film = PresenceFiLM(config.decoder_hidden_size) if conditioning else None
        self.conditioning = conditioning

    def forward(self, pixels: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Pixels must have shape [B, 3, H, W]")
        if probability.shape != (pixels.shape[0], 7):
            raise ValueError("Presence must have shape [B, 7]")
        states = self.encoder(pixels, output_hidden_states=True).hidden_states
        fused = []
        output_size = states[0].shape[-2:]
        for state, projection in zip(states, self.decode_head.linear_projections):
            height, width = state.shape[-2:]
            feature = projection(state).transpose(1, 2).reshape(pixels.shape[0], -1, height, width)
            fused.append(F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False))
        feature = self.decode_head.linear_fuse(torch.cat(fused[::-1], dim=1))
        feature = self.decode_head.activation(self.decode_head.batch_norm(feature))
        feature = self.decode_head.dropout(feature)
        if self.film is not None:
            feature = self.film(feature, probability)
        logits = self.decode_head.classifier(feature)
        return F.interpolate(logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False)
