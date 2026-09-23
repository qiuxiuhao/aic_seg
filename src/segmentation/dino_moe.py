"""DINO-conditioned four-expert soft MoE on the Stage 04 FiLM feature."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.segmentation.dino_conditioning import (
    DECODER_DIM,
    DINO_DIM,
    DirectDinoSegformer,
)


NUM_EXPERTS = 4
ROUTER_HIDDEN = 256
EXPERT_HIDDEN = 192


class DinoSoftRouter(nn.Module):
    """Produce four dense routing weights from one Combined DINO embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(DINO_DIM),
            nn.Linear(DINO_DIM, ROUTER_HIDDEN),
            nn.GELU(),
            nn.Linear(ROUTER_HIDDEN, NUM_EXPERTS),
        )

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[1] != DINO_DIM:
            raise ValueError("Router input must have shape [B, 1536]")
        logits = self.network(embedding)
        weights = torch.softmax(logits, dim=1)
        if logits.shape != (embedding.shape[0], NUM_EXPERTS):
            raise ValueError("Router logits must have shape [B, 4]")
        return logits, weights


class ExpertAdapter(nn.Module):
    """One lightweight 768→192→768 spatial residual adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Conv2d(DECODER_DIM, EXPERT_HIDDEN, kernel_size=1)
        self.activation = nn.GELU()
        self.up = nn.Conv2d(EXPERT_HIDDEN, DECODER_DIM, kernel_size=1)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != DECODER_DIM:
            raise ValueError("Expert input must have shape [B, 768, H, W]")
        return self.up(self.activation(self.down(feature)))


class DinoSoftMoE(nn.Module):
    """Execute all experts and add their DINO-routed weighted residual."""

    def __init__(self) -> None:
        super().__init__()
        self.router = DinoSoftRouter()
        self.experts = nn.ModuleList(ExpertAdapter() for _ in range(NUM_EXPERTS))

    def forward(
        self, feature: torch.Tensor, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, weights = self.router(embedding.to(dtype=feature.dtype))
        delta = torch.zeros_like(feature)
        for index, expert in enumerate(self.experts):
            expert_output = expert(feature)
            if expert_output.shape != feature.shape:
                raise ValueError("Every Expert output must match the 768-channel input feature")
            delta = delta + weights[:, index, None, None, None] * expert_output
        return feature + delta, logits, weights


class DinoSoftMoESegformer(DirectDinoSegformer):
    """Stage 04 Direct DINO FiLM followed by a four-expert soft MoE."""

    def __init__(self, pretrained_dir: str) -> None:
        super().__init__(pretrained_dir)
        self.moe = DinoSoftMoE()

    def routing_weights(self, embedding: torch.Tensor) -> torch.Tensor:
        """Expose router-only inference for utilization reports."""
        return self.moe.router(embedding)[1]

    def forward(
        self,
        pixels: torch.Tensor,
        embedding: torch.Tensor,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if embedding.shape != (pixels.shape[0], DINO_DIM):
            raise ValueError("DINO Combined embedding must have shape [B, 1536]")
        feature = self.decode_feature(pixels)
        film_feature = self.film(feature, embedding)
        moe_feature, _, weights = self.moe(film_feature, embedding)
        logits = self.classify_feature(moe_feature, pixels.shape[-2:])
        return (logits, weights) if return_routing else logits


class RoutingAccumulator:
    """Accumulate dense router utilization without adding a training loss."""

    def __init__(self, num_experts: int = NUM_EXPERTS) -> None:
        self.num_experts = num_experts
        self.weight_sum = torch.zeros(num_experts, dtype=torch.float64)
        self.top1_count = torch.zeros(num_experts, dtype=torch.int64)
        self.entropy_sum = 0.0
        self.sample_count = 0

    def update(self, weights: torch.Tensor) -> None:
        detached = weights.detach().float().cpu()
        if detached.ndim != 2 or detached.shape[1] != self.num_experts:
            raise ValueError(f"Routing weights must have shape [B, {self.num_experts}]")
        if not torch.isfinite(detached).all():
            raise ValueError("Routing weights contain non-finite values")
        if not torch.allclose(
            detached.sum(dim=1), torch.ones(detached.shape[0]), atol=1e-3, rtol=1e-3
        ):
            raise ValueError("Routing weights do not sum to one")
        self.weight_sum += detached.double().sum(dim=0)
        self.top1_count += torch.bincount(
            detached.argmax(dim=1), minlength=self.num_experts
        )
        self.entropy_sum += float(
            (-(detached * detached.clamp_min(1e-12).log()).sum(dim=1)).sum()
        )
        self.sample_count += detached.shape[0]

    def summary(self) -> dict[str, object]:
        if self.sample_count == 0:
            raise ValueError("No routing samples were accumulated")
        mean_weight = (self.weight_sum / self.sample_count).tolist()
        top1_fraction = (self.top1_count.double() / self.sample_count).tolist()
        entropy = self.entropy_sum / self.sample_count
        dominant_expert = max(range(self.num_experts), key=mean_weight.__getitem__)
        return {
            "sample_count": self.sample_count,
            "mean_weight": mean_weight,
            "top1_fraction": top1_fraction,
            "mean_entropy": entropy,
            "normalized_mean_entropy": entropy / math.log(self.num_experts),
            "dominant_expert": dominant_expert,
            "dominant_mean_weight": mean_weight[dominant_expert],
            "router_collapse_threshold": 0.9,
            "router_collapsed": mean_weight[dominant_expert] > 0.9,
        }
