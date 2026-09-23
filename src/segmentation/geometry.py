"""Stage 03 geometry augmentation and its canonical D4 state mapping."""

from __future__ import annotations

import random
from typing import Protocol

import numpy as np


TRANSFORM_STATES = (
    "r0",
    "r90",
    "r180",
    "r270",
    "flip_r0",
    "flip_r90",
    "flip_r180",
    "flip_r270",
)
STATE_TO_INDEX = {state: index for index, state in enumerate(TRANSFORM_STATES)}
TRANSFORM_DEFINITIONS = {
    "r0": "identity",
    "r90": "rotate original 90 degrees counter-clockwise",
    "r180": "rotate original 180 degrees counter-clockwise",
    "r270": "rotate original 270 degrees counter-clockwise",
    "flip_r0": "horizontal flip, then rotate 0 degrees counter-clockwise",
    "flip_r90": "horizontal flip, then rotate 90 degrees counter-clockwise",
    "flip_r180": "horizontal flip, then rotate 180 degrees counter-clockwise",
    "flip_r270": "horizontal flip, then rotate 270 degrees counter-clockwise",
}


class GeometryRng(Protocol):
    def random(self) -> float: ...
    def randrange(self, stop: int) -> int: ...


def canonical_state(horizontal: bool, vertical: bool, turns: int) -> str:
    """Map Stage 03's H-flip, V-flip, rotation sequence to one D4 state."""
    if turns not in range(4):
        raise ValueError("turns must be one of 0, 1, 2, 3")
    if horizontal == vertical:
        canonical_turns = (turns + (2 if horizontal else 0)) % 4
        return f"r{canonical_turns * 90}"
    canonical_turns = (turns + (2 if vertical else 0)) % 4
    return f"flip_r{canonical_turns * 90}"


def apply_transform(array: np.ndarray, state: str) -> np.ndarray:
    """Apply a canonical state: optional horizontal flip, then CCW rotation."""
    if state not in STATE_TO_INDEX:
        raise ValueError(f"Unknown transform state: {state}")
    transformed = np.flip(array, axis=1) if state.startswith("flip_") else array
    angle = int(state.rsplit("r", maxsplit=1)[1])
    return np.rot90(transformed, angle // 90)


def apply_stage03_geometry(
    image: np.ndarray,
    mask: np.ndarray,
    rng: GeometryRng = random,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Preserve Stage 03's exact random calls, probabilities and operation order."""
    horizontal = rng.random() < 0.5
    vertical = rng.random() < 0.5
    turns = rng.randrange(4)
    if horizontal:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    if vertical:
        image, mask = np.flip(image, 0), np.flip(mask, 0)
    image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    return image, mask, canonical_state(horizontal, vertical, turns)
