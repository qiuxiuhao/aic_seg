"""Train a frozen-embedding linear class-presence probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.presence.metrics import foreground_macro_ap


@dataclass
class ProbeResult:
    state_dict: dict[str, torch.Tensor]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    best_epoch: int
    best_val_ap: float
    history: list[dict[str, float | int]]


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Apply the train-only scaler and return eight sigmoid probabilities."""
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            standardized = (features[start : start + batch_size] - scaler_mean) / scaler_std
            batch = torch.from_numpy(np.asarray(standardized, dtype=np.float32)).to(device)
            chunks.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    *,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> ProbeResult:
    """Select the best checkpoint using val_stratified foreground macro AP."""
    if x_train.ndim != 2 or x_val.shape[1] != x_train.shape[1]:
        raise ValueError("Train and validation feature dimensions differ")
    if y_train.shape != (len(x_train), 8) or y_val.shape != (len(x_val), 8):
        raise ValueError("Expected eight-class presence labels")
    positives = y_train.sum(axis=0).astype(np.float32)
    if np.any(positives == 0) or np.any(positives == len(y_train)):
        raise ValueError("Every class needs positive and negative training examples")

    scaler_mean = x_train.mean(axis=0).astype(np.float32)
    scaler_std = x_train.std(axis=0).astype(np.float32)
    scaler_std[scaler_std < 1e-6] = 1.0
    train_features = torch.from_numpy(((x_train - scaler_mean) / scaler_std).astype(np.float32)).to(device)
    train_labels = torch.from_numpy(y_train.astype(np.float32)).to(device)
    torch.manual_seed(seed)
    model = nn.Linear(x_train.shape[1], 8).to(device)
    pos_weight = torch.from_numpy((len(y_train) - positives) / positives).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_ap = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float | int]] = []
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total_loss = 0.0
        for start in range(0, len(x_train), batch_size):
            indices = order[start : start + batch_size].to(device)
            logits = model(train_features[indices])
            loss = criterion(logits, train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        probabilities = predict_probabilities(
            model, x_val, scaler_mean, scaler_std, device, batch_size
        )
        val_ap = foreground_macro_ap(y_val, probabilities)
        history.append({"epoch": epoch, "train_loss": total_loss / len(x_train), "val_foreground_macro_ap": val_ap})
        if val_ap > best_ap + 1e-6:
            best_ap = val_ap
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return ProbeResult(best_state, scaler_mean, scaler_std, best_epoch, best_ap, history)
