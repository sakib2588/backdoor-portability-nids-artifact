"""MNIST patch-trigger backdoor victim and its evaluation (vision control).

This is the positive control: the three detectors are KNOWN to catch a dirty-label
patch backdoor here. If they do not, the detector code is wrong -- fix it before
trusting any null on tabular NIDS. The vision control deliberately uses the classic
dirty-label patch (BadNets), NOT the clean-label SHAP trigger used on NIDS.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.torch_utils import batched_apply


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import datasets

    config.ensure_dirs()
    root = str(config.DATA_RAW / "mnist")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    x_train = (tr.data.float() / 255.0).unsqueeze(1)
    y_train = tr.targets.long()
    x_test = (te.data.float() / 255.0).unsqueeze(1)
    y_test = te.targets.long()
    return x_train, y_train, x_test, y_test


def add_patch(x: torch.Tensor, size: int = 4, value: float = 1.0) -> torch.Tensor:
    out = x.clone()
    out[..., -size:, -size:] = value
    return out


def poison_trainset(
    x: torch.Tensor, y: torch.Tensor, target: int, frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rng = np.random.default_rng(seed)
    non_target = np.where(y.numpy() != target)[0]
    n_poison = int(round(frac * len(non_target)))
    idx = rng.choice(non_target, size=n_poison, replace=False)
    idx = np.sort(idx)

    x_p = x.clone()
    y_p = y.clone()
    x_p[idx] = add_patch(x_p[idx])
    y_p[idx] = target
    return x_p, y_p, idx


class SmallCNN(nn.Module):
    def __init__(self, feat_dim: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(32 * 7 * 7, feat_dim)
        self.fc2 = nn.Linear(feat_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return F.relu(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def train_victim(
    x: torch.Tensor, y: torch.Tensor, seed: int, epochs: int = 3, device: str = "cpu",
    feat_dim: int = 128,
) -> SmallCNN:
    set_seed(seed)
    model = SmallCNN(feat_dim=feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def _predict(model: SmallCNN, x: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    return torch.cat(batched_apply(lambda xb: model(xb).argmax(1).cpu(), x, device))


def clean_accuracy(model: SmallCNN, x: torch.Tensor, y: torch.Tensor, device: str = "cpu") -> float:
    pred = _predict(model, x, device)
    return float((pred == y).float().mean())


def attack_success_rate(
    model: SmallCNN, x: torch.Tensor, y: torch.Tensor, target: int, device: str = "cpu"
) -> float:
    mask = y != target
    x_trig = add_patch(x[mask])
    pred = _predict(model, x_trig, device)
    return float((pred == target).float().mean())


@torch.no_grad()
def penultimate_features(model: SmallCNN, x: torch.Tensor, device: str = "cpu") -> np.ndarray:
    model.eval()
    return np.concatenate(
        batched_apply(lambda xb: model.features(xb).cpu().numpy(), x, device), axis=0)
