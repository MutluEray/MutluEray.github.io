"""EEGNet-sized backbone and the Z2 (mid-sagittal mirror) treatments.

Input trials have shape (batch, channels, time). The backbone returns one logit per trial
(positive = right-hand MI).

MirrorModel(mode='plain')       f(x) = g(x)
MirrorModel(mode='equivariant') f(x) = g(x) - g(Px)

The equivariant version lifts x to a function on Z2, (g(x), g(Px)), and projects it onto the
sign representation, so f(Px) = -f(x) holds exactly, in train and eval mode. Both copies go
through the backbone in one concatenated batch, so BatchNorm sees the mirror-symmetric union
of the two and the identity is not broken by batch statistics.
"""
from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet (Lawhern et al., 2018), single-logit output."""

    def __init__(self, n_channels: int, n_times: int, F1: int = 8, D: int = 2, F2: int = 16,
                 kernel_length: int = 64, dropout: float = 0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_features = self._features(torch.zeros(2, n_channels, n_times)).shape[1]
        self.head = nn.Linear(n_features, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block2(self.block1(x.unsqueeze(1)))
        return z.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x)).squeeze(-1)


class MirrorModel(nn.Module):
    def __init__(self, backbone: nn.Module, perm, mode: str = "plain"):
        super().__init__()
        if mode not in ("plain", "equivariant"):
            raise ValueError(mode)
        self.g = backbone
        self.mode = mode
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "plain":
            return self.g(x)
        n = x.shape[0]
        z = self.g(torch.cat([x, self.mirror(x)], dim=0))
        return z[:n] - z[n:]


def build_model(n_channels: int, n_times: int, perm, treatment: str, dropout: float = 0.25) -> MirrorModel:
    mode = "equivariant" if treatment == "equivariant" else "plain"
    return MirrorModel(EEGNet(n_channels, n_times, dropout=dropout), perm, mode=mode)
