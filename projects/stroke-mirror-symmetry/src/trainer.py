"""Training with early stopping (stall, stuck-at-chance, success) and evaluation helpers."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


@dataclass
class TrainConfig:
    max_epochs: int = 60
    min_epochs: int = 8
    patience: int = 8             # stall: epochs without val-loss improvement
    chance_epoch: int = 12        # stuck-at-chance check happens at this epoch
    chance_threshold: float = 0.55
    success_threshold: float = 0.95
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-3
    dropout: float = 0.25


def balanced_acc(y: np.ndarray, logits: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, (logits > 0).astype(int)))


def safe_auc(y: np.ndarray, logits: np.ndarray) -> float:
    return float(roc_auc_score(y, logits)) if len(np.unique(y)) == 2 else float("nan")


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, device: str, mirror: bool = False,
            batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).float().to(device)
        if mirror:
            xb = model.mirror(xb)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _val_metrics(model, X, y, device):
    logits = predict(model, X, device)
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(y).float()).item()
    return loss, balanced_acc(y, logits)


def fit(model: torch.nn.Module, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
        treatment: str, lam: float, cfg: TrainConfig, seed: int, device: str = "cpu") -> dict:
    """Train in place; restores the best (lowest validation loss) weights. Returns a log dict."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()

    best_loss, best_state, best_epoch = math.inf, copy.deepcopy(model.state_dict()), 0
    best_bacc = 0.0
    stop_reason = "max_epochs"
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), cfg.batch_size):
            b = order[start:start + cfg.batch_size]
            if len(b) < 2:      # BatchNorm needs more than one sample
                continue
            xb = Xtr_t[b].clone()
            yb = ytr_t[b].clone()
            if treatment == "augment":
                flip = torch.from_numpy(rng.random(len(b)) < 0.5)
                if flip.any():
                    xb[flip] = xb[flip][:, model.perm.cpu()]
                    yb[flip] = 1.0 - yb[flip]
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            if lam > 0:
                # penalise the mirror-invariant part S(x) = (f(x) + f(Px)) / 2
                loss = loss + lam * (logits + model(model.mirror(xb))).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        va_loss, va_bacc = _val_metrics(model, Xva, yva, device)
        best_bacc = max(best_bacc, va_bacc)
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch = va_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

        if va_bacc >= cfg.success_threshold:
            stop_reason = "success"
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
            break
        if epoch == cfg.chance_epoch and best_bacc < cfg.chance_threshold:
            stop_reason = "stuck_at_chance"
            break
        if epoch >= cfg.min_epochs and epoch - best_epoch >= cfg.patience:
            stop_reason = "stall"
            break

    model.load_state_dict(best_state)
    return {"epochs": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss,
            "best_val_bacc": best_bacc, "stop_reason": stop_reason}
