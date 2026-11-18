"""Loading processed cohorts and mirror-commuting pre-processing.

Every processed cohort is one .npz in data/processed/ with:
  X             float32 (n_trials, n_channels, n_times), microvolts
  y             int64   (n_trials,)  0 = left-hand MI, 1 = right-hand MI
  subject       int64   (n_trials,)
  ch_names      str     (n_channels,)
  sfreq         float
  affected_side str     (n_trials,)  'L' / 'R' hemiplegic side, '' for healthy subjects
Optional clinical columns (per trial): nihss, mbi, mrs (float), lesion_hemisphere ('L'/'R'/''),
infratentorial, cortical (0/1), handedness ('right'/'left').

All transforms here commute with the mirror permutation P: channel selection on a
mirror-closed set, common-average reference, and one scalar scale per subject.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .channels import mirror_permutation, select_channels

PROCESSED = Path("data/processed")
CLINICAL_KEYS = ("nihss", "mbi", "mrs", "lesion_hemisphere", "infratentorial", "cortical", "handedness")


@dataclass
class Cohort:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    ch_names: list[str]
    sfreq: float
    affected_side: np.ndarray
    perm: np.ndarray
    clinical: dict = field(default_factory=dict)
    dropped_channels: list[str] = field(default_factory=list)


def common_average(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True)


def scale_per_subject(X: np.ndarray, subject: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for s in np.unique(subject):
        m = subject == s
        out[m] = X[m] / (X[m].std() + 1e-8)
    return out


def load_cohort(name: str, channels: str = "all", n_subjects: int = 0, subject_seed: int = 1234,
                exclude: tuple[int, ...] = ()) -> Cohort:
    path = PROCESSED / f"{name}.npz"
    d = np.load(path, allow_pickle=False)
    ch_names = [str(c) for c in d["ch_names"]]
    shared = None
    if channels.startswith("shared"):
        shared_path = PROCESSED / "shared_channels.json"
        if not shared_path.exists():
            raise SystemExit(f"{shared_path} not found: run prep_physionet.py and then prep_stroke.py first")
        shared = json.loads(shared_path.read_text())["channels"]
    idx, dropped = select_channels(ch_names, channels, shared)
    if not idx:
        raise ValueError(f"no channels left for set '{channels}'")

    X = d["X"][:, idx, :].astype(np.float32)
    y = d["y"].astype(np.int64)
    subject = d["subject"].astype(np.int64)
    affected = d["affected_side"].astype(str) if "affected_side" in d.files else np.full(len(y), "")
    clinical = {k: d[k] for k in CLINICAL_KEYS if k in d.files}

    if exclude:
        m = ~np.isin(subject, list(exclude))
        X, y, subject, affected = X[m], y[m], subject[m], affected[m]
        clinical = {k: v[m] for k, v in clinical.items()}

    if n_subjects:
        subs = np.unique(subject)
        if n_subjects < len(subs):
            keep = np.sort(np.random.default_rng(subject_seed).choice(subs, n_subjects, replace=False))
            m = np.isin(subject, keep)
            X, y, subject, affected = X[m], y[m], subject[m], affected[m]
            clinical = {k: v[m] for k, v in clinical.items()}

    names = [ch_names[i] for i in idx]
    X = scale_per_subject(common_average(X), subject).astype(np.float32)
    return Cohort(X=X, y=y, subject=subject, ch_names=names, sfreq=float(d["sfreq"]),
                  affected_side=affected, perm=mirror_permutation(names), clinical=clinical,
                  dropped_channels=dropped)


def canonicalize(X: np.ndarray, y: np.ndarray, affected_side: np.ndarray, perm: np.ndarray):
    """Mirror trials of right-hemiplegic patients and flip their labels.

    Afterwards every patient's affected hand is 'left' (label 0) and the lesion sits in the
    right hemisphere. Returns (X, y, flipped_mask).
    """
    flipped = affected_side == "R"
    if not np.isin(affected_side, ["L", "R"]).all():
        raise ValueError("canonicalization needs affected_side 'L' or 'R' for every trial")
    Xc, yc = X.copy(), y.copy()
    Xc[flipped] = Xc[flipped][:, perm]
    yc[flipped] = 1 - yc[flipped]
    return Xc, yc, flipped
