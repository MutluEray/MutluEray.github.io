"""Grouped cross-subject CV for every symmetry treatment.

Writes one row per (treatment, lam, seed, fold, test subject) to results/<run>/cv.csv and the
test logits (for f(x) and f(Px)) to results/<run>/logits/. Re-running skips finished fits, so
the script can be stopped and resumed.

Secondary check only: the primary analysis is run_ts.py. Early stopping is tight (see
src/trainer.py: stall 8 epochs, stuck-at-chance at epoch 12, max 60).

Examples
  python3 run_cv.py --dataset physionet --channels sensorimotor --n-subjects 50 --seeds 0
  python3 run_cv.py --dataset stroke --channels shared --treatments none augment equivariant canonical
  python3 run_cv.py --dataset stroke --channels shared --treatments soft --lams 0.01 0.1 1 10
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import canonicalize, load_cohort
from src.models import build_model
from src.trainer import TrainConfig, balanced_acc, fit, predict, safe_auc

NEURAL = ("none", "augment", "equivariant", "soft", "canonical")
CLASSICAL = ("csp_lda", "ts_lr")


def make_folds(subjects: np.ndarray, n_folds: int, fold_seed: int = 0):
    subs = np.unique(subjects)
    rng = np.random.default_rng(fold_seed)
    rng.shuffle(subs)
    return [np.sort(f) for f in np.array_split(subs, n_folds)]


def pick_val_subjects(train_subs: np.ndarray, fold: int, frac: float = 0.1) -> np.ndarray:
    n_val = max(3, int(round(frac * len(train_subs))))
    rng = np.random.default_rng(1000 + fold)
    return np.sort(rng.choice(train_subs, n_val, replace=False))


def sym_fraction(logit: np.ndarray, logit_mirror: np.ndarray) -> float:
    """Share of the decision carried by the mirror-invariant part S = (f(x)+f(Px))/2."""
    A = np.abs(logit - logit_mirror) / 2
    S = np.abs(logit + logit_mirror) / 2
    denom = A.mean() + S.mean()
    return float(S.mean() / denom) if denom > 0 else float("nan")


def classical_fit_predict(name, Xtr, ytr, Xte, perm, sfreq):
    import mne
    from mne.decoding import CSP
    from mne.filter import filter_data
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    mne.set_log_level("ERROR")
    band = lambda X: filter_data(X.astype(np.float64), sfreq, 8.0, 30.0, verbose="ERROR")
    Xtr_b, Xte_b = band(Xtr), band(Xte)
    if name == "csp_lda":
        clf = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                            LinearDiscriminantAnalysis())
    else:
        clf = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"),
                            LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(Xtr_b, ytr)
    return clf.decision_function(Xte_b), clf.decision_function(Xte_b[:, perm])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="name of data/processed/<name>.npz")
    p.add_argument("--channels", default="all", choices=["all", "sensorimotor", "shared", "shared_sensorimotor"])
    p.add_argument("--treatments", nargs="+", default=["none", "augment", "equivariant", "csp_lda", "ts_lr"])
    p.add_argument("--lams", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0],
                   help="penalty strengths for the 'soft' treatment")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--only-folds", nargs="+", type=int, default=None, help="run a subset of folds")
    p.add_argument("--n-subjects", type=int, default=0, help="random subset of subjects (0 = all)")
    p.add_argument("--device", default="cpu", help="cpu or mps")
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--threads", type=int, default=4, help="CPU threads (performance cores only is often fastest)")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or 1)))
    cohort = load_cohort(args.dataset, args.channels, args.n_subjects)
    run = f"{args.dataset}_{args.channels}" + (f"_n{args.n_subjects}" if args.n_subjects else "") + (f"_{args.tag}" if args.tag else "")
    out_dir = Path("results") / run
    (out_dir / "logits").mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cv.csv"

    print(f"[{run}] {len(np.unique(cohort.subject))} subjects, {len(cohort.y)} trials, "
          f"{len(cohort.ch_names)} channels, {cohort.X.shape[2]} samples @ {cohort.sfreq:g} Hz")
    print(f"channels: {' '.join(cohort.ch_names)}")
    if cohort.dropped_channels:
        print(f"dropped (no mirror partner): {' '.join(cohort.dropped_channels)}")
    print(f"class balance: {np.bincount(cohort.y).tolist()}")

    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(zip(prev.treatment, prev.lam.round(6), prev.seed, prev.fold))

    jobs = []
    for t in args.treatments:
        if t not in NEURAL + CLASSICAL:
            raise SystemExit(f"unknown treatment {t}")
        lams = args.lams if t == "soft" else [0.0]
        seeds = [0] if t in CLASSICAL else args.seeds    # classical models are deterministic
        for lam in lams:
            for seed in seeds:
                jobs.append((t, float(lam), seed))

    folds = make_folds(cohort.subject, args.folds)
    fold_ids = args.only_folds if args.only_folds is not None else range(len(folds))
    cfg = TrainConfig(max_epochs=args.max_epochs)
    n_ch, n_t = cohort.X.shape[1], cohort.X.shape[2]

    for t, lam, seed in jobs:
        for k in fold_ids:
            if (t, round(lam, 6), seed, k) in done:
                continue
            t0 = time.time()
            test_subs = folds[k]
            train_subs = np.setdiff1d(np.unique(cohort.subject), test_subs)
            val_subs = pick_val_subjects(train_subs, k)
            fit_subs = np.setdiff1d(train_subs, val_subs)

            X, y = cohort.X, cohort.y
            flipped = np.zeros(len(y), dtype=bool)
            if t == "canonical":
                X, y, flipped = canonicalize(X, y, cohort.affected_side, cohort.perm)

            m_fit = np.isin(cohort.subject, fit_subs)
            m_val = np.isin(cohort.subject, val_subs)
            m_te = np.isin(cohort.subject, test_subs)
            info = {"epochs": 0, "best_epoch": 0, "stop_reason": "n/a", "best_val_bacc": np.nan}

            if t in CLASSICAL:
                m_tr = m_fit | m_val
                logit, logit_m = classical_fit_predict(t, X[m_tr], y[m_tr], X[m_te], cohort.perm, cohort.sfreq)
            else:
                model = build_model(n_ch, n_t, cohort.perm, t, dropout=cfg.dropout)
                info = fit(model, X[m_fit], y[m_fit], X[m_val], y[m_val],
                           treatment=t, lam=lam, cfg=cfg, seed=seed, device=args.device)
                logit = predict(model, X[m_te], args.device)
                logit_m = predict(model, X[m_te], args.device, mirror=True)

            # map canonical-frame outputs back to the original left/right frame
            f_te = flipped[m_te]
            logit, logit_m = logit.copy(), logit_m.copy()
            logit[f_te], logit_m[f_te] = -logit[f_te], -logit_m[f_te]
            y_te, s_te = cohort.y[m_te], cohort.subject[m_te]

            np.savez(out_dir / "logits" / f"{t}_lam{lam:g}_s{seed}_f{k}.npz",
                     subject=s_te, y=y_te, logit=logit, logit_mirror=logit_m)
            rows = []
            for s in np.unique(s_te):
                m = s_te == s
                rows.append({"dataset": args.dataset, "channels": args.channels, "treatment": t,
                             "lam": lam, "seed": seed, "fold": k, "subject": int(s),
                             "n_trials": int(m.sum()), "bal_acc": balanced_acc(y_te[m], logit[m]),
                             "auc": safe_auc(y_te[m], logit[m]),
                             "sym_frac": sym_fraction(logit[m], logit_m[m]),
                             "epochs": info["epochs"], "best_epoch": info["best_epoch"],
                             "stop_reason": info["stop_reason"], "val_bacc": info["best_val_bacc"],
                             "n_fit_subjects": len(fit_subs), "n_channels": n_ch})
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            fold_acc = balanced_acc(y_te, logit)
            print(f"{t:<12} lam={lam:<5g} seed={seed} fold={k}: bacc={fold_acc:.3f} "
                  f"stop={info['stop_reason']}@{info['epochs']} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.read_csv(csv_path)
    summary = (res.groupby(["treatment", "lam", "subject"]).bal_acc.mean()
               .groupby(["treatment", "lam"]).agg(["mean", "std", "count"]))
    print("\nper-subject balanced accuracy (averaged over seeds):")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
