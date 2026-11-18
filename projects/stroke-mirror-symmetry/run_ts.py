"""Primary analysis: mirror-symmetry treatments on a Riemannian tangent-space linear decoder.

Why this model. Each trial becomes a covariance matrix per band (mu 8-13, beta 13-30 Hz).
Covariances are re-centred per subject (Zanini et al. 2018; label-free, uses that subject's own
trials), then mapped to the tangent space at the identity. The mirror acts on these tangent
vectors by a permutation Pi. Every tangent vector splits uniquely into
    t_a = (t - Pi t)/2   (sign representation: flips under the mirror)
    t_s = (t + Pi t)/2   (trivial representation: unchanged under the mirror)
which is the isotypic decomposition of the Z2 representation. A linear decoder is exactly
equivariant (f(Pi t) = -f(t)) if and only if it uses t_a only and has no intercept. So every
treatment is a choice of how much weight t_s may carry:

  none         f = w.t + b                      (t_s used freely)
  augment      none, trained on {t, y} + {Pi t, 1-y}
  equivariant  f = w.t_a                        (t_s forbidden)
  soft_a       f = w1.t_a + w2.(a t_s) + b      (L2 penalty on the t_s weights scales as 1/a^2)
  canonical    mirror right-hemiplegic patients and flip their labels, then none

Fits take seconds; no training epochs, no seeds. Regularisation C is chosen per outer fold by
grouped inner CV on the training subjects only.

  python3 run_ts.py --dataset physionet --channels shared --n-subjects 50
  python3 run_ts.py --dataset stroke --channels shared
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.estimation import Covariances
try:                                    # pyriemann >= 0.10
    from pyriemann.geometry.base import invsqrtm, logm
    from pyriemann.geometry.mean import mean_riemann
except ImportError:                     # older pyriemann
    from pyriemann.utils.base import invsqrtm, logm
    from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.data import load_cohort

BANDS = ((8.0, 13.0), (13.0, 30.0))
CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def tangent_features(X, subject, sfreq):
    n, c, _ = X.shape
    iu = np.triu_indices(c)
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    blocks = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        C = Covariances("oas").fit_transform(sosfiltfilt(sos, X, axis=-1))
        T = np.empty((n, len(iu[0])))
        for s in np.unique(subject):
            m = subject == s
            W = invsqrtm(mean_riemann(C[m]))
            L = logm(W @ C[m] @ W)
            T[m] = L[:, iu[0], iu[1]] * weight
        blocks.append(T)
    return np.concatenate(blocks, axis=1)


def vector_permutation(perm, n_channels, n_bands):
    """Index Pi on tangent vectors such that t(P C P^T) = t(C)[Pi]."""
    iu = np.triu_indices(n_channels)
    pos = {(i, j): k for k, (i, j) in enumerate(zip(*iu))}
    base = np.array([pos[tuple(sorted((perm[i], perm[j])))] for i, j in zip(*iu)])
    d = len(base)
    return np.concatenate([base + b * d for b in range(n_bands)])


def feature_map(T, Pi, kind, alpha):
    if kind == "equivariant":
        return (T - T[:, Pi]) / 2
    if kind == "soft":
        return np.hstack([(T - T[:, Pi]) / 2, alpha * (T + T[:, Pi]) / 2])
    return T


def train_set(T, y, kind, Pi):
    if kind == "augment":
        return np.vstack([T, T[:, Pi]]), np.r_[y, 1 - y]
    return T, y


def fit_one(T, y, kind, Pi, alpha, C):
    Tt, yt = train_set(T, y, kind, Pi)
    clf = LogisticRegression(C=C, fit_intercept=(kind != "equivariant"), max_iter=3000)
    clf.fit(feature_map(Tt, Pi, kind, alpha), yt)
    return clf


def fit_select(T, y, g, kind, Pi, alpha, inner_folds=3):
    scores = []
    for C in CS:
        s = []
        for a, b in GroupKFold(inner_folds).split(T, y, g):
            clf = fit_one(T[a], y[a], kind, Pi, alpha, C)
            s.append(balanced_accuracy_score(y[b], clf.predict(feature_map(T[b], Pi, kind, alpha))))
        scores.append(np.mean(s))
    best = CS[int(np.argmax(scores))]
    return fit_one(T, y, kind, Pi, alpha, best), best


def make_folds(subjects, n_folds, fold_seed=0):
    subs = np.unique(subjects)
    if n_folds <= 0 or n_folds >= len(subs):
        return [np.array([s]) for s in subs]                     # leave-one-subject-out
    rng = np.random.default_rng(fold_seed)
    rng.shuffle(subs)
    return [np.sort(f) for f in np.array_split(subs, n_folds)]


def sym_fraction(logit, logit_mirror):
    A = np.abs(logit - logit_mirror) / 2
    S = np.abs(logit + logit_mirror) / 2
    denom = A.mean() + S.mean()
    return float(S.mean() / denom) if denom > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--channels", default="shared",
                   choices=["all", "sensorimotor", "shared", "shared_sensorimotor"])
    p.add_argument("--treatments", nargs="+", default=["none", "augment", "equivariant", "soft", "canonical"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.2, 0.1, 0.05])
    p.add_argument("--folds", type=int, default=10, help="0 = leave-one-subject-out")
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--exclude", nargs="+", type=int, default=[], help="subject ids to leave out")
    p.add_argument("--tag", default="", help="suffix for the output name, e.g. excl")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    t0 = time.time()
    co = load_cohort(args.dataset, args.channels, args.n_subjects, exclude=tuple(args.exclude))
    T = tangent_features(co.X, co.subject, co.sfreq)
    Pi = vector_permutation(co.perm, len(co.ch_names), len(BANDS))
    assert np.array_equal(Pi[Pi], np.arange(len(Pi)))
    run = f"ts_{args.dataset}_{args.channels}" + (f"_n{args.n_subjects}" if args.n_subjects else "") \
        + (f"_{args.tag}" if args.tag else "")
    out_dir = Path("results") / run
    (out_dir / "logits").mkdir(parents=True, exist_ok=True)
    print(f"[{run}] {len(np.unique(co.subject))} subjects, {len(co.y)} trials, {len(co.ch_names)} channels "
          f"({' '.join(co.ch_names)}), {T.shape[1]} tangent features ({time.time() - t0:.0f}s)")

    has_sides = np.isin(co.affected_side, ["L", "R"]).all()
    jobs = []
    for t in args.treatments:
        if t == "canonical" and not has_sides:
            print("skipping 'canonical': no hemiplegic side in this cohort")
            continue
        for a in (args.alphas if t == "soft" else [1.0]):
            jobs.append((t, a))

    folds = make_folds(co.subject, args.folds)
    rows = []
    for t, a in jobs:
        t1 = time.time()
        kind = "none" if t == "canonical" else t
        Tw, yw = T, co.y
        flipped = np.zeros(len(co.y), bool)
        if t == "canonical":
            flipped = co.affected_side == "R"
            Tw, yw = T.copy(), co.y.copy()
            Tw[flipped] = Tw[flipped][:, Pi]
            yw[flipped] = 1 - yw[flipped]
        for k, test_subs in enumerate(folds):
            te = np.isin(co.subject, test_subs)
            clf, C = fit_select(Tw[~te], yw[~te], co.subject[~te], kind, Pi, a)
            logit = clf.decision_function(feature_map(Tw[te], Pi, kind, a))
            logit_m = clf.decision_function(feature_map(Tw[te][:, Pi], Pi, kind, a))
            f = flipped[te]
            logit[f], logit_m[f] = -logit[f], -logit_m[f]          # back to the left/right frame
            y_te, s_te = co.y[te], co.subject[te]
            np.savez(out_dir / "logits" / f"{t}_a{a:g}_f{k}.npz", subject=s_te, y=y_te,
                     logit=logit, logit_mirror=logit_m)
            for s in np.unique(s_te):
                m = s_te == s
                rows.append({"treatment": t, "alpha": a, "fold": k, "subject": int(s), "n_trials": int(m.sum()),
                             "bal_acc": balanced_accuracy_score(y_te[m], (logit[m] > 0).astype(int)),
                             "auc": roc_auc_score(y_te[m], logit[m]) if len(np.unique(y_te[m])) == 2 else np.nan,
                             "sym_frac": sym_fraction(logit[m], logit_m[m]), "C": C})
        print(f"{t:<12} alpha={a:<5g} done ({time.time() - t1:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "cv.csv", index=False)
    res["label"] = np.where(res.treatment == "soft", "soft_a" + res.alpha.map(lambda v: f"{v:g}"), res.treatment)
    wide = res.pivot_table(index="subject", columns="label", values="bal_acc")
    print(f"\nper-subject balanced accuracy ({len(wide)} subjects; total {time.time() - t0:.0f}s)")
    print(f"{'treatment':<14}{'mean':>7}{'sd':>7}{'d_vs_none':>11}{'wilcoxon_p':>12}{'sym_frac':>10}")
    sf = res.groupby("label").sym_frac.mean()
    for lab in wide.columns:
        d = wide[lab] - wide["none"] if "none" in wide else None
        p_ = wilcoxon(d).pvalue if d is not None and lab != "none" and np.any(d != 0) else np.nan
        print(f"{lab:<14}{wide[lab].mean():>7.3f}{wide[lab].std():>7.3f}"
              f"{(d.mean() if d is not None else np.nan):>+11.3f}{p_:>12.3g}{sf[lab]:>10.2f}")
    print(f"\nsaved {out_dir / 'cv.csv'}")


if __name__ == "__main__":
    main()
