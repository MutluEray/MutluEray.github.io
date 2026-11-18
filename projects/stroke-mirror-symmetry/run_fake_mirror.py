"""Control: does a symmetry prior help because of the brain's left-right anatomy, or only
because it regularises the decoder?

For the real mid-sagittal mirror P and for N random "fake mirrors" (random channel pairings
with the same cycle structure as P: same number of swapped pairs and fixed channels), fit the
`equivariant` and `augment` treatments of run_ts.py and record the gain in per-subject
balanced accuracy over the unconstrained decoder. Any involution halves the feature space the
same way, so fake mirrors give the same regularisation without the anatomy.

  real gain >> fake gains  -> the prior carries anatomical information
  real gain ~= fake gains  -> the gain is regularisation only

Permutation p = (1 + #fake gains >= real gain) / (N + 1).

  python3 run_fake_mirror.py --dataset stroke --channels shared --n-fake 30
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_ts import BANDS, feature_map, fit_select, make_folds, tangent_features, vector_permutation
from src.data import load_cohort


def random_involution(perm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(perm)
    n_fixed = int(np.sum(perm == np.arange(n)))
    n_pairs = (n - n_fixed) // 2
    order = rng.permutation(n)
    out = np.arange(n)
    for k in range(n_pairs):
        a, b = order[2 * k], order[2 * k + 1]
        out[a], out[b] = b, a
    return out


def cv_accuracy(T, y, subject, Pi, kind, folds) -> pd.Series:
    acc = {}
    for test in folds:
        te = np.isin(subject, test)
        clf, _ = fit_select(T[~te], y[~te], subject[~te], kind, Pi, 1.0)
        logit = clf.decision_function(feature_map(T[te], Pi, kind, 1.0))
        for s in np.unique(subject[te]):
            m = subject[te] == s
            acc[int(s)] = balanced_accuracy_score(y[te][m], (logit[m] > 0).astype(int))
    return pd.Series(acc).sort_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--channels", default="shared",
                   choices=["all", "sensorimotor", "shared", "shared_sensorimotor"])
    p.add_argument("--n-subjects", type=int, default=0)
    p.add_argument("--n-fake", type=int, default=30)
    p.add_argument("--treatments", nargs="+", default=["equivariant", "augment"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", nargs="+", type=int, default=[], help="subject ids to leave out")
    p.add_argument("--tag", default="", help="suffix for the output name, e.g. excl")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    t0 = time.time()
    co = load_cohort(args.dataset, args.channels, args.n_subjects, exclude=tuple(args.exclude))
    T = tangent_features(co.X, co.subject, co.sfreq)
    n_ch = len(co.ch_names)
    folds = make_folds(co.subject, args.folds)
    rng = np.random.default_rng(args.seed)

    real_Pi = vector_permutation(co.perm, n_ch, len(BANDS))
    base = cv_accuracy(T, co.y, co.subject, real_Pi, "none", folds)
    print(f"[{args.dataset}/{args.channels}] {len(base)} subjects, {n_ch} channels; "
          f"unconstrained decoder {base.mean():.3f}")

    fakes = []
    while len(fakes) < args.n_fake:
        f = random_involution(co.perm, rng)
        if not np.array_equal(f, co.perm):
            fakes.append(f)

    rows = []
    for kind in args.treatments:
        for draw, perm in enumerate([co.perm] + fakes):
            Pi = vector_permutation(perm, n_ch, len(BANDS))
            acc = cv_accuracy(T, co.y, co.subject, Pi, kind, folds)
            rows.append({"treatment": kind, "draw": draw, "is_real": draw == 0,
                         "mean_acc": acc.mean(), "gain": (acc - base).mean()})
        r = pd.DataFrame([x for x in rows if x["treatment"] == kind])
        real = r[r.is_real].gain.iloc[0]
        fake = r[~r.is_real].gain.to_numpy()
        p_perm = (1 + np.sum(fake >= real)) / (len(fake) + 1)
        print(f"{kind:<12} real-mirror gain {real:+.3f} | fake mirrors {fake.mean():+.3f} "
              f"(sd {fake.std():.3f}, range {fake.min():+.3f} to {fake.max():+.3f}) | "
              f"real beats {np.mean(fake < real) * 100:.0f}% of fakes, permutation p = {p_perm:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = Path("results") / f"fake_mirror_{args.dataset}_{args.channels}" \
        f"{'_n' + str(args.n_subjects) if args.n_subjects else ''}{'_' + args.tag if args.tag else ''}.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
