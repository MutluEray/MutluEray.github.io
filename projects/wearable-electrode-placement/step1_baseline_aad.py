"""
Step 1 — reproduce the Ear-SAAD auditory attention decoding (AAD) baselines.
Project: physics-guided wearable electrode placement (C7-B / F17).

Pipeline (follows Geirnaert, Kappel, Kidmose, Sci Rep 2025):
  - setups: scalp (29 incl. M1/M2), around-ear (cEL1-9, cER1-10), in-ear (EL*, ER*)
  - common average reference within each setup (CAR does not depend on the
    original common reference, so it is safe whatever preprocessFromBIDS.m did)
  - per-trial z-scoring of EEG channels and envelopes
  - backward decoder: envelope(t) from EEG(t ... t+400 ms) -> lags 0..8 at 20 Hz
  - covariance with Ledoit-Wolf shrinkage; leave-one-trial-out (6 folds)
  - decision windows 10 / 30 / 60 s, non-overlapping; correct if
    corr(recon, attended) > corr(recon, unattended)
Closed-form decoder: no iterative training, so no early stopping is needed.

Self-checks printed:
  - per-channel std of subject 1 (finds a zero reference channel, e.g. Fp1)
  - NaN fraction
  - "label check": accuracy with attended column = attSpeaker-1; if the mean is
    far below 50 %, the column convention is the other way round.

Run from the project folder:
    python3 step1_baseline_aad.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from pymatreader import read_mat
from scipy.stats import binom
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

FS = 20
LAGS = np.arange(0, 9)            # 0..400 ms post-stimulus at 20 Hz
WINDOWS_S = [10, 30, 60]
PAPER_60S = {"scalp": 83.44, "around_ear": 67.22, "in_ear": 61.11}

SCALP = ["O1", "Pz", "P3", "P7", "CP1", "CP5", "M1", "Cz", "C3", "T7", "FC1", "FC5",
         "F3", "F7", "Fp1", "O2", "P4", "P8", "CP2", "CP6", "M2", "C4", "T8", "FC2",
         "FC6", "Fz", "F4", "F8", "Fp2"]
AROUND = [f"cEL{i}" for i in range(1, 10)] + [f"cER{i}" for i in range(1, 11)]
INEAR = [f"EL{s}" for s in "ABCEIT"] + [f"ER{s}" for s in "ABCEIT"]
SETUPS = {"scalp": SCALP, "around_ear": AROUND, "in_ear": INEAR}
ZERO_STD = 1e-6                   # channels below this std (µV) are dropped (reference)


def zscore(x):
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0)


def lagged(eeg, lags):
    """X[t] = [eeg(t+l) for l in lags]; zero-padded at the end."""
    T, C = eeg.shape
    out = np.zeros((T, C * len(lags)))
    for k, l in enumerate(lags):
        out[: T - l, k * C:(k + 1) * C] = eeg[l:, :]
    return out


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def load_subject(i):
    d = read_mat(str(PRE / f"dataSubject{i}.mat"))
    ch = list(d["channels"])
    trials = [np.asarray(t, float) for t in d["eegTrials"]]
    envs = [np.asarray(e, float) for e in d["envelopes"]]
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    return ch, trials, envs, att


def prepare(trials, ch, names):
    idx = [ch.index(n) for n in names]
    sub = [t[:, idx] for t in trials]
    sd = np.nanstd(np.vstack(sub), axis=0)
    keep = [j for j, s in enumerate(sd) if s > ZERO_STD]
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    out = []
    for t in sub:
        t = t[:, keep]
        t = t - np.nanmean(t, axis=1, keepdims=True)      # CAR within setup
        out.append(zscore(t))
    return out, dropped


def decode_subject(eeg_trials, envs, att, flip=False):
    n = len(eeg_trials)
    X = [lagged(e, LAGS) for e in eeg_trials]
    S = []
    for e, a in zip(envs, att):
        z = zscore(e)
        ia = (a - 1) if not flip else (2 - a)
        S.append((z[:, ia], z[:, 1 - ia]))               # (attended, unattended)
    res = {w: [] for w in WINDOWS_S}
    for k in range(n):
        tr = [j for j in range(n) if j != k]
        Xtr = np.vstack([X[j] for j in tr])
        str_ = np.concatenate([S[j][0] for j in tr])
        cov, _ = ledoit_wolf(Xtr, assume_centered=True)
        w = np.linalg.solve(cov, Xtr.T @ str_ / len(str_))
        rec = X[k] @ w
        sa, su = S[k]
        for win in WINDOWS_S:
            L = win * FS
            for s0 in range(0, len(rec) - L + 1, L):
                sl = slice(s0, s0 + L)
                res[win].append(corr(rec[sl], sa[sl]) > corr(rec[sl], su[sl]))
    return {w: float(np.mean(v)) for w, v in res.items()}, {w: len(v) for w, v in res.items()}


def main():
    subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
    print(f"{len(subjects)} subjects: {subjects}")

    # ---- self-checks on subject 1 ----
    ch, trials, envs, att = load_subject(subjects[0])
    allt = np.vstack(trials)
    std = np.nanstd(allt, axis=0)
    nan_frac = float(np.isnan(allt).mean())
    print(f"\nSubject {subjects[0]}: NaN fraction {nan_frac:.4f}")
    print("per-channel std (µV):")
    for c, s in zip(ch, std):
        print(f"  {c:>7s} {s:9.3f}")

    rows, summary = [], {"subjects": subjects, "nan_fraction_sub1": nan_frac,
                         "std_sub1": dict(zip(ch, map(float, std))), "dropped": {}}
    acc = {s: {w: [] for w in WINDOWS_S} for s in SETUPS}
    n_windows = {}
    label_check = []

    for i in subjects:
        ch, trials, envs, att = load_subject(i)
        row = {"subject": i}
        for sname, names in SETUPS.items():
            eeg, dropped = prepare(trials, ch, names)
            if dropped:
                summary["dropped"].setdefault(sname, {})[i] = dropped
            a, nw = decode_subject(eeg, envs, att)
            n_windows = nw
            for w in WINDOWS_S:
                acc[sname][w].append(a[w])
                row[f"{sname}_{w}s"] = round(100 * a[w], 2)
            if sname == "scalp":
                label_check.append(a[60])
        rows.append(row)
        print(f"sub {i:2d}: " + "  ".join(f"{s} {row[f'{s}_60s']:5.1f}" for s in SETUPS))

    # ---- label convention check ----
    mean_scalp = float(np.mean(label_check))
    summary["label_check_scalp_60s_mean"] = mean_scalp
    if mean_scalp < 0.4:
        print("\nWARNING: scalp accuracy far below chance -> envelope column convention "
              "is probably flipped (attended = 2 - attSpeaker). Tell Claude.")

    # ---- significance threshold per subject (binomial, alpha = 0.05) ----
    n60 = n_windows[60]
    thr = binom.ppf(0.95, n60, 0.5) / n60
    summary["windows_per_subject"] = n_windows
    summary["significance_threshold_60s"] = float(thr)

    print(f"\n=== mean accuracy (%), leave-one-trial-out; 60 s: {n60} windows/subject, "
          f"significance threshold {100*thr:.1f} % ===")
    print(f"{'setup':>11s} " + " ".join(f"{w:>6d}s" for w in WINDOWS_S) + "   paper 60 s   n sig")
    summary["mean"] = {}
    for s in SETUPS:
        m = {w: 100 * float(np.mean(acc[s][w])) for w in WINDOWS_S}
        nsig = int(sum(a > thr for a in acc[s][60]))
        summary["mean"][s] = {**{f"{w}s": m[w] for w in WINDOWS_S},
                              "median_60s": 100 * float(np.median(acc[s][60])),
                              "n_significant_60s": nsig,
                              "paper_60s": PAPER_60S[s],
                              "diff_vs_paper_60s": m[60] - PAPER_60S[s]}
        print(f"{s:>11s} " + " ".join(f"{m[w]:6.1f} " for w in WINDOWS_S)
              + f"   {PAPER_60S[s]:6.2f}      {nsig}/{len(subjects)}")

    ok = all(abs(summary["mean"][s]["diff_vs_paper_60s"]) <= 5 for s in SETUPS)
    summary["go_no_go"] = "GO (all within 5 points of paper)" if ok else \
        "CHECK (at least one setup > 5 points from paper)"
    print(f"\n{summary['go_no_go']}")
    if summary["dropped"]:
        print(f"dropped near-zero channels: {summary['dropped']}")

    with open(RES / "step1_per_subject.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (RES / "step1_baseline.json").write_text(json.dumps(summary, indent=2))
    print("Wrote results/step1_baseline.json and results/step1_per_subject.csv")


if __name__ == "__main__":
    main()
