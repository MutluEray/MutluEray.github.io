"""
Step 3b — is the weak pair-ranking agreement (rho = 0.22) a physics failure or a noise ceiling?
Project: physics-guided wearable electrode placement (C7-B / F17).

Uses step 3 outputs + preprocessed data. No-mastoid inputs throughout (primary analysis).
  1. Noise ceiling: split-half reliability of the MEASURED pair ranking (random 7/8 subject
     splits, 2,000 draws, Spearman-Brown corrected). Max attainable rho ~ sqrt(reliability).
  2. Data-driven ceiling for decoding: ridge (scalp -> each ear channel, leave-one-trial-out),
     bipolar pairs from the ridge predictions, same AAD pipeline -> Spearman vs measured.
     If this is also low, the ranking-relevant signal is not in the scalp at all.
  3. Subject-level test of signal validity (n = 15, one median per subject): physics vs
     spline and vs nearest, one-sided Wilcoxon. Replaces the cell-level p in step 3.
  4. Data quality per around-ear electrode: NaN (rejected) fraction across subjects/trials,
     and physics over-rating per electrode = mean over its pairs of
     rank(physics) - rank(measured). Spearman across the 19 electrodes. EXPLORATORY.
  5. EXPLORATORY (post hoc, labelled as such): rho without pairs that contain electrode 1 or 2.

Closed-form throughout (no training loop -> no early stopping needed).

Run:
    python3 step3b_diagnostics.py
"""

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, LAGS, WIN = 20, np.arange(0, 9), 60
NAN_CH_FRAC = 0.5
INP = "no_mastoids"
N_SPLITS, SEED = 2000, 0
t0 = time.time()

SCALP = ["O1", "Pz", "P3", "P7", "CP1", "CP5", "M1", "Cz", "C3", "T7", "FC1", "FC5",
         "F3", "F7", "Fp1", "O2", "P4", "P8", "CP2", "CP6", "M2", "C4", "T8", "FC2",
         "FC6", "Fz", "F4", "F8", "Fp2"]
EAR = [f"cEL{i}" for i in range(1, 10)] + [f"cER{i}" for i in range(1, 11)]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


pairs_rows = [r for r in read_csv(RES / "step3_pair_decoding.csv") if r["inputs"] == INP]
sig_rows = read_csv(RES / "step3_signal_corr.csv")
ref_rows = {int(r["subject"]): r["shared_ref_used"] for r in read_csv(RES / "step3_shared_ref.csv")}
subjects = sorted({int(r["subject"]) for r in pairs_rows})
pairs = sorted({r["pair"] for r in pairs_rows})
out = {}

# ---------------------------------------------------------------------------
# 1. split-half reliability of the measured ranking
# ---------------------------------------------------------------------------
M = np.full((len(subjects), len(pairs)), np.nan)
Pp = np.full_like(M, np.nan)
for r in pairs_rows:
    i, j = subjects.index(int(r["subject"])), pairs.index(r["pair"])
    M[i, j], Pp[i, j] = r["dr_measured"], r["dr_physics"]

rng = np.random.default_rng(SEED)
rel_m, rel_p = [], []
for _ in range(N_SPLITS):
    perm = rng.permutation(len(subjects))
    a, b = perm[: len(subjects) // 2], perm[len(subjects) // 2:]
    rel_m.append(spearmanr(np.nanmean(M[a], 0), np.nanmean(M[b], 0)).correlation)
    rel_p.append(spearmanr(np.nanmean(Pp[a], 0), np.nanmean(Pp[b], 0)).correlation)
sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan            # Spearman-Brown
r_half_m, r_half_p = float(np.median(rel_m)), float(np.median(rel_p))
rel_full_m, rel_full_p = sb(r_half_m), sb(r_half_p)
rho_phys = spearmanr(np.nanmean(M, 0), np.nanmean(Pp, 0)).correlation
noise_ceiling = float(np.sqrt(max(rel_full_m, 0) * max(rel_full_p, 0)))
out["reliability"] = {
    "measured_split_half_median": r_half_m, "measured_full_SB": rel_full_m,
    "physics_split_half_median": r_half_p, "physics_full_SB": rel_full_p,
    "rho_physics_vs_measured": float(rho_phys),
    "noise_ceiling_rho": noise_ceiling,
    "rho_over_ceiling": float(rho_phys / noise_ceiling) if noise_ceiling > 0 else None}
print(f"measured ranking reliability: split-half {r_half_m:.2f} -> full {rel_full_m:.2f}")
print(f"physics  ranking reliability: split-half {r_half_p:.2f} -> full {rel_full_p:.2f}")
print(f"rho(physics, measured) = {rho_phys:.2f}; noise ceiling ~ {noise_ceiling:.2f}")

# ---------------------------------------------------------------------------
# helpers for the ridge-ceiling decoding
# ---------------------------------------------------------------------------
def zscore(x):
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def lagged(x):
    o = np.zeros((len(x), len(LAGS)))
    for k, l in enumerate(LAGS):
        o[: len(x) - l, k] = x[l:]
    return o


def aad_single(sig_trials, envs, att):
    X = [lagged(zscore(s[:, None])[:, 0]) for s in sig_trials]
    S = [(zscore(e)[:, a - 1], zscore(e)[:, 2 - a]) for e, a in zip(envs, att)]
    drs = []
    for k in range(len(X)):
        tr = [j for j in range(len(X)) if j != k]
        Xtr = np.vstack([X[j] for j in tr])
        s = np.concatenate([S[j][0] for j in tr])
        cov, _ = ledoit_wolf(Xtr, assume_centered=True)
        w = np.linalg.solve(cov, Xtr.T @ s / len(s))
        rec = X[k] @ w
        L = WIN * FS
        for s0 in range(0, len(rec) - L + 1, L):
            sl = slice(s0, s0 + L)
            drs.append(corr(rec[sl], S[k][0][sl]) - corr(rec[sl], S[k][1][sl]))
    return float(np.mean(drs))


# ---------------------------------------------------------------------------
# 2. ridge-ceiling decoding + 4. per-electrode NaN fraction (one pass over data)
# ---------------------------------------------------------------------------
C = np.full_like(M, np.nan)
nan_frac = {e: [] for e in EAR}
inputs = [n for n in SCALP if n not in ("M1", "M2")]
for subj in subjects:
    d = read_mat(str(PRE / f"dataSubject{subj}.mat"))
    ch = list(d["channels"])
    trials = [np.asarray(t, float) for t in d["eegTrials"]]
    envs = [np.asarray(e, float) for e in d["envelopes"]]
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    ref = ref_rows[subj]
    use = [ch.index(n) for n in inputs if n != ref]
    X = [np.nan_to_num(t[:, use] - t[:, [ch.index(ref)]]) for t in trials]
    Y = [t[:, [ch.index(e) for e in EAR]] - t[:, [ch.index("Fp1-cr")]] for t in trials]
    for t in trials:
        for e in EAR:
            nan_frac[e].append(float(np.isnan(t[:, ch.index(e)]).mean()))
    pred = [np.full(y.shape, np.nan) for y in Y]
    for k in range(len(trials)):
        tr = [j for j in range(len(trials)) if j != k]
        Xtr, Ytr = np.vstack([X[j] for j in tr]), np.vstack([Y[j] for j in tr])
        for e in range(len(EAR)):
            m = np.isfinite(Ytr[:, e])
            if m.sum() < 1000:
                continue
            rg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr[m], Ytr[m, e])
            pred[k][:, e] = rg.predict(X[k])
    i = subjects.index(subj)
    for j, p in enumerate(pairs):
        if np.isnan(M[i, j]):
            continue
        a, b = (EAR.index(x) for x in p.split("-"))
        sig = [pt[:, a] - pt[:, b] for pt in pred]
        if any(np.isnan(s).mean() > NAN_CH_FRAC for s in sig):
            continue
        C[i, j] = aad_single(sig, envs, att)
    print(f"subject {subj:2d} ridge-ceiling decoding done ({time.time()-t0:.0f}s)")

rho_ceil = spearmanr(np.nanmean(M, 0), np.nanmean(C, 0)).correlation
rho_ceil_subj = [spearmanr(M[i], C[i], nan_policy="omit").correlation for i in range(len(subjects))]
rho_phys_subj = [spearmanr(M[i], Pp[i], nan_policy="omit").correlation for i in range(len(subjects))]
out["ridge_ceiling_decoding"] = {
    "rho_group": float(rho_ceil),
    "rho_per_subject_median": float(np.nanmedian(rho_ceil_subj)),
    "physics_rho_per_subject_median": float(np.nanmedian(rho_phys_subj)),
    "mean_dr_ceiling": float(np.nanmean(C)), "mean_dr_measured": float(np.nanmean(M)),
    "mean_dr_physics": float(np.nanmean(Pp))}
print(f"\nridge ceiling: rho(group) = {rho_ceil:.2f}; per-subject median "
      f"{np.nanmedian(rho_ceil_subj):.2f} (physics {np.nanmedian(rho_phys_subj):.2f})")

# ---------------------------------------------------------------------------
# 3. subject-level signal-validity test
# ---------------------------------------------------------------------------
def subj_level(method):
    v = {}
    for r in sig_rows:
        if r["inputs"] == INP and r["method"] == method and r["metric"] == "earCAR" \
                and np.isfinite(r["r"]):
            v.setdefault(int(r["subject"]), []).append(r["r"])
    return np.array([np.median(v[s]) for s in subjects])


ph = subj_level("physics")
out["subject_level_signal"] = {"physics_median": float(np.median(ph))}
for other in ("spline", "nearest", "ceiling"):
    o = subj_level(other)
    entry = {"median": float(np.median(o)), "n_subjects_physics_better": int(np.sum(ph > o))}
    if other != "ceiling":
        entry["p_one_sided"] = float(wilcoxon(ph, o, alternative="greater").pvalue)
    out["subject_level_signal"][other] = entry
    print(f"subject-level physics vs {other}: physics better in "
          f"{entry['n_subjects_physics_better']}/15" +
          (f", p = {entry['p_one_sided']:.2g}" if "p_one_sided" in entry else ""))

# ---------------------------------------------------------------------------
# 4. per-electrode over-rating vs data quality (exploratory)
# ---------------------------------------------------------------------------
gm, gp = np.nanmean(M, 0), np.nanmean(Pp, 0)
rank_diff = rankdata(gp) - rankdata(gm)                  # >0: physics rates the pair higher
over = {e: float(np.mean([rank_diff[j] for j, p in enumerate(pairs) if e in p.split("-")]))
        for e in EAR}
nanf = {e: float(np.mean(nan_frac[e])) for e in EAR}
rho_q = spearmanr([over[e] for e in EAR], [nanf[e] for e in EAR]).correlation
out["per_electrode"] = {e: {"physics_overrating_rank": over[e], "nan_fraction": nanf[e]}
                        for e in EAR}
out["overrating_vs_nan_spearman"] = float(rho_q)
print(f"\nper electrode (over-rating = mean rank(physics) - rank(measured) over its pairs):")
for e in EAR:
    print(f"  {e:>6s}  over-rating {over[e]:+6.1f}   NaN fraction {100*nanf[e]:5.1f} %")
print(f"Spearman(over-rating, NaN fraction) across 19 electrodes = {rho_q:.2f}  [exploratory]")

# ---------------------------------------------------------------------------
# 5. exploratory: drop pairs touching electrode 1 or 2
# ---------------------------------------------------------------------------
top = {"cEL1", "cEL2", "cER1", "cER2"}
keep = [j for j, p in enumerate(pairs) if not (set(p.split("-")) & top)]
rho_wo = spearmanr(gm[keep], gp[keep]).correlation
rho_wo_c = spearmanr(gm[keep], np.nanmean(C, 0)[keep]).correlation
out["exploratory_without_top_electrodes"] = {"n_pairs": len(keep), "rho_physics": float(rho_wo),
                                             "rho_ridge_ceiling": float(rho_wo_c)}
print(f"[exploratory] without pairs touching electrodes 1-2 ({len(keep)} pairs): "
      f"rho physics {rho_wo:.2f}, ridge ceiling {rho_wo_c:.2f}")

(RES / "step3b_diagnostics.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
labels = ["physics vs\nmeasured", "ridge ceiling vs\nmeasured", "noise ceiling\n(split-half)"]
vals = [rho_phys, rho_ceil, noise_ceiling]
a1.bar(range(3), vals, color=[BLUE, AQUA, "#c9c9c9"], width=0.55)
for k, v in enumerate(vals):
    a1.text(k, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=10)
a1.axhline(0.5, color=ORANGE, lw=1, ls="--")
a1.text(2.35, 0.51, "pre-registered\nρ > 0.5", color=MUTED, fontsize=8, va="bottom", ha="right")
a1.set_xticks(range(3), labels, color=MUTED, fontsize=9)
a1.set_ylabel("Spearman ρ across 81 pairs (group means)", color=INK)
a1.set_ylim(min(0, min(vals)) - 0.05, 1)
a1.set_title("Can anything rank the around-ear pairs?", fontsize=10, color=INK)
for s in ("top", "right"):
    a1.spines[s].set_visible(False)
x = [100 * nanf[e] for e in EAR]
y = [over[e] for e in EAR]
a2.scatter(x, y, s=40, c=BLUE, edgecolors="white", linewidths=1.5, zorder=3)
for e, xi, yi in zip(EAR, x, y):
    a2.annotate(e.replace("cE", ""), (xi, yi), fontsize=7, color=MUTED,
                xytext=(4, 3), textcoords="offset points")
a2.axhline(0, color=MUTED, lw=0.8)
a2.set_xlabel("rejected samples, % (mean over subjects × trials)", color=INK)
a2.set_ylabel("physics over-rating (mean rank difference)", color=INK)
a2.set_title(f"Exploratory: over-rating vs data quality, ρ = {rho_q:.2f}", fontsize=10, color=INK)
a2.grid(color="#eeeeee", lw=0.6)
for s in ("top", "right"):
    a2.spines[s].set_visible(False)
fig.savefig(FIG / "step3b_diagnostics.png", dpi=150)
print(f"\nWrote results/step3b_diagnostics.json, figures/step3b_diagnostics.png "
      f"({time.time()-t0:.0f}s)")
