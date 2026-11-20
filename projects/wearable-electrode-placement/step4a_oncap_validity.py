"""
Step 4a — on-cap answer key: can physics predict REAL scalp bipolar pairs it never sees?
Project: physics-guided wearable electrode placement (C7-B / F17).

Context: step 3c was NO-GO for off-cap decoding claims (calibrated rho 0.343 < 0.35), so
placement moves on-cap. Before optimising on-cap, the on-cap prediction needs its own test.

For every short real scalp pair (both electrodes present, distance <= 7 cm, shared reference
electrode excluded), per subject:
  measured : AAD from the real bipolar pair (single-channel backward decoder, 0-400 ms lags,
             Ledoit-Wolf, leave-one-trial-out, 60 s windows) -> mean delta-r = r(att) - r(unatt)
  physics  : the two electrodes are REMOVED from the inputs; their signals are predicted from
             the remaining scalp channels (MNE, fsaverage, SNR 3) -> same AAD
  spline   : same, with spherical-spline interpolation
Also:
  - bipolar signal validity: r(predicted pair, measured pair) per trial
  - split-half reliability of the measured pair ranking (noise ceiling)
  - fast objective check: trial-level delta-r (full 600 s held-out trial, no windows) vs the
    windowed delta-r on measured pairs. If they agree (Spearman >= 0.9), step 4b can use the
    closed-form trial-level objective for exhaustive search.

Pre-registered (before running): on-cap rho higher than off-cap (>= 0.4); physics ~ spline
(interpolation, not extrapolation); fast objective agreement >= 0.9.
Go for on-cap placement: physics rho (group means) >= 0.35.

Closed-form decoders, no training loop -> no early stopping needed.

Run:
    python3 step4a_oncap_validity.py
"""

import csv
import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval
from pymatreader import read_mat
from scipy.stats import spearmanr, wilcoxon
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, LAGS, WIN = 20, np.arange(0, 9), 60
SNR, NAN_CH_FRAC = 3.0, 0.5
MAX_DIST = 0.07
GO_RHO, GO_FAST = 0.35, 0.9
N_SPLITS, SEED = 2000, 0
t0 = time.time()

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
L_sc = hm["L_scalp"].astype(np.float64)
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}

# ---------------------------------------------------------------------------
# spline (as in step 3)
# ---------------------------------------------------------------------------
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    c = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    return c[:3]


SPH_C = fit_sphere(P_grid[P_grid[:, 2] > np.percentile(P_grid[:, 2], 20)])
_n = np.arange(1, 51)
_coef = np.r_[0, (2 * _n + 1) / (_n * (_n + 1)) ** 4 / (4 * np.pi)]


def _unit(P):
    U = P - SPH_C
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def spline_matrix(P_from, P_to):
    Uf, Ut = _unit(P_from), _unit(P_to)
    G = legval(np.clip(Uf @ Uf.T, -1, 1), _coef)
    n = len(P_from)
    C = np.zeros((n + 1, n + 1))
    C[:n, :n], C[:n, n], C[n, :n] = G, 1.0, 1.0
    Ci = np.linalg.pinv(C)
    Gt = legval(np.clip(Ut @ Uf.T, -1, 1), _coef)
    return np.c_[Gt, np.ones(len(P_to))] @ Ci[:, :n]


def physics_rows(use, ref_i, targets):
    G = L_sc[use] - L_sc[ref_i]
    lam = np.trace(G @ G.T) / len(use) / SNR ** 2
    return (L_sc[targets] - L_sc[ref_i]) @ G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use)))


# ---------------------------------------------------------------------------
# decoding helpers
# ---------------------------------------------------------------------------
def zscore(x):
    mu, sd = np.nanmean(x, 0, keepdims=True), np.nanstd(x, 0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def corr_valid(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return corr(a[m], b[m]) if m.sum() > 100 else np.nan


def lagged(x):
    o = np.zeros((len(x), len(LAGS)))
    for k, l in enumerate(LAGS):
        o[: len(x) - l, k] = x[l:]
    return o


def aad_single(sig_trials, envs, att):
    """returns (windowed mean delta-r, trial-level mean delta-r)."""
    X = [lagged(zscore(s[:, None])[:, 0]) for s in sig_trials]
    S = [(zscore(e)[:, a - 1], zscore(e)[:, 2 - a]) for e, a in zip(envs, att)]
    win, trial = [], []
    for k in range(len(X)):
        tr = [j for j in range(len(X)) if j != k]
        Xtr = np.vstack([X[j] for j in tr])
        s = np.concatenate([S[j][0] for j in tr])
        cov, _ = ledoit_wolf(Xtr, assume_centered=True)
        w = np.linalg.solve(cov, Xtr.T @ s / len(s))
        rec = X[k] @ w
        trial.append(corr(rec, S[k][0]) - corr(rec, S[k][1]))
        L = WIN * FS
        for s0 in range(0, len(rec) - L + 1, L):
            sl = slice(s0, s0 + L)
            win.append(corr(rec[sl], S[k][0][sl]) - corr(rec[sl], S[k][1][sl]))
    return float(np.mean(win)), float(np.mean(trial))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
rows = []
for s in subjects:
    d = read_mat(str(PRE / f"dataSubject{s}.mat"))
    ch = list(d["channels"])
    trials = [np.asarray(t, float)[:, [ch.index(n) for n in SC]] for t in d["eegTrials"]]
    envs = [np.asarray(e, float) for e in d["envelopes"]]
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    ref = REF[s]
    ref_i = SC.index(ref)
    cand = [i for i in range(len(SC)) if i != ref_i]
    pairs = [(a, b) for a, b in combinations(cand, 2)
             if np.linalg.norm(P_sc[a] - P_sc[b]) <= MAX_DIST]
    for a, b in pairs:
        # skip if either electrode is mostly missing in any trial
        if any(np.isnan(t[:, [a, b]]).mean(0).max() > NAN_CH_FRAC for t in trials):
            continue
        meas, phys, spl, sig_p, sig_s = [], [], [], [], []
        for t in trials:
            dref = t - t[:, [ref_i]]
            nanf = np.isnan(t).mean(0)
            use = [i for i in cand if i not in (a, b) and nanf[i] <= NAN_CH_FRAC]
            Xin = np.nan_to_num(dref[:, use])
            bad = np.isnan(dref[:, use]).any(1)
            Mp = physics_rows(use, ref_i, [a, b])
            Ms = spline_matrix(np.vstack([P_sc[use], P_sc[ref_i]]), P_sc[[a, b]])
            pp = Xin @ Mp.T
            ps = np.c_[Xin, np.zeros(len(Xin))] @ Ms.T
            pp[bad], ps[bad] = np.nan, np.nan
            mb = dref[:, a] - dref[:, b]
            meas.append(mb)
            phys.append(pp[:, 0] - pp[:, 1])
            spl.append(ps[:, 0] - ps[:, 1])
            sig_p.append(corr_valid(phys[-1], mb))
            sig_s.append(corr_valid(spl[-1], mb))
        mw, mt = aad_single(meas, envs, att)
        pw, pt = aad_single(phys, envs, att)
        sw, st = aad_single(spl, envs, att)
        rows.append({"subject": s, "pair": f"{SC[a]}-{SC[b]}",
                     "dist_cm": 100 * float(np.linalg.norm(P_sc[a] - P_sc[b])),
                     "dr_measured": mw, "dr_measured_trial": mt,
                     "dr_physics": pw, "dr_physics_trial": pt,
                     "dr_spline": sw, "dr_spline_trial": st,
                     "sig_r_physics": float(np.nanmedian(sig_p)),
                     "sig_r_spline": float(np.nanmedian(sig_s))})
    print(f"subject {s:2d}: {sum(r['subject'] == s for r in rows)} pairs "
          f"(ref {ref}) ({time.time()-t0:.0f}s)")

with open(RES / "step4a_oncap_pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------------------
# analysis — only pairs available in every subject, so group means are comparable
# (Fp1/Fp2-containing pairs drop out where that electrode was the shared reference)
# ---------------------------------------------------------------------------
from collections import Counter
cnt = Counter(r["pair"] for r in rows)
n_min = max(len(subjects) - 3, 10)
PAIRS = sorted(p for p, c in cnt.items() if c >= n_min)
SUBJ = subjects
idx = {(r["subject"], r["pair"]): r for r in rows}


def mat(key):
    M = np.full((len(SUBJ), len(PAIRS)), np.nan)
    for i, s in enumerate(SUBJ):
        for j, p in enumerate(PAIRS):
            if (s, p) in idx:
                M[i, j] = idx[(s, p)][key]
    return M


Mm, Mp, Ms = mat("dr_measured"), mat("dr_physics"), mat("dr_spline")
Mmt = mat("dr_measured_trial")
gm, gp, gs = (np.nanmean(M, 0) for M in (Mm, Mp, Ms))
rho_p = float(spearmanr(gm, gp).correlation)
rho_s = float(spearmanr(gm, gs).correlation)
rho_p_subj = [spearmanr(Mm[i], Mp[i], nan_policy="omit").correlation for i in range(len(SUBJ))]
rho_s_subj = [spearmanr(Mm[i], Ms[i], nan_policy="omit").correlation for i in range(len(SUBJ))]

rng = np.random.default_rng(SEED)
half = []
for _ in range(N_SPLITS):
    perm = rng.permutation(len(SUBJ))
    a, b = perm[: len(SUBJ) // 2], perm[len(SUBJ) // 2:]
    half.append(spearmanr(np.nanmean(Mm[a], 0), np.nanmean(Mm[b], 0)).correlation)
r_half = float(np.median(half))
rel_full = 2 * r_half / (1 + r_half)

fast_pair = float(spearmanr(gm, np.nanmean(Mmt, 0)).correlation)
ok = np.isfinite(Mm) & np.isfinite(Mmt)
fast_cell = float(spearmanr(Mm[ok], Mmt[ok]).correlation)

sp = np.array([idx[k]["sig_r_physics"] for k in idx if k[1] in PAIRS])
ss = np.array([idx[k]["sig_r_spline"] for k in idx if k[1] in PAIRS])
subj_sp = np.array([np.nanmedian([idx[(s, p)]["sig_r_physics"] for p in PAIRS if (s, p) in idx])
                    for s in SUBJ])
subj_ss = np.array([np.nanmedian([idx[(s, p)]["sig_r_spline"] for p in PAIRS if (s, p) in idx])
                    for s in SUBJ])
p_sig = float(wilcoxon(subj_sp, subj_ss, alternative="two-sided").pvalue)

best_m = PAIRS[int(np.nanargmax(gm))]
best_p = PAIRS[int(np.nanargmax(gp))]
order_m = np.argsort(-gm)
out = {
    "n_subjects": len(SUBJ), "n_pairs": len(PAIRS), "max_dist_cm": 100 * MAX_DIST,
    "pairs_present_in_at_least": n_min,
    "rank_agreement": {"physics_rho_group": rho_p, "spline_rho_group": rho_s,
                       "physics_rho_subject_median": float(np.nanmedian(rho_p_subj)),
                       "spline_rho_subject_median": float(np.nanmedian(rho_s_subj)),
                       "offcap_reference": {"template": 0.22, "calibrated": 0.343}},
    "noise_ceiling": {"split_half": r_half, "full_SB": rel_full,
                      "ceiling_rho_approx": float(np.sqrt(max(rel_full, 0)))},
    "signal_validity_bipolar": {"physics_subject_median": float(np.median(subj_sp)),
                                "spline_subject_median": float(np.median(subj_ss)),
                                "p_two_sided": p_sig},
    "fast_objective": {"spearman_group_means": fast_pair, "spearman_all_cells": fast_cell},
    "mean_dr": {"measured": float(np.nanmean(Mm)), "physics": float(np.nanmean(Mp)),
                "spline": float(np.nanmean(Ms))},
    "best_pair_measured": best_m, "best_pair_physics": best_p,
    "measured_rank_of_physics_best": int(np.where(np.array(PAIRS)[order_m] == best_p)[0][0]) + 1,
    "top5_measured": [PAIRS[j] for j in order_m[:5]],
    "top5_physics": [PAIRS[j] for j in np.argsort(-gp)[:5]],
    "go_placement": bool(rho_p >= GO_RHO), "go_fast_objective": bool(fast_pair >= GO_FAST),
}
print(f"\n{len(PAIRS)} real scalp pairs (<= {100*MAX_DIST:.0f} cm) in >= {n_min} subjects")
print(f"rank agreement (group means): physics {rho_p:.2f}, spline {rho_s:.2f} "
      f"(per-subject medians {np.nanmedian(rho_p_subj):.2f}, {np.nanmedian(rho_s_subj):.2f})")
print(f"noise ceiling: split-half {r_half:.2f} -> full {rel_full:.2f} "
      f"(max rho ~ {np.sqrt(max(rel_full, 0)):.2f})")
print(f"bipolar signal r (subject medians): physics {np.median(subj_sp):.3f}, "
      f"spline {np.median(subj_ss):.3f} (p = {p_sig:.2g})")
print(f"fast objective vs windowed: Spearman {fast_pair:.2f} (group means), "
      f"{fast_cell:.2f} (all cells)")
print(f"best measured pair {best_m}; physics' best {best_p} ranks "
      f"#{out['measured_rank_of_physics_best']} of {len(PAIRS)}")
print(f"top-5 measured: {out['top5_measured']}")
print(f"top-5 physics : {out['top5_physics']}")
print(f"\nGO on-cap placement: {out['go_placement']}  |  GO fast objective: "
      f"{out['go_fast_objective']}")
(RES / "step4a_oncap_validity.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, ORANGE, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#c9c9c9"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
a1.scatter(gm, gp, s=30, c=BLUE, edgecolors="white", linewidths=1.5, zorder=3, label="physics")
a1.scatter(gm, gs, s=30, c=ORANGE, edgecolors="white", linewidths=1.5, zorder=2, label="spline")
lim = [np.nanmin([gm, gp, gs]) - 0.005, np.nanmax([gm, gp, gs]) + 0.005]
a1.plot(lim, lim, ls="--", lw=0.8, color=MUTED)
for j in order_m[:5]:
    a1.annotate(PAIRS[j], (gm[j], gp[j]), fontsize=7, color=MUTED, xytext=(4, 3),
                textcoords="offset points")
a1.set_xlabel("measured pair: mean Δr (att − unatt)", color=INK)
a1.set_ylabel("predicted pair (electrodes removed): mean Δr", color=INK)
a1.set_title(f"{len(PAIRS)} real scalp pairs ≤ {100*MAX_DIST:.0f} cm, group means\n"
             f"Spearman ρ: physics {rho_p:.2f}, spline {rho_s:.2f}", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8)
a1.grid(color="#eeeeee", lw=0.6)
vals = [0.22, 0.343, rho_p, float(np.sqrt(max(rel_full, 0)))]
labs = ["off-cap\ntemplate", "off-cap\ncalibrated", "on-cap\nphysics", "on-cap\nnoise ceiling"]
a2.bar(range(4), vals, color=["#f2b49a", ORANGE, BLUE, GREY], width=0.55)
for k, v in enumerate(vals):
    a2.text(k, v + 0.02, f"{v:.2f}", ha="center", color=INK)
a2.axhline(GO_RHO, color=MUTED, lw=1, ls="--")
a2.text(3.3, GO_RHO + 0.01, f"go ≥ {GO_RHO}", ha="right", va="bottom", fontsize=8, color=MUTED)
a2.set_xticks(range(4), labs, fontsize=9, color=MUTED)
a2.set_ylim(min(0, min(vals)) - 0.05, 1)
a2.set_ylabel("Spearman ρ, predicted vs measured ranking", color=INK)
a2.set_title("Decoding-rank agreement: off-cap vs on-cap", fontsize=10, color=INK)
for ax in (a1, a2):
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
fig.savefig(FIG / "step4a_oncap_validity.png", dpi=150)
print(f"Wrote results/step4a_*.csv/json, figures/step4a_oncap_validity.png "
      f"({time.time()-t0:.0f}s)")
