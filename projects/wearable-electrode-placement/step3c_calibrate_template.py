"""
Step 3c — calibrate the around-ear template geometry, leave-one-subject-out (LOSO).
Project: physics-guided wearable electrode placement (C7-B / F17).

Step 3b showed the pair ranking is not just noise (ridge ceiling rho 0.46, noise ceiling
0.70) but physics reaches only 0.22, with a geometric over-rating pattern (top-front
electrodes). Here 4 template parameters are fitted on 14 subjects and tested on the 15th:
  dy    : extra posterior shift of the arc centre  {-1.0 ... +1.5} cm
  dz    : vertical shift of the arc centre          {-2.0 ... +1.0} cm
  s_up  : vertical semi-axis                        {2.5, 3.0, 3.5} cm
  s_post: posterior semi-axis                       {2.0, 2.5, 3.0} cm
(378 configs; the step-2 template is dy = dz = 0, s_up = 3.5, s_post = 2.5).
Fitting target = SIGNAL validity only (mean ear-CAR correlation, no-mastoid inputs), so
the decoding ranking used for the go/no-go is never fitted on.

Pre-registered (before running): calibrated arc moves down and/or shrinks; calibrated
decoding rho rises but stays below the ridge ceiling (0.46).
Go for off-cap placement claims: calibrated rho (group means, 81 pairs) >= 0.35.

Implementation: lead field at any template point = thin-plate RBF weights x grid lead
field; the projected lead field (grid x scalp inputs) is cached per inverse kernel, and
per-trial covariances make each config evaluation a few small matrix products.
Grid search, no training loop -> no early stopping needed.

Run:
    python3 step3c_calibrate_template.py
"""

import csv
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
from scipy.interpolate import RBFInterpolator
from scipy.stats import spearmanr, wilcoxon
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, LAGS, WIN = 20, np.arange(0, 9), 60
SNR, NAN_CH_FRAC = 3.0, 0.5
GO_RHO = 0.35
BASE = dict(post=0.015, up=0.010)                       # step-2 pinna centre offsets (m)
GRID = dict(dy=np.arange(-0.010, 0.0151, 0.005), dz=np.arange(-0.020, 0.0101, 0.005),
            s_up=[0.025, 0.030, 0.035], s_post=[0.020, 0.025, 0.030])
TEMPLATE = (0.0, 0.0, 0.035, 0.025)
THETA = np.deg2rad(np.linspace(120, -120, 10))
t0 = time.time()

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
EAR = list(hm["names_ear"])
LEFT, RIGHT = EAR[:9], EAR[9:]
L_sc = hm["L_scalp"].astype(np.float64)
L_gr = hm["L_grid"].astype(np.float64)
Pg = hm["pos_grid"]
lpa, rpa = hm["lpa"], hm["rpa"]
INPUTS = [n for n in SC if n not in ("M1", "M2")]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


REF = {int(float(r["subject"])): r["shared_ref_used"] for r in read_csv(RES / "step3_shared_ref.csv")}
pair_rows = [r for r in read_csv(RES / "step3_pair_decoding.csv") if r["inputs"] == "no_mastoids"]
PAIRS = sorted({r["pair"] for r in pair_rows})
MEAS = {(int(float(r["subject"])), r["pair"]): float(r["dr_measured"]) for r in pair_rows}
PHYS0 = {(int(float(r["subject"])), r["pair"]): float(r["dr_physics"]) for r in pair_rows}
DEC_SUBJ = sorted({s for s, _ in MEAS})

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def to_skin(p, side):
    cand = Pg[np.sign(Pg[:, 0]) == side]
    d = np.linalg.norm(cand[:, 1:] - p[1:], axis=1)
    nn = np.argsort(d)[:4]
    w = 1.0 / (d[nn] + 1e-4)
    return np.array([np.sum(w * cand[nn, 0]) / w.sum(), p[1], p[2]])


def arc(dy, dz, s_up, s_post):
    out = []
    for pre, side, n in ((lpa, -1, 9), (rpa, +1, 10)):
        c = pre + np.array([0.0, -(BASE["post"] + dy), BASE["up"] + dz])
        for th in THETA[:n]:
            out.append(to_skin(c + np.array([0.0, -s_post * np.cos(th), s_up * np.sin(th)]), side))
    return np.array(out)


configs = list(itertools.product(GRID["dy"], GRID["dz"], GRID["s_up"], GRID["s_post"]))
i_tpl = int(np.argmin([np.abs(np.array(c) - np.array(TEMPLATE)).sum() for c in configs]))
print(f"{len(configs)} configs; template index {i_tpl}")
rbf = RBFInterpolator(Pg, np.eye(len(Pg)), kernel="thin_plate_spline")
W_all = np.stack([rbf(arc(*c)) for c in configs])          # n_cfg x 19 x n_grid
print(f"interpolation weights ready ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# per-trial statistics
# ---------------------------------------------------------------------------
_gk = {}


def proj(use_idx, ref_i):
    key = (tuple(use_idx), ref_i)
    if key not in _gk:
        G = L_sc[use_idx] - L_sc[ref_i]
        lam = np.trace(G @ G.T) / len(use_idx) / SNR ** 2
        K = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use_idx)))
        _gk[key] = (L_gr @ K, L_sc[ref_i] @ K)
    return _gk[key]


def car_matrix(good):
    E = np.zeros((19, 19))
    for side in (LEFT, RIGHT):
        cols = [EAR.index(n) for n in side if n in good]
        if len(cols) < 2:
            continue
        for c in cols:
            E[c, cols] = -1.0 / len(cols)
            E[c, c] += 1.0
    return E


subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
STATS, DATA = {}, {}
for s in subjects:
    d = read_mat(str(PRE / f"dataSubject{s}.mat"))
    ch = list(d["channels"])
    trials = [np.asarray(t, float) for t in d["eegTrials"]]
    ref = REF[s]
    ref_i = SC.index(ref)
    STATS[s], DATA[s] = [], {"envs": [np.asarray(e, float) for e in d["envelopes"]],
                             "att": np.asarray(d["attSpeaker"]).astype(int).ravel(), "trials": []}
    for t in trials:
        sc = t[:, [ch.index(n) for n in SC]]
        nanf = np.isnan(sc).mean(0)
        use = [SC.index(n) for n in INPUTS if n != ref and nanf[SC.index(n)] <= NAN_CH_FRAC]
        dref = sc[:, use] - sc[:, [ref_i]]
        meas = t[:, [ch.index(e) for e in EAR]] - t[:, [ch.index("Fp1-cr")]]
        good = [e for e, f in zip(EAR, np.isnan(meas).mean(0)) if f <= NAN_CH_FRAC]
        E = car_matrix(good)
        gi = [EAR.index(e) for e in good]
        valid = np.isfinite(dref).all(1) & np.isfinite(meas[:, gi]).all(1)
        X = dref[valid] - dref[valid].mean(0)
        Mc = np.nan_to_num(meas[valid]) @ E.T
        Mc -= Mc.mean(0)
        n = len(X)
        vm = (Mc ** 2).mean(0) if n else np.zeros(19)
        gi = [g for g in gi if vm[g] > 1e-12] if n >= 1000 else []
        STATS[s].append({"use": use, "ref_i": ref_i, "E": E, "good": gi,
                         "Cdd": X.T @ X / max(n, 1), "Cdm": X.T @ Mc / max(n, 1), "vm": vm})
        DATA[s]["trials"].append({"use": use, "ref_i": ref_i, "d": dref})
    print(f"subject {s:2d} stats ready ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# score matrix: mean ear-CAR r per (config, subject)
# ---------------------------------------------------------------------------
SCORE = np.full((len(configs), len(subjects)), np.nan)
N_SKIPPED = {}
for j, s in enumerate(subjects):
    per_trial = []
    for st in STATS[s]:
        if len(st["good"]) < 2:
            N_SKIPPED[s] = N_SKIPPED.get(s, 0) + 1
            continue
        GK, gref = proj(st["use"], st["ref_i"])
        A = np.einsum("ij,cjk->cik", st["E"], W_all @ GK - gref[None, None, :])  # cfg x19 x n
        vp = np.einsum("cik,kl,cil->ci", A, st["Cdd"], A)
        cpm = np.einsum("cik,ki->ci", A, st["Cdm"])
        g = st["good"]
        r = cpm[:, g] / np.sqrt(vp[:, g] * st["vm"][g] + 1e-30)
        per_trial.append(np.nanmean(r, 1))
    if per_trial:
        SCORE[:, j] = np.nanmean(np.vstack(per_trial), 0)
print(f"trials skipped (fewer than 2 usable ear channels): {N_SKIPPED or 'none'}")
print(f"subjects without any usable trial: "
      f"{[s for j, s in enumerate(subjects) if np.isnan(SCORE[0, j])] or 'none'}")
print(f"score matrix ready ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# LOSO selection + held-out signal validity
# ---------------------------------------------------------------------------
chosen = {}
for j, s in enumerate(subjects):
    others = [k for k in range(len(subjects)) if k != j]
    chosen[s] = int(np.nanargmax(np.nanmean(SCORE[:, others], axis=1)))
sig_tpl = np.array([SCORE[i_tpl, j] for j in range(len(subjects))])
sig_cal = np.array([SCORE[chosen[s], j] for j, s in enumerate(subjects)])
ok = np.isfinite(sig_tpl) & np.isfinite(sig_cal)
p_sig = float(wilcoxon(sig_cal[ok], sig_tpl[ok], alternative="greater").pvalue)
best_all = int(np.nanargmax(np.nanmean(SCORE, axis=1)))


def on_edge(i):
    c = configs[i]
    keys = ("dy", "dz", "s_up", "s_post")
    return [k for k, v in zip(keys, c)
            if np.isclose(v, min(GRID[k])) or np.isclose(v, max(GRID[k]))]


EDGE = {s: on_edge(chosen[s]) for s in subjects}
if any(EDGE.values()):
    print(f"WARNING: chosen configs on the grid boundary for {EDGE} -> the optimum may lie "
          f"outside the searched range")
cfg = lambda i: {k: round(100 * v, 1) for k, v in zip(("dy_cm", "dz_cm", "s_up_cm", "s_post_cm"),
                                                      configs[i])}
print(f"\nheld-out signal r: template {np.nanmedian(sig_tpl):.3f} (sanity: ~0.26 expected) "
      f"-> calibrated {np.nanmedian(sig_cal):.3f} "
      f"(better in {np.sum(sig_cal[ok] > sig_tpl[ok])}/{ok.sum()}, p = {p_sig:.2g})")
print(f"best config on all subjects: {cfg(best_all)}")
print("LOSO choices:", {s: cfg(chosen[s]) for s in subjects})

# ---------------------------------------------------------------------------
# held-out decoding ranking with the LOSO-chosen geometry
# ---------------------------------------------------------------------------
def zscore(x):
    mu, sd = np.nanmean(x, 0, keepdims=True), np.nanstd(x, 0, keepdims=True)
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


CAL = {}
for s in DEC_SUBJ:
    W = W_all[chosen[s]]
    preds = []
    for tr in DATA[s]["trials"]:
        GK, gref = proj(tr["use"], tr["ref_i"])
        A = W @ GK - gref[None, :]
        p = np.nan_to_num(tr["d"]) @ A.T
        p[np.isnan(tr["d"]).any(1)] = np.nan
        preds.append(p)
    for pair in PAIRS:
        if (s, pair) not in MEAS:
            continue
        a, b = (EAR.index(x) for x in pair.split("-"))
        CAL[(s, pair)] = aad_single([p[:, a] - p[:, b] for p in preds],
                                    DATA[s]["envs"], DATA[s]["att"])
    print(f"subject {s:2d} calibrated decoding done ({time.time()-t0:.0f}s)")


def group_rho(pred):
    gm = [np.mean([MEAS[(s, p)] for s in DEC_SUBJ if (s, p) in MEAS]) for p in PAIRS]
    gp = [np.mean([pred[(s, p)] for s in DEC_SUBJ if (s, p) in pred]) for p in PAIRS]
    per = [spearmanr([MEAS[(s, p)] for p in PAIRS if (s, p) in MEAS],
                     [pred[(s, p)] for p in PAIRS if (s, p) in MEAS]).correlation
           for s in DEC_SUBJ]
    return float(spearmanr(gm, gp).correlation), float(np.nanmedian(per)), float(np.mean(gp))


rho_t, rho_t_s, dr_t = group_rho(PHYS0)
rho_c, rho_c_s, dr_c = group_rho(CAL)
go = rho_c >= GO_RHO
out = {
    "n_configs": len(configs), "template_config": cfg(i_tpl), "best_config_all": cfg(best_all),
    "loso_configs": {s: cfg(chosen[s]) for s in subjects},
    "signal_heldout": {"template_median": float(np.nanmedian(sig_tpl)),
                       "calibrated_median": float(np.nanmedian(sig_cal)),
                       "n_subjects": int(ok.sum()),
                       "n_better": int(np.sum(sig_cal[ok] > sig_tpl[ok])), "p_one_sided": p_sig},
    "trials_skipped": {str(k): v for k, v in N_SKIPPED.items()},
    "chosen_on_grid_edge": {str(k): v for k, v in EDGE.items()},
    "template_score_per_subject": {str(s): float(sig_tpl[j]) for j, s in enumerate(subjects)},
    "decoding_rank": {"n_subjects": len(DEC_SUBJ), "n_pairs": len(PAIRS),
                      "template_rho_group": rho_t, "template_rho_subject_median": rho_t_s,
                      "calibrated_rho_group": rho_c, "calibrated_rho_subject_median": rho_c_s,
                      "ridge_ceiling_rho_group": 0.46, "noise_ceiling_rho": 0.70,
                      "mean_dr_template": dr_t, "mean_dr_calibrated": dr_c},
    "go_threshold": GO_RHO,
    "go_no_go": ("GO: off-cap placement claims allowed with the calibrated template" if go else
                 "NO-GO for off-cap claims: step 4 on-cap; off-cap layouts reported at signal level"),
}
print(f"\ndecoding ranking (group means, {len(PAIRS)} pairs, {len(DEC_SUBJ)} subjects): "
      f"template rho {rho_t:.2f} -> calibrated {rho_c:.2f} "
      f"(per-subject median {rho_t_s:.2f} -> {rho_c_s:.2f}); ridge ceiling 0.46")
print(out["go_no_go"])
(RES / "step3c_calibration.json").write_text(json.dumps(out, indent=2))
np.savez(RES / "step3c_score_matrix.npz", score=SCORE, configs=np.array(configs),
         subjects=np.array(subjects))

# ---------------------------------------------------------------------------
# figure: template vs calibrated arc (left ear); ranking bars
# ---------------------------------------------------------------------------
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c9c9c9"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
side = Pg[:, 0] < 0
a1.scatter(Pg[side, 1] * 100, Pg[side, 2] * 100, s=3, c="#dddddd")
for i, col, lab in ((i_tpl, ORANGE, "step-2 template"), (best_all, BLUE, "calibrated (all subjects)")):
    P = arc(*configs[i])[:9]
    a1.plot(P[:, 1] * 100, P[:, 2] * 100, "-o", color=col, ms=6, lw=1.5, label=lab,
            markeredgecolor="white")
    for k, p in enumerate(P):
        a1.annotate(str(k + 1), (p[1] * 100, p[2] * 100), fontsize=7, color=MUTED,
                    xytext=(4, 2), textcoords="offset points")
for s in subjects:
    P = arc(*configs[chosen[s]])[:9]
    a1.plot(P[:, 1] * 100, P[:, 2] * 100, "-", color=BLUE, lw=0.5, alpha=0.25)
a1.scatter(lpa[1] * 100, lpa[2] * 100, marker="x", c=INK, s=40, label="preauricular point")
a1.set_xlim(lpa[1] * 100 - 9, lpa[1] * 100 + 5)
a1.set_ylim(lpa[2] * 100 - 6, lpa[2] * 100 + 8)
a1.set_aspect("equal")
a1.set_xlabel("y, cm (anterior →)", color=INK)
a1.set_ylabel("z, cm (up)", color=INK)
a1.set_title("Left cEEGrid: template vs calibrated (thin = LOSO folds)", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8, loc="lower left")
for sp in ("top", "right"):
    a1.spines[sp].set_visible(False)
vals = [rho_t, rho_c, 0.46, 0.70]
labs = ["physics,\ntemplate", "physics,\ncalibrated (LOSO)", "ridge\nceiling", "noise\nceiling"]
a2.bar(range(4), vals, color=[ORANGE, BLUE, AQUA, GREY], width=0.55)
for k, v in enumerate(vals):
    a2.text(k, v + 0.02, f"{v:.2f}", ha="center", color=INK)
a2.axhline(GO_RHO, color=MUTED, lw=1, ls="--")
a2.text(3.3, GO_RHO + 0.01, f"go ≥ {GO_RHO}", ha="right", va="bottom", fontsize=8, color=MUTED)
a2.set_xticks(range(4), labs, fontsize=9, color=MUTED)
a2.set_ylim(min(0, min(vals)) - 0.05, 1)
a2.set_ylabel("Spearman ρ, 81 pairs (group means)", color=INK)
a2.set_title("Decoding-rank agreement with measured ear pairs", fontsize=10, color=INK)
for sp in ("top", "right"):
    a2.spines[sp].set_visible(False)
fig.savefig(FIG / "step3c_calibration.png", dpi=150)
print(f"Wrote results/step3c_calibration.json, figures/step3c_calibration.png "
      f"({time.time()-t0:.0f}s)")
