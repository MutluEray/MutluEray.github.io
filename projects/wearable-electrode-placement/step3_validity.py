"""
Step 3 — can physics predict around-ear EEG from scalp EEG? (the answer-key test)
Project: physics-guided wearable electrode placement (C7-B / F17).

The 19 around-ear channels were recorded but are never used as inputs. From the scalp
channels only, four methods predict them:
  physics : MNE minimum-norm source estimate on fsaverage (SNR = 3), forward-projected to
            the template around-ear positions:  y_hat = (L_e - L_ref) K d
  spline  : spherical-spline interpolation (Perrin 1989, m = 4, 50 Legendre terms); no physics
  nearest : nearest scalp electrode
  ceiling : ridge regression scalp -> ear trained on the other 5 trials (data-driven upper
            bound on what is linearly predictable from the scalp; NOT a competitor)
Two input sets: all scalp channels, and "no mastoids" (M1/M2 removed — they sit under the
cEEGrid's lower end, so with them the test is partly trivial).

Metrics
  (A) signal validity: per ear channel and held-out trial, correlation between predicted and
      measured signals after a same-side ear-CAR (reference-free; the shared Fp1 reference
      would inflate every method). Also within-ear bipolar pairs (B).
  (C) decoding validity: for all 81 within-ear bipolar pairs (36 left + 45 right), AAD from
      the measured pair vs from the physics-predicted pair (single-channel backward decoder,
      lags 0-400 ms, Ledoit-Wolf, leave-one-trial-out). Metric: mean over 60 s windows of
      r(att) - r(unatt) ("delta-r", more stable than accuracy with 60 windows), plus accuracy.
      Spearman rank agreement across pairs (group means, and per subject).
  Shared-reference check: correlation of Fp1-cr (ear amplifier) with Fp1 and Fp2 (scalp);
  the manual says S04 used Fp2.

Pre-registered (plan doc): physics > spline > nearest, most clearly far below the cap;
predicted decoding optimistic but ranking preserved (Spearman rho > 0.5).
Go/no-go: physics no better than spline (paired Wilcoxon on channel x subject medians,
no-mastoid inputs) -> report the negative result; placement study stays on-cap.

Closed-form throughout (no iterative training -> no early stopping needed).

Run:
    python3 step3_validity.py
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
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
HM = ROOT / "data" / "headmodel" / "headmodel.npz"
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

FS, LAGS, WIN = 20, np.arange(0, 9), 60
SNR = 3.0
NAN_CH_FRAC = 0.5          # a channel with > 50 % NaN in a trial is treated as missing
MASTOIDS = ("M1", "M2")
t0 = time.time()

hm = np.load(HM)
SC_NAMES = list(hm["names_scalp"])
EAR_NAMES = list(hm["names_ear"])
L_sc = hm["L_scalp"].astype(np.float64)
L_ear = hm["L_ear"].astype(np.float64)
P_sc, P_ear, P_grid = hm["pos_scalp"], hm["pos_ear"], hm["pos_grid"]
LEFT = [n for n in EAR_NAMES if n.startswith("cEL")]
RIGHT = [n for n in EAR_NAMES if n.startswith("cER")]
PAIRS = [(a, b) for side in (LEFT, RIGHT) for a, b in combinations(side, 2)]

# ---------------------------------------------------------------------------
# spherical spline (Perrin et al. 1989)
# ---------------------------------------------------------------------------
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    return c[:3]


SPH_C = fit_sphere(P_grid[P_grid[:, 2] > np.percentile(P_grid[:, 2], 20)])
N_LEG = 50
_n = np.arange(1, N_LEG + 1)
_coef = np.r_[0, (2 * _n + 1) / (_n * (_n + 1)) ** 4 / (4 * np.pi)]


def _unit(P):
    U = P - SPH_C
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def spline_matrix(P_from, P_to):
    """Matrix M (n_to x n_from) so that v_to = M @ v_from."""
    Uf, Ut = _unit(P_from), _unit(P_to)
    G = legval(np.clip(Uf @ Uf.T, -1, 1), _coef)
    n = len(P_from)
    C = np.zeros((n + 1, n + 1))
    C[:n, :n], C[:n, n], C[n, :n] = G, 1.0, 1.0
    Ci = np.linalg.pinv(C)
    Gt = legval(np.clip(Ut @ Uf.T, -1, 1), _coef)
    return np.c_[Gt, np.ones(len(P_to))] @ Ci[:, :n]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def zscore(x):
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


def corr_valid(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def lagged(x):
    T = len(x)
    out = np.zeros((T, len(LAGS)))
    for k, l in enumerate(LAGS):
        out[: T - l, k] = x[l:]
    return out


def aad_single(sig_trials, envs, att):
    """single-channel backward decoder, leave-one-trial-out; returns (delta_r, accuracy)."""
    X = [lagged(zscore(s[:, None])[:, 0]) for s in sig_trials]
    S = [(zscore(e)[:, a - 1], zscore(e)[:, 2 - a]) for e, a in zip(envs, att)]
    drs, hits = [], []
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
            ra, ru = corr(rec[sl], S[k][0][sl]), corr(rec[sl], S[k][1][sl])
            drs.append(ra - ru)
            hits.append(ra > ru)
    return float(np.mean(drs)), float(np.mean(hits))


_kcache = {}


def physics_map(sc_idx, ref_i):
    """(19 x n_in) matrix: ear (ref-relative) from scalp channels sc_idx (ref-relative)."""
    key = (tuple(sc_idx), ref_i)
    if key not in _kcache:
        G = L_sc[sc_idx] - L_sc[ref_i]
        lam = np.trace(G @ G.T) / len(sc_idx) / SNR ** 2
        K = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(sc_idx)))
        _kcache[key] = (L_ear - L_sc[ref_i]) @ K
    return _kcache[key]


def good_ear(Y):
    """ear channels (names) with <= NAN_CH_FRAC missing samples in this trial."""
    frac = np.isnan(Y).mean(0)
    return [n for n, f in zip(EAR_NAMES, frac) if f <= NAN_CH_FRAC]


def ear_car(Y, good):
    """same-side ear CAR over the `good` channels (same set for measured and predicted)."""
    out = np.full(Y.shape, np.nan)
    for side in (LEFT, RIGHT):
        cols = [EAR_NAMES.index(n) for n in side if n in good]
        if len(cols) < 2:
            continue
        mu = Y[:, cols].mean(1, keepdims=True)          # NaN rows stay NaN
        out[:, cols] = Y[:, cols] - mu
    return out


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
METHODS = ["physics", "spline", "nearest", "ceiling"]
INPUTS = {"all": [n for n in SC_NAMES],
          "no_mastoids": [n for n in SC_NAMES if n not in MASTOIDS]}
chan_rows, pair_rows, ref_rows = [], [], []

for subj in subjects:
    d = read_mat(str(PRE / f"dataSubject{subj}.mat"))
    ch = list(d["channels"])
    trials = [np.asarray(t, float) for t in d["eegTrials"]]
    envs = [np.asarray(e, float) for e in d["envelopes"]]
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()

    # ---- shared-reference check ----
    allt = np.vstack(trials)
    fcr = allt[:, ch.index("Fp1-cr")]
    r_fp1 = corr_valid(fcr, allt[:, ch.index("Fp1")])
    r_fp2 = corr_valid(fcr, allt[:, ch.index("Fp2")])
    ref_name = "Fp1" if (np.nan_to_num(r_fp1) >= np.nan_to_num(r_fp2)) else "Fp2"
    ref_rows.append({"subject": subj, "r_Fp1cr_Fp1": r_fp1, "r_Fp1cr_Fp2": r_fp2,
                     "shared_ref_used": ref_name})
    ref_i = SC_NAMES.index(ref_name)

    sc_cols = [ch.index(n) for n in SC_NAMES]
    ear_cols = [ch.index(n) for n in EAR_NAMES]

    for inp_name, inp_list in INPUTS.items():
        # per trial: scalp referenced to shared ref, ear referenced to Fp1-cr
        meas_trials, pred_trials = [], {m: [] for m in METHODS}
        for k, t in enumerate(trials):
            sc = t[:, sc_cols]
            ear = t[:, ear_cols] - t[:, [ch.index("Fp1-cr")]]
            nan_frac = np.isnan(sc).mean(0)
            use = [SC_NAMES.index(n) for n in inp_list
                   if n != ref_name and nan_frac[SC_NAMES.index(n)] <= NAN_CH_FRAC]
            dref = sc[:, use] - sc[:, [ref_i]]
            # physics
            M = physics_map(use, ref_i)
            pred_trials["physics"].append(np.nan_to_num(dref) @ M.T)
            # spline (ref electrode included as a zero channel)
            S = spline_matrix(np.vstack([P_sc[use], P_sc[ref_i]]), P_ear)
            pred_trials["spline"].append(np.c_[np.nan_to_num(dref), np.zeros(len(dref))] @ S.T)
            # nearest
            P_in = P_sc[use]
            nn = np.argmin(np.linalg.norm(P_ear[:, None] - P_in[None], axis=2), axis=1)
            pred_trials["nearest"].append(np.nan_to_num(dref)[:, nn])
            meas_trials.append(ear)
            nanmask = np.isnan(dref).any(1)
            for m in ("physics", "spline", "nearest"):
                pred_trials[m][-1][nanmask] = np.nan
        # ceiling: ridge from the same scalp input set, leave-one-trial-out, CAR target
        for k in range(len(trials)):
            tr = [j for j in range(len(trials)) if j != k]
            use = [SC_NAMES.index(n) for n in inp_list if n != ref_name]

            def xy(j):
                t = trials[j]
                X = np.nan_to_num(t[:, sc_cols][:, use] - t[:, sc_cols][:, [ref_i]])
                Y = ear_car(meas_trials[j], good_ear(meas_trials[j]))
                return X, Y
            Xs, Ys = zip(*[xy(j) for j in tr])
            Xtr, Ytr = np.vstack(Xs), np.vstack(Ys)
            Xte, _ = xy(k)
            pred = np.full((len(Xte), len(EAR_NAMES)), np.nan)
            for e in range(len(EAR_NAMES)):
                m = np.isfinite(Ytr[:, e])
                if m.sum() < 1000:
                    continue
                rg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xtr[m], Ytr[m, e])
                pred[:, e] = rg.predict(Xte)
            pred_trials["ceiling"].append(pred)

        # ---- (A) per-channel signal validity, ear-CAR ----
        for k in range(len(trials)):
            good = good_ear(meas_trials[k])
            meas = ear_car(meas_trials[k], good)
            for m in METHODS:
                p = pred_trials[m][k] if m == "ceiling" else ear_car(pred_trials[m][k], good)
                for e, name in enumerate(EAR_NAMES):
                    chan_rows.append({"subject": subj, "inputs": inp_name, "trial": k + 1,
                                      "method": m, "channel": name, "metric": "earCAR",
                                      "r": corr_valid(p[:, e], meas[:, e])})
            # (B) bipolar pairs
            for a, b in PAIRS:
                ia, ib = EAR_NAMES.index(a), EAR_NAMES.index(b)
                mb = meas_trials[k][:, ia] - meas_trials[k][:, ib]
                for m in ("physics", "spline", "nearest"):
                    pb = pred_trials[m][k][:, ia] - pred_trials[m][k][:, ib]
                    chan_rows.append({"subject": subj, "inputs": inp_name, "trial": k + 1,
                                      "method": m, "channel": f"{a}-{b}", "metric": "bipolar",
                                      "r": corr_valid(pb, mb)})

        # ---- (C) decoding validity on bipolar pairs (physics vs measured) ----
        for a, b in PAIRS:
            ia, ib = EAR_NAMES.index(a), EAR_NAMES.index(b)
            meas_sig = [mt[:, ia] - mt[:, ib] for mt in meas_trials]
            if any(np.isnan(s).mean() > NAN_CH_FRAC for s in meas_sig):
                continue
            pred_sig = [pt[:, ia] - pt[:, ib] for pt in pred_trials["physics"]]
            dr_m, acc_m = aad_single(meas_sig, envs, att)
            dr_p, acc_p = aad_single(pred_sig, envs, att)
            pair_rows.append({"subject": subj, "inputs": inp_name, "pair": f"{a}-{b}",
                              "side": "L" if a.startswith("cEL") else "R",
                              "dr_measured": dr_m, "acc_measured": acc_m,
                              "dr_physics": dr_p, "acc_physics": acc_p})
    print(f"subject {subj:2d} done  (shared ref {ref_name}: r={max(np.nan_to_num(r_fp1), np.nan_to_num(r_fp2)):.3f})"
          f"  {time.time()-t0:.0f}s")

# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


write_csv(chan_rows, RES / "step3_signal_corr.csv")
write_csv(pair_rows, RES / "step3_pair_decoding.csv")
write_csv(ref_rows, RES / "step3_shared_ref.csv")

summary = {"shared_ref": ref_rows, "signal": {}, "decoding": {}, "tests": {}}


def subj_chan_median(inp, method, metric):
    """median over trials per (subject, channel) -> array keyed consistently."""
    out = {}
    for r in chan_rows:
        if r["inputs"] == inp and r["method"] == method and r["metric"] == metric \
                and np.isfinite(r["r"]):
            out.setdefault((r["subject"], r["channel"]), []).append(r["r"])
    return {k: float(np.median(v)) for k, v in out.items()}


print("\n=== (A/B) signal validity: median r over subject x channel (trial medians) ===")
print(f"{'inputs':>12s} {'metric':>8s} " + " ".join(f"{m:>9s}" for m in METHODS))
for inp in INPUTS:
    for metric in ("earCAR", "bipolar"):
        row = {}
        for m in METHODS:
            v = subj_chan_median(inp, m, metric)
            row[m] = float(np.median(list(v.values()))) if v else np.nan
        summary["signal"][f"{inp}/{metric}"] = row
        print(f"{inp:>12s} {metric:>8s} " + " ".join(f"{row[m]:9.3f}" for m in METHODS))

# paired tests physics vs spline / nearest (no_mastoids, earCAR = primary)
for inp in INPUTS:
    ph = subj_chan_median(inp, "physics", "earCAR")
    for other in ("spline", "nearest"):
        ot = subj_chan_median(inp, other, "earCAR")
        keys = sorted(set(ph) & set(ot))
        a, b = np.array([ph[k] for k in keys]), np.array([ot[k] for k in keys])
        p = float(wilcoxon(a, b, alternative="greater").pvalue) if len(keys) > 5 else np.nan
        summary["tests"][f"{inp}: physics > {other}"] = {
            "n": len(keys), "median_diff": float(np.median(a - b)), "p_one_sided": p,
            "frac_physics_better": float(np.mean(a > b))}

# per-channel (no_mastoids) physics vs spline, for "far below the cap" reading
per_ch = {}
for name in EAR_NAMES:
    vals = {m: [v for (s, c), v in subj_chan_median("no_mastoids", m, "earCAR").items()
                if c == name] for m in METHODS}
    per_ch[name] = {m: float(np.median(v)) if v else np.nan for m, v in vals.items()}
summary["signal"]["per_channel_no_mastoids_earCAR"] = per_ch

print("\n=== (C) decoding validity, within-ear bipolar pairs (physics-predicted vs measured) ===")
for inp in INPUTS:
    rows = [r for r in pair_rows if r["inputs"] == inp]
    pairs = sorted({r["pair"] for r in rows})
    gm = {p: (np.mean([r["dr_measured"] for r in rows if r["pair"] == p]),
              np.mean([r["dr_physics"] for r in rows if r["pair"] == p])) for p in pairs}
    mvals = np.array([gm[p][0] for p in pairs])
    pvals = np.array([gm[p][1] for p in pairs])
    rho_g = spearmanr(mvals, pvals).correlation
    rho_s = []
    for s in subjects:
        rs = [r for r in rows if r["subject"] == s]
        if len(rs) > 10:
            rho_s.append(spearmanr([r["dr_measured"] for r in rs],
                                   [r["dr_physics"] for r in rs]).correlation)
    acc_m = np.mean([r["acc_measured"] for r in rows])
    acc_p = np.mean([r["acc_physics"] for r in rows])
    best_m = pairs[int(np.argmax(mvals))]
    best_p = pairs[int(np.argmax(pvals))]
    rank_of_pred_best = int(np.sum(mvals > gm[best_p][0])) + 1
    summary["decoding"][inp] = {
        "n_pairs": len(pairs), "spearman_group_mean": float(rho_g),
        "spearman_per_subject_median": float(np.nanmedian(rho_s)),
        "spearman_per_subject_iqr": [float(np.nanpercentile(rho_s, 25)),
                                     float(np.nanpercentile(rho_s, 75))],
        "mean_acc_measured": float(acc_m), "mean_acc_physics": float(acc_p),
        "mean_dr_measured": float(mvals.mean()), "mean_dr_physics": float(pvals.mean()),
        "best_pair_measured": best_m, "best_pair_physics": best_p,
        "measured_rank_of_physics_best_pair": rank_of_pred_best,
        "group_means": {p: {"measured": float(gm[p][0]), "physics": float(gm[p][1])}
                        for p in pairs}}
    print(f"{inp:>12s}: Spearman (group means, {len(pairs)} pairs) = {rho_g:.2f};"
          f" per-subject median {np.nanmedian(rho_s):.2f};"
          f" mean acc measured {100*acc_m:.1f} % vs physics {100*acc_p:.1f} %;"
          f" physics' best pair {best_p} ranks #{rank_of_pred_best} of {len(pairs)} measured")

prim = summary["tests"]["no_mastoids: physics > spline"]
go = prim["median_diff"] > 0 and prim["p_one_sided"] < 0.05
summary["go_no_go"] = ("GO: physics beats spline (no-mastoid inputs, ear-CAR)" if go else
                       "NO-GO on physics advantage: report negative result; placement on-cap only")
print("\n" + json.dumps(summary["tests"], indent=1))
print(summary["go_no_go"])
(RES / "step3_validity.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------------------------------------------------------------------
# figure: (left) per-channel median r by method; (right) decoding rank agreement
# ---------------------------------------------------------------------------
COL = {"physics": "#2a78d6", "spline": "#eb6834", "nearest": "#1baf7a", "ceiling": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.6, 1]})
x = np.arange(len(EAR_NAMES))
off = {"physics": -0.24, "spline": -0.08, "nearest": 0.08, "ceiling": 0.24}
for m in METHODS:
    y = [per_ch[n][m] for n in EAR_NAMES]
    ax1.scatter(x + off[m], y, s=36, c=COL[m], label=m, zorder=3,
                edgecolors="white", linewidths=1.5)
ax1.axhline(0, color=MUTED, lw=0.8)
ax1.set_xticks(x, [n.replace("cE", "") for n in EAR_NAMES], fontsize=8, color=MUTED)
ax1.set_ylabel("median r, predicted vs measured", color=INK)
ax1.set_title("Around-ear signals predicted from scalp only (no mastoids, ear-CAR)\n"
              "1 = top-front of the C, 9/10 = bottom-front", fontsize=10, color=INK)
ax1.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
ax1.grid(axis="y", color="#eeeeee", lw=0.6)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

dd = summary["decoding"]["no_mastoids"]["group_means"]
mv = np.array([v["measured"] for v in dd.values()])
pv = np.array([v["physics"] for v in dd.values()])
ax2.scatter(mv, pv, s=28, c=COL["physics"], edgecolors="white", linewidths=1.5, zorder=3)
lim = [min(mv.min(), pv.min()) - 0.005, max(mv.max(), pv.max()) + 0.005]
ax2.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
ax2.set_xlabel("measured pair: mean Δr (att − unatt)", color=INK)
ax2.set_ylabel("physics-predicted pair: mean Δr", color=INK)
ax2.set_title(f"81 within-ear bipolar pairs, group means\nSpearman ρ = "
              f"{summary['decoding']['no_mastoids']['spearman_group_mean']:.2f}",
              fontsize=10, color=INK)
ax2.grid(color="#eeeeee", lw=0.6)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig.savefig(FIG / "step3_validity.png", dpi=150)
print(f"\nWrote results/step3_*.csv, results/step3_validity.json, figures/step3_validity.png "
      f"({time.time()-t0:.0f}s)")
