"""
Step 4b — on-cap placement of a wearable bipolar pair (physics-guided, LOSO-validated).
Project: physics-guided wearable electrode placement (C7-B / F17).

Step 4a: on-cap physics ranks real scalp pairs at rho 0.68 (noise ceiling 0.89) and the
trial-level delta-r agrees with the windowed one at 0.999 -> GO for an exhaustive search.

Search space: candidate grid points within 2.5 cm of a real scalp electrode ("on-cap").
Pair classes: mini (1-3 cm apart) and short (3-7 cm). Every pair is scored per subject with a
closed-form version of the AAD pipeline (single-channel backward decoder, 0-400 ms lags,
leave-one-trial-out, trial-level delta-r = r(att) - r(unatt)), using lagged covariances of the
scalp data and lead-field projections (physics) or spherical-spline weights (robustness).
Population layout = best mean over training subjects; evaluated on the held-out subject (LOSO).

Pre-registered (before running):
  P4' population-optimal pair over left lateral centro-parietal/temporal scalp
      (within ~2 cm of CP5/C3/P7)
  P5  continuous placement beats the best real 10-20 pair by <= 20 % relative delta-r
      (physics world) and <= 3 accuracy points held-out
  P6  mini pairs lose >= 30 % of delta-r against short pairs
  Engine check: closed-form delta-r on real pairs vs step 4a trial-level values, rho >= 0.95
A local continuous refinement (Nelder-Mead in the tangent plane, early stop on stall:
xatol 0.5 mm, fatol 1e-6, max 300 iterations) checks whether off-grid positions add anything.

Run:
    python3 step4b_oncap_placement.py
"""

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legval
from pymatreader import read_mat
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, NL, WIN = 20, 9, 60
SNR, NAN_CH_FRAC = 3.0, 0.5
ONCAP_DIST = 0.025
MINI, SHORT = (0.010, 0.030), (0.030, 0.070)
CHUNK = 8000
t0 = time.time()

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
L_sc = hm["L_scalp"].astype(np.float64)
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
d_near = np.min(np.linalg.norm(P_grid[:, None] - P_sc[None], axis=2), axis=1)
REG = np.where(d_near <= ONCAP_DIST)[0]
P_reg = P_grid[REG]
L_reg = hm["L_grid"][REG].astype(np.float64)
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}
print(f"on-cap region: {len(REG)} grid points ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# spline
# ---------------------------------------------------------------------------
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    return np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0][:3]


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
    Gt = legval(np.clip(Ut @ Uf.T, -1, 1), _coef)
    return np.c_[Gt, np.ones(len(P_to))] @ np.linalg.pinv(C)[:, :n]


_cache = {}


def projections(use, ref_i):
    """physics and spline weights (region x n, scalp x n) for this input set."""
    key = (tuple(use), ref_i)
    if key not in _cache:
        G = L_sc[use] - L_sc[ref_i]
        lam = np.trace(G @ G.T) / len(use) / SNR ** 2
        K = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use)))
        S_reg = spline_matrix(np.vstack([P_sc[use], P_sc[ref_i]]), P_reg)[:, :-1]
        _cache[key] = {"phys": (L_reg - L_sc[ref_i]) @ K, "spline": S_reg,
                       "phys_sc": (L_sc - L_sc[ref_i]) @ K}
    return _cache[key]


# ---------------------------------------------------------------------------
# per-trial statistics
# ---------------------------------------------------------------------------
def zscore(x):
    mu, sd = np.nanmean(x, 0, keepdims=True), np.nanstd(x, 0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
ST = {}
for s in subjects:
    d = read_mat(str(PRE / f"dataSubject{s}.mat"))
    ch = list(d["channels"])
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    ref_i = SC.index(REF[s])
    cand = [i for i in range(len(SC)) if i != ref_i]
    ST[s] = []
    for t, e, a in zip(d["eegTrials"], d["envelopes"], att):
        t = np.asarray(t, float)[:, [ch.index(n) for n in SC]]
        dref = t - t[:, [ref_i]]
        nanf = np.isnan(dref).mean(0)
        use = [i for i in cand if nanf[i] <= NAN_CH_FRAC]
        X = dref[:, use]
        X = X - np.nanmean(X, 0, keepdims=True)
        X = np.nan_to_num(X)
        T = len(X)
        ez = zscore(np.asarray(e, float))
        sa, su = ez[:, a - 1], ez[:, 2 - a]
        R = np.stack([X[: T - k].T @ X[k:] / T for k in range(NL)])         # 9 x n x n
        R = 0.5 * (R + R.transpose(0, 2, 1))                                 # scalar forms only
        qa = np.stack([X[k:].T @ sa[: T - k] / T for k in range(NL)])        # 9 x n
        qu = np.stack([X[k:].T @ su[: T - k] / T for k in range(NL)])
        ST[s].append({"use": use, "ref_i": ref_i, "R": R, "qa": qa, "qu": qu, "T": T,
                      "X": X, "sa": sa, "su": su, **projections(use, ref_i)})
    print(f"subject {s:2d} stats ready ({time.time()-t0:.0f}s)")

TOEP = np.abs(np.subtract.outer(np.arange(NL), np.arange(NL)))


def dr_from_A(st, A_list):
    """closed-form LOTO trial-level delta-r for B candidate signals x = A d (one row each)."""
    nt, B = len(st), A_list[0].shape[0]
    g = np.empty((nt, B, NL))
    ca, cu = np.empty((nt, B, NL)), np.empty((nt, B, NL))
    for t, (tr, A) in enumerate(zip(st, A_list)):
        AR = np.matmul(A[None], tr["R"])                     # 9 x B x n
        gg = np.einsum("kbn,bn->bk", AR, A)
        v0 = np.maximum(gg[:, 0], 1e-30)
        g[t] = gg / v0[:, None]
        ca[t] = (A @ tr["qa"].T) / np.sqrt(v0)[:, None]
        cu[t] = (A @ tr["qu"].T) / np.sqrt(v0)[:, None]
    S = g[:, :, TOEP]                                        # nt x B x 9 x 9
    Tw = np.array([tr["T"] for tr in st], float)
    out = np.zeros(B)
    for k in range(nt):
        trn = [j for j in range(nt) if j != k]
        Ss = np.tensordot(Tw[trn], S[trn], axes=(0, 0))
        cs = np.tensordot(Tw[trn], ca[trn], axes=(0, 0))
        Ss += 1e-6 * Ss[:, :1, :1] * np.eye(NL)
        w = np.linalg.solve(Ss, cs[..., None])[..., 0]
        var = np.maximum(np.einsum("bi,bij,bj->b", w, S[k], w), 1e-30)
        out += (np.einsum("bi,bi->b", w, ca[k]) - np.einsum("bi,bi->b", w, cu[k])) / np.sqrt(var)
    return out / nt


def dr_pairs(s, I, J, kind):
    st = ST[s]
    out = np.empty(len(I))
    for c0 in range(0, len(I), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        out[sl] = dr_from_A(st, [tr[kind][I[sl]] - tr[kind][J[sl]] for tr in st])
    return out


# ---------------------------------------------------------------------------
# engine check on real pairs (vs step 4a pipeline values)
# ---------------------------------------------------------------------------
with open(RES / "step4a_oncap_pairs.csv") as f:
    r4a = list(csv.DictReader(f))
eng_cf, eng_pipe = [], []
REAL_PAIRS = sorted({r["pair"] for r in r4a})
MEAS_CF = {}
for s in subjects:
    st = ST[s]
    for p in REAL_PAIRS:
        a, b = (SC.index(x) for x in p.split("-"))
        if any(a not in tr["use"] or b not in tr["use"] for tr in st):
            continue
        A_list = []
        for tr in st:
            v = np.zeros((1, len(tr["use"])))
            v[0, tr["use"].index(a)], v[0, tr["use"].index(b)] = 1.0, -1.0
            A_list.append(v)
        MEAS_CF[(s, p)] = float(dr_from_A(st, A_list)[0])
for r in r4a:
    k = (int(float(r["subject"])), r["pair"])
    if k in MEAS_CF:
        eng_cf.append(MEAS_CF[k])
        eng_pipe.append(float(r["dr_measured_trial"]))
rho_eng = float(spearmanr(eng_cf, eng_pipe).correlation)
print(f"engine check: closed-form vs step-4a pipeline (trial-level), Spearman {rho_eng:.3f} "
      f"on {len(eng_cf)} subject-pairs (go >= 0.95)")

# ---------------------------------------------------------------------------
# candidate pairs and exhaustive scoring
# ---------------------------------------------------------------------------
D = np.linalg.norm(P_reg[:, None] - P_reg[None], axis=2)
iu = np.triu_indices(len(REG), 1)
dist = D[iu]
CLS = {"mini": (dist >= MINI[0]) & (dist <= MINI[1]),
       "short": (dist > SHORT[0]) & (dist <= SHORT[1])}
PAIRS = {c: (iu[0][m], iu[1][m]) for c, m in CLS.items()}
print({c: len(v[0]) for c, v in PAIRS.items()}, "candidate pairs")
SCORES = {}
for kind in ("phys", "spline"):
    for c, (I, J) in PAIRS.items():
        SCORES[(kind, c)] = np.stack([dr_pairs(s, I, J, kind) for s in subjects])
        print(f"scored {kind}/{c}: {SCORES[(kind, c)].shape} ({time.time()-t0:.0f}s)")

# real pairs in the physics world (re-projected electrodes) and measured (closed form)
RP = [p for p in REAL_PAIRS if all((s, p) in MEAS_CF for s in subjects)]
RP_meas = np.array([[MEAS_CF[(s, p)] for p in RP] for s in subjects])
RP_phys = np.empty_like(RP_meas)
for i, s in enumerate(subjects):
    st = ST[s]
    for j, p in enumerate(RP):
        a, b = (SC.index(x) for x in p.split("-"))
        RP_phys[i, j] = dr_from_A(st, [tr["phys_sc"][[a]] - tr["phys_sc"][[b]] for tr in st])[0]

# ---------------------------------------------------------------------------
# LOSO selection + held-out evaluation (closed form and full windowed pipeline)
# ---------------------------------------------------------------------------
def lagged(x):
    o = np.zeros((len(x), NL))
    for k in range(NL):
        o[: len(x) - k, k] = x[k:]
    return o


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def aad_windowed(sig, st):
    X = [lagged(zscore(x[:, None])[:, 0]) for x in sig]
    S = [(tr["sa"], tr["su"]) for tr in st]
    drs, hits = [], []
    for k in range(len(X)):
        trn = [j for j in range(len(X)) if j != k]
        Xtr = np.vstack([X[j] for j in trn])
        y = np.concatenate([S[j][0] for j in trn])
        cov, _ = ledoit_wolf(Xtr, assume_centered=True)
        w = np.linalg.solve(cov, Xtr.T @ y / len(y))
        rec = X[k] @ w
        L = WIN * FS
        for s0 in range(0, len(rec) - L + 1, L):
            sl = slice(s0, s0 + L)
            ra, ru = corr(rec[sl], S[k][0][sl]), corr(rec[sl], S[k][1][sl])
            drs.append(ra - ru)
            hits.append(ra > ru)
    return float(np.mean(drs)), float(np.mean(hits))


def signals(s, kind, i, j):
    return [tr["X"] @ (tr[kind][i] - tr[kind][j]) for tr in ST[s]]


def real_signals(s, p):
    a, b = (SC.index(x) for x in p.split("-"))
    return [tr["X"][:, tr["use"].index(a)] - tr["X"][:, tr["use"].index(b)] for tr in ST[s]]


loso = []
for h, s in enumerate(subjects):
    oth = [k for k in range(len(subjects)) if k != h]
    row = {"subject": s}
    for c in ("short", "mini"):
        sc = SCORES[("phys", c)]
        best = int(np.argmax(sc[oth].mean(0)))
        I, J = PAIRS[c]
        row[f"virtual_{c}_dr_cf"] = float(sc[h, best])
        dr, acc = aad_windowed(signals(s, "phys", I[best], J[best]), ST[s])
        row[f"virtual_{c}_dr"], row[f"virtual_{c}_acc"] = dr, acc
        row[f"virtual_{c}_pair"] = (int(REG[I[best]]), int(REG[J[best]]))
    bj = int(np.argmax(RP_meas[oth].mean(0)))
    row["real_pair"] = RP[bj]
    row["real_dr_cf_measured"] = float(RP_meas[h, bj])
    row["real_dr_cf_physics"] = float(RP_phys[h, bj])
    dr, acc = aad_windowed(real_signals(s, RP[bj]), ST[s])
    row["real_dr"], row["real_acc"] = dr, acc
    loso.append(row)
    print(f"LOSO subject {s:2d}: real {RP[bj]} acc {100*row['real_acc']:.1f} % | "
          f"virtual short acc {100*row['virtual_short_acc']:.1f} % | "
          f"virtual mini acc {100*row['virtual_mini_acc']:.1f} %")

# ---------------------------------------------------------------------------
# population optimum (all subjects), robustness, refinement
# ---------------------------------------------------------------------------
def nearest_labels(p, k=2):
    dd = np.linalg.norm(P_sc - p, axis=1)
    o = np.argsort(dd)[:k]
    return [f"{SC[i]} ({100*dd[i]:.1f} cm)" for i in o]


pop = {}
for kind in ("phys", "spline"):
    for c in ("short", "mini"):
        sc = SCORES[(kind, c)].mean(0)
        b = int(np.argmax(sc))
        I, J = PAIRS[c]
        pa, pb = P_reg[I[b]], P_reg[J[b]]
        pop[f"{kind}/{c}"] = {"dr_mean": float(sc[b]), "pos_a": pa.tolist(), "pos_b": pb.tolist(),
                              "near_a": nearest_labels(pa), "near_b": nearest_labels(pb),
                              "distance_cm": float(100 * np.linalg.norm(pa - pb)),
                              "top20_dr": np.sort(sc)[-20:][::-1].tolist()}
mid = lambda k: (np.array(pop[k]["pos_a"]) + np.array(pop[k]["pos_b"])) / 2
pop["phys_vs_spline_short_midpoint_distance_cm"] = float(100 * np.linalg.norm(mid("phys/short") -
                                                                             mid("spline/short")))
pop["phys_vs_spline_mini_midpoint_distance_cm"] = float(100 * np.linalg.norm(mid("phys/mini") -
                                                                            mid("spline/mini")))
real_best_pop = RP[int(np.argmax(RP_meas.mean(0)))]
pop["real_best_population"] = {"pair": real_best_pop,
                               "dr_measured_mean": float(RP_meas.mean(0).max()),
                               "dr_physics_mean": float(RP_phys.mean(0)[RP.index(real_best_pop)])}

# continuous refinement of the physics/short optimum
rbf = RBFInterpolator(P_reg, np.eye(len(REG)), kernel="thin_plate_spline")
r_reg = np.linalg.norm(P_reg - SPH_C, axis=1)


def on_skin(p):
    u = _unit(p[None])[0]
    cosang = _unit(P_reg) @ u
    nn = np.argsort(-cosang)[:4]
    w = 1.0 / (1 - cosang[nn] + 1e-6)
    return SPH_C + u * np.sum(w * r_reg[nn]) / w.sum()


def tangent_basis(p):
    n = _unit(p[None])[0]
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


pa0, pb0 = np.array(pop["phys/short"]["pos_a"]), np.array(pop["phys/short"]["pos_b"])
Ba, Bb = tangent_basis(pa0), tangent_basis(pb0)


def pos_from(x):
    pa = on_skin(pa0 + x[0] * Ba[0] + x[1] * Ba[1])
    pb = on_skin(pb0 + x[2] * Bb[0] + x[3] * Bb[1])
    return pa, pb


def neg_obj(x):
    if np.any(np.abs(x) > 0.015):
        return 1.0
    pa, pb = pos_from(x)
    dd = np.linalg.norm(pa - pb)
    if not (SHORT[0] < dd <= SHORT[1]):
        return 1.0
    W = rbf(np.vstack([pa, pb]))
    vals = [dr_from_A(ST[s], [W[:1] @ tr["phys"] - W[1:] @ tr["phys"] for tr in ST[s]])[0]
            for s in subjects]
    return -float(np.mean(vals))


f0 = neg_obj(np.zeros(4))
opt = minimize(neg_obj, np.zeros(4), method="Nelder-Mead",
               options={"xatol": 5e-4, "fatol": 1e-6, "maxiter": 300,
                        "initial_simplex": np.vstack([np.zeros(4), 0.005 * np.eye(4)])})
pa1, pb1 = pos_from(opt.x)
refine = {"dr_grid": -f0, "dr_refined": -float(opt.fun), "gain_relative": float(opt.fun / f0 - 1)
          if f0 != 0 else None, "moved_a_mm": float(1000 * np.linalg.norm(pa1 - pa0)),
          "moved_b_mm": float(1000 * np.linalg.norm(pb1 - pb0)), "iterations": int(opt.nit),
          "stopped": opt.message}
print(f"\nrefinement: grid {-f0:.4f} -> refined {-opt.fun:.4f} "
      f"({100*refine['gain_relative']:+.1f} %), moved {refine['moved_a_mm']:.1f} / "
      f"{refine['moved_b_mm']:.1f} mm in {opt.nit} iterations")

# ---------------------------------------------------------------------------
# summary vs pre-registered predictions
# ---------------------------------------------------------------------------
med = lambda k: float(np.median([r[k] for r in loso]))
gain_phys = (pop["phys/short"]["dr_mean"] / pop["real_best_population"]["dr_physics_mean"] - 1)
acc_gain = med("virtual_short_acc") - med("real_acc")
mini_loss = 1 - pop["phys/mini"]["dr_mean"] / pop["phys/short"]["dr_mean"]
near = pop["phys/short"]["near_a"] + pop["phys/short"]["near_b"]
out = {
    "engine_check_rho": rho_eng, "engine_go": rho_eng >= 0.95,
    "n_region_points": int(len(REG)), "n_pairs": {c: int(len(v[0])) for c, v in PAIRS.items()},
    "population": pop, "refinement": refine,
    "loso_median": {k: med(k) for k in ("real_acc", "virtual_short_acc", "virtual_mini_acc",
                                        "real_dr", "virtual_short_dr", "virtual_mini_dr",
                                        "real_dr_cf_measured", "real_dr_cf_physics",
                                        "virtual_short_dr_cf", "virtual_mini_dr_cf")},
    "loso_real_pairs_chosen": [r["real_pair"] for r in loso],
    "predictions": {
        "P4_nearest_electrodes_of_optimum": near,
        "P5_gain_relative_physics_world": gain_phys,
        "P5_heldout_accuracy_gain_points": 100 * acc_gain,
        "P5_true": bool(gain_phys <= 0.20 and 100 * acc_gain <= 3),
        "P6_mini_loss_relative": mini_loss, "P6_true": bool(mini_loss >= 0.30)},
}
print("\n=== summary ===")
print(f"population optimum (physics, short): {pop['phys/short']['near_a']} / "
      f"{pop['phys/short']['near_b']}, {pop['phys/short']['distance_cm']:.1f} cm apart")
print(f"population optimum (physics, mini) : {pop['phys/mini']['near_a']} / "
      f"{pop['phys/mini']['near_b']}, {pop['phys/mini']['distance_cm']:.1f} cm apart")
print(f"physics vs spline optimum midpoints: short "
      f"{pop['phys_vs_spline_short_midpoint_distance_cm']:.1f} cm, mini "
      f"{pop['phys_vs_spline_mini_midpoint_distance_cm']:.1f} cm apart")
print(f"best real pair (population, measured): {real_best_pop}")
print(f"P5: physics-world gain {100*gain_phys:+.1f} % (<= 20 %?); held-out accuracy gain "
      f"{100*acc_gain:+.1f} points (<= 3?) -> {out['predictions']['P5_true']}")
print(f"P6: mini loses {100*mini_loss:.1f} % vs short (>= 30 %?) -> {out['predictions']['P6_true']}")
print(f"held-out median accuracy (60 s): real {100*med('real_acc'):.1f} %, virtual short "
      f"{100*med('virtual_short_acc'):.1f} %, virtual mini {100*med('virtual_mini_acc'):.1f} %")
(RES / "step4b_placement.json").write_text(json.dumps(out, indent=2, default=str))
with open(RES / "step4b_loso.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(loso[0].keys()))
    w.writeheader()
    w.writerows(loso)

# ---------------------------------------------------------------------------
# figure: (a) top view map, (b) held-out accuracy per subject
# ---------------------------------------------------------------------------
INK, MUTED = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
sc_short = SCORES[("phys", "short")].mean(0)
I, J = PAIRS["short"]
best_touch = np.full(len(REG), -np.inf)
np.maximum.at(best_touch, I, sc_short)
np.maximum.at(best_touch, J, sc_short)
m = np.isfinite(best_touch)
hsc = a1.scatter(P_reg[m, 0] * 100, P_reg[m, 1] * 100, c=best_touch[m], cmap="Greys", s=14)
cb = fig.colorbar(hsc, ax=a1, shrink=0.8)
cb.set_label("best mean Δr of a 3–7 cm pair using this point (physics)", color=MUTED, fontsize=8)
a1.scatter(P_sc[:, 0] * 100, P_sc[:, 1] * 100, s=12, c=MUTED, zorder=3)
for n, p in zip(SC, P_sc):
    a1.annotate(n, (p[0] * 100, p[1] * 100), fontsize=6.5, color=MUTED, xytext=(2, 2),
                textcoords="offset points")
for key, col, lab in (("phys/short", BLUE, "virtual 3–7 cm (physics)"),
                      ("phys/mini", ORANGE, "virtual 1–3 cm (physics)")):
    pa, pb = np.array(pop[key]["pos_a"]), np.array(pop[key]["pos_b"])
    a1.plot([pa[0] * 100, pb[0] * 100], [pa[1] * 100, pb[1] * 100], "-o", color=col, lw=2.5,
            ms=7, markeredgecolor="white", label=lab, zorder=4)
ra, rb = (P_sc[SC.index(x)] for x in real_best_pop.split("-"))
a1.plot([ra[0] * 100, rb[0] * 100], [ra[1] * 100, rb[1] * 100], "-o", color=AQUA, lw=2.5, ms=7,
        markeredgecolor="white", label=f"best real pair ({real_best_pop})", zorder=4)
a1.set_aspect("equal")
a1.set_xlabel("x, cm (right →)", color=INK)
a1.set_ylabel("y, cm (anterior →)", color=INK)
a1.set_title("On-cap placement, population optimum (top view)", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8, loc="lower left")
cats = [("real_acc", "best real\n10-20 pair", AQUA), ("virtual_short_acc", "virtual\n3–7 cm", BLUE),
        ("virtual_mini_acc", "virtual\n1–3 cm", ORANGE)]
for k, (key, lab, col) in enumerate(cats):
    v = np.array([100 * r[key] for r in loso])
    jit = np.linspace(-0.12, 0.12, len(v))
    a2.scatter(k + jit, v, s=26, c=col, edgecolors="white", linewidths=1.5, zorder=3)
    a2.plot([k - 0.22, k + 0.22], [np.median(v)] * 2, color=INK, lw=2)
    a2.text(k + 0.26, np.median(v), f"{np.median(v):.1f}", va="center", fontsize=9, color=INK)
a2.axhline(50, color=MUTED, lw=0.8, ls="--")
a2.set_xticks(range(3), [c[1] for c in cats], fontsize=9, color=MUTED)
a2.set_ylabel("held-out AAD accuracy, 60 s windows (%)", color=INK)
a2.set_title("LOSO: one layout for all users, decoder per user\n(virtual = physics-predicted "
             "signals)", fontsize=10, color=INK)
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.savefig(FIG / "step4b_placement.png", dpi=150)
print(f"Wrote results/step4b_placement.json, results/step4b_loso.csv, "
      f"figures/step4b_placement.png ({time.time()-t0:.0f}s)")
