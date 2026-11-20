"""
Step 4c — sensor noise, fair comparison, placement tolerance, boundary check.
Project: physics-guided wearable electrode placement (C7-B / F17).

Why: step 4b found mini pairs (1-3 cm) lose nothing vs short pairs (3-7 cm). That is an
artifact: virtual electrodes are noise-free linear combinations of the cap, so a smaller
signal costs nothing. Real miniaturisation losses come from small signals competing with
independent electrode/amplifier noise. Step 4b also compared physics-predicted virtual pairs
with a MEASURED real pair (physics is optimistic: +28 % for P3-C3), confounding the gain.

What it does
  1. Sensor-noise model: independent noise per electrode, band-limited like the data
     (4th-order 1-9 Hz Butterworth, zero-phase, at 1 kHz, downsampled to 20 Hz), in-band rms
     sigma per electrode, sweep sigma = 0, 0.25, 0.5, 1, 2, 4 uV. In the closed form it adds
     2 sigma^2 rho_n(tau) to the bipolar autocovariance and nothing to the envelope covariance.
     Data-based UPPER bound for sigma: rms residual of each scalp channel predicted from the
     others by ridge (held-out trial) — includes brain signal not shared with other channels.
  2. Population optimum per sigma for mini and short pairs; mini loss vs short.
  3. Fair comparison at sigma_ref = 1 uV (nominal, chosen, not sourced): LOSO, windowed AAD
     accuracy for the best real pair re-projected through physics + noise vs virtual pairs
     + noise (same treatment). Measured real-pair accuracy reported as anchor.
  4. Placement tolerance at sigma = 0 and sigma_ref: worst and mean delta-r (fraction of
     the optimum) over pairs whose both ends lie within r of the optimum's ends.
  5. Boundary: distance of the optimum to the edge of the on-cap region; EXPLORATORY,
     UNVALIDATED wider region (<= 4 cm from an electrode) — signal-level only (step 3c).

Pre-registered (before running): P6 as tested (noise-free) is false/untestable. New P6':
mini loss reaches 30 % at some sigma <= 1 uV. P7: short pairs keep >= 90 % of max delta-r
within 1 cm. P8: at sigma_ref, physics-world held-out accuracy gain over the real pair
<= 3 points.

Closed-form decoders, grid search; no training loop -> no early stopping needed.

Run:
    python3 step4c_noise_tolerance.py
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
from scipy.signal import butter, sosfiltfilt
from sklearn.covariance import ledoit_wolf
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, NL, WIN = 20, 9, 60
SNR, NAN_CH_FRAC = 3.0, 0.5
ONCAP, WIDE = 0.025, 0.040
MINI, SHORT = (0.010, 0.030), (0.030, 0.070)
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SIG_REF = 1.0
RADII = [0.005, 0.010, 0.015, 0.020, 0.030]
CHUNK, SEED = 8000, 0
t0 = time.time()
rng = np.random.default_rng(SEED)

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
L_sc = hm["L_scalp"].astype(np.float64)
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
d_near_all = np.min(np.linalg.norm(P_grid[:, None] - P_sc[None], axis=2), axis=1)
WREG = np.where(d_near_all <= WIDE)[0]                    # superset; on-cap is a subset
P_w = P_grid[WREG]
L_w = hm["L_grid"][WREG].astype(np.float64)
dn_w = d_near_all[WREG]
ONC = np.where(dn_w <= ONCAP)[0]                          # indices into WREG
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}
print(f"wide region {len(WREG)} points, on-cap {len(ONC)} ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# band-limited noise autocorrelation (matches the preprocessing filter)
# ---------------------------------------------------------------------------
SOS = butter(2, [1, 9], btype="bandpass", fs=1000, output="sos")   # 4th-order band-pass


def band_noise(n20):
    w = rng.standard_normal(n20 * 50 + 2000)
    x = sosfiltfilt(SOS, w)[1000:1000 + n20 * 50:50]
    return x / x.std()


_z = band_noise(200_000)
RHO_N = np.array([np.mean(_z[: len(_z) - k] * _z[k:]) for k in range(NL)])
RHO_N /= RHO_N[0]
print("noise autocorrelation (lags 0-8):", np.round(RHO_N, 3))

# ---------------------------------------------------------------------------
# projections + per-trial statistics (as step 4b)
# ---------------------------------------------------------------------------
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    return np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0][:3]


SPH_C = fit_sphere(P_grid[P_grid[:, 2] > np.percentile(P_grid[:, 2], 20)])
_cache = {}


def projections(use, ref_i):
    key = (tuple(use), ref_i)
    if key not in _cache:
        G = L_sc[use] - L_sc[ref_i]
        lam = np.trace(G @ G.T) / len(use) / SNR ** 2
        K = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use)))
        _cache[key] = {"phys": (L_w - L_sc[ref_i]) @ K, "phys_sc": (L_sc - L_sc[ref_i]) @ K}
    return _cache[key]


def zscore(x):
    mu, sd = np.nanmean(x, 0, keepdims=True), np.nanstd(x, 0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
ST, RESID = {}, []
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
        X = np.nan_to_num(X - np.nanmean(X, 0, keepdims=True))
        T = len(X)
        ez = zscore(np.asarray(e, float))
        sa, su = ez[:, a - 1], ez[:, 2 - a]
        R = np.stack([X[: T - k].T @ X[k:] / T for k in range(NL)])
        R = 0.5 * (R + R.transpose(0, 2, 1))
        qa = np.stack([X[k:].T @ sa[: T - k] / T for k in range(NL)])
        qu = np.stack([X[k:].T @ su[: T - k] / T for k in range(NL)])
        ST[s].append({"use": use, "R": R, "qa": qa, "qu": qu, "T": T, "X": X,
                      "sa": sa, "su": su, **projections(use, ref_i)})
    # data-based upper bound on independent noise: ridge residual per channel (trial 6 held out)
    common = [c for c in ST[s][0]["use"] if all(c in tr["use"] for tr in ST[s])]
    sel = lambda tr: tr["X"][:, [tr["use"].index(c) for c in common]]
    tr_X = np.vstack([sel(tr) for tr in ST[s][:-1]])
    te_X = sel(ST[s][-1])
    for j in range(len(common)):
        cols = [c for c in range(len(common)) if c != j]
        rg = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(tr_X[:, cols], tr_X[:, j])
        RESID.append(float(np.std(te_X[:, j] - rg.predict(te_X[:, cols]))))
    print(f"subject {s:2d} ready ({time.time()-t0:.0f}s)")
sigma_upper = float(np.median(RESID))
print(f"data-based UPPER bound on per-electrode independent noise (median ridge residual rms): "
      f"{sigma_upper:.2f} uV")

TOEP = np.abs(np.subtract.outer(np.arange(NL), np.arange(NL)))


def dr_from_A(st, A_list, sigma=0.0, n_el=2):
    nt, B = len(st), A_list[0].shape[0]
    g = np.empty((nt, B, NL))
    ca, cu = np.empty((nt, B, NL)), np.empty((nt, B, NL))
    noise = n_el * sigma ** 2 * RHO_N
    for t, (tr, A) in enumerate(zip(st, A_list)):
        AR = np.matmul(A[None], tr["R"])
        gg = np.einsum("kbn,bn->bk", AR, A) + noise[None, :]
        v0 = np.maximum(gg[:, 0], 1e-30)
        g[t] = gg / v0[:, None]
        ca[t] = (A @ tr["qa"].T) / np.sqrt(v0)[:, None]
        cu[t] = (A @ tr["qu"].T) / np.sqrt(v0)[:, None]
    S = g[:, :, TOEP]
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


def score(s, I, J, sigma):
    st = ST[s]
    out = np.empty(len(I))
    for c0 in range(0, len(I), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        out[sl] = dr_from_A(st, [tr["phys"][I[sl]] - tr["phys"][J[sl]] for tr in st], sigma)
    return out


def pairs_in(idx):
    P = P_w[idx]
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    iu = np.triu_indices(len(idx), 1)
    dist = D[iu]
    out = {}
    for name, (lo, hi) in (("mini", MINI), ("short", SHORT)):
        m = (dist >= lo) & (dist <= hi) if name == "mini" else (dist > lo) & (dist <= hi)
        out[name] = (idx[iu[0][m]], idx[iu[1][m]])
    return out


PAIRS = pairs_in(ONC)
print({k: len(v[0]) for k, v in PAIRS.items()}, "on-cap candidate pairs")

# ---------------------------------------------------------------------------
# 1-2. noise sweep: population optimum and mini loss
# ---------------------------------------------------------------------------
SWEEP, SCORES = {}, {}
for sig in SIGMAS:
    row = {}
    for c, (I, J) in PAIRS.items():
        M = np.stack([score(s, I, J, sig) for s in subjects])
        SCORES[(sig, c)] = M
        b = int(np.argmax(M.mean(0)))
        row[c] = {"dr": float(M.mean(0)[b]), "i": int(I[b]), "j": int(J[b]),
                  "dist_cm": float(100 * np.linalg.norm(P_w[I[b]] - P_w[J[b]]))}
    row["mini_loss"] = 1 - row["mini"]["dr"] / row["short"]["dr"]
    SWEEP[sig] = row
    print(f"sigma {sig:4.2f} uV: short {row['short']['dr']:.4f} ({row['short']['dist_cm']:.1f} cm), "
          f"mini {row['mini']['dr']:.4f} ({row['mini']['dist_cm']:.1f} cm), "
          f"mini loss {100*row['mini_loss']:5.1f} %  ({time.time()-t0:.0f}s)")
cross = next((s for s in SIGMAS if SWEEP[s]["mini_loss"] >= 0.30), None)

# ---------------------------------------------------------------------------
# 3. fair LOSO comparison at sigma_ref
# ---------------------------------------------------------------------------
with open(RES / "step4a_oncap_pairs.csv") as f:
    REAL = sorted({r["pair"] for r in csv.DictReader(f)})
RP = [p for p in REAL if all(all(SC.index(x) in tr["use"] for x in p.split("-") for tr in ST[s])
                             for s in subjects)]


def real_A(st, p, kind):
    a, b = (SC.index(x) for x in p.split("-"))
    if kind == "meas":
        out = []
        for tr in st:
            v = np.zeros((1, len(tr["use"])))
            v[0, tr["use"].index(a)], v[0, tr["use"].index(b)] = 1.0, -1.0
            out.append(v)
        return out
    return [tr["phys_sc"][[a]] - tr["phys_sc"][[b]] for tr in st]


RM = np.array([[dr_from_A(ST[s], real_A(ST[s], p, "meas"))[0] for p in RP] for s in subjects])
RPn = np.array([[dr_from_A(ST[s], real_A(ST[s], p, "phys"), SIG_REF)[0] for p in RP]
                for s in subjects])


def lagged(x):
    o = np.zeros((len(x), NL))
    for k in range(NL):
        o[: len(x) - k, k] = x[k:]
    return o


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def aad_acc(sig, st):
    X = [lagged(zscore(x[:, None])[:, 0]) for x in sig]
    hits = []
    for k in range(len(X)):
        trn = [j for j in range(len(X)) if j != k]
        Xtr = np.vstack([X[j] for j in trn])
        y = np.concatenate([st[j]["sa"] for j in trn])
        cov, _ = ledoit_wolf(Xtr, assume_centered=True)
        w = np.linalg.solve(cov, Xtr.T @ y / len(y))
        rec = X[k] @ w
        L = WIN * FS
        for s0 in range(0, len(rec) - L + 1, L):
            sl = slice(s0, s0 + L)
            hits.append(corr(rec[sl], st[k]["sa"][sl]) > corr(rec[sl], st[k]["su"][sl]))
    return float(np.mean(hits))


def noisy(sig_trials, sigma):
    return [x + sigma * (band_noise(len(x)) - band_noise(len(x))) for x in sig_trials]


fair = []
for h, s in enumerate(subjects):
    oth = [k for k in range(len(subjects)) if k != h]
    st = ST[s]
    row = {"subject": s}
    for c in ("short", "mini"):
        M = SCORES[(SIG_REF, c)]
        b = int(np.argmax(M[oth].mean(0)))
        I, J = PAIRS[c]
        sig = [tr["X"] @ (tr["phys"][I[b]] - tr["phys"][J[b]]) for tr in st]
        row[f"virtual_{c}_acc"] = aad_acc(noisy(sig, SIG_REF), st)
    bj = int(np.argmax(RPn[oth].mean(0)))
    p = RP[bj]
    a, b = (SC.index(x) for x in p.split("-"))
    sig_p = [tr["X"] @ (tr["phys_sc"][a] - tr["phys_sc"][b]) for tr in st]
    row["real_pair"] = p
    row["real_physics_noisy_acc"] = aad_acc(noisy(sig_p, SIG_REF), st)
    bm = int(np.argmax(RM[oth].mean(0)))
    pm = RP[bm]
    am, bm_ = (SC.index(x) for x in pm.split("-"))
    row["real_measured_pair"] = pm
    row["real_measured_acc"] = aad_acc([tr["X"][:, tr["use"].index(am)] -
                                        tr["X"][:, tr["use"].index(bm_)] for tr in st], st)
    fair.append(row)
    print(f"LOSO {s:2d}: real {p} physics+noise {100*row['real_physics_noisy_acc']:.1f} % | "
          f"measured ({pm}) {100*row['real_measured_acc']:.1f} % | virtual short "
          f"{100*row['virtual_short_acc']:.1f} % | mini {100*row['virtual_mini_acc']:.1f} %")
med = lambda k: float(np.median([r[k] for r in fair]))
gain_fair = 100 * (med("virtual_short_acc") - med("real_physics_noisy_acc"))

# ---------------------------------------------------------------------------
# 4. placement tolerance
# ---------------------------------------------------------------------------
def tolerance(sig, c):
    M = SCORES[(sig, c)].mean(0)
    I, J = PAIRS[c]
    b = int(np.argmax(M))
    pa, pb = P_w[I[b]], P_w[J[b]]
    da = np.maximum(np.linalg.norm(P_w[I] - pa, axis=1), np.linalg.norm(P_w[J] - pb, axis=1))
    db = np.maximum(np.linalg.norm(P_w[I] - pb, axis=1), np.linalg.norm(P_w[J] - pa, axis=1))
    disp = np.minimum(da, db)
    out = {}
    for r in RADII:
        m = disp <= r
        out[f"{100*r:.1f}cm"] = {"worst_frac": float(M[m].min() / M[b]),
                                 "mean_frac": float(M[m].mean() / M[b]), "n": int(m.sum())}
    return out


TOL = {f"sigma={sig}/{c}": tolerance(sig, c) for sig in (0.0, SIG_REF) for c in ("short", "mini")}
p7 = TOL["sigma=0.0/short"]["1.0cm"]["worst_frac"] >= 0.90

# ---------------------------------------------------------------------------
# 5. boundary + exploratory wider region (UNVALIDATED for decoding)
# ---------------------------------------------------------------------------
opt = SWEEP[SIG_REF]["short"]
edge = {"opt_a_dist_to_nearest_electrode_cm": float(100 * dn_w[opt["i"]]),
        "opt_b_dist_to_nearest_electrode_cm": float(100 * dn_w[opt["j"]]),
        "oncap_limit_cm": 100 * ONCAP}
WP = pairs_in(np.arange(len(WREG)))["short"]
Mw = np.stack([score(s, WP[0], WP[1], SIG_REF) for s in subjects]).mean(0)
bw = int(np.argmax(Mw))
edge["wide_region_best_dr"] = float(Mw[bw])
edge["oncap_best_dr"] = opt["dr"]
edge["wide_gain_relative"] = float(Mw[bw] / opt["dr"] - 1)
edge["wide_best_dist_to_nearest_electrode_cm"] = [float(100 * dn_w[WP[0][bw]]),
                                                   float(100 * dn_w[WP[1][bw]])]

def near(p, k=2):
    dd = np.linalg.norm(P_sc - p, axis=1)
    o = np.argsort(dd)[:k]
    return [f"{SC[i]} ({100*dd[i]:.1f} cm)" for i in o]


edge["wide_best_near"] = [near(P_w[WP[0][bw]]), near(P_w[WP[1][bw]])]
print(f"\noptimum ends sit {edge['opt_a_dist_to_nearest_electrode_cm']:.1f} / "
      f"{edge['opt_b_dist_to_nearest_electrode_cm']:.1f} cm from the nearest electrode "
      f"(on-cap limit {100*ONCAP:.1f} cm)")
print(f"[EXPLORATORY, unvalidated] region <= {100*WIDE:.0f} cm: best {Mw[bw]:.4f} vs on-cap "
      f"{opt['dr']:.4f} ({100*edge['wide_gain_relative']:+.1f} %), ends near "
      f"{edge['wide_best_near']}")

out = {
    "rho_noise": RHO_N.tolist(), "sigma_upper_bound_uV": sigma_upper,
    "sweep": {str(k): v for k, v in SWEEP.items()},
    "P6_prime": {"sigma_where_mini_loss_ge_30pct": cross, "true": bool(cross is not None
                                                                       and cross <= 1.0)},
    "fair_loso_median_acc": {k: med(k) for k in ("real_physics_noisy_acc", "real_measured_acc",
                                                 "virtual_short_acc", "virtual_mini_acc")},
    "P8": {"gain_points_physics_world": gain_fair, "true": bool(gain_fair <= 3)},
    "tolerance": TOL, "P7_true": bool(p7), "boundary": edge,
}
print(f"\nP6': mini loss >= 30 % first at sigma = {cross} uV -> {out['P6_prime']['true']}")
print(f"P7: short pairs keep {100*TOL['sigma=0.0/short']['1.0cm']['worst_frac']:.1f} % of max "
      f"within 1 cm (>= 90 %?) -> {p7}")
print(f"P8: physics-world gain {gain_fair:+.1f} points (<= 3?) -> {out['P8']['true']}")
print(f"held-out median accuracy at sigma = {SIG_REF} uV: real (physics+noise) "
      f"{100*med('real_physics_noisy_acc'):.1f} %, real measured {100*med('real_measured_acc'):.1f} %, "
      f"virtual short {100*med('virtual_short_acc'):.1f} %, virtual mini "
      f"{100*med('virtual_mini_acc'):.1f} %")
(RES / "step4c_noise_tolerance.json").write_text(json.dumps(out, indent=2, default=str))
with open(RES / "step4c_fair_loso.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fair[0].keys()))
    w.writeheader()
    w.writerows(fair)

# ---------------------------------------------------------------------------
# figure: (a) noise sweep; (b) tolerance
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
xs = [max(s, 0.1) for s in SIGMAS]
a1.plot(xs, [SWEEP[s]["short"]["dr"] for s in SIGMAS], "-o", color=BLUE, lw=2, ms=6,
        markeredgecolor="white", label="best 3–7 cm pair")
a1.plot(xs, [SWEEP[s]["mini"]["dr"] for s in SIGMAS], "-o", color=ORANGE, lw=2, ms=6,
        markeredgecolor="white", label="best 1–3 cm pair")
a1.axvline(sigma_upper, color=MUTED, lw=0.8, ls=":")
a1.text(sigma_upper, a1.get_ylim()[1], " data upper bound", fontsize=8, color=MUTED, va="top")
a1.set_xscale("log")
a1.set_xticks(xs, ["0"] + [f"{s:g}" for s in SIGMAS[1:]])
a1.set_xlabel("independent noise per electrode, in-band rms (µV)", color=INK)
a1.set_ylabel("population mean Δr (physics)", color=INK)
a1.set_title("Miniaturisation penalty appears only with sensor noise", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8)
a1.grid(color="#eeeeee", lw=0.6)
rr = [100 * r for r in RADII]
for c, col in (("short", BLUE), ("mini", ORANGE)):
    t = TOL[f"sigma={SIG_REF}/{c}"]
    a2.plot(rr, [100 * t[f"{r:.1f}cm"]["worst_frac"] for r in rr], "-o", color=col, lw=2, ms=6,
            markeredgecolor="white", label=f"{'3–7' if c == 'short' else '1–3'} cm, worst case")
    a2.plot(rr, [100 * t[f"{r:.1f}cm"]["mean_frac"] for r in rr], "--", color=col, lw=1.2,
            label=f"{'3–7' if c == 'short' else '1–3'} cm, mean")
a2.axhline(90, color=MUTED, lw=0.8, ls=":")
a2.set_xlabel("misplacement of each electrode (cm)", color=INK)
a2.set_ylabel(f"Δr kept, % of optimum (σ = {SIG_REF} µV)", color=INK)
a2.set_title("Placement tolerance around the population optimum", fontsize=10, color=INK)
a2.legend(frameon=False, fontsize=8)
a2.grid(color="#eeeeee", lw=0.6)
for ax in (a1, a2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.savefig(FIG / "step4c_noise_tolerance.png", dpi=150)
print(f"Wrote results/step4c_noise_tolerance.json, results/step4c_fair_loso.csv, "
      f"figures/step4c_noise_tolerance.png ({time.time()-t0:.0f}s)")
