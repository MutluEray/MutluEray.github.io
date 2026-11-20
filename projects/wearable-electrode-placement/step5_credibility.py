"""
Step 5 — credibility of the on-cap result: conductivity and co-registration.
Project: physics-guided wearable electrode placement (C7-B / F17).

Everything at sigma_noise = 1 uV (step 4c), 3-7 cm ("short") and 1-3 cm ("mini") pairs.
  A. Conductivity: rebuild the 3-layer BEM with other conductivities (S/m, scalp/skull/brain)
       baseline 0.3 / 0.006 / 0.3; skull 0.0042, 0.0105, 0.015; scalp 0.43.
     For each: (i) on-cap answer key — closed-form leave-pair-out Delta-r for the real scalp
     pairs from step 4a vs measured, Spearman of group means; (ii) population optimum;
     (iii) shift of the optimum vs baseline (max over the two ends, orientation-matched);
     (iv) regret = loss of Delta-r if the variant's optimum is used under the baseline model.
  B. Co-registration: 10 random rigid rotations of the cap electrodes about the head centre,
     ~1 cm rms displacement; jittered scalp lead field by RBF interpolation of the grid lead
     field (~1 % error, step 2). Same metrics (shift, regret under the unjittered model).

Pre-registered (before running): P9 — optimum moves < 1.5 cm under every conductivity variant
and on-cap rank agreement stays >= 0.6. P10 — under 1 cm jitter, median optimum shift
< 1.5 cm and median regret < 5 %.

Closed-form decoders, grid search; no training loop -> no early stopping needed.
Each conductivity variant needs a new BEM solution + forward (~1-3 min each).

Run:
    python3 step5_credibility.py
"""

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from pymatreader import read_mat
from scipy.interpolate import RBFInterpolator
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
SUBJ_DIR = ROOT / "data" / "mne_subjects"
FS_DIR = SUBJ_DIR / "fsaverage"
FS, NL = 20, 9
SNR, NAN_CH_FRAC, SIGMA = 3.0, 0.5, 1.0
ONCAP = 0.025
MINI, SHORT = (0.010, 0.030), (0.030, 0.070)
COND = {"baseline": (0.3, 0.006, 0.3), "skull_0.0042": (0.3, 0.0042, 0.3),
        "skull_0.0105": (0.3, 0.0105, 0.3), "skull_0.015": (0.3, 0.015, 0.3),
        "scalp_0.43": (0.43, 0.006, 0.3)}
N_JIT, JIT_RMS = 10, 0.010
CHUNK, SEED = 8000, 0
mne.set_log_level("WARNING")
t0 = time.time()
rng = np.random.default_rng(SEED)

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
L_grid0 = hm["L_grid"].astype(np.float64)
fid = {k: hm[k] for k in ("nasion", "lpa", "rpa")}
d_near = np.min(np.linalg.norm(P_grid[:, None] - P_sc[None], axis=2), axis=1)
REG = np.where(d_near <= ONCAP)[0]
P_reg = P_grid[REG]
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}
with open(RES / "step4a_oncap_pairs.csv") as f:
    REAL = sorted({r["pair"] for r in csv.DictReader(f)})

# noise autocorrelation (as step 4c)
SOS = butter(2, [1, 9], btype="bandpass", fs=1000, output="sos")
_w = rng.standard_normal(200_000 * 50 + 2000)
_z = sosfiltfilt(SOS, _w)[1000:1000 + 200_000 * 50:50]
_z /= _z.std()
RHO_N = np.array([np.mean(_z[: len(_z) - k] * _z[k:]) for k in range(NL)])
RHO_N /= RHO_N[0]
TOEP = np.abs(np.subtract.outer(np.arange(NL), np.arange(NL)))

# ---------------------------------------------------------------------------
# per-trial data statistics (independent of the head model)
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
        X = np.nan_to_num(dref[:, use] - np.nanmean(dref[:, use], 0, keepdims=True))
        T = len(X)
        ez = zscore(np.asarray(e, float))
        sa, su = ez[:, a - 1], ez[:, 2 - a]
        R = np.stack([X[: T - k].T @ X[k:] / T for k in range(NL)])
        R = 0.5 * (R + R.transpose(0, 2, 1))
        qa = np.stack([X[k:].T @ sa[: T - k] / T for k in range(NL)])
        qu = np.stack([X[k:].T @ su[: T - k] / T for k in range(NL)])
        ST[s].append({"use": use, "ref_i": ref_i, "R": R, "qa": qa, "qu": qu, "T": T})
print(f"data statistics ready ({time.time()-t0:.0f}s)")


def dr_from_A(st, A_list, sigma=SIGMA, n_el=2):
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


def kernel(Lsc, use, ref_i):
    G = Lsc[use] - Lsc[ref_i]
    lam = np.trace(G @ G.T) / len(use) / SNR ** 2
    return G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use)))


# candidate pairs (fixed)
D = np.linalg.norm(P_reg[:, None] - P_reg[None], axis=2)
iu = np.triu_indices(len(REG), 1)
dist = D[iu]
PAIRS = {"mini": (iu[0][(dist >= MINI[0]) & (dist <= MINI[1])],
                  iu[1][(dist >= MINI[0]) & (dist <= MINI[1])]),
         "short": (iu[0][(dist > SHORT[0]) & (dist <= SHORT[1])],
                   iu[1][(dist > SHORT[0]) & (dist <= SHORT[1])])}


def evaluate(Lsc, Lreg):
    """population scores (subjects x pairs) for both classes, plus on-cap answer key rho."""
    proj = {}
    for s in subjects:
        proj[s] = []
        for tr in ST[s]:
            K = kernel(Lsc, tr["use"], tr["ref_i"])
            proj[s].append((Lreg - Lsc[tr["ref_i"]]) @ K)
    scores = {}
    for c, (I, J) in PAIRS.items():
        M = np.empty((len(subjects), len(I)))
        for i, s in enumerate(subjects):
            for c0 in range(0, len(I), CHUNK):
                sl = slice(c0, c0 + CHUNK)
                M[i, sl] = dr_from_A(ST[s], [p[I[sl]] - p[J[sl]] for p in proj[s]])
        scores[c] = M
    # on-cap answer key: leave-pair-out physics vs measured, closed form, no added noise
    meas, pred = {}, {}
    for s in subjects:
        for p in REAL:
            a, b = (SC.index(x) for x in p.split("-"))
            if any(a not in tr["use"] or b not in tr["use"] for tr in ST[s]):
                continue
            Am, Ap = [], []
            for tr in ST[s]:
                v = np.zeros((1, len(tr["use"])))
                v[0, tr["use"].index(a)], v[0, tr["use"].index(b)] = 1.0, -1.0
                Am.append(v)
                keep = [k for k, c in enumerate(tr["use"]) if c not in (a, b)]
                K = kernel(Lsc, [tr["use"][k] for k in keep], tr["ref_i"])
                row = np.zeros((1, len(tr["use"])))
                row[0, keep] = ((Lsc[a] - Lsc[tr["ref_i"]]) - (Lsc[b] - Lsc[tr["ref_i"]])) @ K
                Ap.append(row)
            meas[(s, p)] = dr_from_A(ST[s], Am, sigma=0.0)[0]
            pred[(s, p)] = dr_from_A(ST[s], Ap, sigma=0.0)[0]
    from collections import Counter
    cnt = Counter(p for _, p in meas)
    keep_p = [p for p in REAL if cnt[p] >= len(subjects) - 3]
    gm = [np.mean([meas[(s, p)] for s in subjects if (s, p) in meas]) for p in keep_p]
    gp = [np.mean([pred[(s, p)] for s in subjects if (s, p) in pred]) for p in keep_p]
    return scores, float(spearmanr(gm, gp).correlation), len(keep_p)


def optimum(scores, c):
    m = scores[c].mean(0)
    b = int(np.argmax(m))
    I, J = PAIRS[c]
    return b, P_reg[I[b]], P_reg[J[b]], float(m[b])


def shift(pa, pb, qa, qb):
    return float(min(max(np.linalg.norm(pa - qa), np.linalg.norm(pb - qb)),
                     max(np.linalg.norm(pa - qb), np.linalg.norm(pb - qa))))


# ---------------------------------------------------------------------------
# A. conductivity variants
# ---------------------------------------------------------------------------
src = mne.read_source_spaces(FS_DIR / "bem" / "fsaverage-ico-5-src.fif")
surfs0 = mne.read_bem_surfaces(FS_DIR / "bem" / "fsaverage-5120-5120-5120-bem.fif")
names = SC + [f"G{i:04d}" for i in range(len(REG))]
pos = {**dict(zip(SC, P_sc)), **dict(zip(names[len(SC):], P_reg))}
info = mne.create_info(names, 1000.0, "eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head", **fid))

COND_RES, BASE = {}, None
for name, (s_scalp, s_skull, s_brain) in COND.items():
    surfs = [dict(sf) for sf in surfs0]
    for sf in surfs:
        sf["sigma"] = {4: s_scalp, 3: s_skull, 1: s_brain}[sf["id"]]
    bem = mne.make_bem_solution(surfs)
    fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=bem, eeg=True,
                                    meg=False, mindist=5.0, n_jobs=1)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    L = fwd["sol"]["data"].astype(np.float64)
    Lsc, Lreg = L[: len(SC)], L[len(SC):]
    scores, rho, npairs = evaluate(Lsc, Lreg)
    r = {"rho_oncap_answer_key": rho, "n_pairs_answer_key": npairs}
    for c in ("short", "mini"):
        b, pa, pb, v = optimum(scores, c)
        r[c] = {"dr": v, "pos_a": pa.tolist(), "pos_b": pb.tolist(), "idx": b}
    if name == "baseline":
        BASE = {"scores": scores, **{c: r[c] for c in ("short", "mini")}}
    for c in ("short", "mini"):
        ba, bb = np.array(BASE[c]["pos_a"]), np.array(BASE[c]["pos_b"])
        r[c]["shift_cm"] = 100 * shift(np.array(r[c]["pos_a"]), np.array(r[c]["pos_b"]), ba, bb)
        r[c]["regret"] = float(1 - BASE["scores"][c].mean(0)[r[c]["idx"]] / BASE[c]["dr"])
    COND_RES[name] = r
    print(f"{name:>13s}: answer-key rho {rho:.2f} | short shift {r['short']['shift_cm']:.1f} cm, "
          f"regret {100*r['short']['regret']:.1f} % | mini shift {r['mini']['shift_cm']:.1f} cm, "
          f"regret {100*r['mini']['regret']:.1f} %  ({time.time()-t0:.0f}s)")
    del fwd, L, Lsc, Lreg, scores

# ---------------------------------------------------------------------------
# B. co-registration jitter (baseline conductivities; RBF lead field for moved electrodes)
# ---------------------------------------------------------------------------
def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    return np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0][:3]


C = fit_sphere(P_grid[P_grid[:, 2] > np.percentile(P_grid[:, 2], 20)])
rbf = RBFInterpolator(P_grid, L_grid0, kernel="thin_plate_spline")
L_reg0 = L_grid0[REG]
r_grid = np.linalg.norm(P_grid - C, axis=1)
U_grid = (P_grid - C) / r_grid[:, None]


def on_skin(p):
    u = (p - C) / np.linalg.norm(p - C)
    cosang = U_grid @ u
    nn = np.argsort(-cosang)[:4]
    w = 1.0 / (1 - cosang[nn] + 1e-6)
    return C + u * np.sum(w * r_grid[nn]) / w.sum()


JIT = []
for j in range(N_JIT):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    # angle giving ~JIT_RMS rms displacement for these electrodes
    Rm = lambda ang: Rotation.from_rotvec(axis * ang).apply(P_sc - C) + C
    ang = 0.05
    for _ in range(30):
        rms = np.sqrt(np.mean(np.sum((Rm(ang) - P_sc) ** 2, 1)))
        ang *= JIT_RMS / rms
    Pj = np.array([on_skin(p) for p in Rm(ang)])
    Lsc_j = rbf(Pj)
    scores, rho, _ = evaluate(Lsc_j, L_reg0)
    r = {"rms_displacement_cm": float(100 * np.sqrt(np.mean(np.sum((Pj - P_sc) ** 2, 1)))),
         "rho_oncap_answer_key": rho}
    for c in ("short", "mini"):
        b, pa, pb, v = optimum(scores, c)
        ba, bb = np.array(BASE[c]["pos_a"]), np.array(BASE[c]["pos_b"])
        r[c] = {"shift_cm": 100 * shift(pa, pb, ba, bb),
                "regret": float(1 - BASE["scores"][c].mean(0)[b] / BASE[c]["dr"])}
    JIT.append(r)
    print(f"jitter {j+1:2d}: rms {r['rms_displacement_cm']:.2f} cm, answer-key rho {rho:.2f} | "
          f"short shift {r['short']['shift_cm']:.1f} cm, regret {100*r['short']['regret']:.1f} % | "
          f"mini shift {r['mini']['shift_cm']:.1f} cm, regret {100*r['mini']['regret']:.1f} %  "
          f"({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
var = [k for k in COND if k != "baseline"]
p9 = all(COND_RES[k]["short"]["shift_cm"] < 1.5 for k in var) and \
    all(COND_RES[k]["rho_oncap_answer_key"] >= 0.6 for k in COND)
js = [r["short"]["shift_cm"] for r in JIT]
jr = [r["short"]["regret"] for r in JIT]
p10 = float(np.median(js)) < 1.5 and float(np.median(jr)) < 0.05
out = {"sigma_noise_uV": SIGMA, "conductivity": COND_RES, "jitter": JIT,
       "jitter_median": {"short_shift_cm": float(np.median(js)), "short_regret": float(np.median(jr)),
                         "mini_shift_cm": float(np.median([r["mini"]["shift_cm"] for r in JIT])),
                         "mini_regret": float(np.median([r["mini"]["regret"] for r in JIT])),
                         "rho_answer_key": float(np.median([r["rho_oncap_answer_key"] for r in JIT]))},
       "P9_true": bool(p9), "P10_true": bool(p10)}
print(f"\nP9 (conductivity: short optimum shift < 1.5 cm and answer-key rho >= 0.6): {p9}")
print(f"P10 (1 cm jitter: median shift {np.median(js):.1f} cm < 1.5, median regret "
      f"{100*np.median(jr):.1f} % < 5): {p10}")
(RES / "step5_credibility.json").write_text(json.dumps(out, indent=2))

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
labels = list(COND)
x = np.arange(len(labels))
a1.bar(x - 0.18, [COND_RES[k]["short"]["shift_cm"] for k in labels], 0.36, color=BLUE,
       label="optimum shift, 3–7 cm (cm)")
a1.bar(x + 0.18, [100 * COND_RES[k]["short"]["regret"] for k in labels], 0.36, color=ORANGE,
       label="regret under baseline model (%)")
for k, lab in enumerate(labels):
    a1.text(k, -0.6, f"ρ {COND_RES[lab]['rho_oncap_answer_key']:.2f}", ha="center", fontsize=8,
            color=MUTED)
a1.axhline(1.5, color=MUTED, lw=0.8, ls=":")
a1.set_xticks(x, [l.replace("_", "\n") for l in labels], fontsize=8, color=MUTED)
a1.set_title("Conductivity variants (ρ = on-cap answer key)", fontsize=10, color=INK)
a1.legend(frameon=False, fontsize=8)
a1.set_ylim(-1, max(2.5, a1.get_ylim()[1]))
a2.scatter([r["short"]["shift_cm"] for r in JIT], [100 * r["short"]["regret"] for r in JIT],
           s=40, c=BLUE, edgecolors="white", linewidths=1.5, label="3–7 cm", zorder=3)
a2.scatter([r["mini"]["shift_cm"] for r in JIT], [100 * r["mini"]["regret"] for r in JIT],
           s=40, c=ORANGE, edgecolors="white", linewidths=1.5, label="1–3 cm", zorder=3)
a2.axvline(1.5, color=MUTED, lw=0.8, ls=":")
a2.axhline(5, color=MUTED, lw=0.8, ls=":")
a2.set_xlabel("optimum shift (cm)", color=INK)
a2.set_ylabel("regret under the unjittered model (%)", color=INK)
a2.set_title(f"Co-registration jitter, {N_JIT} rigid rotations (~1 cm rms)", fontsize=10,
             color=INK)
a2.legend(frameon=False, fontsize=8)
for ax in (a1, a2):
    ax.grid(color="#eeeeee", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.savefig(FIG / "step5_credibility.png", dpi=150)
print(f"Wrote results/step5_credibility.json, figures/step5_credibility.png "
      f"({time.time()-t0:.0f}s)")
