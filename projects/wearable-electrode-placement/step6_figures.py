"""
Step 6 — site figures for the write-up (reads saved results; recomputes one map).
Project: physics-guided wearable electrode placement (C7-B / F17).

  figures/site_validity.png  : (a) around-ear signals predicted from scalp only, by method;
                               (b) decoding-rank agreement off-cap vs on-cap with ceilings
  figures/site_placement.png : (a) top-view map of the population score of 3-7 cm pairs at
                               1 uV sensor noise, optimum and best real pair;
                               (b) best 1-3 cm vs 3-7 cm pair against sensor noise

No training -> no early stopping needed. ~1 min.

Run:
    python3 step6_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, NL, SNR, NAN_CH_FRAC, SIGMA = 20, 9, 3.0, 0.5, 1.0
ONCAP, SHORT = 0.025, (0.030, 0.070)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#eeeeee"
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c9c9c9"

j3 = json.loads((RES / "step3_validity.json").read_text())
j3b = json.loads((RES / "step3b_diagnostics.json").read_text())
j3c = json.loads((RES / "step3c_calibration.json").read_text())
j4a = json.loads((RES / "step4a_oncap_validity.json").read_text())
j4c = json.loads((RES / "step4c_noise_tolerance.json").read_text())


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------------------
# figure 1: validity
# ---------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.35]})
sig = j3["signal"]["no_mastoids/earCAR"]
meth = [("physics", "Physics\n(head model)", BLUE), ("spline", "Spline\ninterpolation", ORANGE),
        ("nearest", "Nearest\nelectrode", AQUA), ("ceiling", "Data-driven\nceiling", GREY)]
for k, (key, lab, col) in enumerate(meth):
    a1.bar(k, sig[key], color=col, width=0.6)
    a1.text(k, sig[key] + 0.008, f"{sig[key]:.2f}", ha="center", color=INK, fontsize=10)
a1.set_xticks(range(4), [m[1] for m in meth], fontsize=9, color=MUTED)
a1.set_ylabel("median r, predicted vs measured", color=INK)
a1.set_title("Around-ear EEG predicted from scalp EEG only\n(19 cEEGrid channels, mastoids "
             "withheld)", fontsize=10, color=INK)
a1.set_ylim(0, max(sig.values()) * 1.25)
a1.grid(axis="y", color=GRID, lw=0.6)
clean(a1)

off_t = j3c["decoding_rank"]["template_rho_group"]
off_c = j3c["decoding_rank"]["calibrated_rho_group"]
off_r = j3b["ridge_ceiling_decoding"]["rho_group"]
off_n = j3b["reliability"]["noise_ceiling_rho"]
on_p = j4a["rank_agreement"]["physics_rho_group"]
on_s = j4a["rank_agreement"]["spline_rho_group"]
on_n = j4a["noise_ceiling"]["ceiling_rho_approx"]
bars = [("template", off_t, "#f2b49a"), ("calibrated\n(LOSO)", off_c, ORANGE),
        ("ridge\nceiling", off_r, AQUA), ("noise\nceiling", off_n, GREY), (None, 0, None),
        ("physics", on_p, BLUE), ("spline", on_s, YELLOW), ("noise\nceiling", on_n, GREY)]
for k, (lab, v, col) in enumerate(bars):
    if lab is None:
        continue
    a2.bar(k, v, color=col, width=0.62)
    a2.text(k, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=9.5)
a2.axhline(0.35, color=MUTED, lw=0.8, ls="--")
a2.text(7.45, 0.36, "pre-registered go ≥ 0.35", ha="right", va="bottom", fontsize=8, color=MUTED)
a2.set_xticks([k for k, b in enumerate(bars) if b[0]], [b[0] for b in bars if b[0]],
              fontsize=8.5, color=MUTED)
a2.text(1.5, -0.2, "Around the ear (81 pairs, 13 subjects)", ha="center", fontsize=9,
        color=INK, transform=a2.get_xaxis_transform())
a2.text(6.0, -0.2, "On the cap (55 pairs, 15 subjects)", ha="center", fontsize=9, color=INK,
        transform=a2.get_xaxis_transform())
a2.set_ylim(0, 1)
a2.set_ylabel("Spearman ρ, predicted vs measured\ndecoding ranking of electrode pairs",
              color=INK)
a2.set_title("Can the model rank electrode pairs for attention decoding?", fontsize=10,
             color=INK)
a2.grid(axis="y", color=GRID, lw=0.6)
clean(a2)
fig.savefig(FIG / "site_validity.png", dpi=180)
print("wrote figures/site_validity.png")

# ---------------------------------------------------------------------------
# figure 2: placement map (recomputed at 1 uV) + noise sweep
# ---------------------------------------------------------------------------
hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
L_sc = hm["L_scalp"].astype(np.float64)
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
dn = np.min(np.linalg.norm(P_grid[:, None] - P_sc[None], axis=2), axis=1)
REG = np.where(dn <= ONCAP)[0]
P_reg, L_reg = P_grid[REG], hm["L_grid"][REG].astype(np.float64)
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}

rng = np.random.default_rng(0)
SOS = butter(2, [1, 9], btype="bandpass", fs=1000, output="sos")
z = sosfiltfilt(SOS, rng.standard_normal(200_000 * 50 + 2000))[1000:1000 + 200_000 * 50:50]
z /= z.std()
RHO = np.array([np.mean(z[: len(z) - k] * z[k:]) for k in range(NL)])
RHO /= RHO[0]
TOEP = np.abs(np.subtract.outer(np.arange(NL), np.arange(NL)))


def zscore(x):
    mu, sd = np.nanmean(x, 0, keepdims=True), np.nanstd(x, 0, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0)


def dr_from_A(st, A_list):
    nt, B = len(st), A_list[0].shape[0]
    g, ca, cu = (np.empty((nt, B, NL)) for _ in range(3))
    for t, (tr, A) in enumerate(zip(st, A_list)):
        gg = np.einsum("kbn,bn->bk", np.matmul(A[None], tr["R"]), A) + 2 * SIGMA ** 2 * RHO
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
        Ss += 1e-6 * Ss[:, :1, :1] * np.eye(NL)
        w = np.linalg.solve(Ss, np.tensordot(Tw[trn], ca[trn], axes=(0, 0))[..., None])[..., 0]
        var = np.maximum(np.einsum("bi,bij,bj->b", w, S[k], w), 1e-30)
        out += (np.einsum("bi,bi->b", w, ca[k]) - np.einsum("bi,bi->b", w, cu[k])) / np.sqrt(var)
    return out / nt


D = np.linalg.norm(P_reg[:, None] - P_reg[None], axis=2)
iu = np.triu_indices(len(REG), 1)
m = (D[iu] > SHORT[0]) & (D[iu] <= SHORT[1])
I, J = iu[0][m], iu[1][m]
subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
total = np.zeros(len(I))
for s in subjects:
    d = read_mat(str(PRE / f"dataSubject{s}.mat"))
    ch = list(d["channels"])
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    ref_i = SC.index(REF[s])
    st, proj = [], []
    for t, e, a in zip(d["eegTrials"], d["envelopes"], att):
        t = np.asarray(t, float)[:, [ch.index(n) for n in SC]]
        dref = t - t[:, [ref_i]]
        use = [i for i in range(len(SC)) if i != ref_i and np.isnan(dref[:, i]).mean() <= NAN_CH_FRAC]
        X = np.nan_to_num(dref[:, use] - np.nanmean(dref[:, use], 0, keepdims=True))
        T = len(X)
        ez = zscore(np.asarray(e, float))
        sa, su = ez[:, a - 1], ez[:, 2 - a]
        R = np.stack([X[: T - k].T @ X[k:] / T for k in range(NL)])
        st.append({"R": 0.5 * (R + R.transpose(0, 2, 1)), "T": T,
                   "qa": np.stack([X[k:].T @ sa[: T - k] / T for k in range(NL)]),
                   "qu": np.stack([X[k:].T @ su[: T - k] / T for k in range(NL)])})
        G = L_sc[use] - L_sc[ref_i]
        lam = np.trace(G @ G.T) / len(use) / SNR ** 2
        proj.append((L_reg - L_sc[ref_i]) @ G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use))))
    for c0 in range(0, len(I), 8000):
        sl = slice(c0, c0 + 8000)
        total[sl] += dr_from_A(st, [p[I[sl]] - p[J[sl]] for p in proj])
    print(f"subject {s:2d} scored")
pop = total / len(subjects)
best = int(np.argmax(pop))
touch = np.full(len(REG), -np.inf)
np.maximum.at(touch, I, pop)
np.maximum.at(touch, J, pop)

fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.1, 1]})
ok = np.isfinite(touch)
hsc = b1.scatter(P_reg[ok, 0] * 100, P_reg[ok, 1] * 100, c=touch[ok], cmap="Greys", s=13)
cb = fig.colorbar(hsc, ax=b1, shrink=0.75)
cb.set_label("best mean Δr of a 3–7 cm pair through this point", color=MUTED, fontsize=8)
cb.ax.tick_params(colors=MUTED, labelsize=8)
b1.scatter(P_sc[:, 0] * 100, P_sc[:, 1] * 100, s=12, c=MUTED, zorder=3)
for n, p in zip(SC, P_sc):
    b1.annotate(n, (p[0] * 100, p[1] * 100), fontsize=6.5, color=MUTED, xytext=(2, 2),
                textcoords="offset points")
pa, pb = P_reg[I[best]], P_reg[J[best]]
b1.plot([pa[0] * 100, pb[0] * 100], [pa[1] * 100, pb[1] * 100], "-o", color=BLUE, lw=2.5,
        ms=8, markeredgecolor="white", zorder=4, label="model-placed pair (population optimum)")
ra, rb = P_sc[SC.index("P3")], P_sc[SC.index("C3")]
b1.plot([ra[0] * 100, rb[0] * 100], [ra[1] * 100, rb[1] * 100], "-o", color=AQUA, lw=2.5, ms=8,
        markeredgecolor="white", zorder=4, label="best real 10-20 pair (P3–C3)")
b1.set_aspect("equal")
b1.set_xlabel("cm (right →)", color=INK)
b1.set_ylabel("cm (front →)", color=INK)
b1.set_title("Where should a two-electrode wearable sit? (top view, 1 µV sensor noise)",
             fontsize=10, color=INK)
b1.legend(frameon=False, fontsize=8, loc="lower left")
clean(b1)

sw = j4c["sweep"]
sig_list = sorted(float(k) for k in sw)
xs = [max(s, 0.1) for s in sig_list]
b2.plot(xs, [sw[str(s) if str(s) in sw else f"{s}"]["short"]["dr"] for s in sig_list], "-o",
        color=BLUE, lw=2, ms=6, markeredgecolor="white", label="best 3–7 cm pair")
b2.plot(xs, [sw[str(s) if str(s) in sw else f"{s}"]["mini"]["dr"] for s in sig_list], "-o",
        color=ORANGE, lw=2, ms=6, markeredgecolor="white", label="best 1–3 cm pair")
ub = j4c["sigma_upper_bound_uV"]
b2.axvline(ub, color=MUTED, lw=0.8, ls=":")
b2.text(ub * 1.04, b2.get_ylim()[1], f"data upper bound\n{ub:.2f} µV", fontsize=8, color=MUTED,
        va="top")
b2.set_xscale("log")
b2.set_xticks(xs, ["0"] + [f"{s:g}" for s in sig_list[1:]])
b2.minorticks_off()
b2.set_xlabel("independent noise per electrode (µV, in-band rms)", color=INK)
b2.set_ylabel("population mean Δr (attended − unattended r)", color=INK)
b2.set_title("Smaller devices pay for electrode noise", fontsize=10, color=INK)
b2.legend(frameon=False, fontsize=8)
b2.grid(color=GRID, lw=0.6)
clean(b2)
fig.savefig(FIG / "site_placement.png", dpi=180)
print("wrote figures/site_placement.png")
