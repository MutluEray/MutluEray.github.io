"""
Step 7 — export the interactive placement explorer (figures/explorer.html).
Project: physics-guided wearable electrode placement (C7-B / F17).

Scores every on-cap candidate pair (1-3 cm and 3-7 cm) at six electrode-noise levels with
the closed-form decoder (same engine as steps 4b-5), averages over the 15 subjects, and
embeds only these AGGREGATE scores (no EEG, no per-subject values) into
explorer_template.html. Also scores the real 10-20 pairs present in every subject.

Output: figures/explorer.html (single self-contained file, ~2.5 MB). Open it in a browser
to check, then copy to /projects/wearable-electrode-placement/ on the site.
No training -> no early stopping needed. ~3-4 min.

Run:
    python3 step7_export_explorer.py
"""

import base64
import csv
import json
import time
from pathlib import Path

import numpy as np
from pymatreader import read_mat
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "data" / "preprocessed" / "preprocessedData"
RES, FIG = ROOT / "results", ROOT / "figures"
FS, NL, SNR, NAN_CH_FRAC = 20, 9, 3.0, 0.5
ONCAP, MINI, SHORT = 0.025, (0.010, 0.030), (0.030, 0.070)
SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
OFFSET, SCALE = 0.03, 2e-6                       # uint16 code = (dr + OFFSET) / SCALE
CHUNK = 8000
t0 = time.time()

hm = np.load(ROOT / "data" / "headmodel" / "headmodel.npz")
SC = list(hm["names_scalp"])
L_sc = hm["L_scalp"].astype(np.float64)
P_sc, P_grid = hm["pos_scalp"], hm["pos_grid"]
dn = np.min(np.linalg.norm(P_grid[:, None] - P_sc[None], axis=2), axis=1)
REG = np.where(dn <= ONCAP)[0]
P_reg, L_reg = P_grid[REG], hm["L_grid"][REG].astype(np.float64)
with open(RES / "step3_shared_ref.csv") as f:
    REF = {int(float(r["subject"])): r["shared_ref_used"] for r in csv.DictReader(f)}
with open(RES / "step4a_oncap_pairs.csv") as f:
    REAL = sorted({r["pair"] for r in csv.DictReader(f)})

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


def dr_from_A(st, A_list, sigma):
    nt, B = len(st), A_list[0].shape[0]
    g, ca, cu = (np.empty((nt, B, NL)) for _ in range(3))
    for t, (tr, A) in enumerate(zip(st, A_list)):
        gg = np.einsum("kbn,bn->bk", np.matmul(A[None], tr["R"]), A) + 2 * sigma ** 2 * RHO
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


# candidate pairs: mini first, then short
D = np.linalg.norm(P_reg[:, None] - P_reg[None], axis=2)
iu = np.triu_indices(len(REG), 1)
dist = D[iu]
m_mini = (dist >= MINI[0]) & (dist <= MINI[1])
m_short = (dist > SHORT[0]) & (dist <= SHORT[1])
I = np.r_[iu[0][m_mini], iu[0][m_short]]
J = np.r_[iu[1][m_mini], iu[1][m_short]]
n_mini = int(m_mini.sum())
print(f"{len(REG)} points, {n_mini} mini + {int(m_short.sum())} short pairs")

subjects = sorted(int(p.stem.replace("dataSubject", "")) for p in PRE.glob("dataSubject*.mat"))
TOT = np.zeros((len(SIGMAS), len(I)))
REAL_OK = {p: True for p in REAL}
REAL_TOT = {p: np.zeros(len(SIGMAS)) for p in REAL}
for s in subjects:
    d = read_mat(str(PRE / f"dataSubject{s}.mat"))
    ch = list(d["channels"])
    att = np.asarray(d["attSpeaker"]).astype(int).ravel()
    ref_i = SC.index(REF[s])
    st, proj, proj_sc, uses = [], [], [], []
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
        K = G.T @ np.linalg.inv(G @ G.T + lam * np.eye(len(use)))
        proj.append((L_reg - L_sc[ref_i]) @ K)
        proj_sc.append((L_sc - L_sc[ref_i]) @ K)
        uses.append(use)
    for si, sig in enumerate(SIGMAS):
        for c0 in range(0, len(I), CHUNK):
            sl = slice(c0, c0 + CHUNK)
            TOT[si, sl] += dr_from_A(st, [p[I[sl]] - p[J[sl]] for p in proj], sig)
        for p in REAL:
            a, b = (SC.index(x) for x in p.split("-"))
            if any(a not in u or b not in u for u in uses):
                REAL_OK[p] = False
                continue
            REAL_TOT[p][si] += dr_from_A(st, [q[[a]] - q[[b]] for q in proj_sc], sig)[0]
    print(f"subject {s:2d} scored ({time.time()-t0:.0f}s)")

POP = TOT / len(subjects)
codes = np.clip(np.round((POP + OFFSET) / SCALE), 0, 65535).astype("<u2")
b64 = lambda arr: base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()

near_idx = np.argmin(np.linalg.norm(P_reg[:, None] - P_sc[None], axis=2), axis=1)
near_d = np.min(np.linalg.norm(P_reg[:, None] - P_sc[None], axis=2), axis=1)
points = [[round(float(p[0] * 1000), 1), round(float(p[1] * 1000), 1), int(k), round(float(dd * 1000), 1)]
          for p, k, dd in zip(P_reg, near_idx, near_d)]
electrodes = [[n, round(float(p[0] * 1000), 1), round(float(p[1] * 1000), 1)] for n, p in zip(SC, P_sc)]
real = []
for p in REAL:
    if not REAL_OK[p]:
        continue
    a, b = (SC.index(x) for x in p.split("-"))
    real.append({"name": p.replace("-", "–"),
                 "a": [round(float(P_sc[a][0] * 1000), 1), round(float(P_sc[a][1] * 1000), 1)],
                 "b": [round(float(P_sc[b][0] * 1000), 1), round(float(P_sc[b][1] * 1000), 1)],
                 "dr": [round(float(v / len(subjects)), 6) for v in REAL_TOT[p]]})
data = {"points": points, "electrodes": electrodes, "I": b64(I.astype("<u2")),
        "J": b64(J.astype("<u2")), "n_mini": n_mini, "sigmas": SIGMAS,
        "scores": [b64(codes[i]) for i in range(len(SIGMAS))], "offset": OFFSET, "scale": SCALE,
        "real": real}

tpl = (ROOT / "explorer_template.html").read_text()
assert "/*__DATA__*/null" in tpl, "placeholder missing in explorer_template.html"
html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
out = FIG / "explorer.html"
out.write_text(html)

# sanity: best pairs per sigma must match steps 4b/4c
for si, sig in enumerate(SIGMAS):
    for name, sl in (("mini", slice(0, n_mini)), ("short", slice(n_mini, len(I)))):
        k = int(np.argmax(POP[si, sl])) + sl.start
        print(f"sigma {sig:4.2f} {name:>5s}: best dr {POP[si, k]:.4f} "
              f"({100*np.linalg.norm(P_reg[I[k]] - P_reg[J[k]]):.1f} cm)")
best_real = max(real, key=lambda r: r["dr"][3])
print(f"best real pair at 1 uV: {best_real['name']} dr {best_real['dr'][3]:.4f}")
print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {time.time()-t0:.0f}s). "
      f"Open it in a browser to check.")
