"""
Step 2 — template head model, candidate grid, cEEGrid template, lead fields.
Project: physics-guided wearable electrode placement (C7-B / F17).

What it does
  1. Fetches MNE fsaverage (~200 MB into data/mne_subjects/, once): 3-layer BEM + ico-5 source space.
  2. Scalp electrodes: standard 10-05 positions for the 29 Ear-SAAD scalp channels.
  3. Around-ear electrodes: TEMPLATE C-arc behind each ear (the Ear-SAAD manual only has a
     hand sketch, no coordinates). Ellipse around an estimated pinna centre
     (preauricular point + 1.5 cm posterior + 1.0 cm up), semi-axes 2.5 cm (posterior)
     x 3.5 cm (vertical), electrodes evenly from 120 deg (1 = top-front) to -120 deg
     (10 = bottom-front); left side uses positions 1-9. Projected onto the skin surface.
  4. Candidate grid: BEM outer-skin vertices above (preauricular z - 4 cm), excluding the
     face below the nasion.
  5. Fixed-orientation lead fields (cortical normals) for scalp + ear + grid in one forward.
  6. RBF interpolation check: (a) leave-10%-out on the grid, (b) ear template positions
     interpolated from the grid vs their exact forward. Go if median relative error < 5 %.
  7. Saves data/headmodel/headmodel.npz and figures/step2_layout.png.

Lead fields are compared after average referencing over the evaluated set, because an EEG
lead field is only defined up to a common reference.

Run:
    python3 step2_headmodel.py
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.interpolate import RBFInterpolator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "headmodel"
FIG = ROOT / "figures"
RES = ROOT / "results"
for d in (OUT, FIG, RES):
    d.mkdir(parents=True, exist_ok=True)

SCALP = ["O1", "Pz", "P3", "P7", "CP1", "CP5", "M1", "Cz", "C3", "T7", "FC1", "FC5",
         "F3", "F7", "Fp1", "O2", "P4", "P8", "CP2", "CP6", "M2", "C4", "T8", "FC2",
         "FC6", "Fz", "F4", "F8", "Fp2"]
EAR_L = [f"cEL{i}" for i in range(1, 10)]
EAR_R = [f"cER{i}" for i in range(1, 11)]

# cEEGrid template parameters (metres) — assumptions, perturbed in step 5
PINNA_POST, PINNA_UP = 0.015, 0.010
SEMI_POST, SEMI_UP = 0.025, 0.035
THETA = np.deg2rad(np.linspace(120, -120, 10))
GRID_BELOW_LPA = 0.04
RBF_HOLDOUT, RBF_SEED = 0.10, 0
GO_MEDIAN_ERR = 0.05

mne.set_log_level("WARNING")
t0 = time.time()

# ---------------------------------------------------------------------------
# 1. fsaverage
# ---------------------------------------------------------------------------
SUBJ = ROOT / "data" / "mne_subjects"          # fsaverage lives inside the project
SUBJ.mkdir(parents=True, exist_ok=True)
fs_dir = Path(mne.datasets.fetch_fsaverage(subjects_dir=SUBJ, verbose=True))
subjects_dir = fs_dir.parent
bem_sol = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
bem_surf = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem.fif"
src = mne.read_source_spaces(fs_dir / "bem" / "fsaverage-ico-5-src.fif")
skin = [s for s in mne.read_bem_surfaces(bem_surf) if s["id"] == 4][0]   # outer skin
skin_rr = skin["rr"]                                                      # MRI = head (fsaverage)
print(f"fsaverage ready ({time.time()-t0:.0f}s); skin vertices {len(skin_rr)}")

# ---------------------------------------------------------------------------
# 2. scalp positions
# ---------------------------------------------------------------------------
pos = {}
for name in ("standard_1005", "standard_1020"):
    mont = mne.channels.make_standard_montage(name)
    p = mont.get_positions()
    for ch in SCALP:
        if ch not in pos and ch in p["ch_pos"]:
            pos[ch] = np.asarray(p["ch_pos"][ch])
    fid = {k: np.asarray(p[k]) for k in ("nasion", "lpa", "rpa")}
missing = [c for c in SCALP if c not in pos]
if missing:
    raise SystemExit(f"Scalp channels missing from standard montages: {missing}")


def to_skin(point, side):
    """Keep y, z; set x from the 4 nearest skin vertices (in y-z) on that side."""
    cand = skin_rr[np.sign(skin_rr[:, 0]) == side]
    d = np.linalg.norm(cand[:, 1:] - point[1:], axis=1)
    nn = np.argsort(d)[:4]
    w = 1.0 / (d[nn] + 1e-4)
    return np.array([np.sum(w * cand[nn, 0]) / w.sum(), point[1], point[2]])


# ---------------------------------------------------------------------------
# 3. cEEGrid template
# ---------------------------------------------------------------------------
ear_pos = {}
for names, pre, side in ((EAR_L, fid["lpa"], -1), (EAR_R, fid["rpa"], +1)):
    centre = pre + np.array([0.0, -PINNA_POST, PINNA_UP])           # y anterior, z up
    for k, name in enumerate(names):
        th = THETA[k]
        p = centre + np.array([0.0, -SEMI_POST * np.cos(th), SEMI_UP * np.sin(th)])
        ear_pos[name] = to_skin(p, side)

# ---------------------------------------------------------------------------
# 4. candidate grid
# ---------------------------------------------------------------------------
z_min = min(fid["lpa"][2], fid["rpa"][2]) - GRID_BELOW_LPA
face = (skin_rr[:, 1] > fid["nasion"][1] - 0.02) & (skin_rr[:, 2] < fid["nasion"][2])
keep = (skin_rr[:, 2] > z_min) & ~face
grid_rr = skin_rr[keep]
grid_names = [f"G{i:04d}" for i in range(len(grid_rr))]
print(f"candidate grid: {len(grid_rr)} skin vertices")

# ---------------------------------------------------------------------------
# 5. forward
# ---------------------------------------------------------------------------
all_names = SCALP + EAR_L + EAR_R + grid_names
all_pos = {**pos, **ear_pos, **dict(zip(grid_names, grid_rr))}
dig = mne.channels.make_dig_montage(ch_pos=all_pos, nasion=fid["nasion"], lpa=fid["lpa"],
                                    rpa=fid["rpa"], coord_frame="head")
info = mne.create_info(all_names, 1000.0, "eeg")
info.set_montage(dig)
print(f"forward for {len(all_names)} electrodes x {sum(s['nuse'] for s in src)} sources ...")
fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=str(bem_sol),
                                eeg=True, meg=False, mindist=5.0, n_jobs=1)
fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
L = fwd["sol"]["data"].astype(np.float64)                 # n_ch x n_src, V/(A·m)
ch_pos_used = np.array([fwd["info"]["chs"][i]["loc"][:3] for i in range(len(all_names))])
print(f"lead field {L.shape} ({time.time()-t0:.0f}s)")

i_sc = np.arange(len(SCALP))
i_ear = np.arange(len(SCALP), len(SCALP) + len(EAR_L) + len(EAR_R))
i_gr = np.arange(i_ear[-1] + 1, len(all_names))


def avgref(M):
    return M - M.mean(axis=0, keepdims=True)


def rel_err(A, B):
    """row-wise ||A-B|| / ||B|| after average referencing both over the same rows."""
    A, B = avgref(A), avgref(B)
    return np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)


# ---------------------------------------------------------------------------
# 6. RBF interpolation checks
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RBF_SEED)
n_g = len(i_gr)
hold = rng.choice(n_g, int(RBF_HOLDOUT * n_g), replace=False)
train = np.setdiff1d(np.arange(n_g), hold)
Pg, Lg = ch_pos_used[i_gr], L[i_gr]
rbf = RBFInterpolator(Pg[train], Lg[train], kernel="thin_plate_spline", smoothing=0.0)
err_hold = rel_err(np.vstack([rbf(Pg[hold]), Lg[train]]),
                   np.vstack([Lg[hold], Lg[train]]))[: len(hold)]

rbf_full = RBFInterpolator(Pg, Lg, kernel="thin_plate_spline", smoothing=0.0)
P_ear = ch_pos_used[i_ear]
err_ear = rel_err(np.vstack([rbf_full(P_ear), Lg]), np.vstack([L[i_ear], Lg]))[: len(i_ear)]

# nearest grid vertex distance (how far ear template points sit from the grid)
d_ear = np.min(np.linalg.norm(P_ear[:, None] - Pg[None], axis=2), axis=1)
spacing = np.median(np.sort(np.linalg.norm(Pg[:, None] - Pg[None], axis=2), axis=1)[:, 1])

summary = {
    "n_sources": int(L.shape[1]), "n_grid": int(n_g),
    "grid_median_spacing_mm": float(1000 * spacing),
    "rbf_holdout_rel_err": {"median": float(np.median(err_hold)),
                            "p95": float(np.percentile(err_hold, 95)),
                            "max": float(err_hold.max())},
    "rbf_ear_rel_err": {"median": float(np.median(err_ear)),
                        "p95": float(np.percentile(err_ear, 95)),
                        "per_channel": dict(zip(EAR_L + EAR_R, map(float, err_ear)))},
    "ear_to_grid_distance_mm": dict(zip(EAR_L + EAR_R, map(lambda x: float(1000 * x), d_ear))),
    "template": {"pinna_post_m": PINNA_POST, "pinna_up_m": PINNA_UP,
                 "semi_post_m": SEMI_POST, "semi_up_m": SEMI_UP,
                 "theta_deg": np.rad2deg(THETA).tolist()},
    "projection_shift_mm": {n: float(1000 * np.linalg.norm(ch_pos_used[i] - all_pos[n]))
                            for i, n in enumerate(all_names[: len(SCALP) + len(i_ear)])},
}
ok = (summary["rbf_holdout_rel_err"]["median"] < GO_MEDIAN_ERR
      and summary["rbf_ear_rel_err"]["median"] < GO_MEDIAN_ERR)
summary["go_no_go"] = "GO" if ok else "CHECK (interpolation error above 5 %)"

print(f"\ngrid spacing (median NN): {summary['grid_median_spacing_mm']:.1f} mm")
print(f"RBF hold-out rel. error: median {np.median(err_hold):.3f}, "
      f"p95 {np.percentile(err_hold, 95):.3f}")
print(f"RBF ear-template rel. error: median {np.median(err_ear):.3f}, "
      f"p95 {np.percentile(err_ear, 95):.3f}")
big = {k: round(v, 1) for k, v in summary["projection_shift_mm"].items() if v > 5}
print(f"electrodes moved > 5 mm by MNE's scalp projection: {big or 'none'}")
print(summary["go_no_go"])

np.savez_compressed(
    OUT / "headmodel.npz",
    L_scalp=L[i_sc].astype(np.float32), L_ear=L[i_ear].astype(np.float32),
    L_grid=Lg.astype(np.float32),
    pos_scalp=ch_pos_used[i_sc], pos_ear=P_ear, pos_grid=Pg,
    names_scalp=np.array(SCALP), names_ear=np.array(EAR_L + EAR_R),
    nasion=fid["nasion"], lpa=fid["lpa"], rpa=fid["rpa"],
    src_vertices_lh=fwd["src"][0]["vertno"], src_vertices_rh=fwd["src"][1]["vertno"],
)
(RES / "step2_headmodel.json").write_text(json.dumps(summary, indent=2))

# ---------------------------------------------------------------------------
# 7. layout figure (sanity check of template placement)
# ---------------------------------------------------------------------------
C_GRID, C_SCALP, C_EAR = "#c9c9c9", "#2a6fdb", "#d9622b"
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
views = [("Left side (y = anterior, z = up)", 1, 2, lambda p: p[:, 0] < 0.01),
         ("Right side (y = anterior, z = up)", 1, 2, lambda p: p[:, 0] > -0.01),
         ("Top (x = right, y = anterior)", 0, 1, lambda p: np.ones(len(p), bool))]
P_sc = ch_pos_used[i_sc]
for ax, (title, a, b, sel) in zip(axes, views):
    m = sel(Pg)
    ax.scatter(Pg[m, a] * 100, Pg[m, b] * 100, s=4, c=C_GRID, label="candidate grid")
    m = sel(P_sc)
    ax.scatter(P_sc[m, a] * 100, P_sc[m, b] * 100, s=40, c=C_SCALP, label="scalp (10-05)")
    for p, n in zip(P_sc[m], np.array(SCALP)[m]):
        ax.annotate(n, (p[a] * 100, p[b] * 100), fontsize=7, color="#333333",
                    xytext=(3, 3), textcoords="offset points")
    m = sel(P_ear)
    ax.scatter(P_ear[m, a] * 100, P_ear[m, b] * 100, s=40, c=C_EAR,
               label="around-ear (template)")
    for p, n in zip(P_ear[m], np.array(EAR_L + EAR_R)[m]):
        ax.annotate(n.replace("cE", ""), (p[a] * 100, p[b] * 100), fontsize=7,
                    color="#333333", xytext=(3, -8), textcoords="offset points")
    for k in ("lpa", "rpa", "nasion"):
        q = fid[k]
        ax.scatter(q[a] * 100, q[b] * 100, marker="x", s=40, c="#333333")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cm")
    ax.set_aspect("equal")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(loc="lower left", fontsize=8, frameon=False)
fig.suptitle("Step 2 — fsaverage electrode layout (x = fiducials). Around-ear positions are a "
             "template, not measured.", fontsize=11)
fig.savefig(FIG / "step2_layout.png", dpi=150)
print(f"\nWrote data/headmodel/headmodel.npz, results/step2_headmodel.json, "
      f"figures/step2_layout.png ({time.time()-t0:.0f}s total)")
