"""Statistics and figures for the write-up and website.

Reads results/ts_*/cv.csv and results/fake_mirror_*.csv (from run_ts.py / run_fake_mirror.py)
and data/processed/*.npz. Writes figures/*.png and results/summary.md.

  python3 make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

from src.channels import normalize  # noqa: E402
from src.data import canonicalize, load_cohort  # noqa: E402

RES, FIG = Path("results"), Path("figures")
COL = {"healthy": "#2a78d6", "stroke": "#eb6834", "fake": "#a3a29c"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COHORT = {"healthy": ("physionet", "_n50"), "stroke": ("stroke", "")}
CHSETS = {"shared": "29 channels", "shared_sensorimotor": "8 central channels"}
TREAT_LABEL = {"augment": "Mirror\naugmentation", "equivariant": "Exact\nequivariance",
               "soft_a0.1": "Soft\n(a = 0.1)", "canonical": "Lesion-side\ncanonicalization"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight"})


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def load_cv(cohort, chset, tag=""):
    name, suffix = COHORT[cohort]
    path = RES / f"ts_{name}_{chset}{suffix}{'_' + tag if tag else ''}" / "cv.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["label"] = np.where(df.treatment == "soft", "soft_a" + df.alpha.map(lambda v: f"{v:g}"), df.treatment)
    return df.pivot_table(index="subject", columns="label", values="bal_acc")


def load_fake(cohort, chset, tag=""):
    name, suffix = COHORT[cohort]
    path = RES / f"fake_mirror_{name}_{chset}{suffix}{'_' + tag if tag else ''}.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- statistics
def summary_tables(lines, tag=""):
    lines.append(f"\n## Treatment effects{' (' + tag + ')' if tag else ''}\n")
    lines.append("Per-subject balanced accuracy; Δ = treatment − `none`, bootstrap 95% CI, Wilcoxon signed-rank.\n")
    lines.append("| cohort | channels | treatment | mean acc | Δ vs none | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|---|")
    deltas = {}
    for chset in CHSETS:
        for cohort in COHORT:
            w = load_cv(cohort, chset, tag)
            if w is None:
                continue
            for lab in [c for c in ["none", "augment", "equivariant", "canonical", "soft_a0.5", "soft_a0.2",
                                    "soft_a0.1", "soft_a0.05"] if c in w]:
                d = w[lab] - w["none"]
                deltas[(cohort, chset, lab)] = d
                m, lo, hi = boot_ci(d)
                p = wilcoxon(d).pvalue if lab != "none" and np.any(d != 0) else np.nan
                lines.append(f"| {cohort} | {CHSETS[chset]} | {lab} | {w[lab].mean():.3f} | "
                             f"{m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |")
            m, lo, hi = boot_ci(w["none"] - 0.5)
            lines.append(f"| {cohort} | {CHSETS[chset]} | *none vs chance* | {w['none'].mean():.3f} | "
                         f"{m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {wilcoxon(w['none'] - 0.5).pvalue:.3g} |")
    have = [c for c in CHSETS if ("healthy", c, "equivariant") in deltas and ("stroke", c, "equivariant") in deltas]
    if have:
        lines.append("\nCohort contrast of the equivariance gain (Mann–Whitney U on per-subject Δ; confounded by dataset):\n")
    for chset in have:
        a, b = deltas.get(("healthy", chset, "equivariant")), deltas.get(("stroke", chset, "equivariant"))
        if a is not None and b is not None:
            lines.append(f"- {CHSETS[chset]}: healthy {a.mean():+.3f}, stroke {b.mean():+.3f}, "
                         f"p = {mannwhitneyu(a, b).pvalue:.3g}")


def fake_tables(lines, tag=""):
    if not any(load_fake(c, ch, tag) is not None for c in COHORT for ch in CHSETS):
        return
    lines.append(f"\n## Real vs fake mirrors{' (' + tag + ')' if tag else ''}\n")
    lines.append("Gain over `none` for the real mid-sagittal mirror and for random channel pairings with the "
                 "same cycle structure. Permutation p = (1 + #fake ≥ real)/(N + 1).\n")
    lines.append("| cohort | channels | treatment | real gain | fake gain (mean ± sd) | real − fake | p | N fake |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for chset in CHSETS:
        for cohort in COHORT:
            f = load_fake(cohort, chset, tag)
            if f is None:
                continue
            for kind, g in f.groupby("treatment"):
                real = g[g.is_real].gain.iloc[0]
                fake = g[~g.is_real].gain.to_numpy()
                p = (1 + np.sum(fake >= real)) / (len(fake) + 1)
                lines.append(f"| {cohort} | {CHSETS[chset]} | {kind} | {real:+.3f} | {fake.mean():+.3f} ± "
                             f"{fake.std():.3f} | {real - fake.mean():+.3f} | {p:.3f} | {len(fake)} |")


# ---------------------------------------------------------------- figures
def fig_fake_mirror():
    cols = [("healthy", "", "Healthy"), ("stroke", "", "Stroke, all 50")]
    if load_fake("stroke", "shared", "excl") is not None:
        cols.append(("stroke", "excl", "Stroke, 3 high-amplitude\nrecordings removed"))
    fig, axes = plt.subplots(2, len(cols), figsize=(4.2 * len(cols), 5.8), sharex=True, squeeze=False)
    for i, chset in enumerate(CHSETS):
        for j, (cohort, tag, name) in enumerate(cols):
            ax = axes[i, j]
            f = load_fake(cohort, chset, tag)
            if f is None:
                ax.set_axis_off()
                continue
            g = f[f.treatment == "equivariant"]
            real = g[g.is_real].gain.iloc[0] * 100
            fake = g[~g.is_real].gain.to_numpy() * 100
            ax.hist(fake, bins=np.arange(-9, 6, 0.75), color=COL["fake"], edgecolor="white", linewidth=1)
            ax.axvline(real, color=COL[cohort], linewidth=2.5)
            ax.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
            p = (1 + np.sum(fake >= real)) / (len(fake) + 1)
            ax.set_title(f"{name}, {CHSETS[chset]}\nreal {real:+.1f} pp, beats {np.mean(fake < real) * 100:.0f}% "
                         f"of {len(fake)} fakes, p = {p:.3f}", color=INK, fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel("Fake mirrors")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL["fake"]),
               plt.Line2D([], [], color=COL["healthy"], linewidth=2.5),
               plt.Line2D([], [], color=COL["stroke"], linewidth=2.5)]
    fig.legend(handles, ["Fake mirrors (random channel pairings)", "Real mirror, healthy", "Real mirror, stroke"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.supxlabel("Gain in balanced accuracy over no prior, exact equivariance (percentage points)",
                  fontsize=10, color=INK, y=0.03)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig_fake_mirror.png")
    plt.close(fig)


def fig_treatments():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    labs = list(TREAT_LABEL)
    for ax, chset in zip(axes, CHSETS):
        for k, cohort in enumerate(COHORT):
            w = load_cv(cohort, chset)
            if w is None:
                continue
            for i, lab in enumerate(labs):
                if lab not in w:
                    continue
                d = (w[lab] - w["none"]) * 100
                m, lo, hi = boot_ci(d)
                x = i + (k - 0.5) * 0.3
                rng = np.random.default_rng(i)
                ax.scatter(x + rng.uniform(-0.06, 0.06, len(d)), d, s=9, color=COL[cohort], alpha=0.25,
                           linewidths=0)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[cohort], markersize=7,
                            capsize=0, elinewidth=2, markeredgecolor="white", markeredgewidth=1.5,
                            label=cohort.capitalize() if i == 1 else None)
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_xticks(range(len(labs)), [TREAT_LABEL[l] for l in labs], fontsize=8.5)
        ax.set_title(CHSETS[chset].capitalize(), color=INK, fontsize=10, loc="left")
        ax.set_ylim(-25, 25)
    axes[0].set_ylabel("Δ balanced accuracy vs no prior (pp)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(0.02, -0.03, "Dots: patients/subjects. Markers: mean with bootstrap 95% CI.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_treatments.png")
    plt.close(fig)


def fig_soft_curve():
    order = [("none", "1\n(none)"), ("soft_a0.5", "0.5"), ("soft_a0.2", "0.2"), ("soft_a0.1", "0.1"),
             ("soft_a0.05", "0.05"), ("equivariant", "0\n(exact)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, chset in zip(axes, CHSETS):
        for cohort in COHORT:
            w = load_cv(cohort, chset)
            if w is None:
                continue
            stats = [boot_ci(w[l] * 100) if l in w else (np.nan,) * 3 for l, _ in order]
            m = np.array([s[0] for s in stats])
            lo = np.array([s[1] for s in stats])
            hi = np.array([s[2] for s in stats])
            x = np.arange(len(order))
            ax.fill_between(x, lo, hi, color=COL[cohort], alpha=0.12, linewidth=0)
            ax.plot(x, m, color=COL[cohort], linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", label=cohort.capitalize())
        ax.axhline(50, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(order)), [t for _, t in order], fontsize=8.5)
        ax.set_xlabel("a = weight allowed on the mirror-invariant part")
        ax.set_title(CHSETS[chset].capitalize(), color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_soft_curve.png")
    plt.close(fig)


def _montage_info(names):
    try:
        mont = mne.channels.make_standard_montage("colin27_1005")      # newer MNE name
    except ValueError:
        mont = mne.channels.make_standard_montage("standard_1005")
    lookup = {normalize(c): c for c in mont.ch_names}
    std = [lookup[normalize(c)] for c in names]
    info = mne.create_info(std, 128.0, "eeg")
    info.set_montage(mont)
    return info


def _class_logpower(co, canonical=False):
    """Per-subject mean log mu/beta power difference (label 1 - label 0) per channel."""
    sos = butter(4, [8, 30], btype="bandpass", fs=co.sfreq, output="sos")
    X, y = co.X, co.y
    if canonical:
        X, y, _ = canonicalize(X, y, co.affected_side, co.perm)
    lp = np.log(sosfiltfilt(sos, X, axis=-1).var(axis=-1))
    out = []
    for s in np.unique(co.subject):
        m = co.subject == s
        out.append(lp[m & (y == 1)].mean(0) - lp[m & (y == 0)].mean(0))
    return np.array(out)


def fig_lateralization(lines):
    try:
        cos = {"healthy": load_cohort("physionet", "shared", 50), "stroke": load_cohort("stroke", "shared")}
    except Exception as e:  # noqa: BLE001
        print(f"skipping lateralization figure: {e}")
        return
    maps = {"Healthy\nright − left hand": ("healthy", _class_logpower(cos["healthy"])),
            "Stroke\nright − left hand": ("stroke", _class_logpower(cos["stroke"])),
            "Stroke, canonical frame\nunaffected − affected hand": ("stroke", _class_logpower(cos["stroke"], True))}
    vmax = max(np.abs(v.mean(0)).max() for _, v in maps.values())
    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 0.05, 0.45, 1.25], wspace=0.25)
    topo_axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax, (title, (cohort, d)) in zip(topo_axes, maps.items()):
        info = _montage_info(cos[cohort].ch_names)
        im, _ = mne.viz.plot_topomap(d.mean(0), info, axes=ax, show=False, cmap="RdBu_r", vlim=(-vmax, vmax),
                                     contours=0, sensors=True)
        ax.set_title(title, fontsize=9, color=INK)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
    cb.set_label("Δ log mu/beta power", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 5])
    ax.set_title("Lateralization index per subject\n(C4 − C3; > 0 = contralateral ERD)", fontsize=9, color=INK)
    lis = {}
    for k, cohort in enumerate(COHORT):
        names = [normalize(c) for c in cos[cohort].ch_names]
        d = _class_logpower(cos[cohort])
        li = d[:, names.index("C4")] - d[:, names.index("C3")]      # >0 = contralateral ERD
        lis[cohort] = li
        rng = np.random.default_rng(k)
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(li)), li, s=12, color=COL[cohort], alpha=0.5, linewidths=0)
        m, lo, hi = boot_ci(li)
        ax.errorbar(k + 0.28, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[cohort], markersize=7,
                    elinewidth=2, markeredgecolor="white")
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.set_xticks([0, 1], ["Healthy", "Stroke"])
    fig.savefig(FIG / "fig_lateralization.png")
    plt.close(fig)

    lines.append("\n## Lateralization (model-free)\n")
    for cohort, li in lis.items():
        m, lo, hi = boot_ci(li)
        lines.append(f"- {cohort}: mean LI {m:+.3f} [{lo:+.3f}, {hi:+.3f}], Wilcoxon vs 0 p = {wilcoxon(li).pvalue:.3g}, "
                     f"{int((li > 0).sum())}/{len(li)} subjects > 0")
    lines.append(f"- healthy vs stroke: Mann–Whitney p = {mannwhitneyu(lis['healthy'], lis['stroke']).pvalue:.3g}")


def fig_mirror_concept():
    co = load_cohort("stroke", "shared")
    info = _montage_info(co.ch_names)
    try:                                         # same 2D projection as plot_topomap
        from mne.channels.layout import _find_topomap_coords
        pos = _find_topomap_coords(info, picks=list(range(len(info.ch_names))))
    except Exception:  # noqa: BLE001
        xyz = np.array([info.get_montage().get_positions()["ch_pos"][c] for c in info.ch_names])
        xyz = xyz - np.array([0, 0, xyz[:, 2].min()])
        theta = np.arccos(np.clip(xyz[:, 2] / np.linalg.norm(xyz, axis=1), -1, 1))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        pos = np.c_[theta * np.cos(phi), theta * np.sin(phi)]
    rng = np.random.default_rng(3)
    from run_fake_mirror import random_involution
    fake = random_involution(co.perm, rng)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for ax, perm, title in [(axes[0], co.perm, "Real mirror: left ↔ right partner"),
                            (axes[1], fake, "One fake mirror: random pairing")]:
        r = np.abs(pos).max() * 1.15
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=INK2, linewidth=1))
        ax.plot([-0.12 * r, 0, 0.12 * r], [0.98 * r, 1.12 * r, 0.98 * r], color=INK2, linewidth=1)
        from matplotlib.patches import FancyArrowPatch
        for i, j in enumerate(perm):
            if i < j:                            # arcs so nested pairs on one row stay distinct
                bow = 0.12 + 0.25 * np.linalg.norm(pos[i] - pos[j]) / (2 * r)
                ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-", connectionstyle=f"arc3,rad={bow:.2f}",
                                             color=COL["healthy"] if perm is co.perm else COL["fake"],
                                             linewidth=1.5, alpha=0.85))
        ax.scatter(pos[:, 0], pos[:, 1], s=60, color="white", edgecolor=INK, linewidth=1, zorder=3)
        for (x, y), name in zip(pos, info.ch_names):
            ax.text(x, y - 0.06 * r, name, fontsize=6, ha="center", va="top", color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.savefig(FIG / "fig_mirror_concept.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    lines = ["# Results summary", "", "Generated by make_figures.py from results/ and data/processed/."]
    summary_tables(lines)
    fake_tables(lines)
    if any(RES.glob("ts_*_excl")) or any(RES.glob("fake_mirror_*_excl.csv")):
        summary_tables(lines, "excl")
        fake_tables(lines, "excl")
    fig_lateralization(lines)
    for f in (fig_fake_mirror, fig_treatments, fig_soft_curve, fig_mirror_concept):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__} failed: {e}")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures in {FIG}/: {', '.join(sorted(p.name for p in FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
