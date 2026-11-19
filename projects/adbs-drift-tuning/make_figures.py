#!/usr/bin/env python3
"""
make_figures.py — static figures for the project page (renders locally, seconds).

Needs: results/surrogate.json, results/step3_tuned_c.json, results/step3_summary_tuned.csv
Writes to results/figures/:
  fig_tracking_abrupt.png   hero: optimum vs chosen amplitude after a lead shift (one test seed)
  fig_regret_vs_unsafe.png  every tuned algorithm: mean regret vs unsafe evaluations (drift scenarios)
  fig_recovery.png          evaluations needed to re-find the optimum after the shift
  fig_pipeline.png          from NEURON model to tuning study

Usage:
  python3 make_figures.py
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "figures")
sys.path.insert(0, HERE)
import tuning_study as ts  # noqa: E402
from surrogate import Surrogate  # noqa: E402

# validated categorical palette (light surface): fixed order, family -> hue
INK, INK2, MUTED, GRID, SURF = "#1f1f1d", "#5f5e58", "#8a897f", "#e6e5df", "#fcfcfb"
FAMILY = {"stationary": "#2a78d6", "adversarial": "#eb6834", "nonstationary": "#1baf7a", "TV-GP": "#eda100",
          "fixed": INK}
FAM_OF = {"fixed (clinician)": "fixed", "eps-greedy": "stationary", "UCB1": "stationary", "Thompson": "stationary",
          "discounted UCB": "nonstationary", "sliding-window UCB": "nonstationary",
          "Exp3": "adversarial", "Exp3.P": "adversarial", "Exp3.S": "adversarial", "Tsallis-INF": "adversarial",
          "TV-GP-UCB": "TV-GP", "TV-GP-UCB-periodic*": "TV-GP"}
FAM_LABEL = {"stationary": "stationary bandits", "nonstationary": "non-stationary bandits",
             "TV-GP": "time-varying GP", "adversarial": "adversarial (Exp3 family, lec. 7)", "fixed": "fixed setting"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def trajectory(sur, Alg, T, seed):
    """Same loop as tuning_study.run_one, but keeps the chosen arm at every step."""
    rng = np.random.default_rng(seed)
    g = ts.gain_schedule("abrupt", T)
    exp_r = np.array([sur.expected_reward(ts.ARMS, gt, ts.LAM) for gt in g])
    opt = exp_r.argmax(1)
    alg = Alg(T, rng, int(opt[0]))
    chosen = np.empty(T, int)
    reg = 0.0
    for t in range(T):
        i = alg.select(t)
        r, _ = sur.sample_reward(ts.ARMS[i], g[t], ts.LAM, rng)
        alg.update(t, i, ts.to_gain(r))
        chosen[t] = i
        reg += exp_r[t, opt[t]] - exp_r[t, i]
    return g, ts.ARMS[opt], ts.ARMS[chosen], reg


def fig_tracking(sur, T=1000, seed=1000):
    algs = [ts.Fixed, ts.UCB1, ts.TVGPUCB, ts.Exp3S]
    fig, axes = plt.subplots(len(algs) + 1, 1, figsize=(9, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [0.55] + [1] * len(algs)})
    g = ts.gain_schedule("abrupt", T)
    t = np.arange(T)
    ax = axes[0]
    ax.plot(t, g, color=INK, lw=2)
    ax.set_ylabel("electrode\ngain")
    ax.set_ylim(0.6, 1.05)
    ax.annotate("lead shift: 30% less\ncurrent reaches tissue", xy=(T // 3, 0.7), xytext=(T // 3 + 40, 0.86),
                fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    style(ax)
    for ax, Alg in zip(axes[1:], algs):
        g, opt, ch, reg = trajectory(sur, Alg, T, seed)
        col = FAMILY[FAM_OF[Alg.name]]
        unsafe_floor = ts.UNSAFE_MA / g                          # set amplitude above which delivered > 3 mA
        ax.fill_between(t, unsafe_floor, 4.3, color="#f2d9cf", lw=0, zorder=0)
        ax.scatter(t, ch, s=6, color=col, alpha=0.35, lw=0, zorder=2)
        cs = np.concatenate([[0.0], np.cumsum(ch)])            # trailing 25-step mean (no edge droop)
        lo = np.maximum(0, t - 24)
        ma = (cs[t + 1] - cs[lo]) / (t + 1 - lo)
        ax.plot(t, ma, color=col, lw=2, zorder=3)
        ax.step(t, opt, where="post", color=INK, lw=2, ls="--", zorder=4)
        ax.set_ylim(0.3, 4.3)
        ax.set_ylabel("set amplitude\n(mA)")
        ax.set_title(f"{Alg.name.replace('*', '')}  ·  regret {reg:.0f}", loc="left", fontsize=10, color=INK)
        style(ax)
    axes[1].text(5, 3.95, "unsafe zone (delivered > 3 mA, assumed)", fontsize=8, color=INK2, va="top")
    axes[-1].set_xlabel("evaluation (one 2-s stimulation test each)")
    fig.suptitle("Tracking the best DBS amplitude after a lead shift (one test seed)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.945, "dashed: best setting   ·   line: average of the last 25 choices   ·   dots: individual choices",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig_tracking_abrupt.png"), dpi=170)
    plt.close(fig)


def load_summary():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "step3_summary_tuned.csv"))))
    agg = {}
    for r in rows:
        if r["scenario"] == "static":
            continue
        a = agg.setdefault(r["algorithm"], {"reg": [], "uns": [], "rec": None})
        a["reg"].append(float(r["final_dyn_regret"]))
        a["uns"].append(float(r["unsafe_evals"]))
        if r["scenario"] == "abrupt":
            a["rec"] = float(r["recovery_steps"])
    return agg


def fig_scatter(agg):
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    pts = {n: (np.mean(a["uns"]), np.mean(a["reg"])) for n, a in agg.items()}
    # label positions in data coordinates; the crowded bottom-left cluster gets leader lines
    lab = {"UCB1": (16, 12), "discounted UCB": (16, 19), "Thompson": (16, 26), "TV-GP-UCB": (16, 33),
           "TV-GP-UCB-periodic*": (16, 40), "eps-greedy": (16, 52), "sliding-window UCB": (29, 56),
           "fixed (clinician)": (2, 66), "Tsallis-INF": (34, 74), "Exp3": (35, 92), "Exp3.P": (66, 139),
           "Exp3.S": (77, 162)}
    for name, (x, y) in pts.items():
        ax.scatter(x, y, s=70, color=FAMILY[FAM_OF[name]], edgecolor=SURF, lw=2, zorder=3)
        lx, ly = lab.get(name, (x + 2, y + 2))
        text = name.replace("*", " (known period)")
        far = abs(lx - x) > 4 or abs(ly - y) > 6
        ax.annotate(text, (x, y), xytext=(lx, ly), textcoords="data", fontsize=8.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=4) if far else None)
    for fam in ["fixed", "stationary", "nonstationary", "TV-GP", "adversarial"]:
        ax.scatter([], [], s=50, color=FAMILY[fam], label=FAM_LABEL[fam])
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlim(-3, 92)
    ax.set_ylim(0, 175)
    ax.set_xlabel("unsafe evaluations per 1000 (delivered > 3 mA, assumed threshold)")
    ax.set_ylabel("cumulative dynamic regret (lower is better)")
    fig.text(0.01, 0.965, "Noise-model learners sit bottom-left; worst-case learners pay in regret and safety",
             fontsize=11.5, color=INK, fontweight="bold")
    fig.text(0.01, 0.925, "Mean over 4 drift scenarios, 30 test seeds each, all algorithms fairly tuned",
             fontsize=9.5, color=INK2)
    style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(OUT, "fig_regret_vs_unsafe.png"), dpi=170)
    plt.close(fig)


def fig_recovery(agg, horizon=667):
    items = sorted(((a["rec"], n) for n, a in agg.items() if a["rec"] is not None))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, (v, n) in enumerate(items):
        col = FAMILY[FAM_OF[n]]
        ax.barh(k, v, color=col, height=0.66)
        lab = "never (end of run)" if v >= horizon - 1 else f"{v:.0f}"
        ax.text(v + 6, k, lab, va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(items)), [n.replace("*", " (known period)") for _, n in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, horizon * 1.18)
    ax.set_xlabel("evaluations until choices are back within ±0.25 mA of the new best")
    ax.text(0.0, 1.04, "Recovery after the lead shift (abrupt scenario, 30 test seeds)", transform=ax.transAxes,
            fontsize=11.5, color=INK, fontweight="bold")
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_recovery.png"), dpi=170)
    plt.close(fig)


def fig_pipeline():
    steps = [("Biophysical model", "Fleming 2020 cortico-basal\nganglia network (NEURON)"),
             ("Measured response", "49 simulations: beta vs\namplitude, 3 burst patterns"),
             ("Surrogate", "smooth curve + measured\nnoise (SD ≈ 0.05)"),
             ("Drift", "electrode gain: gradual,\nperiodic, combined, abrupt"),
             ("12 tuners", "stationary, non-stationary,\nadversarial, time-varying GP"),
             ("Fair evaluation", "tuned on validation seeds,\nscored on 30 test seeds")]
    loop = ["Plant (virtual patient)", "Sense (beta biomarker)", "Compute", "Plant change", "Decide → Act", "Validate"]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, ((title, sub), stage) in enumerate(zip(steps, loop)):
        x = i + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.18), 0.86, 0.62, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc="#f0fdfa", ec="#0d9488", lw=1.4))
        ax.text(x + 0.43, 0.66, title, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + 0.43, 0.42, sub, ha="center", va="center", fontsize=8, color=INK2)
        ax.text(x + 0.43, 0.06, stage, ha="center", va="center", fontsize=8, color="#0f766e")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.06, 0.49), xytext=(i + 0.92, 0.49),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=170)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    ts.CMULT.update(json.load(open(os.path.join(RESULTS, "step3_tuned_c.json"))))
    sur = Surrogate.load()
    agg = load_summary()
    fig_pipeline()
    fig_scatter(agg)
    fig_recovery(agg)
    fig_tracking(sur)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
