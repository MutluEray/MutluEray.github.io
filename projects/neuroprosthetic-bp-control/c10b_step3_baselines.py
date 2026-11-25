"""
C10-B step 3 — evaluation harness check + clinical baseline controllers (hand-set gains).

Runs on the 40 TUNING patients only (0–39); the 20 test patients stay untouched until
the final comparison in step 4.

Controllers (see c10b_harness.py):
  none      no stimulation (reference)
  C0 fixed  amplitude programmed once so supine SBP = 115 (Phillips 2025 'continuous' arm)
  C1 rule   Christie et al. 2026 v3-style timed amplitude steps (hand-set: Tmin 10 s,
            Tmax 70 s, step 0.1)
  C2 PID    PID to 115 mmHg (hand-set: Kp 0.02 /mmHg, Ki 0.0007, Kd 0, rate limit 0.2/s)

Purpose: (1) check the harness and metrics behave sensibly, (2) see whether the
baselines reproduce the qualitative pattern reported clinically (closed loop beats
fixed amplitude; overshoot above 150 mmHg after tilt with timed rules), (3) time one
episode, which sets the BO budget for step 4. Gains here are NOT tuned; step 4 tunes
every controller family with the same BO budget before any comparison.

Outputs: results/c10b_step3/{baselines.md, per_patient.csv, fig_example.png, fig_metrics.png}
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c10b_harness as H
from c10b_plant import new_sim

OUT = os.path.join("results", "c10b_step3")
os.makedirs(OUT, exist_ok=True)
TUNE = list(range(40))
METRICS = ["in_range", "hyper", "hypo", "peak_tilt", "recovery", "ad_peak", "ad_hyper", "n_changes", "tv", "u_mean", "sat"]
PHASE_METRICS = [f"in_range_{p[0]}" for p in H.PHASES] + [f"hypo_{p[0]}" for p in H.PHASES] + ["hyper_return"]


def main():
    pop, base, info = H.load_population()
    sim = new_sim()
    ctrls = [H.NoStim(), H.FixedAmplitude(), H.RuleBased(), H.PID()]
    rows, traces = [], {}
    t0 = time.time()
    n_ep = 0
    for pid in TUNE:
        inf = info.loc[pid] if info is not None else None
        for c in ctrls:
            o, m = H.run_patient(sim, pop, base, pid, c, info=inf)
            n_ep += 1
            rows.append({"id": pid, "controller": c.name, **m})
            if pid == example_id(info):
                traces[c.name] = o
        if pid % 10 == 9:
            print(f"[{pid + 1}/{len(TUNE)} patients] {(time.time() - t0) / n_ep:.1f} s/episode")
    per_ep = (time.time() - t0) / n_ep
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "per_patient.csv"), index=False)

    lines = ["# C10-B step 3 (v2) — harness check and baseline controllers (hand-set gains, 40 tuning patients)\n",
             f"Episode: 30 min simulated, {per_ep:.1f} s wall per episode ({n_ep} episodes).\n",
             "Median [IQR] across patients. Gains are hand-set, not tuned — no controller comparison is "
             "claimed from this table.\n",
             "| controller | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for c in ctrls:
        d = df[df["controller"] == c.name]
        cells = []
        for k in METRICS:
            q = d[k].quantile([0.25, 0.5, 0.75]).values
            cells.append(f"{q[1]:.1f} [{q[0]:.1f}, {q[2]:.1f}]")
        lines.append(f"| {c.name} | " + " | ".join(cells) + " |")
    lines += ["\n### Per phase (median [IQR])\n",
              "| controller | " + " | ".join(PHASE_METRICS) + " |", "|---" * (len(PHASE_METRICS) + 1) + "|"]
    for c in ctrls:
        d = df[df["controller"] == c.name]
        cells = []
        for k in PHASE_METRICS:
            q = d[k].quantile([0.25, 0.5, 0.75]).values
            cells.append(f"{q[1]:.0f} [{q[0]:.0f}, {q[2]:.0f}]")
        lines.append(f"| {c.name} | " + " | ".join(cells) + " |")
    lines += ["\nMetric definitions: see c10b_harness.py header. in_range/hyper/hypo/ad_hyper/sat in % time; "
              "peak_tilt, ad_peak in mmHg; recovery in s (600 = never within 10 min). "
              f"Amplitude range u in [0, {H.U_MAX:g}].\n"]
    open(os.path.join(OUT, "baselines.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # example patient traces
    ex = example_id(info)
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for name, o in traces.items():
        ax[0].plot(o["t"], o["sbp"], lw=1.0, label=name)
        ax[1].plot(o["t"], o["u"], lw=1.0, label=name)
    ax[0].axhspan(H.LO, H.HI, color="tab:green", alpha=0.12)
    ax[0].axhline(150, color="tab:red", ls="--", lw=0.8)
    ax[0].set_ylabel("SBP (mmHg)"); ax[1].set_ylabel("amplitude u")
    o = next(iter(traces.values()))
    ax[2].plot(o["t"], o["angle"], color="k")
    ax[2].set_ylabel("tilt (deg)"); ax[2].set_xlabel("time (s)")
    for a_ in ax:
        for a, b in H.DROPOUTS:
            a_.axvspan(a, b, color="0.85")
        a_.axvspan(H.AD[0], H.AD[0] + sum(H.AD[1:]), color="tab:red", alpha=0.06)
    ax[0].legend(frameon=False, fontsize=8, ncol=4)
    ax[0].set_title(f"Patient {ex} (median supine SBP): green = target 110–120, red band = AD-like surge, "
                    f"grey = sensor dropout", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_example.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(17, 3.4))
    names = [c.name for c in ctrls]
    for a_, k, lab in zip(ax, ["in_range", "hyper", "hypo", "recovery", "sat"],
                          ["% time in 110–120", "% time > 150", "% time < 90", "recovery after tilt (s)",
                           "% time at u_max"]):
        a_.boxplot([df[df["controller"] == n][k] for n in names], tick_labels=[n.split()[0] for n in names])
        a_.set_title(lab, fontsize=9)
    fig.suptitle("Hand-set baselines on 40 tuning patients (not a tuned comparison)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_metrics.png"), dpi=140); plt.close(fig)
    print(f"[done] {OUT}/baselines.md, per_patient.csv, fig_example.png, fig_metrics.png")


def example_id(info):
    if info is None:
        return 0
    sub = info.loc[TUNE, "supine_sbp"]
    return int((sub - sub.median()).abs().idxmin())


if __name__ == "__main__":
    main()
