"""
C10-B step 6 — figures for the project page and data for the interactive demo.

Usage:  python3 c10b_step6_showcase.py            (~15 min: 200 demo episodes + figures)
        python3 c10b_step6_showcase.py --figs     (figures only, reuses saved demo data; ~1 min)

Inputs: everything from steps 1a–5 (PRCP envelopes, calibrated/SCI theta, population, sysid,
tuned controllers, test_per_patient.csv, cal_sweep.csv, step 5 rows.csv).

Outputs (results/c10b_step6/):
  fig1_credibility.png   healthy model vs PRCP (rapid tilt) + SCI model vs Corbett 1971
  fig2_example.png       one test patient, PID vs PID + posture vs MPC + posture (full + return zoom)
                         patient chosen by a fixed rule: median C2f time > 150 on return among
                         test patients with any (stated in the caption, not picked by eye)
  fig3_paired.png        per-patient % time > 150 and % time in range, PID vs each controller
  fig4_calibration.png   calibration length vs hypertensive time and time in range
  fig5_robustness.png    step 5 heatmap restyled
  captions.md            caption text with the numbers each figure shows
  demo/index.json, demo/p40.json … p59.json   interactive-demo traces (2-s resolution)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import c10b_controllers as C
import c10b_harness as H
from c10b_plant import EESActuator, sci_theta, simulate, new_sim
from c10b_step4b_tune import patient_info
from c10b_step4c_evaluate import load_tuned
import c10b_step5_robustness as R

OUT = os.path.join("results", "c10b_step6")
DEMO = os.path.join(OUT, "demo")
TEST = list(range(40, 60))
DEMO_FAMS = ["none", "C1", "C2f", "C2ff", "C4ff"]
LABEL = {"none": "No stimulation", "C0": "Fixed amplitude", "C1": "Rule-based (Christie-style)", "C2": "PID",
         "C2f": "PID (clinical)", "C2ff": "PID + posture", "C3": "LQI", "C3ff": "LQI + posture",
         "C4": "MPC", "C4ff": "MPC + posture"}
SCEN = {"base": {}, "imu_dropout30": R.VARIANTS["sensor"]["imu_dropout30"]}
# site palette
INK, TEAL, AMBER, GREY, PAPER = "#15232D", "#0B6B66", "#B7791F", "#8A9399", "#F5F7F6"
COL = {"none": "#C3C9CC", "C1": "#6B7B86", "C2": "#A7B0B5", "C2f": INK, "C2ff": AMBER, "C3": "#7FB3AE",
       "C3ff": "#4F9A93", "C4": "#3D6E8F", "C4ff": TEAL}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#55626B",
                         "axes.labelcolor": INK, "xtick.color": "#55626B", "ytick.color": "#55626B",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titleweight": "regular", "legend.frameon": False, "savefig.dpi": 160,
                         "savefig.bbox": "tight"})
    return plt


# ------------------------------------------------------------------ demo traces
def trace(sim, pop, base, pid, fam, tuned, pinf, v):
    p = pop["patients"][pid]
    th = sci_theta(base, **p["plant"])
    act = EESActuator(**p["actuator"])
    ctrl = H.NoStim() if fam == "none" else C.make(fam, tuned[fam])
    angle_fn = v.get("angle", H.angle_fn)
    if "imu" in v:
        ctrl = R.IMU(ctrl, *v["imu"], angle_fn=angle_fn, seed=7 + pid)
    ctrl.reset(pinf)
    o = simulate(sim, th, H.T_END, angle_fn, lambda t, obs: H.quantise(ctrl(t, obs)), actuator=act,
                 sbp_noise_sd=H.NOISE_SD, rng=np.random.default_rng(1000 + pid),
                 dist_fn=H.make_dist_fn(pop["ees_gain_nominal"]), dropout_fn=H.dropout_fn)
    return o, H.metrics(o)


def build_demo():
    os.makedirs(DEMO, exist_ok=True)
    pop, base, info = H.load_population()
    sysid = json.load(open(os.path.join("data", "c10b_sysid.json")))
    tuned, _ = load_tuned()
    sim = new_sim()
    import ctypes
    sim.lib.cv_set_refill.argtypes = [ctypes.c_double]
    sim.lib.cv_set_refill(1.0)
    idx, t0 = [], time.time()
    for pid in TEST:
        pinf = patient_info(info, sysid, pid)
        rec = {"id": pid, "scenarios": {}}
        for sn, v in SCEN.items():
            rec["scenarios"][sn] = {}
            for fam in DEMO_FAMS:
                o, m = trace(sim, pop, base, pid, fam, tuned, pinf, v)
                k = slice(1, None, 2)
                if "t" not in rec:
                    rec["t"] = [int(x) for x in o["t"][k]]
                    rec["angle"] = [round(float(x), 1) for x in o["angle"][k]]
                rec["scenarios"][sn][fam] = {
                    "sbp": [round(float(x), 1) for x in o["sbp"][k]],
                    "u": [round(float(x), 2) for x in o["u"][k]],
                    "m": {kk: round(float(m[kk]), 2) for kk in ("in_range", "hyper", "hypo", "hyper_return", "n_changes", "ad_peak")}}
        act = pop["patients"][pid]["actuator"]
        meta = {"id": pid, "ees_u1_dsbp": round(float(info.loc[pid, "ees_u1_dsbp"]), 1),
                "delay_s": round(float(act["delay_s"]), 1), "tau_s": round(float(act["tau_s"]), 1),
                "supine_sbp": round(float(info.loc[pid, "supine_sbp"]), 0)}
        rec["meta"] = meta
        idx.append(meta)
        json.dump(rec, open(os.path.join(DEMO, f"p{pid}.json"), "w"), separators=(",", ":"))
        print(f"[demo] patient {pid} done ({time.time() - t0:.0f} s)")
    json.dump({"patients": idx, "families": DEMO_FAMS, "labels": {f: LABEL[f] for f in DEMO_FAMS},
               "scenarios": {"base": "Posture sensor working",
                             "imu_dropout30": "Posture sensor frozen at 60° for 30 s after tilt-down"},
               "target": [H.LO, H.HI], "ceiling": 150, "dropouts": H.DROPOUTS,
               "ad": [H.AD[0], H.AD[0] + sum(H.AD[1:])]},
              open(os.path.join(DEMO, "index.json"), "w"), indent=1)


# ------------------------------------------------------------------ figures
def fig1(plt, cap):
    import c10b_step1b_verify as V
    import c10b_step2_sci_population as S2
    sim = new_sim()
    th_cal = np.load(os.path.join("data", "c10b_theta_calibrated.npy"))
    env = pd.read_csv(os.path.join("results", "c10b_step1a", "envelopes.csv"))
    tr = V.scenario(sim, th_cal, "rapid_tilt_up", V.prcp_transition_s("rapid_tilt_up", 2.0))
    pop = json.load(open(os.path.join("data", "c10b_population.json")))
    th_sci = sci_theta(th_cal, **pop["sci_nominal"])
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
    for a_, sig, lab in ((ax[0], "map", "ΔMAP (mmHg)"), (ax[1], "hr", "ΔHR (bpm)")):
        e = env[(env["type"] == "rapid_tilt_up") & (env["signal"] == sig)].sort_values("t_s")
        a_.fill_between(e["t_s"], e["q25"], e["q75"], color=GREY, alpha=0.3, lw=0, label="PRCP, 10 healthy (IQR)")
        a_.plot(e["t_s"], e["median"], color=GREY, lw=1, label="PRCP median")
        a_.plot(V.T, tr[sig], color=TEAL, lw=1.8, label="model")
        a_.axvline(0, color="0.7", lw=0.7, ls=":")
        a_.set_xlabel("time from rapid tilt (s)"); a_.set_ylabel(lab)
    ax[0].set_title("Healthy model vs human tilt data"); ax[0].legend(fontsize=7, loc="lower right")
    rows = []
    for a in (30, 45, 60):
        r = S2.tilt_response(sim, th_sci, a)
        s0, s1, h0, h1 = S2.CORBETT[a]
        rows.append((a, r["dmap"], s1 - s0, r["dhr"], h1 - h0))
    x = np.arange(3)
    for a_, j, k, lab in ((ax[2], 1, 2, "ΔMAP at plateau (mmHg)"), (ax[3], 3, 4, "ΔHR at plateau (bpm)")):
        a_.bar(x - 0.18, [r[k] for r in rows], 0.34, color=GREY, label="Corbett et al. 1971 (group mean)")
        a_.bar(x + 0.18, [r[j] for r in rows], 0.34, color=TEAL, label="model")
        a_.set_xticks(x, ["30°", "45°", "60° (held out)"]); a_.set_ylabel(lab); a_.axhline(0, color="0.6", lw=0.6)
    ax[2].set_title("Spinal-cord-injury model vs published tilt responses"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_credibility.png")); plt.close(fig)
    cap.append("**Fig 1.** Left: model response to rapid head-up tilt vs the PRCP database (10 healthy adults, "
               "PhysioNet). Right: complete cervical SCI model vs Corbett et al. 1971 at 30/45° (fitted) and 60° "
               "(held out; " + ", ".join(f"{a}°: ΔMAP {m:+.0f} vs {t:+d}, ΔHR {hm:+.0f} vs {ht:+d}" for a, m, t, hm, ht in rows) + ").")


def pick_example():
    d = pd.read_csv(os.path.join("results", "c10b_step4c", "test_per_patient.csv"))
    ref = d[(d["family"] == "C2f") & (d["hyper_return"] > 0)].sort_values("hyper_return")
    return int(ref.iloc[len(ref) // 2]["id"]) if len(ref) else TEST[len(TEST) // 2]


def fig2(plt, cap):
    pid = pick_example()
    rec = json.load(open(os.path.join(DEMO, f"p{pid}.json")))
    t, ang = np.array(rec["t"]), np.array(rec["angle"])
    sc = rec["scenarios"]["base"]
    fams = ["C2f", "C2ff", "C4ff"]
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.6, 1], height_ratios=[2.2, 1.2, 0.8], hspace=0.12, wspace=0.12)
    for col, (lo, hi) in enumerate(((0, 1800), (880, 1060))):
        a0 = fig.add_subplot(gs[0, col]); a1 = fig.add_subplot(gs[1, col], sharex=a0); a2 = fig.add_subplot(gs[2, col], sharex=a0)
        m = (t >= lo) & (t <= hi)
        a0.axhspan(H.LO, H.HI, color=TEAL, alpha=0.10, lw=0)
        a0.axhline(150, color="#B23A3A", lw=0.8, ls="--")
        for f in fams:
            a0.plot(t[m], np.array(sc[f]["sbp"])[m], color=COL[f], lw=1.3 if f != "C2f" else 1.1, label=LABEL[f])
            a1.plot(t[m], np.array(sc[f]["u"])[m], color=COL[f], lw=1.0, drawstyle="steps-post")
        a2.fill_between(t[m], 0, ang[m], color=GREY, alpha=0.5, lw=0)
        for a_ in (a0, a1, a2):
            a_.set_xlim(lo, hi)
            for p, q in H.DROPOUTS:
                if q > lo and p < hi:
                    a_.axvspan(max(p, lo), min(q, hi), color="0.9", lw=0)
        plt.setp(a0.get_xticklabels(), visible=False); plt.setp(a1.get_xticklabels(), visible=False)
        a2.set_xlabel("time (s)")
        if col == 0:
            a0.set_ylabel("systolic BP (mmHg)"); a1.set_ylabel("stim. amplitude"); a2.set_ylabel("tilt (°)")
            a0.legend(fontsize=8, ncol=3, loc="lower right")
            a0.set_title(f"Virtual test patient {pid}: 60° tilt, return, 30° tilt, dysreflexia-like surge", loc="left")
        else:
            a0.set_title("Return to supine (zoom)", loc="left")
    fig.savefig(os.path.join(OUT, "fig2_example.png")); plt.close(fig)
    mm = {f: sc[f]["m"] for f in fams}
    cap.append(f"**Fig 2.** Test patient {pid} (chosen by rule: median PID time > 150 mmHg on return among test "
               "patients with any). Green band: 110–120 mmHg target; dashed: 150 mmHg; grey: BP sensor dropouts. "
               + "; ".join(f"{LABEL[f]}: {mm[f]['hyper_return']:.1f} % of the return phase > 150, "
                           f"{mm[f]['in_range']:.0f} % in range" for f in fams) + ".")


def fig3(plt, cap):
    d = pd.read_csv(os.path.join("results", "c10b_step4c", "test_per_patient.csv"))
    ref = d[d["family"] == "C2f"].set_index("id")
    fams = ["C2ff", "C3ff", "C4", "C4ff"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a_, k, lab in ((ax[0], "hyper", "% time > 150 mmHg"), (ax[1], "in_range", "% time in 110–120 mmHg")):
        a_.boxplot([ref[k]] + [d[d["family"] == f].set_index("id").loc[ref.index][k] for f in fams],
                   positions=range(5), widths=0.35, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": "#9AA4AA"}, whiskerprops={"color": "#9AA4AA"},
                   capprops={"color": "#9AA4AA"})
        for j, f in enumerate(fams, start=1):
            y = d[d["family"] == f].set_index("id").loc[ref.index][k]
            for pid in ref.index:
                a_.plot([0, j], [ref.loc[pid, k], y.loc[pid]], color=COL[f], alpha=0.18, lw=0.7)
            a_.scatter(np.full(len(y), j), y, s=10, color=COL[f], zorder=3)
        a_.scatter(np.zeros(len(ref)), ref[k], s=10, color=COL["C2f"], zorder=3)
        a_.set_xticks(range(5), [LABEL["C2f"]] + [LABEL[f] for f in fams], rotation=20, ha="right")
        a_.set_ylabel(lab)
    ax[0].set_title("Hypertensive time: posture feedforward does the work", loc="left")
    ax[1].set_title("Time in range: no controller moves it much", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_paired.png")); plt.close(fig)
    cap.append("**Fig 3.** 20 held-out virtual patients; lines join the same patient. Means % > 150: " +
               ", ".join(f"{LABEL[f]} {d[d['family'] == f]['hyper'].mean():.2f}" for f in ["C2f"] + fams) +
               ". Means % in range: " + ", ".join(f"{LABEL[f]} {d[d['family'] == f]['in_range'].mean():.1f}" for f in ["C2f"] + fams) + ".")


def fig4(plt, cap):
    d = pd.read_csv(os.path.join("results", "c10b_step4c", "cal_sweep.csv"))
    ref = d[d["cal_min"] == 0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
    for a_, k, lab in ((ax[0], "hyper", "% time > 150 mmHg"), (ax[1], "in_range", "% time in range")):
        a_.axhline(ref[k].mean(), color=COL["C2f"], ls="--", lw=1, label=f"{LABEL['C2f']} (no model)")
        for f in ("C4", "C4ff"):
            g = d[d["family"] == f].groupby("cal_min")[k]
            a_.errorbar(g.mean().index, g.mean(), yerr=g.sem(), color=COL[f], marker="o", ms=4, lw=1.4, capsize=2, label=LABEL[f])
        a_.set_xscale("log"); a_.set_xticks([5, 10, 20, 60], ["5", "10", "20", "60"])
        a_.set_xlabel("calibration session (min)"); a_.set_ylabel(lab)
    ax[0].legend(fontsize=7); ax[0].set_title("How much calibration does a model need?", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_calibration.png")); plt.close(fig)
    g = d[d["family"] == "C4ff"].groupby("cal_min")["hyper"].mean()
    cap.append("**Fig 4.** Mean ± SEM over 20 test patients; model identified from a simulated calibration "
               "session of the given length. MPC + posture % > 150: " + ", ".join(f"{int(L)} min {v:.2f}" for L, v in g.items()) + ".")


def fig5(plt, cap):
    d = pd.read_csv(os.path.join("results", "c10b_step5", "rows.csv"))
    names = {"base": "Baseline", "tiltdown_2s": "Tilt-down in 2 s", "tiltdown_30s": "Tilt-down in 30 s",
             "refill_x0.5": "Faster fluid return", "refill_x2": "Slower fluid return",
             "imu_bias+5": "Posture sensor +5° bias", "imu_bias-5": "Posture sensor −5° bias",
             "imu_noise3": "Posture sensor noise 3°", "imu_delay1": "Posture sensor 1 s delay",
             "imu_delay3": "Posture sensor 3 s delay", "imu_dropout30": "Posture sensor frozen 30 s",
             "bp_noise4": "BP noise doubled", "bp_dropout_long": "Long BP dropouts",
             "gain-30": "Stimulation response −30 %", "gain+30": "Stimulation response +30 %"}
    order = [v for g in R.VARIANTS.values() for v in g if v in set(d["variant"])]
    fams = ["C2ff", "C3ff", "C4", "C4ff"]
    M = np.full((len(order), len(fams)), np.nan)
    for i, v in enumerate(order):
        piv = d[d["variant"] == v].groupby("family")["hyper"].mean()
        for j, f in enumerate(fams):
            M[i, j] = 100 * (1 - piv[f] / piv["C2f"]) if piv["C2f"] > 0 else np.nan
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("site", ["#B23A3A", "#FFFFFF", TEAL])
    fig, ax = plt.subplots(figsize=(6.2, 0.33 * len(order) + 1.2))
    ax.imshow(M, cmap=cm, vmin=-100, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 60 else INK)
    ax.set_xticks(range(len(fams)), [LABEL[f] for f in fams], rotation=20, ha="right")
    ax.set_yticks(range(len(order)), [names.get(v, v) for v in order])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("% less time > 150 mmHg than clinical PID", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_robustness.png")); plt.close(fig)
    i = order.index("imu_dropout30") if "imu_dropout30" in order else None
    cap.append("**Fig 5.** Nothing re-tuned for any condition. The benefit holds under changes to the "
               "least-validated phase (tilt-down speed, fluid return), sensor bias/noise/delay and BP noise/dropouts, "
               "and is lost when the posture sensor freezes at 60° after tilt-down" +
               (f" (PID + posture {M[i, 0]:.0f} %, MPC + posture {M[i, 3]:.0f} %)" if i is not None else "") +
               "; LQI + posture is also sensitive to angle noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.figs or not os.path.exists(os.path.join(DEMO, "index.json")):
        build_demo()
    plt = style()
    cap = ["# C10-B figure captions (numbers filled from the result files)\n"]
    for f in (fig1, fig2, fig3, fig4, fig5):
        try:
            f(plt, cap)
            print(f"[fig] {f.__name__} done")
        except Exception as e:  # noqa: BLE001
            print(f"[fig] {f.__name__} FAILED: {type(e).__name__}: {e}")
    open(os.path.join(OUT, "captions.md"), "w").write("\n\n".join(cap) + "\n")
    print(f"[done] {OUT}/ (figures, captions.md, demo/)")


if __name__ == "__main__":
    main()
