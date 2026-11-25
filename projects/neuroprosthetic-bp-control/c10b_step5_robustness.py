"""
C10-B step 5 — does the posture-feedforward effect survive the plant's weakest phase,
sensor error and model mismatch?  (design fixed in the plan doc, §5c, before any run)

Usage:  python3 c10b_step5_robustness.py --group plant     (~30 min)
        python3 c10b_step5_robustness.py --group sensor    (~45 min)
        python3 c10b_step5_robustness.py --group robust    (~30 min)
        python3 c10b_step5_robustness.py --report          (tables, figure, G2; seconds)

Families: C2f (PID reference), C2ff (PID + feedforward), C3ff, C4 (MPC without ff), C4ff,
all with their step 4 tuned parameters and the 20-min calibration models; test patients
40–59; nothing is re-tuned for any variant (the controllers do not know the variant).

Variants
  plant : base | tilt-down fast (2 s ramps) | tilt-down slow (30 s ramps) |
          fluid return ×0.5 (faster refill) | fluid return ×2 (slower refill)
  sensor: IMU bias +5° | IMU bias −5° | IMU noise SD 3° | IMU delay 1 s | IMU delay 3 s |
          IMU dropout 30 s at the 60° tilt-down (angle held at 60°)
  robust: SBP noise SD 4 | long BP dropouts (60 s incl. one at tilt-down) |
          actuator gain −30 % | actuator gain +30 % (after calibration; model not updated)

G2 (pre-specified): the feedforward claim stands if C4ff (and separately C3ff, C2ff) keeps
≥ 50 % relative reduction of % time > 150 mmHg vs C2f in every variant, IMU delay 3 s
excepted. Only variants where C2f has ≥ 0.2 % time > 150 are judged (otherwise "too rare").

Results are appended to results/c10b_step5/rows.csv; a rerun skips finished episodes.
Needs the fluid-return hook: rebuild once with
  python3 -c "import c10b_cvsim as c; c.build(force=True)"
"""

import argparse
import ctypes
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import c10b_controllers as C
import c10b_harness as H
from c10b_plant import EESActuator, sci_theta, simulate, new_sim
from c10b_step4b_tune import patient_info
from c10b_step4c_evaluate import load_tuned

OUT = os.path.join("results", "c10b_step5")
ROWS = os.path.join(OUT, "rows.csv")
TEST = list(range(40, 60))
FAMS = ["C2f", "C2ff", "C3ff", "C4", "C4ff"]
REF = "C2f"
FF_FAMS = ["C2ff", "C3ff", "C4ff"]
G2_RR = 0.50
MIN_REF_HYP = 0.2


def scaled_return(fast_s):
    """ANGLE_BP with both tilt-down ramps replaced by ramps of fast_s seconds."""
    bp = [(0, 0), (300, 0), (310, 60), (900, 60), (900 + fast_s, 0), (1200, 0), (1240, 30), (1500, 30),
          (1500 + fast_s, 0), (1800, 0)]
    ts, a = zip(*bp)
    return lambda t: float(np.interp(t, ts, a))


VARIANTS = {
    "plant": {"base": {}, "tiltdown_2s": {"angle": scaled_return(2)}, "tiltdown_30s": {"angle": scaled_return(30)},
              "refill_x0.5": {"refill": 0.5}, "refill_x2": {"refill": 2.0}},
    "sensor": {"imu_bias+5": {"imu": ("bias", 5.0)}, "imu_bias-5": {"imu": ("bias", -5.0)},
               "imu_noise3": {"imu": ("noise", 3.0)}, "imu_delay1": {"imu": ("delay", 1.0)},
               "imu_delay3": {"imu": ("delay", 3.0)}, "imu_dropout30": {"imu": ("hold", (900.0, 930.0))}},
    "robust": {"bp_noise4": {"noise": 4.0}, "bp_dropout_long": {"dropouts": [(600, 660), (905, 965), (1350, 1380)]},
               "gain-30": {"gain_x": 0.7}, "gain+30": {"gain_x": 1.3}},
}


class IMU:
    """Wraps a controller; corrupts only the angle the controller sees."""

    def __init__(self, inner, kind, val, angle_fn, seed):
        self.inner, self.kind, self.val, self.angle_fn = inner, kind, val, angle_fn
        self.rng = np.random.default_rng(seed)

    def reset(self, info=None):
        self.inner.reset(info)

    def __call__(self, t, obs):
        a = obs["angle"]
        if self.kind == "bias":
            a = a + self.val if a > 1.0 else a          # bias only when tilted (strapped-on IMU offset)
        elif self.kind == "noise":
            a = a + self.rng.normal(0, self.val)
        elif self.kind == "delay":
            a = self.angle_fn(max(t - self.val, 0.0))
        elif self.kind == "hold":
            t0, t1 = self.val
            if t0 <= t < t1:
                a = self.angle_fn(t0)
        o = dict(obs)
        o["angle"] = float(np.clip(a, 0, 90))
        return self.inner(t, o)


def run(sim, pop, base, pid, ctrl, info, v):
    p = pop["patients"][pid]
    th = sci_theta(base, **p["plant"])
    ak = dict(p["actuator"])
    ak["gain"] = ak["gain"] * v.get("gain_x", 1.0)
    act = EESActuator(**ak)
    angle_fn = v.get("angle", H.angle_fn)
    if "imu" in v:
        ctrl = IMU(ctrl, *v["imu"], angle_fn=angle_fn, seed=7 + pid)
    dropouts = v.get("dropouts", H.DROPOUTS)
    sim.lib.cv_set_refill(float(v.get("refill", 1.0)))
    ctrl.reset(info)
    o = simulate(sim, th, H.T_END, angle_fn, lambda t, obs: H.quantise(ctrl(t, obs)), actuator=act,
                 sbp_noise_sd=v.get("noise", H.NOISE_SD), rng=np.random.default_rng(1000 + pid),
                 dist_fn=H.make_dist_fn(pop["ees_gain_nominal"]),
                 dropout_fn=lambda t: any(a <= t < b for a, b in dropouts))
    sim.lib.cv_set_refill(1.0)
    return H.metrics(o)


def run_group(group):
    os.makedirs(OUT, exist_ok=True)
    pop, base, info = H.load_population()
    sysid = json.load(open(os.path.join("data", "c10b_sysid.json")))
    tuned, _ = load_tuned()
    miss = [f for f in FAMS if f not in tuned]
    if miss:
        raise SystemExit(f"Tune first: {miss} (python3 c10b_step4b_tune.py <family>)")
    sim = new_sim()
    try:
        sim.lib.cv_set_refill.argtypes = [ctypes.c_double]
    except AttributeError:
        raise SystemExit('Rebuild needed: python3 -c "import c10b_cvsim as c; c.build(force=True)"')
    done = set()
    if os.path.exists(ROWS):
        d = pd.read_csv(ROWS)
        done = set(zip(d["variant"], d["family"], d["id"]))
    t0, n = time.time(), 0
    for vname, v in VARIANTS[group].items():
        for pid in TEST:
            pinf = patient_info(info, sysid, pid)
            rows = []
            for fam in FAMS:
                if (vname, fam, pid) in done:
                    continue
                m = run(sim, pop, base, pid, C.make(fam, tuned[fam]), pinf, v)
                m["J"] = C.cost_J(m)
                rows.append({"group": group, "variant": vname, "family": fam, "id": pid, **m})
                n += 1
            if rows:
                pd.DataFrame(rows).to_csv(ROWS, mode="a", header=not os.path.exists(ROWS), index=False)
        print(f"[{group}] {vname} done — {n} episodes, {time.time() - t0:.0f} s")
    report()


def report():
    d = pd.read_csv(ROWS)
    order = [v for g in VARIANTS.values() for v in g if v in set(d["variant"])]
    lines = ["# C10-B step 5 — robustness of the posture-feedforward effect\n",
             f"Test patients 40–59; reference {REF} (tuned PID + prefilter). RR = relative reduction of "
             f"% time > 150 mmHg vs {REF}; G2 needs RR ≥ {int(100 * G2_RR)} % for the ff controllers in every "
             f"judged variant except imu_delay3 (judged only if {REF} ≥ {MIN_REF_HYP} % time > 150).\n",
             "| variant | " + " | ".join(f"{f} % >150" for f in FAMS) + " | " +
             " | ".join(f"RR {f}" for f in FAMS if f != REF) + " | in range " + " / ".join(FAMS) + " |",
             "|---" * (1 + len(FAMS) + len(FAMS) - 1 + 1) + "|"]
    verdict = {f: True for f in FF_FAMS}
    rr_tab = {}
    for v in order:
        dv = d[d["variant"] == v]
        piv = dv.pivot_table(index="id", columns="family", values="hyper")
        ir = dv.groupby("family")["in_range"].mean()
        ref = piv[REF].mean() if REF in piv else np.nan
        rr = {f: (1 - piv[f].mean() / ref) if ref > 0 else np.nan for f in FAMS if f != REF and f in piv}
        rr_tab[v] = rr
        judged = ref >= MIN_REF_HYP and v != "imu_delay3"
        for f in FF_FAMS:
            if judged and f in rr and not (rr[f] >= G2_RR):
                verdict[f] = False
        lines.append(f"| {v}{'' if judged else ' (not judged)'} | " +
                     " | ".join(f"{piv[f].mean():.2f}" if f in piv else "—" for f in FAMS) + " | " +
                     " | ".join(f"{100 * rr[f]:+.0f} %" if f in rr and np.isfinite(rr[f]) else "—" for f in FAMS if f != REF) +
                     " | " + " / ".join(f"{ir.get(f, np.nan):.1f}" for f in FAMS) + " |")
    # 5a: PID+ff vs MPC+ff, pooled over judged plant variants and per variant
    lines += ["\n## 5a — PID + feedforward vs model-based + feedforward (paired, % time > 150 and in range)\n",
              "| variant | C2ff vs C4ff: mean % >150 | Wilcoxon p | C2ff vs C4ff: in range | Wilcoxon p |", "|---|---|---|---|---|"]
    for v in order:
        dv = d[d["variant"] == v]
        if not {"C2ff", "C4ff"} <= set(dv["family"]):
            continue
        a = dv[dv["family"] == "C2ff"].set_index("id").sort_index()
        b = dv[dv["family"] == "C4ff"].set_index("id").sort_index()
        ps = []
        for k in ("hyper", "in_range"):
            try:
                ps.append(wilcoxon(a[k], b[k]).pvalue)
            except ValueError:
                ps.append(np.nan)
        lines.append(f"| {v} | {a['hyper'].mean():.2f} vs {b['hyper'].mean():.2f} | {ps[0]:.3g} | "
                     f"{a['in_range'].mean():.1f} vs {b['in_range'].mean():.1f} | {ps[1]:.3g} |")
    lines.append("\n**G2:** " + ", ".join(f"{f} {'PASS' if verdict[f] else 'FAIL'}" for f in FF_FAMS) +
                 (" (incomplete: not all groups run)" if len(order) < sum(len(g) for g in VARIANTS.values()) else ""))
    open(os.path.join(OUT, "step5.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f in FAMS if f != REF]
    M = np.array([[100 * rr_tab[v].get(f, np.nan) for f in fams] for v in order])
    fig, ax = plt.subplots(figsize=(1.3 * len(fams) + 3, 0.4 * len(order) + 1.5))
    im = ax.imshow(M, cmap="RdBu", vmin=-100, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(fams)), fams); ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_title(f"% reduction of time > 150 mmHg vs {REF}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_robustness.png"), dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(VARIANTS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report or not a.group:
        report()
    else:
        run_group(a.group)


if __name__ == "__main__":
    main()
