"""
C10-B step 1b — build the CVSim plant, verify it, validate it against PRCP (G0).

1. Build    : CVSim 21-compartment C code + 3 asserted patches + c10b_cvsim_ext.c
2. Baseline : supine steady state and drift (300 s)
3. Verify   : external-angle tilt (mode 1) vs CVSim's original tilt routine (mode 0),
              75 deg, 2 s raised-cosine, 180 s upright. Expected: equal in supine,
              tilt-up and upright phases; different after tilt-down by design
              (CVSim returns the sequestered volume within the 2 s down-ramp,
              mode 1 uses the same 276 s lag in both directions).
4. Validate : nominal CVSim vs PRCP event-locked envelopes (step 1a) for rapid and
              slow tilt up and down (stand-up shown, but not part of G0: CVSim has
              no muscle pump / active-standing vasodilation).
5. Timing   : wall time of a 30 min episode (sets the compute budget of later steps)

G0 rules (written before the run):
  - verification: tilt-up and upright phases max |dMAP| <= 2 mmHg and |dHR| <= 2 bpm
  - validation (rapid and slow tilt up; MAP and HR only, because Finapres measures
    finger pressure and the model reports aortic pressure, so SBP is shown but not scored):
      * each scored metric within median +/- 1.5 IQR of the PRCP subject distribution
      * >= 70 % of 1 s points (0-180 s) inside [q25 - IQR, q75 + IQR] of the envelope
  If validation fails, step 1c calibrates a few reflex parameters to PRCP (BO), then re-tests.

Outputs: results/c10b_step1b/verification.md, fig_equivalence.png, fig_validation.png,
         model_traces.csv
"""

import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c10b_cvsim import CVSim21, beats_to_grid, raised_cosine_profile

OUT = os.path.join("results", "c10b_step1b")
S1A = os.path.join("results", "c10b_step1a")
os.makedirs(OUT, exist_ok=True)

SETTLE = 300.0
T = np.arange(-60, 241)          # 1 Hz event-locked grid, same as step 1a
BASE = (T >= -60) & (T <= -10)
EARLY = (T >= 0) & (T <= 40)
LATE = (T >= 120) & (T <= 180)
SCORE = (T >= 0) & (T <= 180)
ANGLE = 75.0


# ------------------------------------------------------------------ helpers
def prcp_transition_s(kind, default):
    p = os.path.join(S1A, "events.csv")
    if not os.path.exists(p):
        return default
    ev = pd.read_csv(p)
    v = ev[(ev["type"] == kind) & (ev["status"] == "ok")]["transition_s"].dropna()
    return float(v.median()) if len(v) else default


def ramp(sim, t0, dur, a_from, a_to, shape):
    if shape == "cosine":
        return raised_cosine_profile(sim, t0, dur, a_from, a_to)
    return sim.run(t0 + dur, angle_end=a_to)


def scenario(sim, th, kind, dur):
    """Returns event-locked dict of deltas on T plus absolute baselines."""
    sim.init(th, mode=1)
    up = kind.endswith("_up") or kind == "stand_up"
    shape = "cosine" if kind.startswith("rapid") or kind == "stand_up" else "linear"
    target = 90.0 if kind == "stand_up" else ANGLE
    b = [sim.run(SETTLE, angle_end=0.0)]
    if up:
        t0 = SETTLE
        b.append(ramp(sim, t0, dur, 0.0, target, shape))
        b.append(sim.run(t0 + 245.0))
    else:
        b.append(ramp(sim, SETTLE, 2.0, 0.0, ANGLE, "cosine"))   # get upright first
        t0 = SETTLE + 2.0 + 190.0
        b.append(sim.run(t0))
        b.append(ramp(sim, t0, dur, ANGLE, 0.0, shape))
        b.append(sim.run(t0 + 245.0))
    beats = np.vstack([x for x in b if len(x)])
    g = beats_to_grid(beats, t0 + T)
    out = {}
    for s_model, s in (("sbp", "sbp"), ("map", "map"), ("hr", "hr")):
        y = g[s_model]
        base = y[BASE].mean()
        out[s] = y - base
        out[s + "_base"] = base
    return out


def parse_metrics_md():
    p = os.path.join(S1A, "metrics.md")
    if not os.path.exists(p):
        return {}
    rows = {}
    pat = re.compile(r"\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[-\d.]+\s*\|\s*"
                     r"([-\d.]+) \[([-\d.]+), ([-\d.]+)\]\s*\|\s*[-\d.]+\s*\|\s*"
                     r"([-\d.]+) \[([-\d.]+), ([-\d.]+)\]")
    for line in open(p):
        m = pat.search(line)
        if m:
            ev, sig = m.group(1), m.group(2)
            v = [float(m.group(i)) for i in range(3, 9)]
            rows[(ev, sig)] = {"early": v[0:3], "steady": v[3:6]}
    return rows


def model_metrics(tr, sig):
    y = tr[sig]
    ext = np.max(y[EARLY]) if sig == "hr" else np.min(y[EARLY])
    return {"early": float(ext), "steady": float(np.mean(y[LATE]))}


def within_tukey(val, med, q25, q75, k=1.5):
    iqr = max(q75 - q25, 1e-6)
    return (med - k * iqr) <= val <= (med + k * iqr)


# ------------------------------------------------------------------ main
def main(theta=None, out=OUT, label="CVSim default (initial.c + mapping_ptr, as used by CVSim's main and GUI)"):
    os.makedirs(out, exist_ok=True)
    lines = ["# C10-B — CVSim plant: verification and validation (G0)\n"]
    t_build = time.time()
    sim = CVSim21(force_build=(theta is None))
    th = sim.default_theta() if theta is None else np.asarray(theta, dtype=float)
    lines.append(f"Build + load: {time.time() - t_build:.1f} s. Parameters: {label}. Both reflexes on.\n")

    # ---- 2. baseline
    sim.init(th, mode=1)
    t0 = time.time()
    b = sim.run(SETTLE, angle_end=0.0)
    wall300 = time.time() - t0
    late = b[b[:, 0] > 250, 1:].mean(0)
    prev = b[(b[:, 0] > 200) & (b[:, 0] <= 250), 1:].mean(0)
    lines += ["## Baseline (supine)\n",
              f"Mean over 250–300 s: SAP {late[0]:.1f}, MAP {late[1]:.1f}, DAP {late[2]:.1f} mmHg, HR {late[3]:.1f} bpm. "
              f"Drift vs 200–250 s: MAP {late[1] - prev[1]:+.2f} mmHg, HR {late[3] - prev[3]:+.2f} bpm.\n"]

    # ---- 5. timing
    sim.init(th, mode=1)
    t0 = time.time()
    sim.run(1800.0, angle_end=0.0)
    wall1800 = time.time() - t0
    lines += ["## Timing\n", f"300 s simulated: {wall300:.2f} s wall. 30 min simulated: {wall1800:.1f} s wall.\n"]

    # ---- 3. verification: mode 0 vs mode 1
    th_v = th.copy()
    th_v[91], th_v[92], th_v[94] = ANGLE, 2.0, 180.0
    ts, tstop, tend = SETTLE, SETTLE + 2.0 + 180.0, SETTLE + 400.0
    sim.init(th_v, mode=0, t_start=ts, t_stop=tstop)
    b0 = sim.run(tend)
    sim.init(th_v, mode=1)
    b1 = np.vstack([sim.run(ts, angle_end=0.0), raised_cosine_profile(sim, ts, 2.0, 0.0, ANGLE),
                    sim.run(tstop), raised_cosine_profile(sim, tstop, 2.0, ANGLE, 0.0), sim.run(tend)])
    g = np.arange(ts - 60, tend, 0.5)
    o, e = beats_to_grid(b0, g), beats_to_grid(b1, g)
    lines += ["## Verification: external-angle tilt vs CVSim's original tilt\n",
              "| phase | max abs dSAP | max abs dMAP | max abs dHR | scored |", "|---|---|---|---|---|"]
    ver_ok = True
    for name, lo, hi, scored in [("supine", ts - 60, ts, True), ("tilt up + upright", ts, tstop, True),
                                 ("tilt down + supine", tstop, tend, False)]:
        m = (g >= lo) & (g < hi)
        d = {k: np.abs(e[k] - o[k])[m].max() for k in ("sbp", "map", "hr")}
        ok = d["map"] <= 2.0 and d["hr"] <= 2.0
        if scored:
            ver_ok &= ok
        lines.append(f"| {name} | {d['sbp']:.2f} | {d['map']:.2f} | {d['hr']:.2f} | "
                     f"{('PASS' if ok else 'FAIL') if scored else 'by design (see header)'} |")
    lines.append(f"\nVerification: **{'PASS' if ver_ok else 'FAIL'}**\n")

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for a_, k, lab in ((ax[0], "map", "MAP (mmHg)"), (ax[1], "hr", "HR (bpm)")):
        a_.plot(g, o[k], lw=1.4, label="CVSim original tilt")
        a_.plot(g, e[k], lw=1.0, ls="--", label="external-angle tilt")
        a_.axvspan(ts, tstop, color="0.9")
        a_.set_ylabel(lab)
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel("time (s); grey = upright 75°")
    fig.suptitle("Verification: tilt implementations")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_equivalence.png"), dpi=140)
    plt.close(fig)

    # ---- 4. validation vs PRCP
    env_p = os.path.join(S1A, "envelopes.csv")
    if not os.path.exists(env_p):
        lines.append("PRCP envelopes not found — run step 1a first. Validation skipped.\n")
        open(os.path.join(out, "verification.md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return ver_ok, False
    env = pd.read_csv(env_p)
    met = parse_metrics_md()
    kinds = {"rapid_tilt_up": prcp_transition_s("rapid_tilt_up", 2.0),
             "slow_tilt_up": prcp_transition_s("slow_tilt_up", 50.0),
             "rapid_tilt_down": prcp_transition_s("rapid_tilt_down", 2.0),
             "slow_tilt_down": prcp_transition_s("slow_tilt_down", 50.0),
             "stand_up": 1.0}
    scored = {"rapid_tilt_up", "slow_tilt_up"}
    traces, rows = {}, []
    lines += ["## Validation vs PRCP (Δ from the −60…−10 s baseline)\n",
              "Transition durations taken from the PRCP Angle channel (median): " +
              ", ".join(f"{k} {v:.1f} s" for k, v in kinds.items() if k != "stand_up") +
              ". Stand-up approximated as a 1 s tilt to 90° (not scored).\n",
              "| event | signal | model early | PRCP early median [IQR] | ok | model steady | PRCP steady median [IQR] | ok | envelope coverage 0–180 s | ok |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    g0_ok = True
    csv_rows = []
    for kind, dur in kinds.items():
        tr = scenario(sim, th, kind, dur)
        traces[kind] = tr
        for i, t in enumerate(T):
            csv_rows.append((kind, int(t), tr["sbp"][i], tr["map"][i], tr["hr"][i]))
        for sig in ("sbp", "map", "hr"):
            e_ = env[(env["type"] == kind) & (env["signal"] == sig)].sort_values("t_s")
            if e_.empty:
                continue
            med, q25, q75 = (np.interp(T, e_["t_s"], e_[c]) for c in ("median", "q25", "q75"))
            iqr = q75 - q25
            inside = (tr[sig] >= q25 - iqr) & (tr[sig] <= q75 + iqr)
            cov = float(inside[SCORE].mean())
            mm = model_metrics(tr, sig)
            pm = met.get((kind, sig))
            is_scored = kind in scored and sig in ("map", "hr")
            if pm:
                ok_e = within_tukey(mm["early"], *pm["early"])
                ok_s = within_tukey(mm["steady"], *pm["steady"])
                pe = f"{pm['early'][0]:.1f} [{pm['early'][1]:.1f}, {pm['early'][2]:.1f}]"
                ps = f"{pm['steady'][0]:.1f} [{pm['steady'][1]:.1f}, {pm['steady'][2]:.1f}]"
            else:
                ok_e = ok_s = False
                pe = ps = "—"
            ok_c = cov >= 0.70
            if is_scored:
                g0_ok &= ok_e and ok_s and ok_c
            mark = (lambda x: "PASS" if x else "FAIL") if is_scored else (lambda x: "(info)")
            lines.append(f"| {kind} | {sig} | {mm['early']:+.1f} | {pe} | {mark(ok_e)} | {mm['steady']:+.1f} | "
                         f"{ps} | {mark(ok_s)} | {100 * cov:.0f} % | {mark(ok_c)} |")
            rows.append((kind, sig, med, q25, q75))
    lines.append(f"\nModel baselines (absolute, aortic): " + "; ".join(
        f"{k}: MAP {traces[k]['map_base']:.0f}, HR {traces[k]['hr_base']:.0f}" for k in kinds) +
        ". PRCP baselines are finger (Finapres) pressures, so only changes are compared.\n")
    lines.append(f"\n**Verification {'PASS' if ver_ok else 'FAIL'}; validation {'PASS' if g0_ok else 'FAIL'} "
                 f"→ G0 {'PASS' if (ver_ok and g0_ok) else 'not yet (see step 1c)'}**\n")

    pd.DataFrame(csv_rows, columns=["type", "t_s", "d_sbp", "d_map", "d_hr"]).to_csv(
        os.path.join(out, "model_traces.csv"), index=False)

    show = list(kinds.keys())
    fig, ax = plt.subplots(3, len(show), figsize=(3.6 * len(show), 7.5), sharex=True, squeeze=False)
    lab = {"sbp": "ΔSBP (mmHg)", "map": "ΔMAP (mmHg)", "hr": "ΔHR (bpm)"}
    for j, kind in enumerate(show):
        for i, sig in enumerate(("sbp", "map", "hr")):
            a_ = ax[i, j]
            r = [x for x in rows if x[0] == kind and x[1] == sig]
            if r:
                _, _, med, q25, q75 = r[0]
                a_.fill_between(T, q25, q75, alpha=0.25, lw=0, label="PRCP IQR")
                a_.plot(T, med, lw=1.0, label="PRCP median")
            a_.plot(T, traces[kind][sig], lw=1.6, color="k", label="CVSim")
            a_.axvline(0, color="0.5", lw=0.8, ls="--")
            a_.axhline(0, color="0.8", lw=0.6)
            if j == 0:
                a_.set_ylabel(lab[sig])
        title = kind.replace("_", " ") + (" (scored)" if kind in scored else "")
        ax[0, j].set_title(title, fontsize=10)
        ax[-1, j].set_xlabel("time from event (s)")
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"CVSim ({label}) vs PRCP healthy responses", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_validation.png"), dpi=140)
    plt.close(fig)

    open(os.path.join(out, "verification.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[done] wrote {out}/verification.md, fig_equivalence.png, fig_validation.png, model_traces.csv")
    return ver_ok, g0_ok


if __name__ == "__main__":
    main()
