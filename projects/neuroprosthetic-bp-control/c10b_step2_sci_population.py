"""
C10-B step 2 (v2) — SCI plant, EES actuator scaling, virtual population.

v2 (2026-11-24): v1 passed G-SCI with a resting HR of ~100 bpm, because supine HR was
neither in the loss nor in the acceptance rule (omission). The fit pushed the baroreflex
set point to its upper bound (95 mmHg), so the reflex saw a large pressure deficit even
supine. v2 adds supine HR (target 64 bpm = mean of Corbett's 67/62/62, tol 8 bpm) to the
loss and to G-SCI, flags parameters that end on a bound, logs redraw reasons, and
reports the amplitude headroom needed under 60 deg tilt. This is a stated post-hoc
correction, not a loosened criterion.

Inputs : data/c10b_theta_calibrated.npy (step 1c v2, healthy nominal subject)
Targets: Corbett, Frankel & Harris 1971, J Physiol (PMC1331891) — chronic complete
         cervical SCI, head-up tilt, average mean BP and HR:
           30 deg: MAP 68 -> 44, HR 67 -> 85
           45 deg: MAP 74 -> 36, HR 62 -> 97
           60 deg: MAP 70 -> 44, HR 62 -> 103
         BP fell and HR rose progressively for ~2 min, then plateaued.

A. SCI calibration (BO, 5 parameters) on 30 and 45 deg; 60 deg held out.
     base_res_x   0.4 – 1.0    (arteriolar resistance after loss of sympathetic tone)
     zpfv_add_ml  0 – 1000     (extra splanchnic + leg unstressed volume)
     rr_para      0.003 – 0.015 (vagal HR gain; the healthy fit sits at 0.0033; above
                  ~0.015 the linear vagal arm drives HR to non-physiological values once
                  the sympathetic arms are cut)
     abr_setpoint 60 – 95 mmHg (baroreflex resetting to chronically low BP; default 91)
     hr_nominal   50 – 80 bpm  (HR at the set point, CVSim vec[90]; lower without tonic
                  cardiac sympathetic drive in cervical SCI; added in v2 because a pre-run
                  probe showed resting HR cannot go below ~75 with the healthy value 70)
   Why 30/45 and not 60: Corbett's group means are not monotonic in angle (fall 24, 38,
   26 mmHg at 30, 45, 60 deg), while the hydrostatic load grows with sin(angle); a
   pre-run probe showed no parameter set reaches -26 at 60 deg together with -38 at
   45 deg. 60 deg is therefore reported as held out but does not gate.
   Loss: sum of squared residuals / tolerance^2 for supine MAP (target 70.7 = mean of the
   three supine values, tol 5 mmHg), supine HR (target 63.7, tol 8 bpm), plateau ΔMAP
   (tol 8 mmHg) and plateau ΔHR (tol 8 bpm).
   Plateau = mean of 150–250 s after tilt onset. Tilt ramp 10 s (not reported in 1971).
   Stopping: stall 12 evaluations without >1 % improvement, budget 40. No "success" stop.
   Acceptance (G-SCI): all calibrated residuals within tolerance, including supine HR (v2);
   model meets the OH definition at 60 deg (SBP drop >= 20 mmHg within 3 min); peak HR
   <= 150 bpm in every tilt. 60 deg residuals reported (x1.5 tolerance shown) but not gating.

B. EES actuator scaling on the nominal SCI patient (supine): gain set by bisection so
   that u = 1 gives +30 mmHg steady SBP (200–300 s after the step). Then the static
   amplitude -> SBP curve (linearity R^2; Phillips 2025 reported a linear relation),
   step responses for the swept delay/lag ranges, and how much of the 60 deg fall u = 1
   restores.

C. Virtual population (N = 60, seed 0; patients 0–39 tuning, 40–59 test): per patient
     base_res_x x U(0.85, 1.15), zpfv_add x U(0.8, 1.2) (+ U(0, 100) mL), rr_para x U(0.7, 1.3),
     abr_setpoint + U(-5, 5) mmHg, hr_nominal + U(-5, 5) bpm, blood volume x U(0.92, 1.08), EES gain x U(0.6, 1.4), frac_res U(0.3, 0.7),
     delay U(2, 12) s, tau U(5, 30) s.
   Patients are redrawn if supine MAP is outside 50–100 mmHg, MAP rises on 60 deg tilt,
   or peak HR exceeds 150 bpm.
   Delay/lag ranges are assumptions: no published EES pressor time constants; the 60–70 s
   expert wait times in Christie et al. 2026 suggest responses settle within ~1 min.

Outputs: data/c10b_sci_theta.npy, data/c10b_population.json,
         results/c10b_step2/{step2.md, population.csv, fig_sci_tilt.png,
         fig_ees_static.png, fig_ees_steps.png, fig_population.png}
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c10b_plant import EESActuator, sci_theta, simulate, tilt_profile, new_sim

OUT = os.path.join("results", "c10b_step2")
os.makedirs(OUT, exist_ok=True)

CORBETT = {30: (68, 44, 67, 85), 45: (74, 36, 62, 97), 60: (70, 44, 62, 103)}
SUPINE_MAP = float(np.mean([v[0] for v in CORBETT.values()]))
SUPINE_HR = float(np.mean([v[2] for v in CORBETT.values()]))
TOL = {"supine": 5.0, "supine_hr": 8.0, "dmap": 8.0, "dhr": 8.0}
CAL_ANGLES, HELD_OUT = [30, 45], [60]
P_LO = np.array([0.4, 0.0, 0.003, 60.0, 50.0])
P_HI = np.array([1.0, 1000.0, 0.015, 95.0, 80.0])
P_NAMES = ["base_res_x", "zpfv_add_ml", "rr_para", "abr_setpoint", "hr_nominal"]
T_TILT, T_END = 60.0, 330.0
PLATEAU = (150.0, 250.0)


def tilt_response(sim, th, angle):
    o = simulate(sim, th, T_END, tilt_profile(T_TILT, angle), lambda t, obs: 0.0)
    t = o["t"]
    base = (t >= 10) & (t < T_TILT)
    plat = (t >= T_TILT + PLATEAU[0]) & (t <= T_TILT + PLATEAU[1])
    first3 = (t >= T_TILT) & (t <= T_TILT + 180)
    r = {"supine_map": np.nanmean(o["map"][base]),
         "supine_hr": np.nanmean(o["hr"][base]),
         "dmap": np.nanmean(o["map"][plat]) - np.nanmean(o["map"][base]),
         "dhr": np.nanmean(o["hr"][plat]) - np.nanmean(o["hr"][base]),
         "sbp_drop_3min": np.nanmean(o["sbp"][base]) - np.nanmin(o["sbp"][first3]),
         "hr_max": float(np.nanmax(o["hr"])),
         "trace": o}
    return r


def residuals(r, angle):
    s0, s1, h0, h1 = CORBETT[angle]
    return {"supine": (r["supine_map"] - SUPINE_MAP) / TOL["supine"],
            "supine_hr": (r["supine_hr"] - SUPINE_HR) / TOL["supine_hr"],
            "dmap": (r["dmap"] - (s1 - s0)) / TOL["dmap"],
            "dhr": (r["dhr"] - (h1 - h0)) / TOL["dhr"]}


def sci_loss(sim, base, x01):
    p = P_LO + np.clip(x01, 0, 1) * (P_HI - P_LO)
    th = sci_theta(base, base_res_x=p[0], zpfv_add_ml=p[1], rr_para=p[2], abr_setpoint=p[3], hr_nominal=p[4])
    loss = 0.0
    for a in CAL_ANGLES:
        res = residuals(tilt_response(sim, th, a), a)
        loss += res["dmap"] ** 2 + res["dhr"] ** 2 + (res["supine"] ** 2 + res["supine_hr"] ** 2) / len(CAL_ANGLES)
    return loss, p, th


def propose(X, Y, seed):
    from c10b_step1c_calibrate import propose_bo
    return propose_bo(X, Y, seed)


def calibrate_sci(sim, base, lines):
    try:
        import botorch  # noqa: F401
        use_bo = True
    except ImportError:
        use_bo = False
    from scipy.stats import qmc
    sob = qmc.Sobol(d=len(P_NAMES), scramble=True, seed=1).random(64)
    X, Y, P = [], [], []
    best, last_imp, reason = np.inf, 0, "budget (50 evaluations)"
    t0 = time.time()
    for it in range(50):  # budget
        x = sob[it] if (it < 10 or not use_bo) else propose(X, Y, it)
        loss, p, _ = sci_loss(sim, base, x)
        loss = max(loss, 1e-9)
        X.append(x); Y.append(loss); P.append(p)
        if loss < best * 0.99:
            best, last_imp = loss, it
        print(f"[sci {it:02d}] loss {loss:.3f} best {min(Y):.3f} " +
              " ".join(f"{n}={v:.4g}" for n, v in zip(P_NAMES, p)))
        if it >= 10 and it - last_imp >= 12:
            reason = "stall (12 evaluations without >1 % improvement)"
            break
    i = int(np.argmin(Y))
    p = P[i]
    th = sci_theta(base, base_res_x=p[0], zpfv_add_ml=p[1], rr_para=p[2], abr_setpoint=p[3], hr_nominal=p[4])
    lines += ["## A. SCI calibration to Corbett et al. 1971 (complete cervical SCI)\n",
              f"{'BoTorch BO' if use_bo else 'Sobol search'}; {len(Y)} evaluations; stopped: {reason}; "
              f"{time.time() - t0:.0f} s. Best loss {Y[i]:.3f}.\n",
              "| parameter | value | bounds |", "|---|---|---|"]
    at_b = []
    for n, v, lo, hi in zip(P_NAMES, p, P_LO, P_HI):
        on_b = (v - lo) / (hi - lo) < 0.02 or (hi - v) / (hi - lo) < 0.02
        if on_b:
            at_b.append(n)
        lines.append(f"| {n} | {v:.4g} | [{lo}, {hi}] |{' **on bound**' if on_b else ''}")
    lines.append(f"\nParameters within 2 % of a bound: {', '.join(at_b) if at_b else 'none'}.")
    return th, p


def evaluate_sci(sim, th, lines):
    lines += ["\n| angle | role | supine MAP model / target | supine HR model / target | ΔMAP model / target | "
              "ΔHR model / target | SBP drop in 3 min | within tolerance |", "|---|---|---|---|---|---|---|---|"]
    ok_all, traces = True, {}
    for a in [30, 45, 60]:
        r = tilt_response(sim, th, a)
        traces[a] = r["trace"]
        res = residuals(r, a)
        role = "held out (not gating)" if a in HELD_OUT else "calibrated"
        lim = 1.5 if a in HELD_OUT else 1.0
        ok = all(abs(v) <= lim for v in res.values())
        if a not in HELD_OUT:
            ok_all &= ok
        ok_all &= r["hr_max"] <= 150
        s0, s1, h0, h1 = CORBETT[a]
        lines.append(f"| {a}° | {role} | {r['supine_map']:.0f} / {SUPINE_MAP:.0f} | {r['supine_hr']:.0f} / {SUPINE_HR:.0f} | {r['dmap']:+.0f} / {s1 - s0:+d} | "
                     f"{r['dhr']:+.0f} / {h1 - h0:+d} | {r['sbp_drop_3min']:.0f} mmHg | "
                     f"{'yes' if ok else 'NO'} (x{lim} tol); peak HR {r['hr_max']:.0f} |")
        if a == 60:
            oh = r["sbp_drop_3min"] >= 20
            ok_all &= oh
            lines.append(f"\nOH definition at 60° (SBP drop ≥ 20 mmHg within 3 min): {'met' if oh else 'NOT met'}.")
    lines.append(f"\n**G-SCI: {'PASS' if ok_all else 'FAIL'}**\n")
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
    for j, a in enumerate([30, 45, 60]):
        o = traces[a]
        base = (o["t"] >= 10) & (o["t"] < T_TILT)
        s0, s1, h0, h1 = CORBETT[a]
        ax[0, j].plot(o["t"] - T_TILT, o["map"] - np.nanmean(o["map"][base]), color="k")
        ax[0, j].axhline(s1 - s0, color="tab:red", ls="--", lw=1, label="Corbett 1971 mean")
        ax[1, j].plot(o["t"] - T_TILT, o["hr"] - np.nanmean(o["hr"][base]), color="k")
        ax[1, j].axhline(h1 - h0, color="tab:red", ls="--", lw=1)
        for r_ in (0, 1):
            ax[r_, j].axvspan(*PLATEAU, color="0.92")
            ax[r_, j].axvline(0, color="0.5", lw=0.8, ls=":")
        ax[0, j].set_title(f"{a}° ({'held out' if a in HELD_OUT else 'calibrated'})", fontsize=10)
        ax[1, j].set_xlabel("time from tilt onset (s)")
    ax[0, 0].set_ylabel("ΔMAP (mmHg)"); ax[1, 0].set_ylabel("ΔHR (bpm)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("SCI plant (complete cervical) vs Corbett et al. 1971 — grey: plateau window")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sci_tilt.png"), dpi=140); plt.close(fig)
    return ok_all


def steady_dsbp(sim, th, act, u, angle=0.0):
    o = simulate(sim, th, 360.0, lambda t: angle, lambda t, obs: (u if t >= 60 else 0.0), actuator=act)
    t = o["t"]
    return np.nanmean(o["sbp"][(t >= 260) & (t <= 360)]) - np.nanmean(o["sbp"][(t >= 10) & (t < 60)]), o


def scale_actuator(sim, th, lines):
    lo, hi = 0.0, 6.0
    for _ in range(16):
        g = 0.5 * (lo + hi)
        d, _ = steady_dsbp(sim, th, EESActuator(gain=g, delay_s=5, tau_s=15, frac_res=0.5), 1.0)
        if abs(d - 30.0) < 0.5:
            break
        lo, hi = (g, hi) if d < 30.0 else (lo, g)
    g_nom = g
    us = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5])
    ds = np.array([steady_dsbp(sim, th, EESActuator(gain=g_nom), u)[0] for u in us])
    A = np.vstack([us, np.ones_like(us)]).T
    coef, *_ = np.linalg.lstsq(A, ds, rcond=None)
    r2 = 1 - np.sum((ds - A @ coef) ** 2) / np.sum((ds - ds.mean()) ** 2)
    lin = us <= 1.0
    coef1, *_ = np.linalg.lstsq(A[lin], ds[lin], rcond=None)
    r2_1 = 1 - np.sum((ds[lin] - A[lin] @ coef1) ** 2) / np.sum((ds[lin] - ds[lin].mean()) ** 2)
    # how much of the 60 deg fall does a constant amplitude restore? (headroom for control)
    no = simulate(sim, th, T_END, tilt_profile(T_TILT, 60), lambda t, obs: 0.0)
    plat = (no["t"] >= T_TILT + PLATEAU[0]) & (no["t"] <= T_TILT + PLATEAU[1])
    base = (no["t"] >= 10) & (no["t"] < T_TILT)
    fall = np.nanmean(no["sbp"][plat]) - np.nanmean(no["sbp"][base])
    restore = {}
    for u_c in (1.0, 2.0):
        o_u = simulate(sim, th, T_END, tilt_profile(T_TILT, 60), lambda t, obs, u_c=u_c: u_c if t >= T_TILT else 0.0,
                       actuator=EESActuator(gain=g_nom))
        restore[u_c] = (np.nanmean(o_u["sbp"][plat]) - np.nanmean(o_u["sbp"][base]), o_u)
    fall_u, yes = restore[1.0]
    lines += ["## B. EES actuator (nominal SCI patient)\n",
              f"Gain for +30 mmHg supine SBP at u = 1 (delay 5 s, tau 15 s, frac_res 0.5): **{g_nom:.3f}**.\n",
              f"Static amplitude → ΔSBP: linear fit R² = {r2_1:.3f} for u ≤ 1, {r2:.3f} up to u = 1.5. "
              "Linear largely by construction (additive actuator on a near-linear plant in this range); "
              "consistent with, not validated by, the linear relation in Phillips 2025.\n",
              "| u | " + " | ".join(f"{u:g}" for u in us) + " |", "|---" * (len(us) + 1) + "|",
              "| ΔSBP (mmHg) | " + " | ".join(f"{d:+.1f}" for d in ds) + " |\n",
              f"60° tilt, plateau ΔSBP: {fall:+.1f} mmHg without EES, {restore[1.0][0]:+.1f} with u = 1, "
              f"{restore[2.0][0]:+.1f} with u = 2 (constant from tilt onset). Controllers in step 4 get u ∈ [0, 2].\n"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(us, ds, "o-", color="k")
    ax[0].plot(us, A @ coef1, ls="--", color="tab:blue", label=f"linear fit (u ≤ 1), R²={r2_1:.2f}")
    ax[0].set_xlabel("EES amplitude u (normalised)"); ax[0].set_ylabel("steady ΔSBP (mmHg), supine")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(no["t"] - T_TILT, no["sbp"], label="60° tilt, no EES", color="tab:red")
    ax[1].plot(yes["t"] - T_TILT, yes["sbp"], label="60° tilt, u = 1 from onset", color="k")
    ax[1].plot(restore[2.0][1]["t"] - T_TILT, restore[2.0][1]["sbp"], label="60° tilt, u = 2 from onset", color="0.5")
    ax[1].axvline(0, color="0.5", ls=":", lw=0.8)
    ax[1].set_xlabel("time from tilt onset (s)"); ax[1].set_ylabel("SBP (mmHg)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ees_static.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for d_, tau in [(2, 5), (5, 15), (12, 30)]:
        _, o = steady_dsbp(sim, th, EESActuator(gain=g_nom, delay_s=d_, tau_s=tau), 1.0)
        ax.plot(o["t"] - 60, o["sbp"], label=f"delay {d_} s, tau {tau} s")
    ax.axvline(0, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("time from EES step u = 1 (s)"); ax.set_ylabel("SBP (mmHg)")
    ax.set_title("EES step responses across the swept actuator range", fontsize=10)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_ees_steps.png"), dpi=140); plt.close(fig)
    return g_nom


def build_population(sim, base, p_nom, g_nom, lines, n=60, seed=0):
    rng = np.random.default_rng(seed)
    pats, rows = [], []
    i = 0
    why = {"supine MAP outside 50–100": 0, "MAP rises on tilt": 0, "peak HR > 150": 0}
    while len(pats) < n:
        i += 1
        pp = {"base_res_x": p_nom[0] * rng.uniform(0.85, 1.15),
              "zpfv_add_ml": p_nom[1] * rng.uniform(0.8, 1.2) + rng.uniform(0, 100),
              "rr_para": p_nom[2] * rng.uniform(0.7, 1.3),
              "abr_setpoint": p_nom[3] + rng.uniform(-5, 5),
              "hr_nominal": p_nom[4] + rng.uniform(-5, 5),
              "blood_vol_x": rng.uniform(0.92, 1.08)}
        aa = {"gain": g_nom * rng.uniform(0.6, 1.4), "frac_res": rng.uniform(0.3, 0.7),
              "delay_s": rng.uniform(2, 12), "tau_s": rng.uniform(5, 30)}
        th = sci_theta(base, **pp)
        r = tilt_response(sim, th, 60)
        bad = False
        if not (50 <= r["supine_map"] <= 100):
            why["supine MAP outside 50–100"] += 1; bad = True
        if r["dmap"] >= 0:
            why["MAP rises on tilt"] += 1; bad = True
        if r["hr_max"] > 150:
            why["peak HR > 150"] += 1; bad = True
        if bad:
            continue
        d_ees, _ = steady_dsbp(sim, th, EESActuator(**aa), 1.0)
        k = len(pats)
        pats.append({"id": k, "split": "tune" if k < 40 else "test", "plant": pp, "actuator": aa})
        o = r["trace"]
        base_m = (o["t"] >= 10) & (o["t"] < T_TILT)
        rows.append({"id": k, "split": pats[-1]["split"], **pp, **aa,
                     "supine_sbp": np.nanmean(o["sbp"][base_m]), "supine_map": r["supine_map"],
                     "supine_hr": r["supine_hr"], "hr_max_tilt60": r["hr_max"],
                     "tilt60_dmap": r["dmap"], "tilt60_sbp_drop_3min": r["sbp_drop_3min"],
                     "ees_u1_dsbp": d_ees})
        print(f"[pop {k:02d}] supine MAP {r['supine_map']:.0f}, 60° ΔMAP {r['dmap']:+.0f}, EES u=1 ΔSBP {d_ees:+.0f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "population.csv"), index=False)
    json.dump({"base_theta": "data/c10b_theta_calibrated.npy", "sci_nominal": dict(zip(P_NAMES, map(float, p_nom))),
               "ees_gain_nominal": g_nom, "patients": pats, "redraws": i - n},
              open(os.path.join("data", "c10b_population.json"), "w"), indent=1)
    lines += ["## C. Virtual population\n",
              f"{n} patients ({i - n} redrawn by the plausibility filter: " +
              ", ".join(f"{k} {v}" for k, v in why.items()) + "; a draw can fail several); 0–39 tuning, 40–59 test.\n",
              "| quantity | median | 10th pct | 90th pct |", "|---|---|---|---|"]
    for c in ["supine_sbp", "supine_map", "supine_hr", "hr_max_tilt60", "tilt60_dmap", "tilt60_sbp_drop_3min",
              "ees_u1_dsbp", "delay_s", "tau_s"]:
        lines.append(f"| {c} | {df[c].median():.1f} | {df[c].quantile(.1):.1f} | {df[c].quantile(.9):.1f} |")
    oh = (df["tilt60_sbp_drop_3min"] >= 20).mean()
    lines.append(f"\nShare meeting the OH definition at 60°: {100 * oh:.0f} %.\n")
    fig, ax = plt.subplots(1, 5, figsize=(16, 3))
    for a_, c, lab in zip(ax, ["supine_map", "supine_hr", "tilt60_dmap", "ees_u1_dsbp", "tau_s"],
                          ["supine MAP (mmHg)", "supine HR (bpm)", "60° tilt ΔMAP (mmHg)", "EES u=1 ΔSBP (mmHg)",
                           "actuator tau (s)"]):
        a_.hist(df[c], bins=12, color="0.4")
        a_.set_xlabel(lab)
    fig.suptitle("Virtual SCI population (N = 60)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_population.png"), dpi=140); plt.close(fig)


def main():
    t0 = time.time()
    p_cal = os.path.join("data", "c10b_theta_calibrated.npy")
    if not os.path.exists(p_cal):
        raise SystemExit("Run step 1c first (data/c10b_theta_calibrated.npy missing).")
    base = np.load(p_cal)
    sim = new_sim()
    lines = ["# C10-B step 2 (v2) — SCI plant, EES actuator, virtual population\n",
             "Healthy base: step 1c v2 calibrated CVSim. SCI = complete cervical (all sympathetic efferent "
             "reflex arms cut, vagal HR arm kept).\n"]
    th_sci, p_nom = calibrate_sci(sim, base, lines)
    ok = evaluate_sci(sim, th_sci, lines)
    np.save(os.path.join("data", "c10b_sci_theta.npy"), th_sci)
    g_nom = scale_actuator(sim, th_sci, lines)
    build_population(sim, base, p_nom, g_nom, lines)
    lines.append(f"\nTotal wall time {time.time() - t0:.0f} s.")
    open(os.path.join(OUT, "step2.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[done] {OUT}/step2.md and figures; data/c10b_sci_theta.npy, data/c10b_population.json "
          f"(G-SCI {'PASS' if ok else 'FAIL'})")


if __name__ == "__main__":
    main()
