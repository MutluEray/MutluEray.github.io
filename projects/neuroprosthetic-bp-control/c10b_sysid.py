"""
C10-B step 4a — system identification from a simulated clinic calibration session.

Calibration protocol (per patient, length T_cal):
  first 60 s at u = 0 (baseline), then piecewise-constant random amplitudes
  (levels U(0.3, 2.7), holds U(15, 45) s; the week-3 "excitation" idea), supine for
  the first 40 % of the session, a 30° tilt (20 s ramp) from 40 % to 70 %, supine after.
  SBP noise SD 2 mmHg as in the evaluation scenario; no dropouts.

Model (one per patient): first-order ARX with input delay and a posture input
    y[k+1] = a*y[k] + b*u[k-d] + e*sin(angle[k]) + c
  y = measured SBP (mmHg), u = applied amplitude, d = delay in s (0–20).
  For each d, (a, b, e, c) by least squares on one-step prediction; d is chosen by the
  free-run (simulation) RMSE over the session, because MPC uses multi-step predictions.
  a is clipped to [0, 0.995] for stability.

Lengths: 5, 10, 20 (default for the controllers) and 60 min ("oracle" for the
calibration-length sweep). A separate 10-min validation session (different random
seed, 45° tilt) measures free-run RMSE out of sample.

Outputs: data/c10b_sysid.json (model per patient and length),
         results/c10b_step4a/{sysid.md, fig_fit_example.png}
"""

import json
import os
import time

import numpy as np

import c10b_harness as H
from c10b_plant import new_sim, simulate

OUT = os.path.join("results", "c10b_step4a")
LENGTHS = [5, 10, 20, 60]
D_MAX = 20


def calib_session(sim, th, act, T_cal_min, seed, tilt_deg=30.0):
    T = 60.0 * T_cal_min
    rng = np.random.default_rng(seed)
    t_switch, levels, t = [], [], 60.0
    while t < T:
        t_switch.append(t)
        levels.append(rng.uniform(0.3, 2.7))
        t += rng.uniform(15, 45)
    t_switch, levels = np.array(t_switch), np.array(levels)

    def u_fn(tt, obs):
        if tt < 60.0:
            return 0.0
        i = np.searchsorted(t_switch, tt, side="right") - 1
        return H.quantise(levels[max(i, 0)])

    a0, a1 = 0.4 * T, 0.7 * T

    def angle_fn(tt):
        return float(np.interp(tt, [0, a0, a0 + 20, a1, a1 + 20, T + 1], [0, 0, tilt_deg, tilt_deg, 0, 0]))

    return simulate(sim, th, T, angle_fn, u_fn, actuator=act, sbp_noise_sd=H.NOISE_SD,
                    rng=np.random.default_rng(seed + 7))


def simulate_arx(m, y0, u, s, d):
    y = np.empty(len(u))
    y[0] = y0
    for k in range(len(u) - 1):
        uk = u[k - d] if k - d >= 0 else 0.0
        y[k + 1] = m["a"] * y[k] + m["b"] * uk + m["e"] * s[k] + m["c"]
    return y


def fit_arx(o):
    y, u = o["sbp_meas"], o["u"]
    s = np.sin(np.deg2rad(o["angle"]))
    best = None
    for d in range(D_MAX + 1):
        k = np.arange(d, len(y) - 1)
        X = np.column_stack([y[k], u[k - d], s[k], np.ones(len(k))])
        coef, *_ = np.linalg.lstsq(X, y[k + 1], rcond=None)
        a, b, e, c = coef
        if not (0.0 < a < 0.995):
            a = float(np.clip(a, 0.0, 0.995))
            X2 = np.column_stack([u[k - d], s[k], np.ones(len(k))])
            b, e, c = np.linalg.lstsq(X2, y[k + 1] - a * y[k], rcond=None)[0]
        m = {"a": float(a), "b": float(b), "e": float(e), "c": float(c), "d": int(d)}
        r1 = float(np.sqrt(np.mean((X @ np.array([a, b, e, c]) - y[k + 1]) ** 2)))
        yf = simulate_arx(m, y[0], u, s, d)
        rf = float(np.sqrt(np.mean((yf - y) ** 2)))
        if best is None or rf < best["rmse_free"]:
            best = {**m, "rmse_1step": r1, "rmse_free": rf}
    a = best["a"]
    best["gain_per_u"] = best["b"] / (1 - a)
    best["tau_s"] = float(-1.0 / np.log(a)) if a > 0 else 0.0
    best["tilt_gain_per_sin"] = best["e"] / (1 - a)
    return best


def validate(m, o):
    s = np.sin(np.deg2rad(o["angle"]))
    yf = simulate_arx(m, o["sbp_meas"][0], o["u"], s, m["d"])
    return float(np.sqrt(np.mean((yf - o["sbp_meas"]) ** 2)))


def main():
    os.makedirs(OUT, exist_ok=True)
    pop, base, info = H.load_population()
    sim = new_sim()
    res, rows = {}, []
    t0 = time.time()
    ex = None
    for pid in range(len(pop["patients"])):
        th, act = H.patient_plant(pop, base, pid)
        val = calib_session(sim, th, act, 10, seed=50_000 + pid, tilt_deg=45.0)
        res[str(pid)] = {}
        for L in LENGTHS:
            o = calib_session(sim, th, act, L, seed=10_000 * L + pid)
            m = fit_arx(o)
            m["rmse_val"] = validate(m, val)
            res[str(pid)][str(L)] = m
            rows.append((pid, L, m["d"], m["gain_per_u"], m["tau_s"], m["rmse_free"], m["rmse_val"]))
            if pid == 0 and L == 20:
                ex = (o, m, val)
        if pid % 10 == 9:
            print(f"[{pid + 1}/{len(pop['patients'])}] {time.time() - t0:.0f} s")
    json.dump(res, open(os.path.join("data", "c10b_sysid.json"), "w"), indent=1)

    true = {p["id"]: p["actuator"] for p in pop["patients"]}
    lines = ["# C10-B step 4a — system identification from calibration sessions\n",
             f"{len(pop['patients'])} patients × {LENGTHS} min; ARX(1) + input delay + posture input; "
             f"{time.time() - t0:.0f} s.\n",
             "Median [IQR] across patients. Delay error = identified d − true actuator delay "
             "(the ARX delay also absorbs part of the actuator lag, so a positive bias is expected).\n",
             "| T_cal (min) | identified delay (s) | delay error (s) | gain (mmHg per u) | tau (s) | "
             "free-run RMSE in session | free-run RMSE validation |", "|---|---|---|---|---|---|---|"]
    rows = np.array(rows, dtype=float)
    for L in LENGTHS:
        r = rows[rows[:, 1] == L]
        derr = r[:, 2] - np.array([true[int(p)]["delay_s"] for p in r[:, 0]])

        def q(x):
            a, b, c = np.percentile(x, [25, 50, 75])
            return f"{b:.1f} [{a:.1f}, {c:.1f}]"
        lines.append(f"| {L} | {q(r[:, 2])} | {q(derr)} | {q(r[:, 3])} | {q(r[:, 4])} | {q(r[:, 5])} | {q(r[:, 6])} |")
    open(os.path.join(OUT, "sysid.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o, m, val = ex
    fig, ax = plt.subplots(2, 2, figsize=(12, 5.5), sharex="col")
    for j, (oo, title) in enumerate([(o, "calibration session (20 min)"), (val, "validation session (10 min, 45°)")]):
        s = np.sin(np.deg2rad(oo["angle"]))
        yf = simulate_arx(m, oo["sbp_meas"][0], oo["u"], s, m["d"])
        ax[0, j].plot(oo["t"], oo["sbp_meas"], lw=0.7, color="0.5", label="measured SBP")
        ax[0, j].plot(oo["t"], yf, lw=1.4, color="k", label="ARX free run")
        ax[0, j].set_title(title, fontsize=10)
        ax[1, j].plot(oo["t"], oo["u"], color="tab:blue", lw=1)
        ax2 = ax[1, j].twinx()
        ax2.plot(oo["t"], oo["angle"], color="tab:red", lw=1)
        ax2.set_ylabel("tilt (deg)", color="tab:red")
        ax[1, j].set_ylabel("amplitude u")
        ax[1, j].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("SBP (mmHg)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Patient 0: d={m['d']} s, gain {m['gain_per_u']:.1f} mmHg/u, tau {m['tau_s']:.0f} s", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_fit_example.png"), dpi=140); plt.close(fig)
    print(f"[done] data/c10b_sysid.json, {OUT}/sysid.md, fig_fit_example.png")


if __name__ == "__main__":
    main()
