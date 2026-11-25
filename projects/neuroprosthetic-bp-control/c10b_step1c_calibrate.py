"""
C10-B step 1c (v2) — calibrate the nominal CVSim subject to PRCP with Bayesian optimisation.

v2 changes (2026-11-24), method only — the G0 criteria from step 1b are unchanged:
  - "G0 criteria met" is no longer a stopping rule (v1 stopped after 3 evaluations
    on a lucky Sobol point). BO now minimises the loss over the full budget and
    stops only on stall / stuck; G0 is checked once, on the chosen model.
  - 2 extra parameters that set baroreflex loop gain (and hence damping):
    scale on the arterial-baroreflex resistance gains vec[4..7] and on the
    venous-tone gains vec[8..11].
  - RR sympathetic gain lower bound 0.0 (v1 ended on its 0.004 bound).
  - descriptive shape metrics (not pass/fail, added before this run):
      * rapid tilt up: MAP overshoot = max ΔMAP in 15–60 s
      * slow tilt up : HR t63 = time for ΔHR to reach 63 % of its 120–180 s plateau

Why: CVSim's default reflexes ring after a rapid tilt (MAP overshoot ~+17 mmHg at
~25 s, HR oscillation) and adapt too fast to a slow tilt. Loop dynamics are what the
controller comparison exercises, so the healthy plant should not carry that artefact.

Calibrated on : rapid tilt up + slow tilt up (MAP and HR, 0–180 s)
Held out      : rapid and slow tilt down, stand-up (reported by the step 1b re-run)

Parameters (bounds around the CVSim default; defaults inside the bounds):
  vec[2]      RR sympathetic gain                     0.0 – 0.020  (default 0.012)
  vec[3]      RR parasympathetic gain                 0.003 – 0.018 (default 0.009)
  vec[14]     carotid sensed-pressure offset (mmHg)   -30 – -5     (default -25)
  vec[1]      arterial baroreflex saturation scale    10 – 30      (default 18)
  vec[4..7]   x scale, ABR resistance gains           0.3 – 1.5    (default 1)
  vec[8..11]  x scale, ABR venous-tone gains          0.3 – 1.5    (default 1)

Loss: mean over {rapid, slow} x {MAP, HR} of the squared error to the PRCP median
curve divided by that curve's IQR (floor 1), 0–180 s.

BO: Matern-5/2 ARD GP on -log(loss), LogEI, 10 Sobol starts, at most 60 evaluations.
Early stopping: stall = 15 evaluations without > 1 % improvement of the best loss;
stuck = after 20 evaluations the best loss is not below the default model's.
Without BoTorch: Sobol search, same budget and stopping rules.

Outputs:
  data/c10b_theta_calibrated.npy          full 153-parameter vector
  results/c10b_step1c/calibration.md      parameters, loss, shape metrics, stopping reason
  results/c10b_step1c/calibration.json
  results/c10b_step1c/fig_bo_trace.png
  results/c10b_step1c/verification.md + figures (step 1b re-run on the calibrated model)
"""

import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import c10b_step1b_verify as V
from c10b_cvsim import CVSim21

OUT = os.path.join("results", "c10b_step1c")
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)

# (name, kind, indices, lo, hi): kind "set" writes the value, "scale" multiplies the defaults
PARAMS = [("rr_symp_gain", "set", [2], 0.0, 0.020),
          ("rr_para_gain", "set", [3], 0.003, 0.018),
          ("carotid_offset", "set", [14], -30.0, -5.0),
          ("abr_scale", "set", [1], 10.0, 30.0),
          ("abr_res_gain_x", "scale", [4, 5, 6, 7], 0.3, 1.5),
          ("abr_vt_gain_x", "scale", [8, 9, 10, 11], 0.3, 1.5)]
LO = np.array([p[3] for p in PARAMS])
HI = np.array([p[4] for p in PARAMS])
CAL_EVENTS = ["rapid_tilt_up", "slow_tilt_up"]
MAX_EVALS, N_INIT, STALL, STUCK_AT = 60, 10, 15, 20


def load_targets():
    env = pd.read_csv(os.path.join(V.S1A, "envelopes.csv"))
    met = V.parse_metrics_md()
    tg = {}
    for k in CAL_EVENTS:
        for s in ("map", "hr"):
            e = env[(env["type"] == k) & (env["signal"] == s)].sort_values("t_s")
            med, q25, q75 = (np.interp(V.T, e["t_s"], e[c]) for c in ("median", "q25", "q75"))
            tg[(k, s)] = (med, q25, q75)
    return tg, met


def default_x(base):
    x = []
    for name, kind, idx, lo, hi in PARAMS:
        v = base[idx[0]] if kind == "set" else 1.0
        x.append((v - lo) / (hi - lo))
    return np.clip(np.array(x), 0, 1)


def make_theta(base, x01):
    th = base.copy()
    vals = LO + np.clip(x01, 0, 1) * (HI - LO)
    for (name, kind, idx, _, _), v in zip(PARAMS, vals):
        for i in idx:
            th[i] = v if kind == "set" else base[i] * v
    return th, vals


def _smooth(y, w):
    """Centred moving average (w samples at 1 Hz); edges use the available points."""
    k = np.ones(w)
    num = np.convolve(np.nan_to_num(y), k, mode="same")
    den = np.convolve(np.isfinite(y).astype(float), k, mode="same")
    return num / np.maximum(den, 1)


def shape_metrics(traces):
    """Descriptive shape metrics on event-locked traces (dict kind -> {'map','hr'}).
    Curves are smoothed first (5 s for the overshoot, 11 s for t63) so PRCP noise
    does not set the value."""
    out = {}
    if "rapid_tilt_up" in traces:
        m = (V.T >= 15) & (V.T <= 60)
        out["rapid_map_overshoot"] = float(np.max(_smooth(traces["rapid_tilt_up"]["map"], 5)[m]))
    if "slow_tilt_up" in traces:
        hr = _smooth(traces["slow_tilt_up"]["hr"], 11)
        plateau = float(np.mean(hr[V.LATE]))
        post = (V.T >= 0)
        idx = np.where(post & (hr >= 0.63 * plateau))[0] if plateau > 0 else []
        out["slow_hr_t63_s"] = float(V.T[idx[0]]) if len(idx) else float("nan")
    return out


def prcp_shape_metrics(tg):
    tr = {k: {"map": tg[(k, "map")][0], "hr": tg[(k, "hr")][0]} for k in CAL_EVENTS}
    return shape_metrics(tr)


def evaluate(sim, th, tg, met, durs, return_traces=False):
    loss, ok_all, traces = [], True, {}
    for k in CAL_EVENTS:
        tr = V.scenario(sim, th, k, durs[k])
        traces[k] = tr
        for s in ("map", "hr"):
            med, q25, q75 = tg[(k, s)]
            iqr = np.maximum(q75 - q25, 1.0)
            loss.append(np.mean(((tr[s] - med) / iqr)[V.SCORE] ** 2))
            mm = V.model_metrics(tr, s)
            pm = met.get((k, s))
            band = q75 - q25
            cov = np.mean(((tr[s] >= q25 - band) & (tr[s] <= q75 + band))[V.SCORE])
            ok = cov >= 0.70 and pm is not None and V.within_tukey(mm["early"], *pm["early"]) \
                and V.within_tukey(mm["steady"], *pm["steady"])
            ok_all &= ok
    if return_traces:
        return float(np.mean(loss)), ok_all, traces
    return float(np.mean(loss)), ok_all


def propose_bo(X, Y, seed):
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import ScaleKernel, MaternKernel
    torch.manual_seed(seed)
    tx = torch.tensor(np.array(X), dtype=torch.double)
    ty = -torch.tensor(np.log(np.array(Y)), dtype=torch.double).unsqueeze(-1)  # maximise -log loss
    gp = SingleTaskGP(tx, ty, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=tx.shape[1])),
                      outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = LogExpectedImprovement(gp, best_f=ty.max())
    bounds = torch.stack([torch.zeros(tx.shape[1]), torch.ones(tx.shape[1])]).double()
    cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=256)
    return cand.detach().numpy().ravel()


def main():
    try:
        import botorch  # noqa: F401
        use_bo = True
    except ImportError:
        use_bo = False
        print("[info] BoTorch not found -> Sobol search fallback")

    from scipy.stats import qmc
    tg, met = load_targets()
    durs = {k: V.prcp_transition_s(k, 2.0 if k.startswith("rapid") else 50.0) for k in CAL_EVENTS}
    sim = CVSim21()
    base = sim.default_theta()

    t0 = time.time()
    loss0, ok0, tr0 = evaluate(sim, base, tg, met, durs, return_traces=True)
    print(f"[default] loss {loss0:.3f}  G0-criteria {'met' if ok0 else 'not met'}  ({time.time() - t0:.1f} s/eval)")

    X, Y, OK = [default_x(base)], [loss0], [ok0]
    sob = qmc.Sobol(d=len(PARAMS), scramble=True, seed=0).random(128)
    reason, last_improve, best = f"budget ({MAX_EVALS} evaluations)", 0, loss0
    for it in range(1, MAX_EVALS):
        if it <= N_INIT or not use_bo:
            x = sob[it]
        else:
            try:
                x = propose_bo(X, Y, seed=it)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] BO proposal failed ({e}); Sobol point instead")
                x = sob[it]
        th, vals = make_theta(base, x)
        loss, ok = evaluate(sim, th, tg, met, durs)
        X.append(x); Y.append(loss); OK.append(ok)
        if loss < best * 0.99:
            best, last_improve = loss, it
        print(f"[{it:02d}] loss {loss:.3f} best {min(Y):.3f} G0={'y' if ok else 'n'} " +
              " ".join(f"{p[0]}={v:.4g}" for p, v in zip(PARAMS, vals)))
        if it > N_INIT and it - last_improve >= STALL:
            reason = f"stall ({STALL} evaluations without >1 % improvement)"
            break
        if it >= STUCK_AT and min(Y) >= loss0:
            reason = "stuck (no improvement over the default model)"
            break

    Y = np.array(Y)
    i_best = int(np.argmin(Y))
    th_best, vals_best = make_theta(base, X[i_best])
    loss_b, ok_b, tr_b = evaluate(sim, th_best, tg, met, durs, return_traces=True)
    np.save(os.path.join("data", "c10b_theta_calibrated.npy"), th_best)

    sh_p, sh_0, sh_b = prcp_shape_metrics(tg), shape_metrics(tr0), shape_metrics(tr_b)
    at_bound = [p[0] for p, x in zip(PARAMS, X[i_best]) if x < 0.02 or x > 0.98]

    lines = ["# C10-B step 1c (v2) — BO calibration of the nominal subject to PRCP\n",
             f"Method: {'BoTorch GP (Matern 5/2 ARD) + LogEI' if use_bo else 'Sobol search (BoTorch missing)'}; "
             f"{len(Y)} evaluations; stopped: **{reason}**; wall {time.time() - t0:.0f} s.\n",
             f"Default model loss {loss0:.3f} (G0 criteria {'met' if ok0 else 'not met'}); "
             f"best model loss {loss_b:.3f} (G0 criteria {'met' if ok_b else 'not met'}; "
             f"best at evaluation {i_best}).\n",
             "| parameter | indices | default | calibrated | bounds |", "|---|---|---|---|---|"]
    for (name, kind, idx, lo, hi), v in zip(PARAMS, vals_best):
        d = f"{base[idx[0]]:.4g}" if kind == "set" else "×1"
        c = f"{v:.4g}" if kind == "set" else f"×{v:.3f}"
        lines.append(f"| {name} | {idx} | {d} | {c} | [{lo}, {hi}] |")
    lines.append(f"\nParameters within 2 % of a bound: {', '.join(at_bound) if at_bound else 'none'}.\n")
    lines += ["## Shape metrics (descriptive, not pass/fail)\n",
              "| metric | PRCP median curve | CVSim default | calibrated |", "|---|---|---|---|"]
    for k in sh_p:
        lines.append(f"| {k} | {sh_p[k]:.1f} | {sh_0.get(k, float('nan')):.1f} | {sh_b.get(k, float('nan')):.1f} |")
    lines.append("\nCalibrated on rapid and slow tilt up only; tilt down and stand-up in the re-run below are held out.\n")
    open(os.path.join(OUT, "calibration.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"params": {p[0]: float(v) for p, v in zip(PARAMS, vals_best)}, "loss": float(loss_b),
               "default_loss": float(loss0), "g0_criteria_met": bool(ok_b), "reason": reason,
               "shape": {"prcp": sh_p, "default": sh_0, "calibrated": sh_b},
               "history": Y.tolist(), "g0_history": [bool(o) for o in OK]},
              open(os.path.join(OUT, "calibration.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(Y, "o-", ms=3, lw=0.8, label="loss")
    ax.plot(np.minimum.accumulate(Y), lw=1.6, label="best so far")
    ax.axhline(loss0, color="0.5", ls="--", lw=0.8, label="CVSim default")
    ax.axvline(N_INIT + 0.5, color="0.8", lw=0.8)
    ax.set_xlabel("evaluation (left of grey line: Sobol starts)"); ax.set_ylabel("IQR-normalised MSE")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bo_trace.png"), dpi=140)
    plt.close(fig)
    print("\n".join(lines))

    print("\n[re-run step 1b checks on the calibrated model]")
    V.main(theta=th_best, out=OUT, label="calibrated to PRCP median (step 1c v2)")


if __name__ == "__main__":
    main()
