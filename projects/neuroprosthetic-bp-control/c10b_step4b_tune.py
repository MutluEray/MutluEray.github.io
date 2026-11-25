"""
C10-B step 4b — Bayesian-optimisation tuning of one controller family (equal budget for all).

Usage:  python3 c10b_step4b_tune.py C2            (C1, C2, C2f, C2ff, C3, C3ff, C4, C4ff)
        python3 c10b_step4b_tune.py C2 --evals 25

Protocol (identical for every family, fixed before any tuning run):
  - tuning subset: 8 of the 40 tuning patients, spread over the EES response range
    (sorted by supine ΔSBP at u = 1, evenly spaced); test patients 40–59 never used
  - objective: mean J over the subset, J = (100 − in_range) + 5·hyper + 2·hypo + 0.5·tv
  - evaluation 0 = the hand-set default (step 3 values for C1/C2), then 7 Sobol points,
    then BoTorch GP (Matern 5/2, ARD) + LogEI on −log J (course week 4 recipe)
  - model-based families use each patient's ARX model from a 20-min calibration session
    (step 4a); the controller never sees the true plant
  - early stopping: budget (default 25 evaluations) or stall = 10 evaluations without
    > 1 % improvement of the best mean J, counted from the end of the Sobol phase at the
    earliest (v2 fix: v1 counted from evaluation 0, so C2 and C4 got only 3–5 BO proposals)
  - a controller that throws (e.g. Riccati failure) scores J = 500 for that patient

Outputs: results/c10b_step4b/tuned_<FAMILY>.json, fig_bo_<FAMILY>.png
"""

import argparse
import json
import os
import time

import numpy as np

import c10b_controllers as C
import c10b_harness as H
from c10b_plant import new_sim

OUT = os.path.join("results", "c10b_step4b")
CAL_MIN = 20
N_INIT_SOBOL = 7
STALL = 10


def tuning_subset(info, n=8):
    tune = info.loc[list(range(40))].sort_values("ees_u1_dsbp")
    idx = np.round(np.linspace(0, len(tune) - 1, n)).astype(int)
    return [int(i) for i in tune.index[idx]]


def patient_info(info, sysid, pid, cal_min=CAL_MIN):
    d = dict(info.loc[pid])
    if sysid is not None:
        d["model"] = sysid[str(pid)][str(cal_min)]
    return d


def evaluate(sim, pop, base, info, sysid, fam, params, pids):
    Js, ms = [], []
    for pid in pids:
        try:
            c = C.make(fam, params)
            _, m = H.run_patient(sim, pop, base, pid, c, info=patient_info(info, sysid, pid))
            Js.append(C.cost_J(m)); ms.append(m)
        except Exception as e:  # noqa: BLE001
            print(f"   [warn] patient {pid}: {type(e).__name__}: {e} -> J = 500")
            Js.append(500.0)
    keys = ["in_range", "hyper", "hypo", "tv", "sat"]
    agg = {k: float(np.mean([m[k] for m in ms])) for k in keys} if ms else {}
    return float(np.mean(Js)), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(C.FAMILIES))
    ap.add_argument("--evals", type=int, default=25)
    args = ap.parse_args()
    fam = args.family
    os.makedirs(OUT, exist_ok=True)

    pop, base, info = H.load_population()
    sysid = None
    if fam in C.NEEDS_MODEL:
        p = os.path.join("data", "c10b_sysid.json")
        if not os.path.exists(p):
            raise SystemExit("Run step 4a first (data/c10b_sysid.json missing).")
        sysid = json.load(open(p))
    pids = tuning_subset(info)
    print(f"[{fam}] tuning subset: {pids}")
    sim = new_sim()

    try:
        import botorch  # noqa: F401
        use_bo = True
    except ImportError:
        use_bo = False
        print("[info] BoTorch not found -> Sobol search")
    from scipy.stats import qmc
    from c10b_step1c_calibrate import propose_bo

    dim = len(C.FAMILIES[fam]["space"])
    sob = qmc.Sobol(d=dim, scramble=True, seed=3).random(64)
    X, Y, A = [], [], []
    best, last_imp, reason = np.inf, 0, f"budget ({args.evals} evaluations)"
    t0 = time.time()
    for it in range(args.evals):
        if it == 0:
            x = C.to_x01(fam, C.DEFAULTS[fam])
        elif it <= N_INIT_SOBOL or not use_bo:
            x = sob[it]
        else:
            try:
                x = propose_bo(X, Y, seed=it)
            except Exception as e:  # noqa: BLE001
                print(f"   [warn] BO proposal failed ({e}); Sobol point instead")
                x = sob[it]
        p = C.to_params(fam, x)
        J, agg = evaluate(sim, pop, base, info, sysid, fam, p, pids)
        X.append(np.asarray(x)); Y.append(max(J, 1e-6)); A.append(agg)
        if J < best * 0.99:
            best, last_imp = J, it
        print(f"[{fam} {it:02d}] J {J:7.2f} best {min(Y):7.2f} | " +
              " ".join(f"{k}={v:.3g}" for k, v in p.items()) +
              (f" | in {agg['in_range']:.0f} hyper {agg['hyper']:.1f} hypo {agg['hypo']:.1f} tv {agg['tv']:.0f}" if agg else "") +
              f"  ({time.time() - t0:.0f} s)")
        if it - max(last_imp, N_INIT_SOBOL) >= STALL:
            reason = f"stall ({STALL} evaluations without > 1 % improvement)"
            break

    i = int(np.argmin(Y))
    res = {"family": fam, "params": C.to_params(fam, X[i]), "J_subset": float(Y[i]),
           "J_default": float(Y[0]), "default_params": C.DEFAULTS[fam], "metrics_subset": A[i],
           "evaluations": len(Y), "stopped": reason, "subset": pids, "cal_min": CAL_MIN if sysid else None,
           "wall_s": time.time() - t0, "history": [float(y) for y in Y]}
    json.dump(res, open(os.path.join(OUT, f"tuned_{fam}.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(Y, "o-", ms=3, lw=0.8, label="mean J on subset")
    ax.plot(np.minimum.accumulate(Y), lw=1.6, label="best so far")
    ax.axhline(Y[0], color="0.5", ls="--", lw=0.8, label="hand-set default")
    ax.axvline(N_INIT_SOBOL + 0.5, color="0.85", lw=0.8)
    ax.set_xlabel("evaluation"); ax.set_ylabel("mean J"); ax.set_title(f"BO tuning — {fam}", fontsize=10)
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_bo_{fam}.png"), dpi=140); plt.close(fig)
    print(f"\n[{fam}] best J {Y[i]:.2f} (default {Y[0]:.2f}) at evaluation {i}; stopped: {reason}; "
          f"{time.time() - t0:.0f} s\nparams: {res['params']}\n[done] {OUT}/tuned_{fam}.json")


if __name__ == "__main__":
    main()
