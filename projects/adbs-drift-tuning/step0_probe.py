#!/usr/bin/env python3
"""
step0_probe.py — Step 0 go/no-go: does Fleming 2020 run on this Mac, and how fast?

Two separate runs (NEURON cannot rebuild the network inside one process):
  1) python3 step0_probe.py --model-dir fleming2020 --make-ss
       builds the network, integrates 6 s to steady state, writes steady_state.bin
  2) python3 step0_probe.py --model-dir fleming2020 --probe
       reloads the steady state, applies 4 open-loop DBS settings for 2 s each,
       reports beta ARV per setting and wall-clock cost per simulated second,
       and projects the cost of the Step 2 lookup table.

Writes results/step0_ss.json and results/step0_probe.json (next to this script).
"""
import argparse
import json
import os
import platform
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PROBE_SETTINGS = [  # (amp mA, freq Hz)
    (0.0, 0.0),      # DBS off -> parkinsonian beta baseline
    (2.5, 130.0),    # Fleming's constant-amplitude reference
    (1.0, 130.0),    # weak stimulation
    (2.5, 60.0),     # low frequency
]


def projection(wall_per_sim_s, seg_ms):
    per_eval_min = wall_per_sim_s * seg_ms / 1000.0 / 60.0
    rows = []
    for n_amp, n_freq, n_states, reps in [(6, 4, 5, 1), (8, 5, 6, 1), (8, 5, 6, 2)]:
        n = n_amp * n_freq * n_states * reps
        rows.append({"grid": f"{n_amp} amp x {n_freq} freq x {n_states} drift states x {reps} rep",
                     "evaluations": n, "hours_1_core": round(n * per_eval_min / 60.0, 2)})
    return per_eval_min, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-ss", action="store_true")
    g.add_argument("--probe", action="store_true")
    ap.add_argument("--ss-ms", type=float, default=6000.0, help="steady-state duration (keep 6000 except for smoke tests)")
    ap.add_argument("--seg-ms", type=float, default=2000.0, help="duration of each probe setting")
    ap.add_argument("--settle-ms", type=float, default=500.0)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    from cbg_model import CBGModel  # imported after argparse; it chdirs before importing NEURON

    if args.make_ss:
        t0 = time.perf_counter()
        m = CBGModel(args.model_dir, mode="steady_state", ss_ms=args.ss_ms)
        build_s = time.perf_counter() - t0
        print(f"network built in {build_s:.1f} s; integrating {args.ss_ms/1000:.1f} s to steady state ...", flush=True)
        wall = m.make_steady_state()
        out = {"build_s": build_s, "ss_sim_s": args.ss_ms / 1000.0, "ss_wall_s": wall,
               "wall_s_per_sim_s": wall / (args.ss_ms / 1000.0), "machine": platform.platform(),
               "processor": platform.processor()}
        json.dump(out, open(os.path.join(RESULTS, "step0_ss.json"), "w"), indent=2)
        print(json.dumps(out, indent=2))
        print("steady_state.bin written in", os.path.abspath(args.model_dir))
        return

    t0 = time.perf_counter()
    m = CBGModel(args.model_dir, mode="eval", ss_ms=args.ss_ms)
    build_s = time.perf_counter() - t0
    m.load_steady_state()
    print(f"network built in {build_s:.1f} s; steady state loaded", flush=True)
    rows = []
    for amp, freq in PROBE_SETTINGS:
        r = m.evaluate(amp, freq, dur_ms=args.seg_ms, settle_ms=args.settle_ms)
        rows.append(r)
        print(f"  amp={amp:.2f} mA  f={freq:5.1f} Hz  beta_ARV={r['beta_arv']:.3e}  "
              f"wall={r['wall_s']:.1f} s  ({r['wall_s_per_sim_s']:.1f} s per simulated s)", flush=True)
    wps = sum(r["wall_s"] for r in rows) / (len(rows) * args.seg_ms / 1000.0)
    per_eval_min, proj = projection(wps, args.seg_ms)
    base = rows[0]["beta_arv"]
    out = {"build_s": build_s, "wall_s_per_sim_s": wps, "per_eval_min": per_eval_min,
           "beta_off": base, "rows": rows,
           "beta_suppression_vs_off": {f"{r['amp_mA']}mA_{r['freq_Hz']}Hz": (1 - r["beta_arv"] / base) if base > 0 else None for r in rows[1:]},
           "lookup_table_projection": proj, "machine": platform.platform(), "processor": platform.processor()}
    json.dump(out, open(os.path.join(RESULTS, "step0_probe.json"), "w"), indent=2)
    print("\nbeta suppression vs DBS off:")
    for k, v in out["beta_suppression_vs_off"].items():
        print(f"  {k}: {v:+.1%}" if v is not None else f"  {k}: n/a")
    print(f"\n{wps:.1f} wall-s per simulated s  ->  {per_eval_min:.1f} min per {args.seg_ms/1000:.1f}-s evaluation")
    for p in proj:
        print(f"  {p['grid']}: {p['evaluations']} evals ~ {p['hours_1_core']} h on 1 core")
    print("\nSend me results/step0_ss.json and results/step0_probe.json")


if __name__ == "__main__":
    main()
