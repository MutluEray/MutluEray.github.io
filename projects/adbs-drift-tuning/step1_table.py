#!/usr/bin/env python3
"""
step1_table.py — Step 1 go/no-go: does the best DBS setting move with dopaminergic state?

Runs a pilot response table on the Fleming 2020 model in parallel (one fresh process
per evaluation, each starting from steady_state.bin), checkpointing every result to
results/table_<name>.csv so it can be stopped (Ctrl-C) and resumed at any time.

Grid (pilot):  amp {0.5,1.0,1.5,2.0,2.5,3.0} mA x freq {60,130,180} Hz
               x beta_drive {0.01,0.02,0.03}  (+ DBS-off baseline per beta_drive)
Electrode/encapsulation drift (stim_gain) needs no extra runs: delivered amplitude
= amp x gain, so it is read off the amplitude axis at analysis time.

Usage:
  python3 step1_table.py --model-dir fleming2020 --run          # parallel pilot
  python3 step1_table.py --model-dir fleming2020 --analyze      # summary + figure
"""
import argparse
import csv
import itertools
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

AMPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
FREQS = [60.0, 130.0, 180.0]
BETA_DRIVES = [0.01, 0.02, 0.03]
FIELDS = ["beta_drive", "amp_mA", "freq_Hz", "burst_file", "beta_arv", "teed", "n_pulses", "wall_s"]


def perf_cores():
    if platform.system() == "Darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip())
        except Exception:
            pass
    return max(1, (os.cpu_count() or 2) - 1)


def task_list(burst_files):
    tasks = []
    for bf in burst_files:
        for bd in BETA_DRIVES:
            tasks.append((bd, 0.0, 0.0, bf))                       # baseline (DBS off)
            for a, f in itertools.product(AMPS, FREQS):
                tasks.append((bd, a, f, bf))
    return tasks


def key(bd, a, f, bf):
    return f"{float(bd):.4f}|{float(a):.3f}|{float(f):.1f}|{int(bf)}"


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {key(r["beta_drive"], r["amp_mA"], r["freq_Hz"], r["burst_file"]) for r in csv.DictReader(fh)}


def run_one(model_dir, bd, amp, freq, bf, seg_ms, settle_ms):
    """Runs in a fresh process: build network, load steady state, one evaluation."""
    from cbg_model import CBGModel
    m = CBGModel(model_dir, mode="eval", beta_drive=bd,
                 burst_file=f"burst_times_{bf}.txt", level_file=f"burst_level_{bf}.txt")
    m.load_steady_state()
    r = m.evaluate(amp, freq, dur_ms=seg_ms, settle_ms=settle_ms)
    return {"beta_drive": bd, "amp_mA": amp, "freq_Hz": freq, "burst_file": bf,
            "beta_arv": r["beta_arv"], "teed": r["teed"], "n_pulses": r["n_pulses"], "wall_s": r["wall_s"]}


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"table_{args.name}.csv")
    done = done_keys(path)
    todo = [t for t in task_list(args.burst_files) if key(*t) not in done]
    workers = args.workers or perf_cores()
    print(f"{len(done)} done, {len(todo)} to run, {workers} parallel workers -> {path}", flush=True)
    if not todo:
        return
    new_file = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    t0 = time.time()
    n_ok = 0
    ctx = mp.get_context("spawn")                     # fresh interpreter -> fresh NEURON per task
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run_one, args.model_dir, *t, args.seg_ms, args.settle_ms): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                row = fut.result()
            except Exception as e:                    # keep going; failed cells are retried on resume
                print(f"  FAILED {t}: {e}", flush=True)
                continue
            w.writerow(row)
            fh.flush()
            n_ok += 1
            el = time.time() - t0
            eta = el / n_ok * (len(todo) - n_ok)
            print(f"  [{n_ok}/{len(todo)}] bd={t[0]:.3f} amp={t[1]:.1f} f={t[2]:.0f} beta={row['beta_arv']:.3e} "
                  f"| elapsed {el/60:.0f} min, ETA {eta/60:.0f} min", flush=True)
    fh.close()
    print("done. Next: python3 step1_table.py --model-dir", args.model_dir, "--analyze")


def analyze(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(RESULTS, f"table_{args.name}.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in FIELDS:
            r[k] = float(r[k])
    global AMPS, FREQS, BETA_DRIVES
    BETA_DRIVES = sorted({r["beta_drive"] for r in rows})
    AMPS = sorted({r["amp_mA"] for r in rows if r["amp_mA"] > 0})
    FREQS = sorted({r["freq_Hz"] for r in rows if r["amp_mA"] > 0})
    # only (amp, freq) cells measured in EVERY state with a baseline -> fair comparison of optima
    states = [bd for bd in BETA_DRIVES if any(abs(r["beta_drive"] - bd) < 1e-9 and r["amp_mA"] == 0 for r in rows)]
    common = None
    for bd in states:
        cells = {(r["amp_mA"], r["freq_Hz"]) for r in rows if abs(r["beta_drive"] - bd) < 1e-9 and r["amp_mA"] > 0}
        common = cells if common is None else common & cells
    teed_ref = max(r["teed"] for r in rows) or 1.0
    lambdas = [0.0, 0.1, 0.3, 1.0]                    # energy weight in reward
    lines = ["# Step 1 pilot summary", "",
             "reward = -(beta / beta_off_same_state) - lambda * TEED / TEED_max  (higher is better)",
             "optima are compared only over (amp, freq) cells measured in every state", ""]

    fig, axes = plt.subplots(1, len(BETA_DRIVES), figsize=(4.2 * len(BETA_DRIVES), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    best = {}
    noise_sd = []
    for j, bd in enumerate(BETA_DRIVES):
        sub = [r for r in rows if abs(r["beta_drive"] - bd) < 1e-9]
        offs = {r["burst_file"]: r["beta_arv"] for r in sub if r["amp_mA"] == 0.0}
        if not offs:
            lines.append(f"beta_drive={bd}: baseline missing, skipped")
            continue
        # ratio to the stimulation-off baseline of the SAME burst pattern, then mean/SD over patterns
        cell = {}
        for r in sub:
            if r["amp_mA"] > 0 and r["burst_file"] in offs:
                cell.setdefault((r["amp_mA"], r["freq_Hz"]), []).append((r["beta_arv"] / offs[r["burst_file"]], r["teed"]))
        grid = np.full((len(FREQS), len(AMPS)), np.nan)
        lines.append(f"beta_drive={bd}: beta_off={np.mean(list(offs.values())):.3e} "
                     f"(burst patterns: {sorted(int(b) for b in offs)})")
        for (a, f), v in sorted(cell.items()):
            ratios = [x[0] for x in v]
            grid[FREQS.index(f), AMPS.index(a)] = np.mean(ratios)
            if len(ratios) > 1:
                noise_sd.append(np.std(ratios, ddof=1))
                lines.append(f"    {a:.1f} mA @ {f:.0f} Hz: beta/off = {np.mean(ratios):.3f} +/- {np.std(ratios, ddof=1):.3f} (n={len(ratios)})")
        for lam in lambdas:
            cands = [(-np.mean([x[0] for x in v]) - lam * v[0][1] / teed_ref, a, f)
                     for (a, f), v in cell.items() if (a, f) in common]
            if cands:
                s, a, f = max(cands)
                best[(bd, lam)] = (a, f)
                lines.append(f"  lambda={lam:<4} best = {a:.1f} mA @ {f:.0f} Hz  (reward {s:+.3f})")
        ax = axes[j]
        im = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1.2, cmap="viridis",
                       extent=[-0.5, len(AMPS) - 0.5, -0.5, len(FREQS) - 0.5])
        ax.set_xticks(range(len(AMPS)), [f"{a:g}" for a in AMPS])
        ax.set_yticks(range(len(FREQS)), [f"{int(f)}" for f in FREQS])
        ax.set_xlabel("amplitude (mA)")
        ax.set_title(f"beta_drive = {bd}")
        if j == 0:
            ax.set_ylabel("frequency (Hz)")
    fig.colorbar(im, ax=list(axes), label="beta / beta_off (mean over burst patterns)")
    if noise_sd:
        lines.append("")
        lines.append(f"## Noise: median SD of beta/off across burst patterns = {np.median(noise_sd):.3f} "
                     f"(max {np.max(noise_sd):.3f}, {len(noise_sd)} cells)")
    out_png = os.path.join(RESULTS, f"step1_{args.name}_heatmaps.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    lines.append("")
    lines.append(f"## Go/no-go  (states compared: {states}; {len(common or [])} common cells)")
    if len(states) < 2:
        lines.append("  fewer than 2 complete states -> no verdict yet")
    moved = {lam: len({best[(bd, lam)] for bd in states if (bd, lam) in best}) > 1 for lam in lambdas}
    for lam in lambdas:
        lines.append(f"  lambda={lam}: optimum {'MOVES' if moved[lam] else 'does NOT move'} across beta_drive")
    only_low_f = bool(best) and len(FREQS) > 1 and all(best[k][1] == min(FREQS) for k in best)
    lines.append(f"  lowest frequency wins everywhere: {only_low_f}  (if True, frequency axis is degenerate in this model)")
    txt = "\n".join(lines)
    open(os.path.join(RESULTS, f"step1_{args.name}_summary.txt"), "w").write(txt + "\n")
    print(txt)
    print(f"\nfigure: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--workers", type=int, default=0, help="default: number of performance cores")
    ap.add_argument("--burst-files", type=int, nargs="+", default=[1], help="Fleming burst patterns 1-10 (replicates)")
    ap.add_argument("--seg-ms", type=float, default=2000.0)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--amps", type=float, nargs="+", help="override amplitude grid (mA)")
    ap.add_argument("--freqs", type=float, nargs="+", help="override frequency grid (Hz)")
    ap.add_argument("--beta-drives", type=float, nargs="+", help="override beta_drive levels")
    args = ap.parse_args()
    global AMPS, FREQS, BETA_DRIVES
    AMPS = args.amps or AMPS
    FREQS = args.freqs or FREQS
    BETA_DRIVES = args.beta_drives or BETA_DRIVES
    run(args) if args.run else analyze(args)


if __name__ == "__main__":
    main()
