#!/usr/bin/env python3
"""
surrogate.py — Step 2: fast surrogate of the Fleming 2020 model for the tuning study.

Built from the NEURON response table (results/table_pilot.csv):
  * beta/beta_off vs delivered amplitude at 130 Hz, beta_drive 0.02, each run normalised by
    the DBS-off baseline of its own burst pattern, averaged over burst patterns
  * shape-preserving (PCHIP) interpolation through the means, measured SD per amplitude as
    the evaluation noise (Gaussian, floored)
  * electrode/encapsulation drift: delivered amplitude = set amplitude x gain(t)
  * reward = -(beta/beta_off) - lambda * TEED / TEED_REF   (same definition as step 1)

Delivered amplitudes above the largest measured amplitude are clamped to its response
(no extrapolation of benefit); the energy term still grows, so such arms are never optimal.

Usage:
  python3 surrogate.py --build            # fit, save results/surrogate.json, print optima vs gain
  python3 surrogate.py --plot             # results/surrogate_fit.png (curve + reward vs gain)
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TEED_REF = 3.0 ** 2 * 180.0 * 0.06          # same normaliser as step1_table.py
PW = 0.06
NOISE_FLOOR = 0.03


def teed(amp, freq=130.0):
    return amp ** 2 * freq * PW


class Surrogate:
    def __init__(self, amps, mean, sd, freq=130.0):
        self.amps, self.mean, self.sd, self.freq = np.asarray(amps), np.asarray(mean), np.asarray(sd), freq
        self._f = PchipInterpolator(self.amps, self.mean, extrapolate=False)
        self._s = PchipInterpolator(self.amps, self.sd, extrapolate=False)
        self.a_max = float(self.amps.max())

    def beta_ratio(self, amp_eff):
        a = np.clip(amp_eff, 0.0, self.a_max)
        return self._f(a)

    def noise_sd(self, amp_eff):
        a = np.clip(amp_eff, 0.0, self.a_max)
        return np.maximum(self._s(a), NOISE_FLOOR)

    def expected_reward(self, amp_set, gain=1.0, lam=1.0):
        a_eff = np.asarray(amp_set) * gain
        return -self.beta_ratio(a_eff) - lam * teed(a_eff, self.freq) / TEED_REF

    def sample_reward(self, amp_set, gain, lam, rng):
        a_eff = amp_set * gain
        ratio = self.beta_ratio(a_eff) + rng.normal(0.0, self.noise_sd(a_eff))
        return float(-ratio - lam * teed(a_eff, self.freq) / TEED_REF), float(ratio)

    def to_json(self):
        return {"amps": self.amps.tolist(), "mean": self.mean.tolist(), "sd": self.sd.tolist(),
                "freq": self.freq, "teed_ref": TEED_REF, "noise_floor": NOISE_FLOOR}

    @classmethod
    def load(cls, path=os.path.join(RESULTS, "surrogate.json")):
        d = json.load(open(path))
        return cls(d["amps"], d["mean"], d["sd"], d["freq"])


def build_from_table(table, beta_drive=0.02, freq=130.0):
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(table))]
    rows = [r for r in rows if abs(r["beta_drive"] - beta_drive) < 1e-9]
    offs = {r["burst_file"]: r["beta_arv"] for r in rows if r["amp_mA"] == 0.0}
    by_amp = {0.0: [1.0]}
    for r in rows:
        if r["amp_mA"] > 0 and abs(r["freq_Hz"] - freq) < 1e-9 and r["burst_file"] in offs:
            by_amp.setdefault(r["amp_mA"], []).append(r["beta_arv"] / offs[r["burst_file"]])
    base = np.array(list(offs.values()))
    off_sd = float(base.std(ddof=1) / base.mean()) if len(base) > 1 else NOISE_FLOOR
    amps = sorted(by_amp)
    mean = [float(np.mean(by_amp[a])) for a in amps]
    sd = []
    for a in amps:
        v = by_amp[a]
        if a == 0.0:
            sd.append(off_sd)
        elif len(v) > 1:
            sd.append(float(np.std(v, ddof=1)))
        else:
            sd.append(np.nan)                  # single run: filled below
    known = [s for s in sd if not np.isnan(s)]
    fill = float(np.median(known)) if known else 0.1
    sd = [fill if np.isnan(s) else s for s in sd]
    n = {a: len(by_amp[a]) for a in amps}
    return Surrogate(amps, mean, sd, freq), n


def optima_table(s, gains, lams, grid):
    out = []
    for lam in lams:
        for g in gains:
            r = s.expected_reward(grid, g, lam)
            i = int(np.nanargmax(r))
            out.append({"lambda": lam, "gain": g, "best_set_mA": float(grid[i]), "delivered_mA": float(grid[i] * g),
                        "reward": float(r[i])})
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--plot", action="store_true")
    ap.add_argument("--table", default=os.path.join(RESULTS, "table_pilot.csv"))
    ap.add_argument("--beta-drive", type=float, default=0.02)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    grid = np.round(np.arange(0.5, 4.0001, 0.25), 2)       # arm set used in step 3
    gains = [1.0, 0.9, 0.8, 0.7, 0.6]
    lams = [0.5, 1.0, 2.0]

    if args.build:
        s, n = build_from_table(args.table, args.beta_drive)
        json.dump(s.to_json(), open(os.path.join(RESULTS, "surrogate.json"), "w"), indent=2)
        print("delivered amp (mA) | beta/off mean | SD    | n runs")
        for a, m, sd in zip(s.amps, s.mean, s.sd):
            print(f"  {a:4.1f}             | {m:.3f}         | {sd:.3f} | {n[a]}")
        print("\noptimal SET amplitude vs electrode gain (arms 0.5-4.0 mA, step 0.25):")
        for r in optima_table(s, gains, lams, grid):
            print(f"  lambda={r['lambda']:<4} gain={r['gain']:.1f}: set {r['best_set_mA']:.2f} mA "
                  f"(delivered {r['delivered_mA']:.2f}), reward {r['reward']:+.3f}")
        print("\nsaved results/surrogate.json")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = Surrogate.load()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.linspace(0, s.a_max, 200)
    a1.fill_between(x, s.beta_ratio(x) - s.noise_sd(x), s.beta_ratio(x) + s.noise_sd(x), alpha=0.25)
    a1.plot(x, s.beta_ratio(x), lw=2)
    a1.errorbar(s.amps, s.mean, yerr=s.sd, fmt="o", capsize=3)
    a1.set_xlabel("delivered amplitude (mA), 130 Hz")
    a1.set_ylabel("beta / beta_off")
    a1.set_title("Fleming 2020 response (mean ± SD)")
    for gval in gains:
        r = s.expected_reward(grid, gval, 1.0)
        i = int(np.nanargmax(r))
        line, = a2.plot(grid, r, label=f"gain {gval:.1f}")
        a2.plot(grid[i], r[i], "o", color=line.get_color())
    a2.set_xlabel("set amplitude (mA)")
    a2.set_ylabel("expected reward (lambda = 1)")
    a2.set_title("optimum moves as electrode gain drops")
    a2.legend(fontsize=8)
    out = os.path.join(RESULTS, "surrogate_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure:", out)


if __name__ == "__main__":
    main()
