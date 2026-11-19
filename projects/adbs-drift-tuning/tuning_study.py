#!/usr/bin/env python3
"""
tuning_study.py — Step 3: online-learning tuners tracking the DBS amplitude optimum under
electrode/encapsulation drift, on the NEURON-fitted surrogate (results/surrogate.json).

Arms: set amplitude 0.5-4.0 mA (step 0.25, K=15) at 130 Hz. Each step = one 2-s evaluation
returning a noisy reward r = -(beta/beta_off) - lambda*TEED/TEED_REF (lambda = 1).
Delivered amplitude = set x gain(t). Unsafe evaluation = delivered > 3.0 mA (assumed side-effect
threshold = top of the measured range; stated assumption).

Drift scenarios (gain(t), T steps), following the Oxford TV-BayesOpt protocol + an abrupt shift:
  static     1.0
  gradual    1.0 -> 0.6 linear (encapsulation)
  periodic   0.8 + 0.15 sin(2 pi t / 100)        (protocol match; weak physiological meaning)
  combined   gradual + periodic component
  abrupt     1.0, then 0.7 from T/3 (lead migration)

Algorithm families:
  fixed      clinician setting (optimum at t=0), never changes
  stationary eps-greedy, UCB1, Gaussian Thompson sampling
  nonstat.   discounted UCB, sliding-window UCB
  adversarial (lec 7) Exp3, Exp3.P, Exp3.S, Tsallis-INF
  TV-GP      TV-GP-UCB (forgetting kernel, Bogunovic et al. 2016);
             TV-GP-UCB-periodic (forgetting x periodic temporal kernel, Oxford TV-BayesOpt-style;
             given the true period = privileged information, flagged in results)
Bandits use gains x = 1 - loss in [0,1], loss = clip((-r - 0.5)/1.0, 0, 1). Hyperparameters are
theory defaults or fixed a priori (listed in HYPER); nothing is tuned on the test scenarios.

Metrics: dynamic regret (sum of oracle expected reward - expected reward of chosen arm),
unsafe evaluations, mean |set - optimal set|, recovery time after the abrupt shift.

Usage:
  python3 tuning_study.py --run            # all algorithms x scenarios x seeds
  python3 tuning_study.py --run --quick    # 5 seeds, T=500 smoke run (~1 min)
  python3 tuning_study.py --plot
  python3 tuning_study.py --tune           # fair tuning on validation seeds (2000+)
  python3 tuning_study.py --tune --only "Exp3" --grid 30 100   # extend the grid for some algorithms
  python3 tuning_study.py --run --tuned    # test seeds (1000+) with tuned multipliers
  python3 tuning_study.py --plot --tuned
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from surrogate import Surrogate  # noqa: E402

ARMS = np.round(np.arange(0.5, 4.0001, 0.25), 2)
K = len(ARMS)
LAM = 1.0
UNSAFE_MA = 3.0
HYPER = {"eps": 0.1, "ducb_gamma": 0.99, "ducb_xi": 0.5, "swucb_tau": 100, "exp3s_switches": 5,
         "exp3p_delta": 0.05, "gp_len_mA": 0.5, "gp_sig2": 0.05, "gp_noise": 0.08, "gp_eps": 0.02,
         "gp_eps_periodic": 0.005, "gp_window": 200, "gp_window_periodic": 300, "gp_beta": 2.0,
         "period": 100, "ts_sigma": 0.08}


CMULT = {}                    # per-algorithm exploration / learning-rate multiplier (fair tuning)
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def cm(obj):
    return CMULT.get(obj.name, 1.0)


def to_gain(r):
    return 1.0 - float(np.clip((-r - 0.5) / 1.0, 0.0, 1.0))


# ------------------------------------------------------------------ scenarios
def gain_schedule(name, T):
    t = np.arange(T)
    if name == "static":
        return np.ones(T)
    if name == "gradual":
        return 1.0 - 0.4 * t / (T - 1)
    if name == "periodic":
        return 0.8 + 0.15 * np.sin(2 * np.pi * t / HYPER["period"])
    if name == "combined":
        return 1.0 - 0.3 * t / (T - 1) + 0.08 * np.sin(2 * np.pi * t / HYPER["period"])
    if name == "abrupt":
        g = np.ones(T)
        g[T // 3:] = 0.7
        return g
    raise ValueError(name)


SCENARIOS = ["static", "gradual", "periodic", "combined", "abrupt"]


# ------------------------------------------------------------------ algorithms
class Fixed:
    name = "fixed (clinician)"

    def __init__(self, T, rng, arm0):
        self.a = arm0

    def select(self, t):
        return self.a

    def update(self, t, i, x):
        pass


class EpsGreedy:
    name = "eps-greedy"

    def __init__(self, T, rng, arm0):
        self.rng, self.n, self.s = rng, np.zeros(K), np.zeros(K)

    def select(self, t):
        if t < K:
            return t
        if self.rng.random() < min(1.0, HYPER["eps"] * cm(self)):
            return int(self.rng.integers(K))
        return int(np.argmax(self.s / self.n))

    def update(self, t, i, x):
        self.n[i] += 1
        self.s[i] += x


class UCB1(EpsGreedy):
    name = "UCB1"

    def select(self, t):
        if t < K:
            return t
        return int(np.argmax(self.s / self.n + cm(self) * np.sqrt(2 * math.log(t + 1) / self.n)))


class Thompson(EpsGreedy):
    name = "Thompson"

    def select(self, t):
        s2 = (HYPER["ts_sigma"] * cm(self)) ** 2
        prec = 1.0 + self.n / s2                        # N(0.5, 1) prior
        mean = (0.5 + self.s / s2) / prec
        return int(np.argmax(mean + self.rng.standard_normal(K) / np.sqrt(prec)))


class DUCB:
    name = "discounted UCB"

    def __init__(self, T, rng, arm0):
        self.n, self.s = np.zeros(K), np.zeros(K)

    def select(self, t):
        if t < K:
            return t
        n_tot = self.n.sum()
        return int(np.argmax(self.s / np.maximum(self.n, 1e-9)
                             + cm(self) * 2 * np.sqrt(HYPER["ducb_xi"] * math.log(max(n_tot, 1.0001)) / np.maximum(self.n, 1e-9))))

    def update(self, t, i, x):
        g = HYPER["ducb_gamma"]
        self.n *= g
        self.s *= g
        self.n[i] += 1
        self.s[i] += x


class SWUCB:
    name = "sliding-window UCB"

    def __init__(self, T, rng, arm0):
        self.hist = []

    def select(self, t):
        if t < K:
            return t
        tau = HYPER["swucb_tau"]
        h = self.hist[-tau:]
        n, s = np.zeros(K), np.zeros(K)
        for i, x in h:
            n[i] += 1
            s[i] += x
        if (n == 0).any():
            return int(np.argmin(n))
        return int(np.argmax(s / n + cm(self) * np.sqrt(0.6 * math.log(min(t, tau)) / n)))

    def update(self, t, i, x):
        self.hist.append((i, x))


class Exp3:
    name = "Exp3"

    def __init__(self, T, rng, arm0):
        self.rng = rng
        self.gamma = min(1.0, math.sqrt(K * math.log(K) / ((math.e - 1) * T)))
        self.logw = np.zeros(K)

    def probs(self):
        w = np.exp(self.logw - self.logw.max())
        return (1 - self.gamma) * w / w.sum() + self.gamma / K

    def select(self, t):
        self.p = self.probs()
        return int(self.rng.choice(K, p=self.p))

    def update(self, t, i, x):
        self.logw[i] += cm(self) * self.gamma * (x / self.p[i]) / K


class Exp3P(Exp3):
    name = "Exp3.P"

    def __init__(self, T, rng, arm0):
        self.rng, self.T = rng, T
        self.alpha = 2 * math.sqrt(math.log(K * T / HYPER["exp3p_delta"]))
        self.gamma = min(0.6, 2 * math.sqrt(0.6 * K * math.log(K) / T))
        self.logw = np.full(K, self.alpha * self.gamma / 3 * math.sqrt(T / K))

    def update(self, t, i, x):
        xhat = np.zeros(K)
        xhat[i] = x / self.p[i]
        self.logw += cm(self) * self.gamma / (3 * K) * (xhat + self.alpha / (self.p * math.sqrt(K * self.T)))


class Exp3S(Exp3):
    name = "Exp3.S"

    def __init__(self, T, rng, arm0):
        self.rng = rng
        S = HYPER["exp3s_switches"]
        self.a = 1.0 / T
        self.gamma = min(1.0, math.sqrt(K * (S * math.log(K * T) + math.e) / ((math.e - 1) * T)))
        self.w = np.ones(K) / K

    def probs(self):
        return (1 - self.gamma) * self.w / self.w.sum() + self.gamma / K

    def update(self, t, i, x):
        xhat = np.zeros(K)
        xhat[i] = x / self.p[i]
        W = self.w.sum()
        self.w = self.w * np.exp(cm(self) * self.gamma * xhat / K) + math.e * self.a / K * W
        self.w /= self.w.sum()                          # renormalise (scale-invariant update)


class TsallisINF:
    name = "Tsallis-INF"

    def __init__(self, T, rng, arm0):
        self.rng, self.L = rng, np.zeros(K)

    def select(self, t):
        eta = cm(self) * 2.0 / math.sqrt(t + 1)
        L = self.L
        x = L.min() - 2.0 / eta                         # start below min so all p_i <= 1
        for _ in range(50):
            d = L - x
            p = 4.0 / (eta ** 2 * d ** 2)
            f = p.sum() - 1.0
            if abs(f) < 1e-10:
                break
            x -= f / (8.0 / eta ** 2 * (1.0 / d ** 3)).sum()
        p = np.clip(p, 1e-12, None)
        self.p = p / p.sum()
        return int(self.rng.choice(K, p=self.p))

    def update(self, t, i, x):
        loss = 1.0 - x
        self.L[i] += (loss - 0.5) / self.p[i] + 0.5     # reduced-variance estimator
        self.L[np.arange(K) != i] += 0.5


class TVGPUCB:
    name = "TV-GP-UCB"
    periodic = False

    def __init__(self, T, rng, arm0):
        self.X, self.tt, self.y = [], [], []
        self.eps = HYPER["gp_eps_periodic"] if self.periodic else HYPER["gp_eps"]
        self.W = HYPER["gp_window_periodic"] if self.periodic else HYPER["gp_window"]

    def _kt(self, t1, t2):
        d = np.abs(t1[:, None] - t2[None, :])
        k = (1 - self.eps) ** (d / 2.0)
        if self.periodic:
            k = k * np.exp(-2 * np.sin(np.pi * d / HYPER["period"]) ** 2 / 0.5 ** 2)
        return k

    def _ka(self, a1, a2):
        return HYPER["gp_sig2"] * np.exp(-0.5 * (a1[:, None] - a2[None, :]) ** 2 / HYPER["gp_len_mA"] ** 2)

    def select(self, t):
        if t < 3:
            return [2, 6, 10][t]                         # 1.0, 2.0, 3.0 mA seed points
        X, tt, y = np.array(self.X[-self.W:]), np.array(self.tt[-self.W:], float), np.array(self.y[-self.W:])
        mu0 = y.mean()
        Kxx = self._ka(X, X) * self._kt(tt, tt) + HYPER["gp_noise"] ** 2 * np.eye(len(X))
        c = cho_factor(Kxx)
        Ksx = self._ka(ARMS, X) * self._kt(np.full(K, float(t)), tt)
        mean = mu0 + Ksx @ cho_solve(c, y - mu0)
        var = HYPER["gp_sig2"] - np.einsum("ij,ji->i", Ksx, cho_solve(c, Ksx.T))
        return int(np.argmax(mean + cm(self) * math.sqrt(HYPER["gp_beta"]) * np.sqrt(np.maximum(var, 1e-12))))

    def update(self, t, i, x):
        self.X.append(ARMS[i])
        self.tt.append(t)
        self.y.append(x)


class TVGPUCBPeriodic(TVGPUCB):
    name = "TV-GP-UCB-periodic*"
    periodic = True


ALGOS = [Fixed, EpsGreedy, UCB1, Thompson, DUCB, SWUCB, Exp3, Exp3P, Exp3S, TsallisINF, TVGPUCB, TVGPUCBPeriodic]


# ------------------------------------------------------------------ runner
def run_one(sur, Alg, scen, T, seed):
    rng = np.random.default_rng(seed)
    g = gain_schedule(scen, T)
    exp_r = np.array([sur.expected_reward(ARMS, gt, LAM) for gt in g])      # T x K
    opt = exp_r.argmax(1)
    alg = Alg(T, rng, int(opt[0]))
    regret = np.empty(T)
    chosen = np.empty(T, int)
    for t in range(T):
        i = alg.select(t)
        r, _ = sur.sample_reward(ARMS[i], g[t], LAM, rng)
        alg.update(t, i, to_gain(r))
        regret[t] = exp_r[t, opt[t]] - exp_r[t, i]
        chosen[t] = i
    unsafe = int(np.sum(ARMS[chosen] * g > UNSAFE_MA + 1e-9))
    track = float(np.mean(np.abs(ARMS[chosen] - ARMS[opt])))
    rec = np.nan
    if scen == "abrupt":
        t0 = T // 3
        ok = np.abs(ARMS[chosen[t0:]] - ARMS[opt[t0:]]) <= 0.25
        ma = np.convolve(ok.astype(float), np.ones(20) / 20, mode="valid")
        hit = np.where(ma >= 0.8)[0]
        rec = float(hit[0]) if len(hit) else float(T - t0)
    return np.cumsum(regret), unsafe, track, rec


def tune(args):
    """Pick each algorithm's multiplier c on VALIDATION seeds (2000+), mean regret over all scenarios.
    Test seeds (1000+) are never used here. The periodic GP inherits TV-GP-UCB's c (cost)."""
    import json
    sur = Surrogate.load()
    T, seeds = args.T, args.tune_seeds
    os.makedirs(RESULTS, exist_ok=True)
    grid = args.grid or C_GRID
    best_path = os.path.join(RESULTS, "step3_tuned_c.json")
    tune_path = os.path.join(RESULTS, "step3_tuning.csv")
    best = json.load(open(best_path)) if (args.only and os.path.exists(best_path)) else {}
    rows = list(csv.DictReader(open(tune_path))) if (args.only and os.path.exists(tune_path)) else []
    for r in rows:
        r["c"], r["mean_val_regret"] = float(r["c"]), float(r["mean_val_regret"])
    t0 = time.time()
    for Alg in ALGOS:
        if Alg in (Fixed, TVGPUCBPeriodic):
            continue
        if args.only and Alg.name not in args.only:
            continue
        scores = {r["c"]: r["mean_val_regret"] for r in rows if r["algorithm"] == Alg.name}
        for c in grid:
            if c in scores:
                continue
            CMULT.clear()
            CMULT[Alg.name] = c
            reg = [run_one(sur, Alg, scen, T, 2000 + s)[0][-1] for scen in SCENARIOS for s in range(seeds)]
            scores[c] = float(np.mean(reg))
            rows.append({"algorithm": Alg.name, "c": c, "mean_val_regret": scores[c]})
            print(f"  {Alg.name:22s} c={c:<5} val regret {scores[c]:7.1f}  [{time.time() - t0:.0f} s]", flush=True)
        best[Alg.name] = min(scores, key=scores.get)
    best[TVGPUCBPeriodic.name] = best.get(TVGPUCB.name, 1.0)
    CMULT.clear()
    rows.sort(key=lambda r: (r["algorithm"], r["c"]))
    with open(os.path.join(RESULTS, "step3_tuning.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["algorithm", "c", "mean_val_regret"])
        w.writeheader()
        w.writerows(rows)
    json.dump(best, open(best_path, "w"), indent=2)
    print("chosen multipliers:", best)
    tried = {}
    for r in rows:
        tried.setdefault(r["algorithm"], []).append(r["c"])
    edge = [k for k, v in best.items() if k in tried and v in (min(tried[k]), max(tried[k]))]
    if edge:
        print("NOTE: chosen at the edge of the grid (true optimum may lie beyond):", edge)
    print("saved results/step3_tuned_c.json")


def run(args):
    sur = Surrogate.load()
    if args.tuned:
        import json
        CMULT.update(json.load(open(os.path.join(RESULTS, "step3_tuned_c.json"))))
        print("using tuned multipliers:", CMULT)
    T = 500 if args.quick else args.T
    seeds = 5 if args.quick else args.seeds
    os.makedirs(RESULTS, exist_ok=True)
    curves, rows = {}, []
    t0 = time.time()
    for scen in SCENARIOS:
        for Alg in ALGOS:
            cr, un, tr, rc = [], [], [], []
            for s in range(seeds):
                c, u, tk, r = run_one(sur, Alg, scen, T, 1000 + s)
                cr.append(c); un.append(u); tr.append(tk); rc.append(r)
            cr = np.array(cr)
            curves[f"{scen}|{Alg.name}"] = cr.mean(0)
            curves[f"{scen}|{Alg.name}|se"] = cr.std(0, ddof=1) / math.sqrt(seeds)
            rows.append({"scenario": scen, "algorithm": Alg.name, "seeds": seeds, "T": T,
                         "final_dyn_regret": float(cr[:, -1].mean()),
                         "se": float(cr[:, -1].std(ddof=1) / math.sqrt(seeds)),
                         "unsafe_evals": float(np.mean(un)), "mean_track_err_mA": float(np.mean(tr)),
                         "recovery_steps": float(np.nanmean(rc)) if scen == "abrupt" else ""})
            print(f"  {scen:9s} {Alg.name:22s} regret {rows[-1]['final_dyn_regret']:8.1f} ± {rows[-1]['se']:5.1f}"
                  f"  unsafe {rows[-1]['unsafe_evals']:6.1f}  track {rows[-1]['mean_track_err_mA']:.2f} mA"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
    tag = "quick" if args.quick else ("tuned" if args.tuned else "main")
    np.savez(os.path.join(RESULTS, f"step3_curves_{tag}.npz"), **curves)
    with open(os.path.join(RESULTS, f"step3_summary_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved results/step3_summary_{tag}.csv and step3_curves_{tag}.npz")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = "quick" if args.quick else ("tuned" if args.tuned else "main")
    d = np.load(os.path.join(RESULTS, f"step3_curves_{tag}.npz"))
    fams = {"fixed (clinician)": "k", "eps-greedy": "#9e9e9e", "UCB1": "#7f7f7f", "Thompson": "#5f5f5f",
            "discounted UCB": "#1f77b4", "sliding-window UCB": "#17becf", "Exp3": "#d62728", "Exp3.P": "#ff7f0e",
            "Exp3.S": "#e377c2", "Tsallis-INF": "#8c564b", "TV-GP-UCB": "#2ca02c", "TV-GP-UCB-periodic*": "#98df8a"}
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(4 * len(SCENARIOS), 3.8), sharey=False)
    for ax, scen in zip(axes, SCENARIOS):
        for name, col in fams.items():
            m = d[f"{scen}|{name}"]
            se = d[f"{scen}|{name}|se"]
            x = np.arange(len(m))
            ax.plot(x, m, color=col, lw=1.6, label=name)
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.15)
        ax.set_title(scen)
        ax.set_xlabel("evaluation")
    axes[0].set_ylabel("cumulative dynamic regret")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    out = os.path.join(RESULTS, f"step3_regret_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("figure:", out)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--plot", action="store_true")
    g.add_argument("--tune", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tuned", action="store_true", help="use multipliers from --tune")
    ap.add_argument("--tune-seeds", type=int, default=6)
    ap.add_argument("--only", nargs="+", help="re-tune only these algorithms (names as printed), merging into saved results")
    ap.add_argument("--grid", type=float, nargs="+", help="multiplier values to try (default C_GRID)")
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    if args.tune:
        tune(args)
    elif args.run:
        run(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
