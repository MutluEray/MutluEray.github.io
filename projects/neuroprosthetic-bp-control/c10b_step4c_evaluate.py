"""
C10-B step 4c — final comparison on the 20 held-out test patients and the G1 decision.

Usage:  python3 c10b_step4c_evaluate.py              (main comparison, ~15 min)
        python3 c10b_step4c_evaluate.py --cal-sweep  (calibration-length sweep, ~10 min)

Main comparison: every tuned family (results/c10b_step4b/tuned_*.json), plus no
stimulation and C0 fixed amplitude, on test patients 40–59 (never used for tuning).
Model-based families use 20-min calibration models (step 4a).

G1 rule (pre-specified in the plan, made operational here before any test run):
  Let OOR = 100 − in_range (% time out of 110–120) and HYP = hyper (% time > 150).
  For C4 and for C4ff versus the BO-tuned PID reference (C2 or C2f, lower tuning-subset J), relative reduction
  RR = 1 − mean(X_model) / mean(X_PID) over the test patients, with a patient
  bootstrap 95 % CI (2000 resamples) and a Wilcoxon signed-rank test.
  G1 PASS if, for C4 or C4ff, RR ≥ 30 % on OOR or on HYP (HYP only if mean HYP of PID
  ≥ 0.5 % time, otherwise too rare to compare), AND the same holds in at least 2 of 3
  tertiles of the patients' true actuator delay, AND (cal sweep) the 20-min model keeps
  ≥ 70 % of the 60-min model's J advantage over PID.
  The calibration part is evaluated by --cal-sweep; the main run reports a provisional G1.

Outputs: results/c10b_step4c/{test_per_patient.csv, summary.md, g1.md, fig_box.png,
         fig_delay.png, fig_example.png, cal_sweep.csv, cal_sweep.md}
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import c10b_controllers as C
import c10b_harness as H
from c10b_plant import new_sim
from c10b_step4b_tune import patient_info

OUT = os.path.join("results", "c10b_step4c")
TEST = list(range(40, 60))
ORDER = ["none", "C0", "C1", "C2", "C2f", "C2ff", "C3", "C3ff", "C4", "C4ff"]
MET = ["J", "in_range", "hyper", "hypo", "tv", "n_changes", "sat", "peak_tilt", "recovery",
       "hyper_return", "ad_peak", "in_range_tilt60", "in_range_return", "in_range_tilt30"]


def load_tuned():
    tuned, jsub = {}, {}
    for f in glob.glob(os.path.join("results", "c10b_step4b", "tuned_*.json")):
        r = json.load(open(f))
        tuned[r["family"]] = r["params"]
        jsub[r["family"]] = r["J_subset"]
    return tuned, jsub


def pid_reference(jsub):
    """The PID comparator for G1 = whichever PID family (C2, C2f) has the lower J on the
    TUNING subset (decided without looking at test patients). Using the stronger PID can
    only make G1 harder to pass."""
    cands = [f for f in ("C2", "C2f") if f in jsub]
    return min(cands, key=lambda f: jsub[f]) if cands else None


def controller_for(fam, tuned):
    if fam == "none":
        return H.NoStim()
    if fam == "C0":
        return H.FixedAmplitude()
    return C.make(fam, tuned[fam])


def boot_rr(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    rr = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        if b[i].mean() > 0:
            rr.append(1 - a[i].mean() / b[i].mean())
    return np.percentile(rr, [2.5, 97.5]) if rr else (np.nan, np.nan)


def rel_red(a, b):
    return 1 - np.mean(a) / np.mean(b) if np.mean(b) > 0 else np.nan


def main_comparison(sim, pop, base, info, sysid, tuned, ref):
    fams = [f for f in ORDER if f in ("none", "C0") or f in tuned]
    missing = [f for f in ORDER if f not in fams]
    rows, ex = [], {}
    t0 = time.time()
    ex_pid = TEST[len(TEST) // 2]
    for pid in TEST:
        pinf = patient_info(info, sysid, pid)
        for fam in fams:
            o, m = H.run_patient(sim, pop, base, pid, controller_for(fam, tuned), info=pinf)
            m["J"] = C.cost_J(m)
            rows.append({"id": pid, "family": fam, "delay_s": pop["patients"][pid]["actuator"]["delay_s"], **m})
            if pid == ex_pid:
                ex[fam] = o
        print(f"[test patient {pid}] {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test_per_patient.csv"), index=False)

    lines = ["# C10-B step 4c — held-out test patients (n = 20)\n",
             f"Families: {', '.join(fams)}" + (f"; not tuned yet: {', '.join(missing)}" if missing else "") + ".\n",
             "Mean ± SD across patients.\n",
             "| family | " + " | ".join(MET) + " |", "|---" * (len(MET) + 1) + "|"]
    for f in fams:
        d = df[df["family"] == f]
        lines.append(f"| {f} | " + " | ".join(f"{d[k].mean():.1f} ± {d[k].std():.1f}" for k in MET) + " |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- G1 (provisional, calibration part from --cal-sweep)
    g = ["# C10-B — G1 publication decision (provisional until --cal-sweep)\n",
         f"PID reference: **{ref}** (lower J on the tuning subset).\n"]
    if ref is None:
        g.append("No PID family tuned yet — no G1 evaluation.")
    else:
        pid_df = df[df["family"] == ref].set_index("id")
        tert = pd.qcut(pid_df["delay_s"].rank(method="first"), 3, labels=["short", "mid", "long"])
        g += ["| model | metric | mean PID | mean model | RR | 95 % CI | Wilcoxon p | RR ≥ 30 % in delay tertiles (short/mid/long) |",
              "|---|---|---|---|---|---|---|---|"]
        any_pass = False
        for f in ("C4", "C4ff", "C3", "C3ff", "C2ff"):
            if f not in fams:
                continue
            md = df[df["family"] == f].set_index("id").loc[pid_df.index]
            for k, lab in (("oor", "OOR"), ("hyper", "HYP")):
                a = (100 - md["in_range"]) if k == "oor" else md["hyper"]
                b = (100 - pid_df["in_range"]) if k == "oor" else pid_df["hyper"]
                if k == "hyper" and b.mean() < 0.5:
                    g.append(f"| {f} | {lab} | {b.mean():.2f} | {a.mean():.2f} | — | — | — | PID hyper < 0.5 % time: not compared |")
                    continue
                rr = rel_red(a, b)
                lo, hi = boot_rr(a.values, b.values)
                try:
                    p = wilcoxon(a.values, b.values).pvalue
                except ValueError:
                    p = np.nan
                tt = [rel_red(a[tert == t], b[tert == t]) for t in ["short", "mid", "long"]]
                n_ok = sum(x >= 0.30 for x in tt if np.isfinite(x))
                ok = rr >= 0.30 and n_ok >= 2 and f in ("C4", "C4ff")
                any_pass |= ok
                g.append(f"| {f} | {lab} | {b.mean():.2f} | {a.mean():.2f} | {100 * rr:+.0f} % | "
                         f"[{100 * lo:+.0f}, {100 * hi:+.0f}] % | {p:.3g} | " +
                         "/".join(f"{100 * x:+.0f}" for x in tt) + f" ({n_ok}/3) |")
        g.append("\nC3/C3ff/C2ff are reported for context; G1 is defined on C4/C4ff.")
        g += ["\n## Exploratory (not part of G1): hypertension-related secondary metrics vs PID reference\n",
              "| family | % time > 150 | > 150 on return to supine (%) | AD-surge peak SBP | peak SBP at tilt-up | Wilcoxon p (> 150 on return) |",
              "|---|---|---|---|---|---|"]
        for f in [x for x in fams if x not in ("none", "C0")]:
            md = df[df["family"] == f].set_index("id").loc[pid_df.index]
            try:
                p = wilcoxon(md["hyper_return"].values, pid_df["hyper_return"].values).pvalue if f != ref else np.nan
            except ValueError:
                p = np.nan
            g.append(f"| {f}{' (ref)' if f == ref else ''} | {md['hyper'].mean():.2f} | {md['hyper_return'].mean():.2f} | "
                     f"{md['ad_peak'].mean():.1f} | {md['peak_tilt'].mean():.1f} | {p:.3g} |")
        g.append(f"\n**Provisional G1 (comparison part): {'PASS' if any_pass else 'FAIL'}** — the calibration part "
                 "is decided by `--cal-sweep`.")
    open(os.path.join(OUT, "g1.md"), "w").write("\n".join(g) + "\n")
    print("\n".join(g))
    figures(df, fams, ex, ex_pid)


def figures(df, fams, ex, ex_pid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(18, 3.6))
    for a_, k, lab in zip(ax, ["in_range", "hyper", "hypo", "tv", "J"],
                          ["% time in 110–120", "% time > 150", "% time < 90", "amplitude total variation", "cost J"]):
        a_.boxplot([df[df["family"] == f][k] for f in fams], tick_labels=fams)
        a_.set_title(lab, fontsize=9)
        a_.tick_params(axis="x", labelsize=7, rotation=45)
    fig.suptitle("Held-out test patients (n = 20), BO-tuned controllers", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_box.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for f in [x for x in fams if x not in ("none", "C0")]:
        d = df[df["family"] == f].sort_values("delay_s")
        ax[0].plot(d["delay_s"], 100 - d["in_range"], "o-", ms=3, lw=0.8, label=f)
        ax[1].plot(d["delay_s"], d["J"], "o-", ms=3, lw=0.8, label=f)
    ax[0].set_xlabel("true actuator delay (s)"); ax[0].set_ylabel("% time out of 110–120")
    ax[1].set_xlabel("true actuator delay (s)"); ax[1].set_ylabel("cost J")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_delay.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for f, o in ex.items():
        if f in ("none",):
            continue
        ax[0].plot(o["t"], o["sbp"], lw=0.9, label=f)
        ax[1].plot(o["t"], o["u"], lw=0.9, label=f)
    o = next(iter(ex.values()))
    ax[2].plot(o["t"], o["angle"], color="k")
    ax[0].axhspan(H.LO, H.HI, color="tab:green", alpha=0.12); ax[0].axhline(150, color="tab:red", ls="--", lw=0.8)
    ax[0].set_ylabel("SBP (mmHg)"); ax[1].set_ylabel("amplitude u"); ax[2].set_ylabel("tilt (deg)")
    ax[2].set_xlabel("time (s)")
    for a_ in ax:
        for a, b in H.DROPOUTS:
            a_.axvspan(a, b, color="0.85")
        a_.axvspan(H.AD[0], H.AD[0] + sum(H.AD[1:]), color="tab:red", alpha=0.06)
    ax[0].legend(frameon=False, fontsize=7, ncol=7)
    ax[0].set_title(f"Test patient {ex_pid}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_example.png"), dpi=140); plt.close(fig)


def cal_sweep(sim, pop, base, info, sysid, tuned, ref):
    rows = []
    t0 = time.time()
    for pid in TEST:
        pinf = patient_info(info, sysid, pid)
        _, m = H.run_patient(sim, pop, base, pid, controller_for(ref, tuned), info=pinf)
        rows.append({"id": pid, "family": ref, "cal_min": 0, "J": C.cost_J(m), **m})
        for fam in [f for f in ("C4ff", "C4") if f in tuned]:
            for L in (5, 10, 20, 60):
                pinf = patient_info(info, sysid, pid, cal_min=L)
                _, m = H.run_patient(sim, pop, base, pid, controller_for(fam, tuned), info=pinf)
                rows.append({"id": pid, "family": fam, "cal_min": L, "J": C.cost_J(m), **m})
        print(f"[cal sweep, patient {pid}] {time.time() - t0:.0f} s")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "cal_sweep.csv"), index=False)
    j_pid = df[df["family"] == ref]["J"].mean()
    lines = ["# C10-B — calibration-length sweep (test patients)\n",
             f"Mean J of tuned PID ({ref}): {j_pid:.1f}. Advantage = mean J(PID) − mean J(model, T_cal).\n",
             "| family | T_cal (min) | mean J | in_range | hyper | advantage over PID | share of 60-min advantage |",
             "|---|---|---|---|---|---|---|"]
    ok_any = False
    for fam in [f for f in ("C4ff", "C4") if f in tuned]:
        adv60 = j_pid - df[(df["family"] == fam) & (df["cal_min"] == 60)]["J"].mean()
        for L in (5, 10, 20, 60):
            d = df[(df["family"] == fam) & (df["cal_min"] == L)]
            adv = j_pid - d["J"].mean()
            share = adv / adv60 if adv60 > 0 else np.nan
            if L == 20 and adv60 > 0 and share >= 0.70:
                ok_any = True
            lines.append(f"| {fam} | {L} | {d['J'].mean():.1f} | {d['in_range'].mean():.1f} | {d['hyper'].mean():.2f} | "
                         f"{adv:+.1f} | {100 * share:.0f} % |")
    lines.append(f"\n**G1 calibration part (20-min model keeps ≥ 70 % of the 60-min advantage, C4 or C4ff): "
                 f"{'PASS' if ok_any else 'FAIL'}**")
    open(os.path.join(OUT, "cal_sweep.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-sweep", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pop, base, info = H.load_population()
    sysid = json.load(open(os.path.join("data", "c10b_sysid.json")))
    tuned, jsub = load_tuned()
    ref = pid_reference(jsub)
    print(f"tuned families found: {sorted(tuned)}; PID reference: {ref}")
    sim = new_sim()
    if args.cal_sweep:
        cal_sweep(sim, pop, base, info, sysid, tuned, ref)
    else:
        main_comparison(sim, pop, base, info, sysid, tuned, ref)
    print(f"[done] {OUT}/")


if __name__ == "__main__":
    main()
