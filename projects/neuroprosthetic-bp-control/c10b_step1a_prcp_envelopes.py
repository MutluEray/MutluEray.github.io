"""
C10-B step 1a — PRCP event-locked response envelopes (the G0 validation target).

For every maneuver in PRCP (rapid tilt up, slow tilt up, stand-up, and the
returns to supine) this script extracts beat-to-beat SBP, MAP and HR changes
relative to a pre-event baseline, resampled to 1 Hz, and summarises them across
subjects (per-subject mean first, then median and IQR across subjects).

Event timing:
  - tilts: detected from the 'Angle' channel (10 % / 50 % / 90 % crossings),
    so mislabelled or duplicated annotations don't matter; transition duration
    decides rapid (< 10 s) vs slow (>= 20 s)
  - stand-up / back to supine: from the 'anI' annotations (the table angle
    does not change)

Cleaning:
  - beats marked invalid in step 0a are dropped
  - beats whose interbeat interval deviates > 30 % from the local median
    (11 beats) are dropped (removes the HR spikes from missed onsets)
  - Finapres off / fade / artefact / signal-break periods are excluded
    (from the annotation to the matching 'on/back' annotation + 60 s, or ±60 s)

Inputs (from step 0a):
  data/prcp/*.hea/.dat
  results/c10b_step0/prcp_beats.csv, prcp_annotations.csv
Outputs:
  results/c10b_step1a/events.csv              one row per detected event
  results/c10b_step1a/envelopes.csv           t, event type, signal, median, q25, q75, n_subjects
  results/c10b_step1a/metrics.md              nadir / steady-state table per event type
  results/c10b_step1a/fig_envelopes.png
"""

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join("data", "prcp")
IN_DIR = os.path.join("results", "c10b_step0")
OUT_DIR = os.path.join("results", "c10b_step1a")
os.makedirs(OUT_DIR, exist_ok=True)

PRE, POST = 60, 240            # seconds around the event
BASE = (-60, -10)              # baseline window (s)
T = np.arange(-PRE, POST + 1)  # 1 Hz grid
SIGNALS = ["sbp", "map", "hr"]
BAD_WORDS = ("finapres", "fade", "artifact", "artefact", "break in signal")


# ---------------------------------------------------------------- cleaning
def clean_beats(b):
    b = b[b["valid"] == 1].copy().sort_values("t_s")
    ibi = 60.0 / b["hr"].to_numpy()
    med = pd.Series(ibi).rolling(11, center=True, min_periods=3).median().to_numpy()
    ok = np.abs(ibi - med) / med < 0.30
    return b[ok]


def bad_intervals(ann):
    """Excluded time intervals (s) from Finapres / artefact annotations."""
    out = []
    a = ann.sort_values("t_s").reset_index(drop=True)
    for i, row in a.iterrows():
        txt = str(row["aux_note"]).lower()
        if not any(w in txt for w in BAD_WORDS):
            continue
        if "on" in txt.split()[-1:] or "back" in txt:
            continue  # handled as the end of an 'off'/'fade' interval
        t0 = row["t_s"]
        end = t0 + 60.0
        # look for a matching 'switched ... on' / 'back' within 5 min
        for _, r2 in a[(a["t_s"] > t0) & (a["t_s"] < t0 + 300)].iterrows():
            t2 = str(r2["aux_note"]).lower()
            if ("on" in t2.split()[-1:] or "back" in t2) and ("finapres" in t2 or "signal" in t2):
                end = r2["t_s"] + 60.0
                break
        out.append((t0 - 60.0, end))
    return out


def in_bad(t, intervals):
    return any(a <= t <= b for a, b in intervals)


# ---------------------------------------------------------------- events
def tilt_events_from_angle(rec_path):
    rec = wfdb.rdrecord(rec_path)
    names = [n.lower() for n in rec.sig_name]
    if "angle" not in names:
        return [], None
    fs = float(rec.fs)
    ang = rec.p_signal[:, names.index("angle")].astype(float)
    # 1 Hz median to remove noise
    n = int(len(ang) // fs)
    a1 = np.array([np.nanmedian(ang[int(i * fs):int((i + 1) * fs)]) for i in range(n)])
    lo, hi = np.nanpercentile(a1, 5), np.nanpercentile(a1, 95)
    rng = hi - lo
    info = (float(lo), float(hi))
    if rng < 5:  # no real tilt in this record
        return [], info
    f = (a1 - lo) / rng
    ev = []
    for direction in ("up", "down"):
        s = f if direction == "up" else 1 - f
        cross = np.where((s[:-1] < 0.5) & (s[1:] >= 0.5))[0]
        for c in cross:
            i10 = c
            while i10 > 0 and s[i10] > 0.1:
                i10 -= 1
            i90 = c + 1
            while i90 < len(s) - 1 and s[i90] < 0.9:
                i90 += 1
            dur = i90 - i10
            kind = "rapid" if dur < 10 else ("slow" if dur >= 20 else "medium")
            ev.append((float(i10), f"{kind}_tilt_{direction}", float(dur)))
    return ev, info


def posture_events_from_ann(ann):
    ev = []
    for _, r in ann.iterrows():
        txt = str(r["aux_note"]).lower()
        if "stand" in txt:
            ev.append((float(r["t_s"]), "stand_up", np.nan))
        elif "supine" in txt and "tilt" not in txt:
            ev.append((float(r["t_s"]), "back_to_supine", np.nan))
    return ev


# ---------------------------------------------------------------- traces
def event_trace(b, t0):
    t = b["t_s"].to_numpy() - t0
    m = (t >= -PRE - 5) & (t <= POST + 5)
    if m.sum() < 50:
        return None
    out = {}
    for s in SIGNALS:
        y = np.interp(T, t[m], b[s].to_numpy()[m], left=np.nan, right=np.nan)
        base = np.nanmean(y[(T >= BASE[0]) & (T <= BASE[1])])
        out[s] = y - base
        out[s + "_base"] = base
    return out


def main():
    beats = pd.read_csv(os.path.join(IN_DIR, "prcp_beats.csv"), dtype={"record": str})
    anns = pd.read_csv(os.path.join(IN_DIR, "prcp_annotations.csv"), dtype={"record": str})
    rows, traces = [], []
    for rec in sorted(beats["record"].unique()):
        b = clean_beats(beats[beats["record"] == rec])
        a = anns[anns["record"] == rec]
        bad = bad_intervals(a)
        tilt_ev, info = tilt_events_from_angle(os.path.join(DATA_DIR, rec))
        print(f"[{rec}] angle range {info}, tilt events {len(tilt_ev)}, "
              f"excluded intervals {[(round(x), round(y)) for x, y in bad]}")
        for t0, kind, dur in tilt_ev + posture_events_from_ann(a):
            if any(in_bad(t0 + dt, bad) for dt in (-PRE, 0, 60, 120)):
                rows.append((rec, t0, kind, dur, "excluded_artefact"))
                continue
            tr = event_trace(b, t0)
            if tr is None:
                rows.append((rec, t0, kind, dur, "excluded_no_beats"))
                continue
            rows.append((rec, t0, kind, dur, "ok"))
            traces.append((rec, kind, tr))

    ev = pd.DataFrame(rows, columns=["record", "t0_s", "type", "transition_s", "status"])
    ev.to_csv(os.path.join(OUT_DIR, "events.csv"), index=False)
    print("\n[events]\n", ev.groupby(["type", "status"]).size())

    types = ["rapid_tilt_up", "slow_tilt_up", "stand_up", "rapid_tilt_down", "slow_tilt_down", "back_to_supine"]
    env_rows, met = [], []
    for k in types:
        tk = [x for x in traces if x[1] == k]
        if not tk:
            continue
        subj = sorted({x[0] for x in tk})
        for s in SIGNALS:
            per_subj = np.array([np.nanmean([x[2][s] for x in tk if x[0] == r], axis=0) for r in subj])
            med = np.nanmedian(per_subj, axis=0)
            q25, q75 = np.nanpercentile(per_subj, 25, axis=0), np.nanpercentile(per_subj, 75, axis=0)
            for i, t in enumerate(T):
                env_rows.append((k, s, int(t), med[i], q25[i], q75[i], len(subj)))
            early = (T >= 0) & (T <= 40)
            late = (T >= 120) & (T <= 180)
            ext = np.nanmin if s != "hr" else np.nanmax
            ext_vals = [ext(p[early]) for p in per_subj]
            t_ext = [int(T[early][np.nanargmin(p[early]) if s != "hr" else np.nanargmax(p[early])])
                     for p in per_subj]
            ss = [np.nanmean(p[late]) for p in per_subj]
            base = [np.nanmean([x[2][s + "_base"] for x in tk if x[0] == r]) for r in subj]
            met.append((k, s, len(subj), len(tk), np.median(base), np.median(ext_vals),
                        np.percentile(ext_vals, 25), np.percentile(ext_vals, 75),
                        np.median(t_ext), np.median(ss), np.percentile(ss, 25), np.percentile(ss, 75)))

    env = pd.DataFrame(env_rows, columns=["type", "signal", "t_s", "median", "q25", "q75", "n_subjects"])
    env.to_csv(os.path.join(OUT_DIR, "envelopes.csv"), index=False)

    lines = ["# PRCP event-locked responses (C10-B step 1a)\n",
             "Per-subject mean first, then median [IQR] across subjects. Δ relative to the -60…-10 s baseline. "
             "Early extreme = nadir for SBP/MAP, peak for HR, within 0–40 s. Steady state = mean of 120–180 s.\n",
             "| event | signal | subjects | events | baseline | early extreme Δ | t of extreme (s) | steady-state Δ |",
             "|---|---|---|---|---|---|---|---|"]
    for m in met:
        lines.append(f"| {m[0]} | {m[1]} | {m[2]} | {m[3]} | {m[4]:.1f} | {m[5]:.1f} [{m[6]:.1f}, {m[7]:.1f}] | "
                     f"{m[8]:.0f} | {m[9]:.1f} [{m[10]:.1f}, {m[11]:.1f}] |")
    with open(os.path.join(OUT_DIR, "metrics.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    show = [k for k in ["rapid_tilt_up", "slow_tilt_up", "stand_up"] if k in env["type"].unique()]
    fig, ax = plt.subplots(3, len(show), figsize=(4 * len(show), 7.5), sharex=True, squeeze=False)
    units = {"sbp": "ΔSBP (mmHg)", "map": "ΔMAP (mmHg)", "hr": "ΔHR (bpm)"}
    for j, k in enumerate(show):
        for i, s in enumerate(SIGNALS):
            e = env[(env["type"] == k) & (env["signal"] == s)]
            ax[i, j].fill_between(e["t_s"], e["q25"], e["q75"], alpha=0.25, lw=0)
            ax[i, j].plot(e["t_s"], e["median"], lw=1.4)
            ax[i, j].axvline(0, color="0.5", lw=0.8, ls="--")
            ax[i, j].axhline(0, color="0.8", lw=0.6)
            if j == 0:
                ax[i, j].set_ylabel(units[s])
        ax[0, j].set_title(f"{k.replace('_', ' ')} (n={int(e['n_subjects'].iloc[0])} subjects)")
        ax[-1, j].set_xlabel("time from event (s)")
    fig.suptitle("PRCP healthy responses: median and IQR across subjects (G0 target)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_envelopes.png"), dpi=140)
    print(f"\n[done] wrote {OUT_DIR}/events.csv, envelopes.csv, metrics.md, fig_envelopes.png")


if __name__ == "__main__":
    main()
