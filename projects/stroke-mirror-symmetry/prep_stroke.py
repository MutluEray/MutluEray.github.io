"""Stroke cohort: Liu et al. 2024 (figshare 21679035), 50 acute stroke patients, left vs right hand MI.

Same pipeline as the healthy cohort: 4-40 Hz band-pass, resample to 128 Hz, epoch 0.5-3.5 s
after motor-imagery onset, microvolts. CPz (reference), EOG and the marker channel are dropped,
leaving 29 EEG channels.

Labels. The EDFs carry only a stage marker (last channel: 1 instruction, 2 imagery, 3 break).
Left/right comes from the task-level events.tsv, which lists one fixed L/R order. The script
checks that each recording's imagery onsets match that template's timing, then runs a
within-subject label sanity check (if labels were wrong, decoding would sit at 50%).

Also writes data/processed/shared_channels.json (stroke montage present in physionet.npz).

  python3 prep_stroke.py --root data/raw/stroke
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.channels import mirror_closed, normalize

DROP = {"CPZ", "HEOL", "HEOR", "VEOR", "VEOL", ""}


def lesion_hemisphere(loc: str) -> str:
    s = str(loc).lower()
    left, right = "left" in s, "right" in s
    return "L" if left and not right else "R" if right and not left else ""


def is_infratentorial(loc: str) -> int:
    s = str(loc).lower()
    return int(any(k in s for k in ("pons", "medulla", "cerebell")))


def is_cortical(loc: str) -> int:
    s = str(loc).lower()
    return int(any(k in s for k in ("cortex", "lobe")))


def template(events_tsv: Path):
    ev = pd.read_csv(events_tsv, sep="\t")
    mi = ev[ev["value"] == 2]
    return mi["onset"].to_numpy(float) / 1000.0, mi["trial_type"].to_numpy(int) - 1  # ms -> s, 0=left


def mi_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    """Rising edges of marker value 2 (imagery onset). Marker values are 1/2/3 in microvolt units
    after MNE's scaling; rounding to the nearest integer removes EDF quantisation glitches."""
    marker_name = "" if "" in raw.ch_names else raw.ch_names[-1]
    m = raw.get_data(picks=[raw.ch_names.index(marker_name)])[0]
    scale = 1e-6 if np.nanmax(np.abs(m)) < 1e-3 else 1.0
    level = np.rint(m / scale).astype(int)
    is_mi = level == 2
    return np.flatnonzero(is_mi & ~np.r_[False, is_mi[:-1]])


def match_to_template(onsets_s: np.ndarray, t_onsets: np.ndarray, tol: float):
    """Match each template imagery onset to a recorded onset (seconds).

    Tries every recorded onset as the anchor for template trial 1 and keeps the anchor that
    matches the most template trials within `tol`. Returns (indices into onsets_s or None,
    max error over matched trials, number matched, first unmatched template trial).
    """
    rel_tpl = t_onsets - t_onsets[0]
    best = (None, np.inf, -1, 0)
    for anchor in onsets_s:
        target = anchor + rel_tpl
        j = np.searchsorted(onsets_s, target)
        cand = np.stack([np.clip(j - 1, 0, len(onsets_s) - 1), np.clip(j, 0, len(onsets_s) - 1)])
        d = np.abs(onsets_s[cand] - target)
        pick = cand[np.argmin(d, axis=0), np.arange(len(target))]
        err = np.abs(onsets_s[pick] - target)
        ok = err <= tol
        n_ok = int(ok.sum())
        if n_ok > best[2] or (n_ok == best[2] and err[ok].max(initial=0) < best[1]):
            first_bad = int(np.argmin(ok)) + 1 if not ok.all() else 0
            best = (pick if ok.all() and len(np.unique(pick)) == len(pick) else None,
                    float(err[ok].max(initial=0)), n_ok, first_bad)
    return best


def load_subject(edf: Path, t_onsets, t_labels, l_freq, h_freq, sfreq, tmin, tmax, tol):
    """Returns (X, y, ch_names, info) or (None, None, None, info) if the recording is excluded."""
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    onsets = mi_onsets(raw)
    fs = raw.info["sfreq"]
    pick, err, n_ok, first_bad = match_to_template(onsets / fs, t_onsets, tol)
    info = {"n_markers": len(onsets), "n_matched": n_ok, "timing_err": err, "first_bad": first_bad}
    if pick is None:
        return None, None, None, info
    onsets = onsets[pick]

    eeg = [c for c in raw.ch_names if normalize(c) not in DROP]
    raw.pick(eeg)
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    events = np.c_[onsets, np.zeros_like(onsets), t_labels + 1]
    raw, events = raw.resample(sfreq, events=events, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id={"left": 1, "right": 2}, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose="ERROR")
    n_times = int(round((tmax - tmin) * sfreq))
    X = epochs.get_data(copy=False)[:, :, :n_times] * 1e6
    y = epochs.events[:, 2] - 1
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names, info


def label_sanity_check(X, y, subject, sfreq):
    """Within-subject 5-fold CV with tangent space + logistic regression (mu/beta band)."""
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from scipy.signal import butter, sosfiltfilt
    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    sos = butter(4, [8, 30], btype="bandpass", fs=sfreq, output="sos")
    Xb = sosfiltfilt(sos, X, axis=-1)
    accs = []
    for s in np.unique(subject):
        m = subject == s
        clf = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(C=0.1, max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        accs.append(cross_val_score(clf, Xb[m], y[m], cv=cv, scoring="balanced_accuracy").mean())
    accs = np.array(accs)
    p = wilcoxon(accs - 0.5).pvalue
    print(f"\nlabel sanity check (within-subject, 5-fold, TS+LR 8-30 Hz): mean {accs.mean():.3f}, "
          f"median {np.median(accs):.3f}, {int((accs > 0.5).sum())}/{len(accs)} subjects above 0.5, "
          f"Wilcoxon vs 0.5 p = {p:.2g}")
    print("  (labels scrambled -> mean ~0.50; the dataset paper reports 0.56-0.72 within-subject)")
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/stroke")
    p.add_argument("--out", default="data/processed/stroke.npz")
    p.add_argument("--l-freq", type=float, default=4.0)
    p.add_argument("--h-freq", type=float, default=40.0)
    p.add_argument("--sfreq", type=float, default=128.0)
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    p.add_argument("--max-timing-error", type=float, default=0.05,
                   help="seconds; recordings whose imagery onsets deviate more from the label template are excluded")
    args = p.parse_args()
    root = Path(args.root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    parts = pd.read_csv(root / "participants.tsv", sep="\t").set_index("Participant_ID")
    t_onsets, t_labels = template(root / "task-motor-imagery_events.tsv")
    print(f"template: {len(t_onsets)} imagery trials, left {int((t_labels == 0).sum())}, right {int((t_labels == 1).sum())}")

    edfs = sorted(root.rglob("sub-*_task-motor-imagery_eeg.edf"))
    cols = {k: [] for k in ("X", "y", "subject", "affected", "nihss", "mbi", "mrs", "lesion",
                            "infra", "cortical", "hand")}
    ch_ref = None
    excluded = []
    for edf in edfs:
        sid = edf.name.split("_")[0]
        subj = int(sid.split("-")[1])
        row = parts.loc[sid]
        try:
            X, y, ch, info = load_subject(edf, t_onsets, t_labels, args.l_freq, args.h_freq,
                                          args.sfreq, args.tmin, args.tmax, args.max_timing_error)
        except Exception as e:  # noqa: BLE001
            print(f"{sid}: EXCLUDED (could not load: {e})", flush=True)
            excluded.append(sid)
            continue
        if X is None:
            print(f"{sid}: EXCLUDED: {info['n_markers']} imagery markers, only {info['n_matched']}/{len(t_onsets)} "
                  f"match the label template within {args.max_timing_error * 1000:.0f} ms "
                  f"(first unmatched trial {info['first_bad']})", flush=True)
            excluded.append(sid)
            continue
        err = info["timing_err"]
        if ch_ref is None:
            ch_ref = ch
        elif ch != ch_ref:
            raise RuntimeError(f"{sid}: channel order differs")
        side = {"left": "L", "right": "R"}[str(row["ParalysisSide"]).strip().lower()]
        n = len(y)
        cols["X"].append(X); cols["y"].append(y); cols["subject"].append(np.full(n, subj))
        cols["affected"].append(np.full(n, side))
        for key, col in (("nihss", "NIHSS"), ("mbi", "MBI"), ("mrs", "mRS")):
            cols[key].append(np.full(n, float(row[col])))
        cols["lesion"].append(np.full(n, lesion_hemisphere(row["StrokeLocation"])))
        cols["infra"].append(np.full(n, is_infratentorial(row["StrokeLocation"])))
        cols["cortical"].append(np.full(n, is_cortical(row["StrokeLocation"])))
        cols["hand"].append(np.full(n, str(row["Handedness"]).strip().lower()))
        flag = f"  ({info['n_markers']} markers, extra ignored)" if info["n_markers"] != len(t_onsets) else ""
        print(f"{sid}: {n} trials, paralysis {side}, timing error vs template {err * 1000:.1f} ms, "
              f"median channel SD {np.median(X.std(axis=(0, 2))):.1f} uV{flag}", flush=True)

    X = np.concatenate(cols["X"]); y = np.concatenate(cols["y"]); subject = np.concatenate(cols["subject"])
    out = dict(X=X, y=y, subject=subject, ch_names=np.array(ch_ref), sfreq=args.sfreq,
               affected_side=np.concatenate(cols["affected"]),
               nihss=np.concatenate(cols["nihss"]), mbi=np.concatenate(cols["mbi"]),
               mrs=np.concatenate(cols["mrs"]), lesion_hemisphere=np.concatenate(cols["lesion"]),
               infratentorial=np.concatenate(cols["infra"]), cortical=np.concatenate(cols["cortical"]),
               handedness=np.concatenate(cols["hand"]))
    np.savez(args.out, **out)
    per_subj = pd.DataFrame({"subject": subject, "side": out["affected_side"]}).drop_duplicates()
    print(f"\nsaved {args.out}: X {X.shape}, {len(np.unique(subject))} subjects, channels {ch_ref}")
    print(f"paralysis side: {per_subj.side.value_counts().to_dict()}")
    print(f"excluded ({len(excluded)}): {' '.join(excluded) if excluded else 'none'}")
    (Path(args.out).parent / "stroke_excluded.json").write_text(json.dumps({"timing_mismatch": excluded}, indent=1))

    # channels shared with the healthy cohort
    shared = [normalize(c) for c in ch_ref]
    phys = Path(args.out).parent / "physionet.npz"
    if phys.exists():
        phys_names = {normalize(c) for c in np.load(phys)["ch_names"]}
        shared = [c for c in shared if c in phys_names]
    keep, drop = mirror_closed(shared)
    (Path(args.out).parent / "shared_channels.json").write_text(json.dumps({"channels": keep}, indent=1))
    print(f"shared mirror-closed channels ({len(keep)}): {' '.join(keep)}" + (f"; dropped {drop}" if drop else ""))
    if not phys.exists():
        print("note: physionet.npz not found, shared list = stroke montage only; re-run after prep_physionet.py")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_sanity_check(X, y, subject, args.sfreq)


if __name__ == "__main__":
    main()
