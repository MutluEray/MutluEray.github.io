"""Stroke cohort from the authors' sourcedata .mat files (per-patient labels).

Why: the EDF route takes left/right labels from one task-level template, and that failed the
label sanity check. The authors' own code (code/PlotAccuracy.m) loads sourcedata/sub-XX/*.mat
with a struct `eeg` holding `rawdata` (trials x 33 channels x samples, column 33 = stage marker,
2 = imagery onset) and `label` (1 = left, 2 = right). This script does the same and writes the
same data/processed/stroke.npz that run_ts.py reads, with the same pipeline as before
(4-40 Hz, 128 Hz, 0.5-3.5 s after imagery onset, CPz/EOG/marker dropped).

It also prints, per patient, how often the old template labels agree with the real ones.

  unzip -q data/raw/stroke/sourcedata.zip -d data/raw/stroke
  python3 prep_stroke_mat.py --root data/raw/stroke
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from prep_stroke import (DROP, is_cortical, is_infratentorial, label_sanity_check, lesion_hemisphere,
                         template)
from src.channels import mirror_closed, normalize

# column order of rawdata = EDF channel order (code: channel = [1:17 19:30], 18 = CPz, 33 = marker)
CHANNELS = ["FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4", "FT7", "FT8", "Cz", "C3",
            "C4", "T3", "T4", "CPz", "CP3", "CP4", "TP7", "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1",
            "O2", "HEOL", "HEOR", "MARKER"]


def load_mat(path: Path):
    """Return rawdata as (trials, 33, samples) and labels (trials,) in {1, 2}."""
    try:
        import scipy.io as sio
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        eeg = d["eeg"]
        raw, label = np.asarray(eeg.rawdata, float), np.asarray(eeg.label).ravel()
    except NotImplementedError:          # MATLAB v7.3 file
        import h5py
        with h5py.File(path, "r") as f:
            raw = np.array(f["eeg"]["rawdata"], float).transpose()   # h5py reverses axes
            label = np.array(f["eeg"]["label"]).ravel()
    n_trials = len(label)
    ch_axis = [i for i, s in enumerate(raw.shape) if s == len(CHANNELS)]
    tr_axis = [i for i, s in enumerate(raw.shape) if s == n_trials]
    if not ch_axis or not tr_axis:
        raise RuntimeError(f"{path.name}: rawdata shape {raw.shape} does not match {n_trials} trials x 33 channels")
    t_ax = tr_axis[0]
    c_ax = next(i for i in ch_axis if i != t_ax)
    s_ax = ({0, 1, 2} - {t_ax, c_ax}).pop()
    return raw.transpose(t_ax, c_ax, s_ax), label.astype(int)


def load_subject(path: Path, fs: float, l_freq, h_freq, sfreq, tmin, tmax):
    raw3, label = load_mat(path)
    n_tr, _, n_s = raw3.shape
    cont = raw3.transpose(1, 0, 2).reshape(len(CHANNELS), n_tr * n_s)      # trials back to back
    marker = np.rint(cont[-1]).astype(int)
    is_mi = marker == 2
    onsets = np.flatnonzero(is_mi & ~np.r_[False, is_mi[:-1]])
    if len(onsets) != n_tr:
        # fall back to one onset per trial: first sample of value 2 inside each trial
        onsets = []
        for k in range(n_tr):
            idx = np.flatnonzero(np.rint(raw3[k, -1]).astype(int) == 2)
            if len(idx) == 0:
                raise RuntimeError(f"{path.name}: trial {k + 1} has no imagery marker")
            onsets.append(k * n_s + idx[0])
        onsets = np.asarray(onsets)
    eeg_idx = [i for i, c in enumerate(CHANNELS) if normalize(c) not in DROP and c != "MARKER"]
    names = [CHANNELS[i] for i in eeg_idx]
    info = mne.create_info(names, fs, "eeg")
    raw = mne.io.RawArray(cont[eeg_idx] * 1e-6, info, verbose="ERROR")          # uV -> V
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    events = np.c_[onsets, np.zeros_like(onsets), label]
    raw, events = raw.resample(sfreq, events=events, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id={"left": 1, "right": 2}, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose="ERROR")
    n_times = int(round((tmax - tmin) * sfreq))
    X = epochs.get_data(copy=False)[:, :, :n_times] * 1e6
    y = epochs.events[:, 2] - 1
    return X.astype(np.float32), y.astype(np.int64), names, label - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/stroke")
    p.add_argument("--out", default="data/processed/stroke.npz")
    p.add_argument("--fs", type=float, default=500.0)
    p.add_argument("--l-freq", type=float, default=4.0)
    p.add_argument("--h-freq", type=float, default=40.0)
    p.add_argument("--sfreq", type=float, default=128.0)
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    args = p.parse_args()
    root = Path(args.root)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    parts = pd.read_csv(root / "participants.tsv", sep="\t").set_index("Participant_ID")
    _, t_labels = template(root / "task-motor-imagery_events.tsv")
    mats = sorted(root.rglob("sub-*/*.mat"))
    if not mats:
        raise SystemExit(f"no sub-*/*.mat under {root}: unzip sourcedata.zip there first")

    cols = {k: [] for k in ("X", "y", "subject", "affected", "nihss", "mbi", "mrs", "lesion", "infra",
                            "cortical", "hand")}
    ch_ref, skipped, agree, seen = None, [], [], set()
    for path in mats:
        sid = next(part for part in path.parts if part.startswith("sub-"))[:6]
        subj = int(sid.split("-")[1])
        if subj in seen:
            print(f"{sid}: more than one .mat file, keeping the first ({path.name} ignored)", flush=True)
            continue
        try:
            X, y, ch, lab = load_subject(path, args.fs, args.l_freq, args.h_freq, args.sfreq, args.tmin, args.tmax)
        except Exception as e:  # noqa: BLE001
            print(f"{sid}: SKIPPED ({e})", flush=True)
            skipped.append(sid)
            continue
        seen.add(subj)
        ch_ref = ch_ref or ch
        row = parts.loc[sid]
        side = {"left": "L", "right": "R"}[str(row["ParalysisSide"]).strip().lower()]
        n = len(y)
        tpl_agree = float((lab == t_labels).mean()) if len(lab) == len(t_labels) else np.nan
        agree.append(tpl_agree)
        cols["X"].append(X); cols["y"].append(y); cols["subject"].append(np.full(n, subj))
        cols["affected"].append(np.full(n, side))
        for key, col in (("nihss", "NIHSS"), ("mbi", "MBI"), ("mrs", "mRS")):
            cols[key].append(np.full(n, float(row[col])))
        cols["lesion"].append(np.full(n, lesion_hemisphere(row["StrokeLocation"])))
        cols["infra"].append(np.full(n, is_infratentorial(row["StrokeLocation"])))
        cols["cortical"].append(np.full(n, is_cortical(row["StrokeLocation"])))
        cols["hand"].append(np.full(n, str(row["Handedness"]).strip().lower()))
        print(f"{sid}: {n} trials (left {int((y == 0).sum())}, right {int((y == 1).sum())}), paralysis {side}, "
              f"template-label agreement {tpl_agree:.2f}, median channel SD {np.median(X.std(axis=(0, 2))):.1f} uV",
              flush=True)

    X = np.concatenate(cols["X"]); y = np.concatenate(cols["y"]); subject = np.concatenate(cols["subject"])
    np.savez(args.out, X=X, y=y, subject=subject, ch_names=np.array(ch_ref), sfreq=args.sfreq,
             affected_side=np.concatenate(cols["affected"]), nihss=np.concatenate(cols["nihss"]),
             mbi=np.concatenate(cols["mbi"]), mrs=np.concatenate(cols["mrs"]),
             lesion_hemisphere=np.concatenate(cols["lesion"]), infratentorial=np.concatenate(cols["infra"]),
             cortical=np.concatenate(cols["cortical"]), handedness=np.concatenate(cols["hand"]))
    sides = pd.Series(np.concatenate(cols["affected"]), index=subject).groupby(level=0).first()
    print(f"\nsaved {args.out}: X {X.shape}, {len(np.unique(subject))} subjects; paralysis side {sides.value_counts().to_dict()}")
    print(f"skipped ({len(skipped)}): {' '.join(skipped) if skipped else 'none'}")
    a = np.array(agree)
    print(f"template vs real labels: mean agreement {np.nanmean(a):.2f}; "
          f"{int(np.sum(a == 1))} patients identical, {int(np.sum(a < 0.6))} patients <= chance-level agreement")

    shared = [normalize(c) for c in ch_ref]
    phys = Path(args.out).parent / "physionet.npz"
    if phys.exists():
        phys_names = {normalize(c) for c in np.load(phys)["ch_names"]}
        shared = [c for c in shared if c in phys_names]
    keep, _ = mirror_closed(shared)
    (Path(args.out).parent / "shared_channels.json").write_text(json.dumps({"channels": keep}, indent=1))
    print(f"shared mirror-closed channels ({len(keep)})")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_sanity_check(X, y, subject, args.sfreq)


if __name__ == "__main__":
    main()
