"""Healthy reference cohort: PhysioNet EEG Motor Movement/Imagery, imagined left vs right fist.

Runs 4, 8, 12 (T1 = left fist imagery, T2 = right fist imagery). Same pipeline as the stroke
cohort: 4-40 Hz band-pass, resample to 128 Hz, epoch 0.5-3.5 s after cue onset, microvolts.
Subjects 88, 89, 92, 100, 104 are excluded (different sampling rate / incomplete runs).

Downloads ~0.8 GB into data/raw/physionet on first run and writes data/processed/physionet.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci

RUNS = [4, 8, 12]
EXCLUDE = {88, 89, 92, 100, 104}


def load_subject(subj: int, raw_dir: Path, l_freq: float, h_freq: float, sfreq: float,
                 tmin: float, tmax: float):
    paths = eegbci.load_data(subj, RUNS, path=str(raw_dir), update_path=False, verbose="ERROR")
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    raw = mne.concatenate_raws(raws, verbose="ERROR")
    eegbci.standardize(raw)
    raw.pick("eeg")
    raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")
    raw.resample(sfreq, verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2}, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id={"left": 1, "right": 2}, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose="ERROR")
    n_times = int(round((tmax - tmin) * sfreq))
    X = epochs.get_data(copy=False)[:, :, :n_times] * 1e6     # V -> microvolts
    y = epochs.events[:, 2] - 1                                  # 0 = left, 1 = right
    return X.astype(np.float32), y.astype(np.int64), epochs.ch_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/physionet")
    p.add_argument("--out", default="data/processed/physionet.npz")
    p.add_argument("--subjects", nargs="+", type=int, default=None, help="default: all usable 1-109")
    p.add_argument("--l-freq", type=float, default=4.0)
    p.add_argument("--h-freq", type=float, default=40.0)
    p.add_argument("--sfreq", type=float, default=128.0)
    p.add_argument("--tmin", type=float, default=0.5)
    p.add_argument("--tmax", type=float, default=3.5)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or [s for s in range(1, 110) if s not in EXCLUDE]

    Xs, ys, ss, ch_ref = [], [], [], None
    for subj in subjects:
        X, y, ch = load_subject(subj, raw_dir, args.l_freq, args.h_freq, args.sfreq, args.tmin, args.tmax)
        if ch_ref is None:
            ch_ref = ch
        elif ch != ch_ref:
            raise RuntimeError(f"subject {subj}: channel order differs from subject {subjects[0]}")
        Xs.append(X), ys.append(y), ss.append(np.full(len(y), subj))
        print(f"S{subj:03d}: {len(y)} trials (left {int((y == 0).sum())}, right {int((y == 1).sum())})", flush=True)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    subject = np.concatenate(ss)
    np.savez(args.out, X=X, y=y, subject=subject, ch_names=np.array(ch_ref), sfreq=args.sfreq,
             affected_side=np.full(len(y), ""))
    print(f"\nsaved {args.out}: X {X.shape}, {len(np.unique(subject))} subjects, {len(ch_ref)} channels")


if __name__ == "__main__":
    main()
