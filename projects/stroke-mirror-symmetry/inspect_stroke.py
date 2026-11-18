"""Print the structure of the downloaded stroke dataset (Liu et al. 2024, figshare 21679035).

Only reads files; never runs anything from the download. Output goes to the terminal and to
results/inspect_stroke.txt (send that file back so the loader can be written against the real
format).

  python3 inspect_stroke.py --root data/raw/stroke
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path


def show_text(path: Path, max_lines: int = 80):
    print(f"\n----- {path.name} -----")
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception as e:  # noqa: BLE001
        print(f"(could not read: {e})")
        return
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")


def show_tree(root: Path, max_files: int = 60):
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"\n===== {len(files)} files under {root} =====")
    for p in files[:max_files]:
        print(f"{p.stat().st_size:>12,d}  {p.relative_to(root)}")
    if len(files) > max_files:
        exts = Counter(p.suffix.lower() for p in files)
        print(f"... ({len(files) - max_files} more) extensions: {dict(exts)}")
    return files


def show_zip(path: Path, max_entries: int = 40):
    print(f"\n----- zip listing: {path.name} -----")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for n in names[:max_entries]:
            print(n)
        if len(names) > max_entries:
            print(f"... ({len(names) - max_entries} more entries)")
        # print their analysis code as text (never executed)
        if path.name.startswith("code"):
            for n in names:
                if n.lower().endswith((".py", ".m", ".txt", ".md")):
                    print(f"\n----- {n} (first 150 lines) -----")
                    text = z.read(n).decode(errors="replace").splitlines()
                    print("\n".join(text[:150]))


def show_edf(path: Path):
    import mne
    import numpy as np

    print(f"\n===== EDF: {path} =====")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    print(f"sfreq {raw.info['sfreq']} Hz, {raw.n_times} samples ({raw.times[-1]:.1f} s), {len(raw.ch_names)} channels")
    print("channels:", raw.ch_names)
    print("channel types:", dict(Counter(raw.get_channel_types())))
    ann = raw.annotations
    print(f"annotations: {len(ann)}; descriptions: {dict(Counter(ann.description))}")
    if len(ann):
        print("first annotations (onset s, duration s, description):")
        for a in list(ann)[:12]:
            print(f"  {a['onset']:.3f}  {a['duration']:.3f}  {a['description']}")
    for ch in raw.ch_names:
        low = ch.lower()
        if any(k in low for k in ("trig", "mark", "stim", "event", "status")):
            data = raw.get_data(picks=[ch])[0]
            vals, counts = np.unique(np.round(data, 6), return_counts=True)
            print(f"possible marker channel '{ch}': {len(vals)} unique values; most common: "
                  f"{sorted(zip(counts, vals), reverse=True)[:10]}")
            nz = np.flatnonzero(np.diff(data) != 0)
            print(f"  first change points (s): {[round(float(i) / raw.info['sfreq'], 3) for i in nz[:12]]}")


def show_mat(path: Path):
    print(f"\n===== MAT: {path} =====")
    try:
        import scipy.io as sio
        d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        for k, v in d.items():
            if not k.startswith("__"):
                print(f"{k}: {type(v).__name__} {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            f.visititems(lambda n, o: print(n, getattr(o, "shape", ""), getattr(o, "dtype", "")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/stroke")
    args = p.parse_args()
    root = Path(args.root)
    out = Path("results/inspect_stroke.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    with redirect_stdout(buf):
        files = show_tree(root)
        for p_ in files:
            if p_.suffix.lower() in (".tsv", ".json", ".md", ".txt") and p_.stat().st_size < 200_000:
                show_text(p_)
        for p_ in files:
            if p_.suffix.lower() == ".zip":
                show_zip(p_)
        edfs = [p_ for p_ in files if p_.suffix.lower() in (".edf", ".bdf")]
        for p_ in edfs[:2]:
            show_edf(p_)
        mats = [p_ for p_ in files if p_.suffix.lower() == ".mat"]
        for p_ in mats[:2]:
            show_mat(p_)
        if not edfs and not mats:
            print("\nNo .edf/.bdf/.mat files found: unzip edffile.zip (and sourcedata.zip if needed) first.")
    text = buf.getvalue()
    out.write_text(text)
    sys.stdout.write(text)
    print(f"\n(saved to {out})")


if __name__ == "__main__":
    main()
