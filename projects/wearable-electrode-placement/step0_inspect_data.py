"""
Step 0 — download and inspect the Ear-SAAD dataset (Geirnaert, Kappel, Kidmose 2025).
Project: physics-guided wearable electrode placement (C7-B / F17).

What it does
  1. Downloads preprocessedData.zip (~416 MB), experiment-manual.pdf and
     preprocessFromBIDS.m from Zenodo (record 16536441) with `requests`
     (urllib failed SSL on this Mac). Skips files already present.
  2. Unzips the preprocessed data and prints the structure of every .mat file.
  3. Reads the 12.2 GB BIDS zip REMOTELY (HTTP range requests via `remotezip`):
     extracts small metadata files (+ events.tsv) and sub-01's EEGLAB .set header
     (~14 MB, no .fdt), then reads its chanlocs (electrode labels + X/Y/Z).
  4. Writes results/step0_summary.json, results/step0_structure.txt,
     results/step0_chanlocs_sub-01.csv.

Run from the project folder:
    python3 step0_inspect_data.py
"""

import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PRE = DATA / "preprocessed"
META = DATA / "bids_meta"
RES = ROOT / "results"
for d in (RAW, PRE, META, RES):
    d.mkdir(parents=True, exist_ok=True)

ZENODO = "https://zenodo.org/records/16536441/files/{}?download=1"
SMALL_FILES = ["experiment-manual.pdf", "preprocessFromBIDS.m"]
PRE_ZIP = "preprocessedData.zip"
BIDS_ZIP = "bids_dataset.zip"
SET_SUB = "sub-01"

META_KEYWORDS = ("electrodes", "channels", "coordsystem", "participants",
                 "dataset_description", "readme", "events", "eeg.json")
META_MAX_BYTES = 2_000_000

summary = {"downloads": {}, "mat_files": {}, "bids_meta": {}, "chanlocs": {}, "notes": []}
structure_lines = []


def log(msg=""):
    print(msg)
    structure_lines.append(str(msg))


# ----------------------------------------------------------------------------
# 1. Downloads (requests + certifi)
# ----------------------------------------------------------------------------
def download(name, dest):
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[skip] {name} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    log(f"[download] {name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(ZENODO.format(name), stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done, t0, last = 0, time.time(), 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 2:
                        last = time.time()
                        pct = f"{100*done/total:5.1f}%" if total else ""
                        rate = done / 1e6 / max(time.time() - t0, 1e-6)
                        print(f"    {done/1e6:8.1f} MB {pct}  {rate:5.1f} MB/s", end="\r")
        tmp.rename(dest)
        print()
        log(f"    saved {dest.stat().st_size/1e6:.1f} MB")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"    FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


for name in SMALL_FILES + [PRE_ZIP]:
    summary["downloads"][name] = download(name, RAW / name)

# ----------------------------------------------------------------------------
# 2. Unzip + inspect preprocessed .mat files
# ----------------------------------------------------------------------------
pre_zip = RAW / PRE_ZIP
if pre_zip.exists():
    marker = PRE / ".unzipped"
    if not marker.exists():
        log(f"[unzip] {PRE_ZIP}")
        with zipfile.ZipFile(pre_zip) as z:
            z.extractall(PRE)
        marker.write_text("ok")
    else:
        log("[skip] preprocessed data already unzipped")
else:
    summary["notes"].append("preprocessedData.zip missing; .mat inspection skipped")


def read_mat_any(path):
    """pymatreader handles v7 and v7.3; fall back to scipy for v7."""
    try:
        from pymatreader import read_mat
        return read_mat(str(path)), "pymatreader"
    except ImportError:
        from scipy.io import loadmat
        d = loadmat(path, squeeze_me=True, simplify_cells=True)
        return {k: v for k, v in d.items() if not k.startswith("__")}, "scipy"


def preview(x, max_items=40):
    try:
        a = np.asarray(x)
        if a.dtype.kind in "US" and a.size <= 80:
            return repr(a.ravel().tolist())
        if a.dtype.kind in "fiub":
            if a.size <= max_items:
                return repr(np.round(a.ravel(), 4).tolist())
            return f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} mean={np.nanmean(a):.4g}"
    except Exception:  # noqa: BLE001
        pass
    return ""


def describe(obj, name, depth=0, max_depth=7):
    pad = "  " * depth
    if depth > max_depth:
        log(f"{pad}{name}: ... (max depth)")
        return
    if isinstance(obj, dict):
        log(f"{pad}{name}: dict keys={list(obj.keys())[:30]}")
        for k, v in list(obj.items())[:30]:
            describe(v, k, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        log(f"{pad}{name}: list len={len(obj)}")
        if obj and all(isinstance(x, str) for x in obj):
            log(f"{pad}  values: {obj[:80]}")
            return
        for i, el in enumerate(obj[:3]):
            describe(el, f"{name}[{i}]", depth + 1, max_depth)
        if len(obj) > 3:
            log(f"{pad}  ... ({len(obj) - 3} more)")
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            log(f"{pad}{name}: object array shape={obj.shape}")
            for i, el in enumerate(obj.ravel()[:3]):
                describe(el, f"{name}[{i}]", depth + 1, max_depth)
        else:
            log(f"{pad}{name}: array shape={obj.shape} dtype={obj.dtype} {preview(obj)}")
    elif isinstance(obj, str):
        log(f"{pad}{name}: str '{obj[:150]}'")
    else:
        log(f"{pad}{name}: {type(obj).__name__} {repr(obj)[:150]}")


mat_files = sorted(PRE.rglob("*.mat"))
log(f"\n=== {len(mat_files)} .mat files in {PRE} ===")
for i, p in enumerate(mat_files):
    rel = str(p.relative_to(PRE))
    summary["mat_files"][rel] = {"bytes": p.stat().st_size}
    log(f"\n--- {rel} ({p.stat().st_size/1e6:.1f} MB) ---")
    try:
        d, reader = read_mat_any(p)
        summary["mat_files"][rel]["reader"] = reader
        summary["mat_files"][rel]["keys"] = list(d.keys())
        if i < 2:                       # full structure for the first two files
            for k, v in d.items():
                describe(v, k)
        else:
            log(f"keys: {list(d.keys())}")
    except Exception as e:  # noqa: BLE001
        summary["mat_files"][rel]["error"] = str(e)
        log(f"ERROR reading: {e}")

# ----------------------------------------------------------------------------
# 3. BIDS metadata + sub-01 .set header via remote zip
# ----------------------------------------------------------------------------
log("\n=== BIDS zip (remote) ===")
set_path = None
try:
    from remotezip import RemoteZip
    with RemoteZip(ZENODO.format(BIDS_ZIP)) as rz:
        infos = rz.infolist()
        picked = []
        for info in infos:
            fn = info.filename.lower()
            if info.is_dir():
                continue
            if info.file_size <= META_MAX_BYTES and any(k in fn for k in META_KEYWORDS):
                rz.extract(info, META)
                picked.append(info.filename)
            if fn.endswith(".set") and f"{SET_SUB}_" in fn:
                log(f"[remote extract] {info.filename} ({info.file_size/1e6:.1f} MB)")
                rz.extract(info, META)
                set_path = META / info.filename
        summary["bids_meta"]["n_entries"] = len(infos)
        summary["bids_meta"]["extracted"] = picked
        log(f"extracted {len(picked)} metadata files -> data/bids_meta/")
except Exception as e:  # noqa: BLE001
    summary["bids_meta"]["error"] = str(e)
    log(f"remote zip failed: {e}")

# Print the small text files for one subject in full
for pattern in ("README", "dataset_description.json", "participants.tsv",
                f"{SET_SUB}/**/*channels.tsv", f"{SET_SUB}/**/*eeg.json",
                f"{SET_SUB}/**/*events.tsv", f"{SET_SUB}/**/*events.json"):
    for f in sorted(META.rglob(pattern)):
        log(f"\n--- {f.relative_to(META)} ---")
        try:
            log(f.read_text(errors="replace"))
        except Exception as e:  # noqa: BLE001
            log(f"(could not read: {e})")

# ----------------------------------------------------------------------------
# 4. chanlocs from the EEGLAB .set header
# ----------------------------------------------------------------------------
def as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    return [x]


if set_path is not None and set_path.exists():
    log(f"\n=== chanlocs from {set_path.name} ===")
    try:
        d, reader = read_mat_any(set_path)
        eeg = d.get("EEG", d)
        for key in ("nbchan", "srate", "pnts", "trials", "ref"):
            if key in eeg:
                log(f"{key}: {eeg[key]}")
                summary["chanlocs"][key] = str(eeg[key])
        cl = eeg.get("chanlocs")
        rows = []
        if isinstance(cl, dict):                      # dict of per-field lists
            n = len(as_list(cl.get("labels")))
            for j in range(n):
                rows.append({k: as_list(cl.get(k))[j] if j < len(as_list(cl.get(k))) else ""
                             for k in ("labels", "type", "X", "Y", "Z",
                                       "sph_theta", "sph_phi", "sph_radius")})
        elif isinstance(cl, (list, tuple, np.ndarray)):  # list of per-channel dicts
            for c in as_list(cl):
                if isinstance(c, dict):
                    rows.append({k: c.get(k, "") for k in ("labels", "type", "X", "Y", "Z",
                                                          "sph_theta", "sph_phi", "sph_radius")})
        n_xyz = sum(1 for r in rows
                    if all(isinstance(r[k], (int, float)) and np.isfinite(r[k])
                           for k in ("X", "Y", "Z")))
        summary["chanlocs"]["n_channels"] = len(rows)
        summary["chanlocs"]["n_with_xyz"] = n_xyz
        log(f"{len(rows)} channels, {n_xyz} with finite X/Y/Z")
        for r in rows:
            log(f"  {str(r['labels']):>10s}  type={r['type']}  "
                f"X={r['X']}  Y={r['Y']}  Z={r['Z']}")
        if rows:
            with open(RES / f"step0_chanlocs_{SET_SUB}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        # what else is stored in the header?
        log(f"\nEEG header fields: {list(eeg.keys())}")
        if "event" in eeg:
            ev = eeg["event"]
            types = as_list(ev.get("type")) if isinstance(ev, dict) else \
                [e.get("type") for e in as_list(ev) if isinstance(e, dict)]
            log(f"events: {len(types)}; unique types: {sorted(set(map(str, types)))[:30]}")
    except Exception as e:  # noqa: BLE001
        summary["chanlocs"]["error"] = str(e)
        log(f"could not read .set: {e}")
else:
    summary["notes"].append(".set header not extracted")

# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------
(RES / "step0_structure.txt").write_text("\n".join(structure_lines))
(RES / "step0_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nWrote results/step0_summary.json, results/step0_structure.txt"
      f" and results/step0_chanlocs_{SET_SUB}.csv (if chanlocs were found).")
print("Send me step0_structure.txt and step0_summary.json.")
