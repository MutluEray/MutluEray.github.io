"""
C10-B step 0b — inspect the CVSim archive from PhysioNet (no unzipping needed).

Usage:
  python3 c10b_step0b_cvsim_inspect.py data/cvsim.zip

Lists the archive, finds source / parameter / example-output files, and counts
lines mentioning the things the Python port needs (tilt, gravity, hydrostatic,
baroreflex, sympathetic, vagal, compartment parameters). Writes:
  results/c10b_step0/cvsim_inventory.md
and extracts only small text files (source, parameters, example outputs) to
  data/cvsim_src/
so the port in step 1 can read the real equations and parameter values.
"""

import os
import re
import sys
import zipfile

if len(sys.argv) < 2:
    sys.exit("Usage: python3 c10b_step0b_cvsim_inspect.py data/cvsim.zip")

ZIP = sys.argv[1]
OUT_DIR = os.path.join("results", "c10b_step0")
SRC_DIR = os.path.join("data", "cvsim_src")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SRC_DIR, exist_ok=True)

TEXT_EXT = (".c", ".h", ".java", ".txt", ".par", ".dat", ".csv", ".m", ".md", ".cfg", ".ini", ".json", ".xml")
KEYWORDS = ["tilt", "gravity", "hydrostatic", "baroreflex", "cardiopulmonary", "sympathetic",
            "vagal", "parasympathetic", "compliance", "resistance", "unstressed", "elastance",
            "21", "six", "compartment", "orthostatic", "lbnp"]
MAX_EXTRACT_BYTES = 2_000_000  # skip large binaries / big outputs

if not zipfile.is_zipfile(ZIP):
    sys.exit(f"{ZIP} is not a zip file. Re-download it (see the curl command) or use the "
             f"'Download the ZIP file' link on https://physionet.org/content/cvsim/")

lines = ["# CVSim archive inventory (C10-B step 0b)\n", f"Archive: `{ZIP}`\n"]
with zipfile.ZipFile(ZIP) as z:
    infos = z.infolist()
    lines.append(f"Entries: {len(infos)}\n")
    lines.append("## Top-level tree (first 250 entries)\n")
    for inf in infos[:250]:
        lines.append(f"- `{inf.filename}` ({inf.file_size} B)")

    lines.append("\n## Text files with keyword hits\n")
    lines.append("| file | size (B) | hits (keyword:count) |")
    lines.append("|---|---|---|")
    extracted = 0
    for inf in infos:
        name = inf.filename
        if inf.is_dir() or not name.lower().endswith(TEXT_EXT):
            continue
        if inf.file_size > MAX_EXTRACT_BYTES:
            lines.append(f"| `{name}` | {inf.file_size} | (skipped: large) |")
            continue
        try:
            txt = z.read(inf).decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            lines.append(f"| `{name}` | {inf.file_size} | read error {e} |")
            continue
        low = txt.lower()
        hits = {k: len(re.findall(r"\b" + re.escape(k) + r"\b", low)) for k in KEYWORDS}
        hits = {k: v for k, v in hits.items() if v}
        lines.append(f"| `{name}` | {inf.file_size} | "
                     f"{', '.join(f'{k}:{v}' for k, v in sorted(hits.items(), key=lambda kv: -kv[1])) or '—'} |")
        dest = os.path.join(SRC_DIR, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(txt)
        extracted += 1

lines.append(f"\nExtracted {extracted} text files to `{SRC_DIR}`.")
with open(os.path.join(OUT_DIR, "cvsim_inventory.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[done] wrote {OUT_DIR}/cvsim_inventory.md ({extracted} text files extracted to {SRC_DIR})")
print("Send back: cvsim_inventory.md (paste or attach).")
