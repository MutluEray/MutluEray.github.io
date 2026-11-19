#!/usr/bin/env python3
"""
patch_pynn.py — make PyNN 0.9.4 + neo 0.8 run Fleming et al. (2020) on a modern
stack (Python 3.11, NumPy 2.x, SciPy 1.16, NEURON 9) without a separate environment.

What it does (idempotent, safe to re-run):
  1. Copies Fleming's Updated_PyNN_Files into the installed pyNN package
     (steady-state save/restore + multi-source cortical cells).
  1b. Back-ports PyNN 0.13's mechanism loader (finds NEURON 9 / macOS .dylib builds).
  2. Back-ports the PyNN 0.10 fix for NEURON >= 8.1: adds __new__ to the four
     hclass spike-source classes in pyNN/neuron/cells.py.
  3. Replaces collections.Sized etc. with collections.abc.* (removed in Python 3.10).
  4. Replaces NumPy aliases removed in NumPy 2 (np.float, np.int, np.NaN, ...) in
     pyNN, neo, lazyarray and nrnutils only. No other package is touched.
  5. Replaces PyNN 0.9.4's .mod files with PyNN 0.13's (same mechanisms, only
     C-API guards added so they compile under NEURON 9), then compiles them.

Usage:
  python3 patch_pynn.py --model-dir fleming2020
--model-dir may be any folder that contains the model (nested ModelDB zip folders are fine).
"""
import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys

NEW_METHOD = (
    "    def __new__(cls, *arg, **kwargs):\n"
    "        return super().__new__(cls, *arg, **kwargs)\n\n"
)
HCLASS_NAMES = ["RandomSpikeSource", "RandomPoissonRefractorySpikeSource",
                "RandomGammaSpikeSource", "VectorSpikeSource"]

NUMPY2_SUBS = [
    # removed builtin aliases (np.float -> float, ...); (?![\w(]) leaves np.float64 etc. alone
    (r"\b(?:np|numpy)\.(float|int|bool|object|complex|str)\b(?![\w(])", r"\1"),
    (r"\b(?:np|numpy)\.long\b(?![\w(])", "int"),
    (r"\b(?:np|numpy)\.unicode\b(?![\w(])", "str"),
    (r"\b(np|numpy)\.float_\b", r"\1.float64"),
    (r"\b(np|numpy)\.complex_\b", r"\1.complex128"),
    (r"\b(np|numpy)\.NaN\b", r"\1.nan"),
    (r"\b(np|numpy)\.NAN\b", r"\1.nan"),
    (r"\b(np|numpy)\.Inf\b", r"\1.inf"),
    (r"\b(np|numpy)\.PINF\b", r"\1.inf"),
    (r"\b(np|numpy)\.product\(", r"\1.prod("),
    (r"\b(np|numpy)\.alltrue\(", r"\1.all("),
    (r"\b(np|numpy)\.sometrue\(", r"\1.any("),
    # NumPy 2 refuses generators in stack functions: vstack((x for y in z)) -> vstack([x for y in z])
    (r"\b(np|numpy)\.(vstack|hstack|stack|column_stack|dstack|concatenate)\(\(([^()]*(?:\([^()]*\))?[^()]*\bfor\b[^()]*)\)\)",
     r"\1.\2([\3])"),
]


def find_model_root(start):
    """Return the folder that contains Fleming's model files (searches subfolders)."""
    start = os.path.abspath(start)
    for root, _, files in os.walk(start):
        if "Cortical_Basal_Ganglia_Cell_Classes.py" in files and os.path.isdir(os.path.join(root, "Updated_PyNN_Files")):
            return root
    sys.exit(f"Could not find Fleming 2020 model files under {start}.\n"
             "Download the model zip from https://modeldb.science/262046 and unzip it into that folder.")


def package_dir(name):
    """Locate an installed package without importing it (old neo cannot import before patching)."""
    spec = importlib.util.find_spec(name)
    if spec is None:
        sys.exit(f"Package '{name}' is not installed for {sys.executable}")
    if spec.submodule_search_locations:          # package -> its folder
        return list(spec.submodule_search_locations)[0]
    return spec.origin                           # single-file module (e.g. lazyarray.py)


def check_pynn_version(sp):
    txt = open(os.path.join(sp, "__init__.py")).read()
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", txt)
    if not m or not m.group(1).startswith("0.9.4"):
        sys.exit(f"Expected PyNN 0.9.4, found {m.group(1) if m else 'unknown'}.")


def copy_fleming_files(model_root, sp):
    src = os.path.join(model_root, "Updated_PyNN_Files")
    for rel in ["common/control.py", "neuron/simulator.py", "neuron/__init__.py",
                "neuron/standardmodels/electrodes.py"]:
        shutil.copyfile(os.path.join(src, rel), os.path.join(sp, rel))
        print(f"  copied {rel}")


LOADER_NEW = '''def load_mechanisms(path):
    """Load NMODL mechanisms from path (back-ported from PyNN 0.13: delegates to
    neuron.load_mechanisms, which knows NEURON 9 / macOS library layouts)."""
    import neuron as _neuron
    global nrn_dll_loaded
    if path in nrn_dll_loaded:
        logger.warning("Mechanisms already loaded from path: %s" % path)
        return
    if not _neuron.load_mechanisms(path):
        raise IOError("NEURON mechanisms not found in %s. You may need to run 'nrnivmodl' in this directory." % path)
    nrn_dll_loaded.append(path)


'''


def backport_mechanism_loader(sp):
    path = os.path.join(sp, "neuron", "simulator.py")
    text = open(path).read()
    if "back-ported from PyNN 0.13" in text:
        print("  mechanism loader already back-ported")
        return
    start = text.find("def load_mechanisms(path):")
    end = text.find("def is_point_process(obj):")
    if start == -1 or end == -1:
        sys.exit("Could not locate load_mechanisms() in pyNN/neuron/simulator.py")
    open(path, "w").write(text[:start] + LOADER_NEW + text[end:])
    print("  mechanism loader back-ported from PyNN 0.13 (NEURON 9 / macOS)")


def add_new_methods(sp):
    path = os.path.join(sp, "neuron", "cells.py")
    text = open(path).read()
    changed = 0
    for name in HCLASS_NAMES:
        m = re.search(rf"class {name}\(hclass\(h\.\w+\)\):\n", text)
        if m is None:
            sys.exit(f"Could not find class {name} in {path}")
        block_end = text.find("\nclass ", m.end())
        block = text[m.end():block_end if block_end != -1 else len(text)]
        if "def __new__" in block:
            continue
        init_pos = block.find("    def __init__")
        next_def = block.find("\n    def ", init_pos + 1)
        insert_at = m.end() + (next_def + 1 if next_def != -1 else len(block))
        text = text[:insert_at] + NEW_METHOD + text[insert_at:]
        changed += 1
    open(path, "w").write(text)
    print(f"  __new__ added to {changed} class(es) in pyNN/neuron/cells.py")


def rewrite_tree(d, subs):
    n = 0
    paths = [d] if os.path.isfile(d) else [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
    for p in paths:
        if True:
            if not p.endswith(".py"):
                continue
            t = open(p, encoding="utf-8", errors="surrogateescape").read()
            t2 = t
            for pat, rep in subs:
                t2 = re.sub(pat, rep, t2)
            if t2 != t:
                open(p, "w", encoding="utf-8", errors="surrogateescape").write(t2)
                n += 1
    return n


def fix_collections_abc(sp):
    names = "Sized|Iterable|Mapping|MutableMapping|Sequence|Callable|Hashable|Container"
    n = 0
    for root, _, files in os.walk(sp):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            t = open(p).read()
            t2 = re.sub(rf"\bcollections\.({names})\b", r"collections.abc.\1", t)
            if t2 != t:
                if "import collections.abc" not in t2:
                    t2 = t2.replace("import collections\n", "import collections\nimport collections.abc\n", 1)
                open(p, "w").write(t2)
                n += 1
    print(f"  collections -> collections.abc fixed in {n} file(s)")


def guard_pkg_resources(neo_dir):
    """neo 0.8 imports pkg_resources (removed in setuptools >= 81) in one unused IO module."""
    p = os.path.join(neo_dir, "io", "neuralynxio_v1.py")
    t = open(p).read()
    if "packaging.version import parse as parse_version" in t:
        print("  neo pkg_resources guard already applied")
        return
    t = t.replace("import pkg_resources\n", "from packaging.version import parse as parse_version  # was pkg_resources\n", 1)
    t = re.sub(r"if hasattr\(pkg_resources, 'pkg_resources'\):\n\s+parse_version = pkg_resources\.pkg_resources\.parse_version\nelse:\n\s+parse_version = pkg_resources\.parse_version\n",
               "", t)
    open(p, "w").write(t)
    print("  neo: pkg_resources replaced by packaging.version (setuptools >= 81 safe)")


def update_mod_files_for_neuron9(sp):
    """PyNN 0.9.4's NMODL files do not compile with NEURON 9 (C++ mechanisms).
    PyNN 0.13 ships the same mechanisms with only C-API compatibility guards added
    (RANDCAST, NRN_VERSION_GTEQ_8_2_0, scop_random(), THREADSAFE); names and
    parameters are unchanged. Fetch PyNN 0.13's source once and copy its .mod files."""
    import glob
    import tarfile
    import tempfile
    nmodl = os.path.join(sp, "neuron", "nmodl")
    marker = os.path.join(nmodl, ".mods_from_pynn_0.13")
    if os.path.exists(marker):
        print("  NMODL files already updated for NEURON 9")
        return
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:",
                            "PyNN==0.13.0", "-d", tmp, "-q"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            sys.exit("Could not download PyNN 0.13.0 source (needed only for its .mod files). Check internet.")
        sdist = glob.glob(os.path.join(tmp, "*.tar.gz"))[0]
        with tarfile.open(sdist) as tf:
            members = [m for m in tf.getmembers() if "/pyNN/neuron/nmodl/" in m.name and m.name.endswith(".mod")]
            n = 0
            for m in members:
                name = os.path.basename(m.name)
                dst = os.path.join(nmodl, name)
                if os.path.exists(dst):                      # only replace mechanisms 0.9.4 already has
                    open(dst, "wb").write(tf.extractfile(m).read())
                    n += 1
    open(marker, "w").write("mod files copied from PyNN 0.13.0\n")
    print(f"  {n} NMODL file(s) replaced with PyNN 0.13 versions (NEURON 9 compatible)")


def compile_pynn_mechanisms(sp):
    nmodl = os.path.join(sp, "neuron", "nmodl")
    print(f"  running nrnivmodl in {nmodl}")
    r = subprocess.run(["nrnivmodl"], cwd=nmodl, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit("nrnivmodl failed for PyNN mechanisms (Xcode command line tools installed?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="folder containing the Fleming 2020 model (ModelDB 262046)")
    args = ap.parse_args()
    model_root = find_model_root(args.model_dir)
    print(f"model files at {model_root}")
    sp = package_dir("pyNN")
    check_pynn_version(sp)
    print(f"PyNN at {sp}")
    copy_fleming_files(model_root, sp)
    backport_mechanism_loader(sp)
    add_new_methods(sp)
    fix_collections_abc(sp)
    n = sum(rewrite_tree(package_dir(p), NUMPY2_SUBS) for p in ["pyNN", "neo", "lazyarray", "nrnutils"])
    print(f"  NumPy-2 aliases fixed in {n} file(s) (pyNN, neo, lazyarray, nrnutils)")
    guard_pkg_resources(package_dir("neo"))
    update_mod_files_for_neuron9(sp)
    compile_pynn_mechanisms(sp)
    print("Done. Next: compile the model mechanisms with nrnivmodl inside the model folder:")
    print(f"  cd \"{model_root}\" && nrnivmodl")


if __name__ == "__main__":
    main()
