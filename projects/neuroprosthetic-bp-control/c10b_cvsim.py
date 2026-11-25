"""
C10-B — build and Python wrapper for the CVSim 21-compartment model.

The plant is CVSim's own C code (PhysioNet CVSim 1.0.0, Heldt et al.), patched
in three small, asserted places and compiled together with c10b_cvsim_ext.c
into a shared library loaded through ctypes. Nothing in the hemodynamic or
reflex equations is rewritten.

Patches (each must match exactly once, or the build stops):
  equation.c  - the original tilt() is renamed tilt_orig(); eqns_ptr() calls
                tilt_dispatch() (external-angle tilt in mode 1, original in mode 0)
  reflex.c    - arteriolar resistances get  + ees_dr[k]
              - zero-pressure filling volumes get  - ees_dv[k]
                (k: 0 upper body, 1 kidney, 2 splanchnic, 3 legs)

Use:
  from c10b_cvsim import CVSim21
  sim = CVSim21()                     # builds on first use
  th = sim.default_theta()
  sim.init(th)                        # mode 1 = external tilt angle
  beats = sim.run(300.0)              # supine settle
  beats = sim.run(302.0, angle_end=75)  # 2 s ramp to 75 deg
"""

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(HERE, "data", "cvsim_src", "cvsim-1.0.0", "cvsim", "devel", "C", "21-comp-backend")
BUILD = os.path.join(HERE, "build", "cvsim21")
EXT_C = os.path.join(HERE, "c10b_cvsim_ext.c")
LIB = os.path.join(HERE, "build", "libcvsim21.so")

N_PARAM = 153
MAX_BEATS = 20000

# ------------------------------------------------------------------ patches
EQ_PATCHES = [
    ('#include "equation.h"',
     '#include "equation.h"\n'
     'int tilt_dispatch(Data_vector *, Parameter_vector *, Tilt_vector *, double, double);'),
    ("int tilt(Data_vector *p, Parameter_vector *theta, Tilt_vector *tilt,",
     "int tilt_orig(Data_vector *p, Parameter_vector *theta, Tilt_vector *tilt,"),
    ("    tilt(p, theta, &t, tiltStartTime, tiltStopTime);",
     "    tilt_dispatch(p, theta, &t, tiltStartTime, tiltStopTime);"),
]
RF_PATCHES = [
    ('#include "reflex.h"', '#include "reflex.h"\nextern double ees_dr[4], ees_dv[4];'),
    ("(*r).resistance[0]  = (*theta).vec[53]", "(*r).resistance[0]  = ees_dr[0] + (*theta).vec[53]"),
    ("(*r).resistance[1]  = (*theta).vec[55]", "(*r).resistance[1]  = ees_dr[1] + (*theta).vec[55]"),
    ("(*r).resistance[2]  = (*theta).vec[57]", "(*r).resistance[2]  = ees_dr[2] + (*theta).vec[57]"),
    ("(*r).resistance[3]  = (*theta).vec[59]", "(*r).resistance[3]  = ees_dr[3] + (*theta).vec[59]"),
    ("(*r).volume[0] = (*theta).vec[77]", "(*r).volume[0] = -ees_dv[0] + (*theta).vec[77]"),
    ("(*r).volume[1] = (*theta).vec[78]", "(*r).volume[1] = -ees_dv[1] + (*theta).vec[78]"),
    ("(*r).volume[2] = (*theta).vec[79]", "(*r).volume[2] = -ees_dv[2] + (*theta).vec[79]"),
    ("(*r).volume[3] = (*theta).vec[80]", "(*r).volume[3] = -ees_dv[3] + (*theta).vec[80]"),
]
SOURCES = ["sim/equation.c", "sim/reflex.c", "sim/estimate.c", "sim/rkqc.c",
           "sim/simulator.c", "main/initial.c"]


def _patch(path, patches):
    txt = open(path).read()
    for old, new in patches:
        n = txt.count(old)
        if n != 1:
            raise RuntimeError(f"patch target found {n}x (need 1) in {os.path.basename(path)}:\n  {old}")
        txt = txt.replace(old, new)
    open(path, "w").write(txt)


def build(force=False, verbose=True):
    """Copy CVSim C sources, patch, compile into build/libcvsim21.so."""
    if os.path.exists(LIB) and not force:
        if os.path.getmtime(LIB) > max(os.path.getmtime(EXT_C), os.path.getmtime(__file__)):
            return LIB
    if not os.path.isdir(SRC_ROOT):
        sys.exit(f"CVSim source not found at {SRC_ROOT}. Run step 0b first.")
    if os.path.isdir(BUILD):
        shutil.rmtree(BUILD)
    shutil.copytree(SRC_ROOT, BUILD)
    _patch(os.path.join(BUILD, "sim", "equation.c"), EQ_PATCHES)
    _patch(os.path.join(BUILD, "sim", "reflex.c"), RF_PATCHES)
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-O2", "-shared", "-fPIC", "-std=gnu99", "-w",
           "-I", os.path.join(BUILD, "sim"), "-I", os.path.join(BUILD, "main"),
           "-o", LIB] + [os.path.join(BUILD, s) for s in SOURCES] + [EXT_C, "-lm"]
    if verbose:
        print("[build]", " ".join(os.path.relpath(c, HERE) if os.path.isabs(c) else c for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr)
        sys.exit("[build] compilation failed (on macOS: run `xcode-select --install` if cc is missing)")
    return LIB


class CVSim21:
    """One simulator instance per process (CVSim's reflex code keeps static state)."""

    def __init__(self, force_build=False):
        build(force=force_build)
        self.lib = ctypes.CDLL(LIB)
        dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        L = self.lib
        L.cv_default_theta.argtypes = [dp]
        L.cv_init.argtypes = [dp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
        L.cv_init.restype = ctypes.c_int
        L.cv_run.argtypes = [ctypes.c_double, ctypes.c_double, dp, dp, dp, ctypes.c_int]
        L.cv_run.restype = ctypes.c_int
        L.cv_time.restype = ctypes.c_double
        L.cv_effectors.argtypes = [dp]
        assert L.cv_n_parameter() == N_PARAM
        self._buf = np.zeros(MAX_BEATS * 5)

    def default_theta(self):
        th = np.zeros(N_PARAM)
        self.lib.cv_default_theta(th)
        return th

    def init(self, theta, ab=True, cp=True, mode=1, t_start=-1.0, t_stop=-1.0):
        """mode 1: external tilt angle (use run(..., angle_end)); mode 0: CVSim's
        original tilt with theta[91] angle, theta[92] tilt time, start/stop times."""
        th = np.ascontiguousarray(theta, dtype=np.float64)
        self.lib.cv_init(th, int(ab), int(cp), int(mode), float(t_start), float(t_stop))

    def run(self, t_end, angle_end=None, ees_dr=(0, 0, 0, 0), ees_dv=(0, 0, 0, 0)):
        """Advance to t_end. Angle ramps linearly to angle_end (default: hold).
        Returns beats as array (n, 5): t, SAP, MAP, DAP, HR."""
        if angle_end is None:
            angle_end = self.effectors()["angle"]
        dr = np.ascontiguousarray(ees_dr, dtype=np.float64)
        dv = np.ascontiguousarray(ees_dv, dtype=np.float64)
        n = self.lib.cv_run(float(t_end), float(angle_end), dr, dv, self._buf, MAX_BEATS)
        return self._buf[: 5 * n].reshape(n, 5).copy()

    def time(self):
        return self.lib.cv_time()

    def effectors(self):
        out = np.zeros(11)
        self.lib.cv_effectors(out)
        return {"resistance": out[0:4], "volume": out[4:8], "hr_reflex": out[8],
                "angle": out[9], "vloss": out[10]}


def beats_to_grid(beats, t_grid):
    """Linear interpolation of beat numerics onto a time grid (s). Returns dict."""
    if len(beats) < 2:
        return {k: np.full(len(t_grid), np.nan) for k in ("sbp", "map", "dbp", "hr")}
    t = beats[:, 0]
    return {"sbp": np.interp(t_grid, t, beats[:, 1]), "map": np.interp(t_grid, t, beats[:, 2]),
            "dbp": np.interp(t_grid, t, beats[:, 3]), "hr": np.interp(t_grid, t, beats[:, 4])}


def raised_cosine_profile(sim, t0, duration, a_from, a_to, dt=0.25):
    """Drive the angle along CVSim's raised-cosine tilt shape with small linear pieces."""
    out = []
    n = max(1, int(round(duration / dt)))
    for i in range(1, n + 1):
        s = i / n
        a = a_from + (a_to - a_from) * (1 - np.cos(np.pi * s)) / 2
        out.append(sim.run(t0 + s * duration, angle_end=a))
    return np.vstack([b for b in out if len(b)]) if any(len(b) for b in out) else np.zeros((0, 5))
