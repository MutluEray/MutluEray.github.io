"""10-10 electrode names and the mid-sagittal mirror (the Z2 group action on electrodes).

Convention: odd numbers are left-hemisphere, even numbers right, 'Z' is midline.
The mirror of an odd number k is k+1, of an even number k is k-1 (C3<->C4, FC5<->FC6,
T7<->T8, FP1<->FP2, ...). Midline channels map to themselves.
"""
from __future__ import annotations

import re

import numpy as np

_PATTERN = re.compile(r"^([A-Z]+?)(\d+|Z)$")

# Old 10-20 names -> 10-10 names, so the two datasets can be matched.
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

SENSORIMOTOR_PREFIXES = ("FC", "C", "CP")


def normalize(name: str) -> str:
    """Upper-case 10-10 name without dots, spaces, 'EEG' prefixes or reference suffixes."""
    n = str(name).strip().upper().replace(".", "").replace(" ", "")
    for prefix in ("EEG-", "EEG"):
        if n.startswith(prefix) and len(n) > len(prefix):
            n = n[len(prefix):]
            break
    for suffix in ("-REF", "-LE", "-AVG", "-A1", "-A2", "-M1", "-M2", "-CPZ"):
        if n.endswith(suffix) and len(n) > len(suffix):
            n = n[: -len(suffix)]
            break
    return _ALIASES.get(n, n)


def mirror_name(name: str) -> str | None:
    """Mirror partner of a normalized 10-10 name, or None if the name is not a 10-10 electrode."""
    m = _PATTERN.match(name)
    if m is None:
        return None
    prefix, num = m.groups()
    if num == "Z":
        return name
    k = int(num)
    if k == 0:
        return None
    return f"{prefix}{k + 1 if k % 2 else k - 1}"


def prefix_of(name: str) -> str | None:
    m = _PATTERN.match(name)
    return None if m is None else m.group(1)


def mirror_closed(names: list[str]) -> tuple[list[str], list[str]]:
    """Split channels into (kept, dropped): kept channels have their mirror partner present."""
    norm = [normalize(c) for c in names]
    present = set(norm)
    keep, drop = [], []
    for raw, n in zip(names, norm):
        partner = mirror_name(n)
        (keep if partner is not None and partner in present else drop).append(raw)
    return keep, drop


def mirror_permutation(names: list[str]) -> np.ndarray:
    """Index array perm such that x[perm] is the mirrored trial (x has channels on axis 0)."""
    norm = [normalize(c) for c in names]
    index = {n: i for i, n in enumerate(norm)}
    if len(index) != len(norm):
        raise ValueError(f"duplicate channel names after normalisation: {norm}")
    perm = []
    for n in norm:
        partner = mirror_name(n)
        if partner is None or partner not in index:
            raise ValueError(f"channel {n} has no mirror partner in this montage")
        perm.append(index[partner])
    perm = np.asarray(perm, dtype=np.int64)
    if not np.array_equal(perm[perm], np.arange(len(perm))):
        raise ValueError("mirror map is not an involution")
    return perm


def select_channels(names: list[str], which: str, shared: list[str] | None = None) -> tuple[list[int], list[str]]:
    """Indices of the channels to use, restricted to a mirror-closed set.

    which: 'all' | 'sensorimotor' (FC/C/CP rows) | 'shared' (normalized names in `shared`)
           | 'shared_sensorimotor' (shared AND FC/C/CP rows).
    Returns (indices into `names`, names dropped for lacking a mirror partner).
    """
    norm = [normalize(c) for c in names]
    if which == "all":
        idx = [i for i, n in enumerate(norm) if mirror_name(n) is not None]
    elif which == "sensorimotor":
        idx = [i for i, n in enumerate(norm) if prefix_of(n) in SENSORIMOTOR_PREFIXES]
    elif which in ("shared", "shared_sensorimotor"):
        if not shared:
            raise ValueError(f"channels='{which}' needs a list of shared channel names")
        wanted = {normalize(s) for s in shared}
        idx = [i for i, n in enumerate(norm) if n in wanted]
        if which == "shared_sensorimotor":
            idx = [i for i in idx if prefix_of(norm[i]) in SENSORIMOTOR_PREFIXES]
    else:
        raise ValueError(f"unknown channel set: {which}")
    sub = [names[i] for i in idx]
    keep, drop = mirror_closed(sub)
    keep_set = set(keep)
    idx = [i for i in idx if names[i] in keep_set]
    return idx, drop
