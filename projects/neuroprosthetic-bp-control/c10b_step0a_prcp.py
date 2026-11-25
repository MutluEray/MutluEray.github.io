"""
C10-B step 0a — PhysioNet PRCP (Physiologic Response to Changes in Posture) inventory.

Downloads PRCP (~54 MB, 10 healthy subjects, slow tilt / fast tilt / stand-up),
extracts beat-to-beat SBP / DBP / MAP / HR from the ABP waveform using the
shipped 'wabp' beat-onset annotations, lists the maneuver annotations ('anI'),
and writes an inventory, a beat table and one figure per record.

Outputs (relative to where you run it):
  data/prcp/                      raw PhysioNet files
  results/c10b_step0/prcp_inventory.md
  results/c10b_step0/prcp_beats.csv
  results/c10b_step0/prcp_annotations.csv
  results/c10b_step0/fig_<record>.png
"""

import os
import sys
import csv
import numpy as np

try:
    import wfdb
except ImportError:
    sys.exit("wfdb missing. Run: python3 -m pip install wfdb")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = "prcp"
DATA_DIR = os.path.join("data", "prcp")
OUT_DIR = os.path.join("results", "c10b_step0")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Plausibility limits for a single beat (Finapres calibration gaps produce garbage)
SBP_RANGE = (60.0, 250.0)
DBP_RANGE = (20.0, 160.0)
IBI_RANGE = (0.3, 2.0)  # seconds


def download():
    have = [f for f in os.listdir(DATA_DIR) if f.endswith(".hea")]
    if have:
        print(f"[download] {len(have)} headers already in {DATA_DIR}, skipping download")
        return
    print("[download] fetching PRCP from PhysioNet (~54 MB) ...")
    wfdb.dl_database(DB, DATA_DIR, records="all", annotators="all")
    print("[download] done")


def list_records():
    recs = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".hea"))
    return recs


def pick_abp_channel(sig_names):
    for i, n in enumerate(sig_names):
        if "abp" in n.lower() or n.lower() in ("bp", "art", "pressure"):
            return i
    for i, n in enumerate(sig_names):
        if "bp" in n.lower() or "press" in n.lower():
            return i
    return None


def read_ann(path, ext):
    try:
        return wfdb.rdann(path, ext)
    except Exception as e:  # noqa: BLE001
        print(f"   [warn] could not read {ext} for {os.path.basename(path)}: {e}")
        return None


def beats_from_onsets(abp, fs, onsets):
    rows = []
    for k in range(len(onsets) - 1):
        a, b = int(onsets[k]), int(onsets[k + 1])
        if b <= a + 2 or b > len(abp):
            continue
        seg = abp[a:b]
        if not np.all(np.isfinite(seg)):
            continue
        ibi = (b - a) / fs
        sbp = float(np.max(seg))
        dbp = float(np.min(seg))
        mapp = float(np.mean(seg))
        ok = (SBP_RANGE[0] <= sbp <= SBP_RANGE[1] and DBP_RANGE[0] <= dbp <= DBP_RANGE[1]
              and IBI_RANGE[0] <= ibi <= IBI_RANGE[1] and sbp > dbp + 10)
        rows.append((a / fs, sbp, dbp, mapp, 60.0 / ibi, int(ok)))
    return rows


def main():
    download()
    recs = list_records()
    print(f"[records] {len(recs)}: {recs}")

    inv = ["# PRCP inventory (C10-B step 0a)\n",
           f"Records: {len(recs)}\n",
           "| record | fs (Hz) | duration (min) | channels | ABP ch | beats | valid beats (%) | anI annotations |",
           "|---|---|---|---|---|---|---|---|"]
    all_beats = []
    all_anns = []
    aux_vocab = {}

    for r in recs:
        path = os.path.join(DATA_DIR, r)
        try:
            rec = wfdb.rdrecord(path)
        except Exception as e:  # noqa: BLE001
            print(f"[{r}] cannot read record: {e}")
            continue
        fs = float(rec.fs)
        names = list(rec.sig_name)
        dur_min = rec.sig_len / fs / 60.0
        ich = pick_abp_channel(names)
        print(f"[{r}] fs={fs} dur={dur_min:.1f} min channels={names} units={rec.units} -> ABP ch={ich}")

        # Maneuver annotations
        ann = read_ann(path, "anI")
        ann_txt = []
        if ann is not None:
            for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                aux = (aux or "").strip()
                all_anns.append((r, s / fs, sym, aux))
                key = f"{sym}|{aux}"
                aux_vocab[key] = aux_vocab.get(key, 0) + 1
                ann_txt.append(f"{s / fs:.0f}s:{sym}{('/' + aux) if aux else ''}")

        # Beats
        nb, nvalid, beats = 0, 0, []
        if ich is not None:
            abp = rec.p_signal[:, ich].astype(float)
            on = read_ann(path, "wabp")
            if on is not None and len(on.sample) > 2:
                beats = beats_from_onsets(abp, fs, on.sample)
                nb = len(beats)
                nvalid = sum(b[-1] for b in beats)
                for b in beats:
                    all_beats.append((r,) + b)

        pct = 100.0 * nvalid / nb if nb else 0.0
        inv.append(f"| {r} | {fs:g} | {dur_min:.1f} | {', '.join(names)} | {ich} | {nb} | {pct:.0f} | "
                   f"{'; '.join(ann_txt) if ann_txt else '—'} |")

        # Figure: beat-to-beat SBP/DBP/MAP and HR with annotation markers
        if beats:
            b = np.array([x[:5] for x in beats if x[-1] == 1])
            if len(b):
                fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
                t = b[:, 0] / 60.0
                ax[0].plot(t, b[:, 1], lw=0.8, label="SBP")
                ax[0].plot(t, b[:, 3], lw=0.8, label="MAP")
                ax[0].plot(t, b[:, 2], lw=0.8, label="DBP")
                ax[0].set_ylabel("mmHg")
                ax[0].legend(loc="upper right", fontsize=8, frameon=False)
                ax[1].plot(t, b[:, 4], lw=0.8, color="k")
                ax[1].set_ylabel("HR (bpm)")
                ax[1].set_xlabel("time (min)")
                if ann is not None:
                    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note):
                        for a_ in ax:
                            a_.axvline(s / fs / 60.0, color="0.6", lw=0.6, ls="--")
                        ax[0].text(s / fs / 60.0, ax[0].get_ylim()[1], (aux or sym)[:12],
                                   fontsize=6, rotation=90, va="top", ha="right", color="0.4")
                fig.suptitle(f"PRCP {r}: beat-to-beat pressure and heart rate")
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, f"fig_{r}.png"), dpi=130)
                plt.close(fig)

    inv.append("\n## Annotation vocabulary (symbol|aux_note: count)\n")
    for k, v in sorted(aux_vocab.items(), key=lambda kv: -kv[1]):
        inv.append(f"- `{k}`: {v}")

    with open(os.path.join(OUT_DIR, "prcp_inventory.md"), "w") as f:
        f.write("\n".join(inv) + "\n")
    with open(os.path.join(OUT_DIR, "prcp_beats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record", "t_s", "sbp", "dbp", "map", "hr", "valid"])
        w.writerows(all_beats)
    with open(os.path.join(OUT_DIR, "prcp_annotations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record", "t_s", "symbol", "aux_note"])
        w.writerows(all_anns)

    print(f"\n[done] wrote {OUT_DIR}/prcp_inventory.md, prcp_beats.csv, prcp_annotations.csv and figures")
    print("Send back: prcp_inventory.md (paste) + 2 figures (one tilt, one stand-up record).")


if __name__ == "__main__":
    main()
