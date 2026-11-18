"""Render web/mirror_flip.gif: what equivariance means, in three seconds.

The healthy grand-average mu/beta pattern for right-hand imagery (power drop over the left
hemisphere) turns about the midline like a card. Halfway, it becomes its mirror image, the label
switches to "left hand", and a decoder's output flips sign: f(mirror x) = -f(x).

  python3 make_gif.py            # ~20 s, writes web/mirror_flip.gif
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

from make_figures import _class_logpower, _montage_info  # noqa: E402
from src.data import load_cohort  # noqa: E402

INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def idw_grid(x, y, v, n=160, sigma=0.16):
    """Gaussian-weighted interpolation (smooth, no bullseyes around electrodes)."""
    gx, gy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    d2 = (gx[..., None] - x) ** 2 + (gy[..., None] - y) ** 2
    w = np.exp(-d2 / (2 * sigma ** 2)) + 1e-12
    img = (w * v).sum(-1) / w.sum(-1)
    img[gx ** 2 + gy ** 2 > 1] = np.nan
    return img


def main():
    from mne.channels.layout import _find_topomap_coords
    co = load_cohort("physionet", "shared", 50)
    info = _montage_info(co.ch_names)
    pos = _find_topomap_coords(info, picks=list(range(len(info.ch_names))))
    pos = pos / (np.abs(pos).max() * 1.12)
    # right-hand imagery relative to the two-class mean = half the right-minus-left difference
    m = _class_logpower(co).mean(0) / 2
    img = idw_grid(pos[:, 0], pos[:, 1], m)
    vmax = np.nanmax(np.abs(img))

    hold, turn = 14, 22
    frames = [0.0] * hold + list(np.linspace(0, 1, turn)) + [1.0] * hold + list(np.linspace(1, 0, turn))
    fig = plt.figure(figsize=(5.4, 3.2), dpi=110)
    ax = fig.add_axes([0.02, 0.06, 0.58, 0.8])
    bx = fig.add_axes([0.70, 0.18, 0.22, 0.58])

    def draw(t):
        ax.clear(), bx.clear()
        c = np.cos(np.pi * t)                       # card turning: +1 -> -1
        sx = max(abs(c), 0.04)
        shown = img if c >= 0 else img[:, ::-1]
        ax.imshow(shown, extent=[-sx, sx, -1, 1], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="bilinear")
        ax.add_patch(Ellipse((0, 0), 2 * sx, 2, fill=False, color=INK2, linewidth=1.4))
        ax.plot([-0.12 * sx, 0, 0.12 * sx], [0.99, 1.13, 0.99], color=INK2, linewidth=1.4)
        ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.1, 1.2), ax.set_axis_off()
        right = c >= 0
        ax.set_title("Right-hand imagery" if right else "Mirror image = left-hand imagery",
                     fontsize=11, color=BLUE if right else ORANGE, pad=2)
        f = 1.0 * c
        bx.add_patch(Rectangle((0, 0), 0.6, f, color=BLUE if f >= 0 else ORANGE))
        bx.axhline(0, color=INK2, linewidth=1)
        bx.set_xlim(-0.2, 0.8), bx.set_ylim(-1.15, 1.15)
        bx.set_xticks([]), bx.set_yticks([-1, 0, 1], ["left", "0", "right"])
        for s in ("top", "right", "bottom"):
            bx.spines[s].set_visible(False)
        bx.set_title("decoder output\nf(mirror x) = −f(x)", fontsize=9, color=INK)

    anim = FuncAnimation(fig, draw, frames=frames, interval=60)
    out = Path("web") / "mirror_flip.gif"
    out.parent.mkdir(exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=16))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
