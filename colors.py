"""Gradient blue dither for the terminal (24-bit color), shared by twitch.py and learnfacts.py.
Color is on when printing to a terminal. Force it through a pipe with FORCE_COLOR=1, turn it off with NO_COLOR=1."""
import os, sys

BLUES = [(8, 20, 70), (18, 52, 150), (36, 100, 214), (86, 164, 255), (160, 210, 255)]
NEON = [(255, 16, 140), (255, 20, 90), (255, 36, 36), (255, 96, 0), (255, 165, 20)]  # pink -> red -> orange
BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
COLOR = (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))) and not os.environ.get("NO_COLOR")


def rgb(c, bg=False):
    return f"\x1b[{48 if bg else 38};2;{c[0]};{c[1]};{c[2]}m" if COLOR else ""


def reset():
    return "\x1b[0m" if COLOR else ""


def dither(text, row=0, lo=0.0, hi=1.0, palette=None):
    """Left-to-right gradient, quantized to 5 colors with a 4x4 Bayer ordered dither (blue unless told otherwise)."""
    palette = palette or BLUES
    if not COLOR:
        return f"[{text}]"
    n, out = max(1, len(text) - 1), []
    for i, ch in enumerate(text):
        t = (lo + (hi - lo) * i / n) * (len(palette) - 1)
        base, frac = int(t), t - int(t)
        step = frac > (BAYER[row % 4][i % 4] + 0.5) / 16
        c = palette[min(base + step, len(palette) - 1)]
        fg = (255, 255, 255) if sum(c) < 420 else (20, 6, 30)
        out.append(rgb(c, bg=True) + rgb(fg) + ch)
    return "".join(out) + reset()


def bar(score, width=10, palette=None):
    """Stasis meter: full blocks fading into dither shades."""
    palette = palette or BLUES
    filled = score * width
    cells = []
    for i in range(width):
        f = max(0.0, min(1.0, filled - i))
        ch = " ░▒▓█"[round(f * 4)]
        cells.append(rgb(palette[min(4, 1 + i * 4 // width)]) + ch)
    return "".join(cells) + reset()


def dim(text):
    return f"\x1b[38;2;110;120;140m{text}\x1b[0m" if COLOR else text
