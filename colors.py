"""Gradient blue dither for the terminal (24-bit color), shared by twitch.py and learnfacts.py.
Color is on when printing to a terminal. Force it through a pipe with FORCE_COLOR=1, turn it off with NO_COLOR=1."""
import os, sys

BLUES = [(8, 20, 70), (18, 52, 150), (36, 100, 214), (86, 164, 255), (160, 210, 255)]
BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
COLOR = (sys.stdout.isatty() or bool(os.environ.get("FORCE_COLOR"))) and not os.environ.get("NO_COLOR")


def rgb(c, bg=False):
    return f"\x1b[{48 if bg else 38};2;{c[0]};{c[1]};{c[2]}m" if COLOR else ""


def reset():
    return "\x1b[0m" if COLOR else ""


def dither(text, row=0, lo=0.0, hi=1.0):
    """Left-to-right blue gradient, quantized to 5 blues with a 4x4 Bayer ordered dither."""
    if not COLOR:
        return f"[{text}]"
    n, out = max(1, len(text) - 1), []
    for i, ch in enumerate(text):
        t = (lo + (hi - lo) * i / n) * (len(BLUES) - 1)
        base, frac = int(t), t - int(t)
        step = frac > (BAYER[row % 4][i % 4] + 0.5) / 16
        c = BLUES[min(base + step, len(BLUES) - 1)]
        fg = (255, 255, 255) if sum(c) < 420 else (6, 18, 60)
        out.append(rgb(c, bg=True) + rgb(fg) + ch)
    return "".join(out) + reset()


def bar(score, width=10):
    """Stasis meter: full blocks fading into dither shades."""
    filled = score * width
    cells = []
    for i in range(width):
        f = max(0.0, min(1.0, filled - i))
        ch = " ░▒▓█"[round(f * 4)]
        cells.append(rgb(BLUES[min(4, 1 + i * 4 // width)]) + ch)
    return "".join(cells) + reset()


def dim(text):
    return f"\x1b[38;2;110;120;140m{text}\x1b[0m" if COLOR else text
