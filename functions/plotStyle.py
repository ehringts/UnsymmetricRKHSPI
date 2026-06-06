"""Shared Matplotlib style for the publication-quality PDF figures."""

from __future__ import annotations


FIGURE_DPI = 2000


def rc_parameters() -> dict:
    """Return the Matplotlib rc parameters used by all figure scripts."""
    return {
        "text.usetex"        : True,
        "font.family"       : "serif",
        "font.serif"        : ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "axes.titlesize"    : 18,
        "axes.labelsize"    : 16,
        "xtick.labelsize"   : 13,
        "ytick.labelsize"   : 13,
        "legend.fontsize"   : 12,
        "lines.linewidth"   : 1.6,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    }
