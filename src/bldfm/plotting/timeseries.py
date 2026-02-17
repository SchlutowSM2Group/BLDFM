"""Temporal footprint evolution plots."""

import numpy as np
import matplotlib.pyplot as plt

from ._common import ensure_ax
from .footprint import extract_percentile_contour


def plot_footprint_timeseries(results, grid, pcts=None, ax=None, title=None, level=0):
    """Plot temporal evolution of footprint extent.

    Parameters
    ----------
    results : list of dict
        Output of run_bldfm_timeseries (list of result dicts for one tower).
    grid : tuple (X, Y, Z)
        Coordinate arrays (from first result, assumed constant).
    pcts : list of float, optional
        Percentile fractions to track (default [0.5, 0.8]).
    ax : matplotlib Axes, optional
    title : str, optional
    level : int
        Z-index to use when footprint fields are 3D. Default 0 (surface).

    Returns
    -------
    ax : matplotlib Axes
    """
    if pcts is None:
        pcts = [0.5, 0.8]

    ax = ensure_ax(ax)

    timestamps = [r["timestamp"] for r in results]
    x_pos = np.arange(len(timestamps))

    for pct in pcts:
        areas = []
        for r in results:
            _, area = extract_percentile_contour(r["flx"], grid, pct, level=level)
            areas.append(area / 1e6)  # m^2 -> km^2
        ax.plot(x_pos, areas, "o-", label=f"{int(pct*100)}%")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t) for t in timestamps], rotation=45, ha="right")
    ax.set_ylabel("Footprint area [km$^2$]")
    ax.set_xlabel("Timestamp")
    ax.legend()
    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax
