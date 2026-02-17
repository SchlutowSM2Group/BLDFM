"""Meteorological data visualizations."""

import numpy as np
import matplotlib.pyplot as plt

from ._common import optional_import


def plot_wind_rose(wind_speed, wind_dir, ax=None, bins=None, title=None):
    """Plot a wind rose from meteorological data.

    Requires the optional ``windrose`` package.

    Parameters
    ----------
    wind_speed : array-like
        Wind speed values.
    wind_dir : array-like
        Wind direction values (degrees, meteorological convention).
    ax : WindroseAxes, optional
    bins : array-like, optional
        Speed bins.
    title : str, optional

    Returns
    -------
    ax : WindroseAxes
    """
    windrose = optional_import("windrose", "windrose")
    WindroseAxes = windrose.WindroseAxes

    ws = np.asarray(wind_speed)
    wd = np.asarray(wind_dir)

    if ax is None:
        fig = plt.figure()
        ax = WindroseAxes.from_ax(fig=fig)

    if bins is None:
        bins = np.linspace(0, np.ceil(ws.max()), 6)

    ax.bar(wd, ws, bins=bins, normed=True, opening=0.8, edgecolor="white")
    ax.set_legend(title="Wind speed [m/s]")
    if title:
        ax.set_title(title)
    return ax
