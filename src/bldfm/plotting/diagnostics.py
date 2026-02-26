"""Diagnostic and analysis plots: convergence, vertical profiles, vertical slices."""

import numpy as np
import matplotlib.pyplot as plt

from ._common import ensure_ax


def plot_convergence(
    grid_sizes,
    errors,
    fits=None,
    marker="o",
    ax=None,
    xlabel="$h$ [m]",
    ylabel="Relative RMSE",
    title=None,
):
    """Log-log convergence plot.

    Parameters
    ----------
    grid_sizes : array-like
        Effective grid spacings (h).
    errors : array-like
        Error values (e.g., RMSE).
    fits : list of (func, params, label), optional
        Curve fits to overlay. Each element is a tuple:
        (function, parameter_dict, label_string).
        Example: (lambda h, a, b: a * h**b, {"a": 0.1, "b": 2.0}, "$O(h^2)$")
    marker : str
        Marker style for data points (default "o").
    ax : matplotlib Axes, optional
    xlabel : str
        X-axis label (default "$h$ [m]").
    ylabel : str
        Y-axis label (default "Relative RMSE").
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    ax = ensure_ax(ax)

    grid_sizes = np.asarray(grid_sizes)
    errors = np.asarray(errors)

    ax.loglog(grid_sizes, errors, marker=marker, linestyle="-", label="Data")

    if fits is not None:
        for func, params, label in fits:
            h_fit = np.linspace(grid_sizes.min(), grid_sizes.max(), 100)
            y_fit = func(h_fit, **params)
            ax.loglog(h_fit, y_fit, linestyle="--", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if title:
        ax.set_title(title)

    return ax


def plot_vertical_profiles(
    z_list, profiles_list, labels, meas_height=None, figsize=None, title=None
):
    r"""Multi-panel (1x2) profile plot: wind speed and diffusivity vs height.

    Left panel: wind speed \|U\| = sqrt(u² + v²) vs z.
    Right panel: Kz vs z.

    Parameters
    ----------
    z_list : list of array-like
        List of z arrays (one per stability condition).
    profiles_list : list of tuple
        List of profile tuples (u, v, Kx, Ky, Kz), one per stability condition.
    labels : list of str
        Legend labels for each profile (e.g., "L = -10 m").
    meas_height : float, optional
        Measurement height to mark with a horizontal dashed line.
    figsize : tuple, optional
        Figure size (width, height). Default (10, 5).
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes (1x2)
    """
    if figsize is None:
        figsize = (10, 5)

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(z_list))]

    for z, profs, label, color in zip(z_list, profiles_list, labels, colors):
        u, v, Kx, Ky, Kz = profs
        speed = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2)
        axes[0].plot(speed, z, marker="+", linestyle="-", label=label, color=color)
        axes[1].plot(Kz, z, marker="", linestyle="-", label=label, color=color)

    if meas_height is not None:
        for ax in axes:
            ax.axhline(meas_height, color="gray", linestyle="--", linewidth=1, zorder=0)

    axes[0].set_xlabel("|U| [m/s]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Wind speed profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Kz [m\u00b2/s]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("Diffusivity profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    return fig, axes


def plot_vertical_slice(
    field,
    grid,
    slice_axis,
    slice_index,
    ax=None,
    cmap="viridis",
    title=None,
    xlabel=None,
    ylabel=None,
):
    """2D slice from a 3D field.

    Parameters
    ----------
    field : ndarray (nz, ny, nx)
        3-D field to slice.
    grid : tuple (X, Y, Z)
        3-D coordinate arrays from the solver.
    slice_axis : str
        Axis to slice along: "x", "y", or "z".
    slice_index : int
        Index along the slice axis.
    ax : matplotlib Axes, optional
    cmap : str
        Colormap name (default "viridis").
    title : str, optional
    xlabel : str, optional
        X-axis label (auto-generated if None).
    ylabel : str, optional
        Y-axis label (auto-generated if None).

    Returns
    -------
    ax : matplotlib Axes
    """
    X, Y, Z = grid
    ax = ensure_ax(ax)

    if slice_axis == "y":
        slice_data = field[:, slice_index, :]
        x_coord = X[:, slice_index, :] if X.ndim == 3 else X
        z_coord = Z[:, slice_index, :] if Z.ndim == 3 else Z
        pm = ax.pcolormesh(x_coord, z_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "x [m]"
        if ylabel is None:
            ylabel = "z [m]"

    elif slice_axis == "x":
        slice_data = field[:, :, slice_index]
        y_coord = Y[:, :, slice_index] if Y.ndim == 3 else Y
        z_coord = Z[:, :, slice_index] if Z.ndim == 3 else Z
        pm = ax.pcolormesh(y_coord, z_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "y [m]"
        if ylabel is None:
            ylabel = "z [m]"

    elif slice_axis == "z":
        slice_data = field[slice_index, :, :]
        x_coord = X[slice_index, :, :] if X.ndim == 3 else X
        y_coord = Y[slice_index, :, :] if Y.ndim == 3 else Y
        pm = ax.pcolormesh(x_coord, y_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "x [m]"
        if ylabel is None:
            ylabel = "y [m]"

    else:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got {slice_axis}")

    ax.figure.colorbar(pm, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax
