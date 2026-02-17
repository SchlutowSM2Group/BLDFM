"""Shared plotting utilities used across submodules."""

import importlib

import matplotlib.pyplot as plt

from ..utils import get_logger

logger = get_logger("plotting")


def ensure_ax(ax=None, **subplots_kw):
    """Return existing axes or create new figure + axes."""
    if ax is None:
        _, ax = plt.subplots(**subplots_kw)
    return ax


def unpack_grid_2d(grid):
    """Extract X, Y from a (X, Y, Z) grid tuple."""
    X, Y, _ = grid
    return X, Y


def format_colorbar_scientific(cbar, label=None):
    """Apply scientific notation formatting to a colorbar."""
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    if label:
        cbar.set_label(label)


def optional_import(module_name, package_hint):
    """Import an optional dependency, raising a helpful error if missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"{module_name} is required for this feature. "
            f"Install it with: pip install {package_hint}"
        )
