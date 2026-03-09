"""Shared plotting utilities used across submodules.

.. deprecated::
    All functions have moved to ``abltk.plotting.axes`` and
    ``abltk.plotting.helpers``. These wrappers will be removed in a future release.
"""

import warnings


def ensure_ax(ax=None, **subplots_kw):
    warnings.warn(
        "bldfm.plotting._common.ensure_ax is deprecated, "
        "use abltk.plotting.axes.ensure_ax",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.axes import ensure_ax as _new

    return _new(ax, **subplots_kw)


def unpack_grid_2d(grid):
    warnings.warn(
        "bldfm.plotting._common.unpack_grid_2d is deprecated, "
        "use abltk.plotting.helpers.unpack_grid_2d",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.helpers import unpack_grid_2d as _new

    return _new(grid)


def format_colorbar_scientific(cbar, label=None):
    warnings.warn(
        "bldfm.plotting._common.format_colorbar_scientific is deprecated, "
        "use abltk.plotting.axes.format_colorbar_scientific",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.axes import format_colorbar_scientific as _new

    return _new(cbar, label)


def optional_import(module_name, package_hint):
    warnings.warn(
        "bldfm.plotting._common.optional_import is deprecated, "
        "use abltk.plotting.helpers.optional_import",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.helpers import optional_import as _new

    return _new(module_name, package_hint)


def _maybe_slice_level(field, grid, level=0):
    warnings.warn(
        "bldfm.plotting._common._maybe_slice_level is deprecated, "
        "use abltk.plotting.helpers._maybe_slice_level",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.helpers import _maybe_slice_level as _new

    return _new(field, grid, level)
