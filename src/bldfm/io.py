"""
NetCDF I/O for BLDFM footprint results.

.. deprecated::
    Use ``abltk.io.netcdf`` instead. These wrappers delegate to abltk
    and will be removed in a future version.
"""

import warnings


def save_footprints_to_netcdf(results, config, filepath):
    """Save multitower results to a CF-compliant NetCDF file.

    .. deprecated::
        Use ``abltk.io.netcdf.save_footprints_to_netcdf`` instead.
    """
    warnings.warn(
        "bldfm.io.save_footprints_to_netcdf is deprecated, "
        "use abltk.io.netcdf.save_footprints_to_netcdf",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.io.netcdf import save_footprints_to_netcdf as _save

    return _save(results, config, filepath)


def load_footprints_from_netcdf(filepath):
    """Load footprint results from a NetCDF file.

    .. deprecated::
        Use ``abltk.io.netcdf.load_footprints_from_netcdf`` instead.
    """
    warnings.warn(
        "bldfm.io.load_footprints_from_netcdf is deprecated, "
        "use abltk.io.netcdf.load_footprints_from_netcdf",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.io.netcdf import load_footprints_from_netcdf as _load

    return _load(filepath)
