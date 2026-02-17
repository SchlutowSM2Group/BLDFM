"""
Disk cache for Green's function results.

When footprint=True, the solver output depends only on the vertical profiles,
domain geometry, and measurement point — not on the surface flux field. Caching
these results avoids redundant solves when re-running with the same PBL/domain
configuration.
"""

import hashlib
import numpy as np
from pathlib import Path

from .utils import get_logger

logger = get_logger("cache")


class GreensFunctionCache:
    """Disk-based cache for Green's function solver outputs.

    Cache key is a SHA-256 hash of the solver inputs that determine the
    Green's function: vertical grid, profiles, domain, modes, measurement
    point, halo, and precision.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cache files. Created if it does not exist.
    """

    def __init__(self, cache_dir=".bldfm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, z, profiles, domain, modes, meas_pt, halo, precision):
        """Compute SHA-256 hash from solver inputs."""
        h = hashlib.sha256()
        h.update(np.asarray(z).tobytes())
        for arr in profiles:
            h.update(np.asarray(arr).tobytes())
        h.update(np.asarray(domain).tobytes())
        h.update(np.asarray(modes).tobytes())
        h.update(np.asarray(meas_pt).tobytes())
        h.update(str(halo).encode())
        h.update(precision.encode())
        return h.hexdigest()

    def get(self, z, profiles, domain, modes, meas_pt, halo, precision):
        """Look up cached result.

        Returns
        -------
        tuple or None
            (grid, conc, flx) if cached, None on miss.
        """
        key = self._compute_key(z, profiles, domain, modes, meas_pt, halo, precision)
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            logger.debug("Cache hit: %s", key[:12])
            data = np.load(path)
            grid = (data["X"], data["Y"], data["Z"])
            return grid, data["conc"], data["flx"]
        logger.debug("Cache miss: %s", key[:12])
        return None

    def put(
        self, z, profiles, domain, modes, meas_pt, halo, precision, grid, conc, flx
    ):
        """Store a result in the cache."""
        key = self._compute_key(z, profiles, domain, modes, meas_pt, halo, precision)
        path = self.cache_dir / f"{key}.npz"
        X, Y, Z = grid
        np.savez(path, X=X, Y=Y, Z=Z, conc=conc, flx=flx)
        logger.debug("Cached: %s", key[:12])

    def clear(self):
        """Remove all cached files."""
        count = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cache entries", count)
