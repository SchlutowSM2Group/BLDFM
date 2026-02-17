"""Memory profiling tests for the BLDFM solver and FFT subsystem.

Each test runs its own isolated solver calls with explicit gc.collect()
and tracemalloc start/stop for clean measurements. Metrics are printed
with a ``MEMORY`` prefix so they are greppable in CI logs.
"""

import gc
import tracemalloc

import numpy as np
import psutil
import pytest

from bldfm.fft_manager import get_fft_manager, reset_fft_manager
from bldfm.pbl_model import vertical_profiles
from bldfm.solver import steady_state_transport_solver
from bldfm.utils import ideal_source

pytestmark = pytest.mark.memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SINGLE_SOLVE_THRESHOLD_MB = 500  # generous ceiling for conftest-scale solve


def _rss_mb():
    """Return current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024**2


def _run_solve(footprint=True):
    """Run one solver call at conftest scale (128x64, modes 128x64, nz=16)."""
    nxy = (128, 64)
    modes = (128, 64)
    nz = 16
    domain = (500.0, 250.0)
    src_pt = (250.0, 125.0)
    meas_height = 10.0
    wind = (5.0, 0.0)
    ustar = 0.4

    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="MOST")

    return steady_state_transport_solver(
        srf_flx,
        z,
        profs,
        domain,
        nz - 1,
        modes=modes,
        footprint=footprint,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_solve_memory():
    """Measure peak memory for a single footprint solve."""
    gc.collect()
    rss_before = _rss_mb()

    tracemalloc.start()
    _run_solve(footprint=True)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _rss_mb()
    peak_tracemalloc_mb = peak_bytes / 1024**2
    rss_delta = rss_after - rss_before

    print(
        f"\nMEMORY peak_tracemalloc={peak_tracemalloc_mb:.1f}MB "
        f"rss_before={rss_before:.1f}MB "
        f"rss_after={rss_after:.1f}MB "
        f"rss_delta={rss_delta:.1f}MB"
    )

    assert (
        rss_delta < SINGLE_SOLVE_THRESHOLD_MB
    ), f"Single solve RSS delta {rss_delta:.1f}MB exceeds {SINGLE_SOLVE_THRESHOLD_MB}MB"


def test_sequential_solves_no_leak():
    """Run 10 sequential solves and check RSS doesn't grow monotonically."""
    n_solves = 10
    rss_values = []

    # Warm up (first solve triggers numba compilation + FFTW planning)
    _run_solve(footprint=True)
    gc.collect()

    for _ in range(n_solves):
        _run_solve(footprint=True)
        gc.collect()
        rss_values.append(_rss_mb())

    rss_str = ", ".join(f"{v:.1f}" for v in rss_values)
    print(f"\nMEMORY sequential_rss=[{rss_str}]MB")

    first_rss = rss_values[0]
    final_rss = rss_values[-1]
    growth_ratio = final_rss / first_rss

    print(
        f"MEMORY first={first_rss:.1f}MB final={final_rss:.1f}MB "
        f"growth_ratio={growth_ratio:.3f}"
    )

    assert growth_ratio < 1.20, (
        f"RSS grew by {(growth_ratio - 1) * 100:.1f}% over {n_solves} solves "
        f"(first={first_rss:.1f}MB, final={final_rss:.1f}MB)"
    )


def test_cache_keepalive_impact():
    """Compare memory impact of different pyFFTW cache keepalive settings.

    This test is informational -- it prints the delta but does not assert
    a hard threshold, since the impact depends on array sizes.
    """
    # --- keepalive = 30s (current default) ---
    reset_fft_manager()
    get_fft_manager(num_threads=1, cache_keepalive=30)
    gc.collect()
    rss_before_30 = _rss_mb()
    _run_solve(footprint=True)
    rss_after_30 = _rss_mb()

    # --- keepalive = 1s ---
    reset_fft_manager()
    get_fft_manager(num_threads=1, cache_keepalive=1)
    gc.collect()
    rss_before_1 = _rss_mb()
    _run_solve(footprint=True)
    rss_after_1 = _rss_mb()

    delta_30 = rss_after_30 - rss_before_30
    delta_1 = rss_after_1 - rss_before_1

    print(
        f"\nMEMORY rss_keepalive30={delta_30:.1f}MB "
        f"rss_keepalive1={delta_1:.1f}MB "
        f"delta={delta_30 - delta_1:.1f}MB"
    )

    # Reset to default state
    reset_fft_manager()
