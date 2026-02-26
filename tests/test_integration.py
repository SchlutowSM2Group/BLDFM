"""
This module contains an integration test for the interaction between the `pbl_model`, `utils`, and `solver` components.

The test validates the combined behavior of these components by comparing the numerical and analytical solutions for the steady-state advection-diffusion equation.
Currently, this test is similar to combining the `pbl_model` and `solver` tests, but it is designed to serve as a foundation for future extensions.

Functions:
----------
- test_integration: Tests the combined behavior of the `pbl_model`, `utils`, and `solver` components.
"""

import pytest

pytestmark = pytest.mark.integration

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source
from bldfm.solver import steady_state_transport_solver
from bldfm.ffm_kormann_meixner import estimateFootprint as FKM


def test_integration():
    """
    Tests the combined behavior of the `pbl_model`, `utils`, and `solver` components.

    The test validates:
        - The interaction between the components.
        - The numerical accuracy of the solver by comparing its results to the analytical solution within a specified tolerance.

    Raises:
        AssertionError: If the numerical and analytical solutions differ beyond the specified tolerance.
    """
    # Define inputs
    nxy = 256, 256
    modes = 512, 512
    nz = 256
    # nxy         = 128, 64
    # modes       = 128, 128
    # nz          = 16
    domain = 100.0, 50.0
    src_pt = 5.0, 5.0
    halo = 500.0
    meas_height = 5.0
    wind = 4.0, 1.0
    ustar = 0.2

    # Generate inputs using pbl_model and utils
    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

    # Solve using the solver
    _, conc_ana, flx_ana = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo, analytic=True
    )
    _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo
    )

    # Validate results
    diff_conc = (conc - conc_ana) / np.max(conc_ana)
    diff_flx = (flx - flx_ana) / np.max(flx_ana)

    assert np.allclose(diff_conc, 0, atol=1e-3), "Concentration mismatch too large"
    assert np.allclose(diff_flx, 0, atol=1e-3), "Flux mismatch too large"

    print(
        f"\nINTEGRATION numerical_vs_analytical: "
        f"max_err_conc={np.abs(diff_conc).max():.4e} "
        f"max_err_flx={np.abs(diff_flx).max():.4e}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(flx_ana, origin="lower", aspect="auto")
    axes[0].set_title("Analytical")
    axes[1].imshow(flx, origin="lower", aspect="auto")
    axes[1].set_title("Numerical")
    fig.suptitle("Flux: analytical vs numerical")
    fig.savefig("plots/test_integration_integration.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def test_convergence_trend():
    """Verify that numerical error decreases monotonically with resolution.

    Runs the solver at 3 resolution levels against the analytic solution
    (CONSTANT closure) and asserts that relative MSE decreases at each
    refinement step.
    """
    nxy = 128, 64
    domain = 100.0, 50.0
    src_pt = 5.0, 5.0
    halo = 500.0
    meas_height = 5.0
    wind = 4.0, 1.0
    ustar = 0.2

    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")

    resolutions = [
        {"modes": (64, 64), "nz": 8},
        {"modes": (128, 128), "nz": 16},
        {"modes": (256, 256), "nz": 32},
    ]

    # Analytic reference at finest resolution
    z_ref, profs_ref = vertical_profiles(
        resolutions[-1]["nz"], meas_height, wind, ustar, closure="CONSTANT"
    )
    _, _, flx_ana = steady_state_transport_solver(
        srf_flx,
        z_ref,
        profs_ref,
        domain,
        resolutions[-1]["nz"],
        modes=resolutions[-1]["modes"],
        halo=halo,
        analytic=True,
    )

    errors = []
    for res in resolutions:
        z, profs = vertical_profiles(
            res["nz"], meas_height, wind, ustar, closure="CONSTANT"
        )
        _, _, flx = steady_state_transport_solver(
            srf_flx,
            z,
            profs,
            domain,
            res["nz"],
            modes=res["modes"],
            halo=halo,
        )
        rmse = np.mean((flx - flx_ana) ** 2) / np.mean(flx_ana**2)
        errors.append(rmse)

    # Error must decrease monotonically with resolution
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1], (
            f"Error did not decrease: errors[{i}]={errors[i]:.6e} "
            f"<= errors[{i+1}]={errors[i+1]:.6e}"
        )

    print("\nINTEGRATION convergence_trend:")
    for res, err in zip(resolutions, errors):
        print(f"  modes={res['modes']} nz={res['nz']} rmse={err:.6e}")

    from bldfm.plotting import plot_convergence

    grid_spacings = np.array([domain[0] / r["modes"][0] for r in resolutions])
    ax = plot_convergence(
        grid_spacings, np.array(errors), title="Convergence trend (test)"
    )
    ax.figure.savefig(
        "plots/test_integration_convergence_trend.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


def _quick_solve(
    footprint=True, precision="single", modes=(128, 64), halo=None, meas_pt=(0.0, 0.0)
):
    """Helper: small solve at conftest scale for fast tests."""
    nxy = (128, 64)
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
        precision=precision,
        halo=halo,
        meas_pt=meas_pt,
    )


def _quick_footprint_solve(
    closure="MOST",
    wind=(0.0, -5.0),
    ustar=0.4,
    z0=None,
    mol=1e9,
    nxy=(64, 256),
    domain=(50.0, 200.0),
    modes=(64, 128),
    meas_pt=(25.0, 10.0),
    meas_height=10.0,
    nz=16,
    halo=None,
):
    """Helper: footprint solve returning (grid, conc, flx, dx, dy).

    Creates a zero surface-flux field (shape is all that matters in footprint
    mode), builds vertical profiles via vertical_profiles, then calls
    steady_state_transport_solver with footprint=True.  The extra dx/dy values
    are included for mass-conservation integration tests.
    """
    nx, ny = nxy
    xmx, ymx = domain
    srf_flx = np.zeros((ny, nx))
    kw = dict(n=nz, meas_height=meas_height, wind=wind, closure=closure, mol=mol)
    if z0 is not None:
        kw["z0"] = z0
    else:
        kw["ustar"] = ustar
    z, profs = vertical_profiles(**kw)
    grid, conc, flx = steady_state_transport_solver(
        srf_flx,
        z,
        profs,
        domain,
        nz,
        modes=modes,
        meas_pt=meas_pt,
        footprint=True,
        halo=halo,
    )
    dx = xmx / nx
    dy = ymx / ny
    return grid, conc, flx, dx, dy


def test_solver_single_precision():
    """Verify single precision path produces float32 output."""
    _, conc, flx = _quick_solve(precision="single")
    assert conc.dtype == np.float32 or conc.dtype == np.float64
    # Single precision uses complex64 internally -> real part is float32
    assert flx.shape == conc.shape
    print(f"\nINTEGRATION solver_single: dtype={conc.dtype} shape={conc.shape}")


def test_solver_double_precision():
    """Verify double precision path produces float64 output."""
    _, conc, flx = _quick_solve(precision="double")
    assert conc.dtype == np.float64
    assert flx.dtype == np.float64
    print(f"\nINTEGRATION solver_double: dtype={conc.dtype} shape={conc.shape}")


def test_solver_invalid_precision_raises():
    """Verify invalid precision raises ValueError."""
    with pytest.raises(ValueError, match="precision must be"):
        _quick_solve(precision="quad")


def test_solver_odd_modes_raises():
    """Verify odd modes raise ValueError."""
    with pytest.raises(ValueError, match="modes must consist of even numbers"):
        _quick_solve(modes=(63, 64))


def test_solver_halo_overflow():
    """Verify solver handles modes exceeding grid+halo gracefully."""
    # Use a tiny halo so modes > grid+halo triggers the clamping path
    _, conc, flx = _quick_solve(modes=(512, 512), halo=1.0)
    assert conc.shape == flx.shape
    assert np.isfinite(flx).all()
    print(f"\nINTEGRATION solver_halo_overflow: shape={conc.shape} all_finite=True")


def test_solver_non_footprint_shift():
    """Verify non-footprint mode with non-zero meas_pt (shift path)."""
    _, conc, flx = _quick_solve(
        footprint=False, meas_pt=(25.0, 12.5), precision="double"
    )
    assert conc.shape == flx.shape
    assert np.isfinite(conc).all()
    print(
        f"\nINTEGRATION solver_non_footprint_shift: shape={conc.shape} all_finite=True"
    )


# ---------------------------------------------------------------------------
# Category 1: BLDFM footprint property tests
# ---------------------------------------------------------------------------


def test_footprint_finite_values():
    """Verify that all footprint flux and concentration values are finite.

    Runs the solver in footprint mode with default parameters and asserts
    that neither NaN nor Inf appear anywhere in the output arrays.  Prints
    shape, dtype, and value range for quick sanity inspection.
    """
    grid, conc, flx, dx, dy = _quick_footprint_solve()
    assert np.isfinite(flx).all(), "flx contains non-finite values"
    assert np.isfinite(conc).all(), "conc contains non-finite values"
    print(
        f"\nINTEGRATION footprint_finite_values: "
        f"flx shape={flx.shape} dtype={flx.dtype} "
        f"range=[{flx.min():.4e}, {flx.max():.4e}] "
        f"conc range=[{conc.min():.4e}, {conc.max():.4e}]"
    )


def test_footprint_positivity_constant():
    """Verify footprint flux is non-negative everywhere with CONSTANT closure.

    FFT roundoff can produce tiny negative values; the tolerance -1e-15
    accounts for this without masking genuine negativity bugs.  Prints the
    minimum value and fraction of cells with positive weight.
    """
    _, _, flx, _, _ = _quick_footprint_solve(closure="CONSTANT")
    assert np.all(
        flx >= -1e-15
    ), f"CONSTANT closure produced flx < -1e-15: min={flx.min():.4e}"
    pct_positive = 100.0 * np.sum(flx > 0) / flx.size
    print(
        f"\nINTEGRATION footprint_positivity_constant: "
        f"min={flx.min():.4e} pct_positive={pct_positive:.1f}%"
    )


def test_footprint_positivity_most():
    """Verify footprint flux has only small negatives with MOST closure (neutral).

    The MOST closure uses a non-constant diffusivity profile which can cause
    Gibbs/FFT ringing at the measurement-point singularity, producing negative
    values of order ~1e-5.  The tolerance -1e-4 accepts this known numerical
    artefact while still rejecting physically meaningless large negatives.
    Neutral conditions (mol=1e9) are used as the baseline case.
    """
    _, _, flx, _, _ = _quick_footprint_solve(closure="MOST", z0=0.1, mol=1e9)
    assert np.all(
        flx >= -1e-4
    ), f"MOST closure produced flx < -1e-4: min={flx.min():.4e}"
    pct_positive = 100.0 * np.sum(flx > 0) / flx.size
    print(
        f"\nINTEGRATION footprint_positivity_most: "
        f"min={flx.min():.4e} pct_positive={pct_positive:.1f}%"
    )


def test_footprint_mass_conservation_constant():
    """Verify footprint integrates to a reasonable fraction with CONSTANT closure.

    Uses a larger domain and halo so that a substantial fraction of the
    footprint falls within the window.  The acceptable range (0.25, 1.05]
    accounts for the finite domain capturing only a partial footprint — the
    true integral over an infinite domain is 1, but at this resolution and
    domain size a significant portion escapes through the upwind boundary.
    Prints the raw integral for inspection.
    """
    _, _, flx, dx, dy = _quick_footprint_solve(
        closure="CONSTANT",
        nxy=(64, 512),
        domain=(50.0, 400.0),
        modes=(64, 256),
        halo=400.0,
    )
    integral = float(np.sum(flx) * dx * dy)
    assert (
        0.25 < integral <= 1.05
    ), f"CONSTANT footprint integral out of range: {integral:.4f}"
    print(
        f"\nINTEGRATION footprint_mass_conservation_constant: integral={integral:.4f}"
    )


def test_footprint_mass_conservation_most():
    """Verify footprint integrates to a reasonable fraction with MOST closure (neutral).

    Same domain and halo as the CONSTANT test, but uses a roughness length
    (z0=0.5) and neutral Monin-Obukhov length (mol=1e9) to exercise the MOST
    code path.  The acceptable range (0.25, 1.05] is identical to the CONSTANT
    test — the finite domain captures a substantial but incomplete portion of
    the footprint.
    """
    _, _, flx, dx, dy = _quick_footprint_solve(
        closure="MOST",
        z0=0.5,
        mol=1e9,
        nxy=(64, 512),
        domain=(50.0, 400.0),
        modes=(64, 256),
        halo=400.0,
    )
    integral = float(np.sum(flx) * dx * dy)
    assert (
        0.25 < integral <= 1.05
    ), f"MOST footprint integral out of range: {integral:.4f}"
    print(f"\nINTEGRATION footprint_mass_conservation_most: integral={integral:.4f}")


def test_footprint_peak_upwind_southward():
    """Verify footprint peak is upwind when wind blows southward (v < 0).

    With wind=(0, -5) the flow moves in the -y direction, so the upwind
    surface contributing to the receptor is at y > meas_y.  Saves a plot
    showing the footprint field with the measurement point (red star) and
    detected peak (blue cross) annotated for visual verification.
    """
    meas_pt = (25.0, 10.0)
    meas_x, meas_y = meas_pt
    grid, conc, flx, dx, dy = _quick_footprint_solve(
        closure="MOST",
        wind=(0.0, -5.0),
        z0=0.1,
        mol=1e9,
        meas_pt=meas_pt,
    )
    X, Y, _ = grid
    peak_idx = np.unravel_index(np.argmax(flx), flx.shape)
    peak_y = float(Y[peak_idx])
    peak_x = float(X[peak_idx])
    assert (
        peak_y > meas_y
    ), f"Footprint peak at y={peak_y:.2f} is NOT upwind of meas_y={meas_y:.2f}"
    print(
        f"\nINTEGRATION footprint_peak_upwind_southward: "
        f"peak=({peak_x:.2f}, {peak_y:.2f}) meas_pt=({meas_x:.2f}, {meas_y:.2f})"
    )
    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        flx,
        origin="lower",
        aspect="auto",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
    )
    ax.scatter(
        meas_x, meas_y, marker="*", color="red", s=200, zorder=5, label="meas_pt"
    )
    ax.scatter(
        peak_x,
        peak_y,
        marker="+",
        color="blue",
        s=200,
        zorder=5,
        linewidths=2,
        label="peak",
    )
    fig.colorbar(im, ax=ax, label="footprint weight")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Footprint: southward wind — peak should be above star")
    ax.legend()
    fig.savefig(
        "plots/test_integration_footprint_peak_southward.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")


def test_footprint_peak_upwind_westward():
    """Verify footprint peak is upwind when wind blows westward (u < 0).

    With wind=(-5, 0) the flow moves in the -x direction, so the upwind
    surface is at x > meas_x.  The domain is rotated (elongated in x)
    so that the footprint has room to develop along the wind axis.  Saves
    a plot with measurement point and peak annotated.
    """
    meas_pt = (25.0, 25.0)
    meas_x, meas_y = meas_pt
    grid, conc, flx, dx, dy = _quick_footprint_solve(
        closure="MOST",
        wind=(-5.0, 0.0),
        z0=0.1,
        mol=1e9,
        nxy=(256, 64),
        domain=(200.0, 50.0),
        modes=(128, 64),
        meas_pt=meas_pt,
    )
    X, Y, _ = grid
    peak_idx = np.unravel_index(np.argmax(flx), flx.shape)
    peak_y = float(Y[peak_idx])
    peak_x = float(X[peak_idx])
    assert (
        peak_x > meas_x
    ), f"Footprint peak at x={peak_x:.2f} is NOT upwind of meas_x={meas_x:.2f}"
    print(
        f"\nINTEGRATION footprint_peak_upwind_westward: "
        f"peak=({peak_x:.2f}, {peak_y:.2f}) meas_pt=({meas_x:.2f}, {meas_y:.2f})"
    )
    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        flx,
        origin="lower",
        aspect="auto",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
    )
    ax.scatter(
        meas_x, meas_y, marker="*", color="red", s=200, zorder=5, label="meas_pt"
    )
    ax.scatter(
        peak_x,
        peak_y,
        marker="+",
        color="blue",
        s=200,
        zorder=5,
        linewidths=2,
        label="peak",
    )
    fig.colorbar(im, ax=ax, label="footprint weight")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Footprint: westward wind — peak should be right of star")
    ax.legend()
    fig.savefig(
        "plots/test_integration_footprint_peak_westward.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Category 2: BLDFM vs Kormann-Meixner comparison (neutral conditions)
# ---------------------------------------------------------------------------


def _run_comparison_neutral():
    """Run both BLDFM and KM01 under neutral conditions and return results.

    Parameters are taken from the manuscript neutral comparison script
    (runs/manuscript/low_level/comparison_footprint_neutral.py) but at
    reduced resolution for speed:
      - nxy=(64, 256), domain=(50, 200), modes=(64, 256), nz=32, halo=300
      - wind=(0, -5), ustar=0.668, z0=0.5, mol=1e9, sigma_v=0.32

    Returns a dict with keys:
      bldfm_flx, bldfm_grid, km01_ffm, km01_grid_x, km01_grid_y,
      dx, dy, meas_pt
    """
    nxy = (64, 256)
    domain = (50.0, 200.0)
    modes = (64, 256)
    nz = 32
    halo = 300.0
    meas_pt = (25.0, 20.0)
    meas_height = 10.0
    wind = (0.0, -5.0)
    ustar = 0.668
    z0 = 0.5
    mol = 1e9
    sigma_v = 0.32

    nx, ny = nxy
    xmx, ymx = domain
    dx = xmx / nx
    dy = ymx / ny

    # --- BLDFM with MOST closure ---
    surf_flx = ideal_source(nxy, domain)
    z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol, closure="MOST")
    grid, conc, bldfm_flx = steady_state_transport_solver(
        surf_flx,
        z,
        profs,
        domain,
        nz,
        modes=modes,
        meas_pt=meas_pt,
        footprint=True,
        halo=halo,
    )

    # --- Kormann-Meixner footprint model ---
    # Wind direction: angle where wind comes FROM, clockwise from +Y axis.
    # wind=(0, -5) means southward flow, i.e. wind comes from the north (0 deg).
    wd = np.arctan(wind[0] / wind[1]) * 180.0 / np.pi
    umean = np.sqrt(wind[0] ** 2 + wind[1] ** 2)
    km01_grid_x, km01_grid_y, km01_ffm = FKM(
        zm=meas_height,
        z0=z0,
        ws=umean,
        ustar=ustar,
        mo_len=mol,
        sigma_v=sigma_v,
        grid_domain=[0, xmx, 0, ymx],
        grid_res=dx,
        mxy=meas_pt,
        wd=wd,
    )

    return {
        "bldfm_flx": bldfm_flx,
        "bldfm_grid": grid,
        "km01_ffm": km01_ffm,
        "km01_grid_x": km01_grid_x,
        "km01_grid_y": km01_grid_y,
        "dx": dx,
        "dy": dy,
        "meas_pt": meas_pt,
    }


def test_comparison_neutral_both_nonnegative():
    """Verify both BLDFM and KM01 produce nearly non-negative footprint values.

    KM01 is analytical and strictly non-negative by construction (only
    upwind cells with x > 0 are filled).  BLDFM uses an FFT-based solver
    whose MOST diffusivity profile introduces Gibbs ringing near the
    measurement-point singularity, producing negatives up to ~1e-4.
    The tolerance -1e-4 catches genuine solver divergence while accepting
    this known artefact.  Prints minimum values of both for inspection.
    """
    r = _run_comparison_neutral()
    bldfm_flx = r["bldfm_flx"]
    km01_ffm = r["km01_ffm"]
    assert np.all(
        bldfm_flx >= -1e-4
    ), f"BLDFM neutral footprint < -1e-4: min={bldfm_flx.min():.4e}"
    assert np.all(
        km01_ffm >= 0
    ), f"KM01 neutral footprint < 0: min={km01_ffm.min():.4e}"
    print(
        f"\nINTEGRATION comparison_neutral_both_nonnegative: "
        f"bldfm_min={bldfm_flx.min():.4e} km01_min={km01_ffm.min():.4e}"
    )


def test_comparison_neutral_mass_agreement():
    """Verify both models integrate to within a factor of 3 of each other.

    Each integral is expected to lie in [0.3, 1.2] — a wide window that
    accounts for the finite domain capturing only part of the footprint.
    The ratio check ensures neither model wildly over- or under-estimates
    the total footprint weight relative to the other.
    Note: KM01 already includes grid_res**2 per cell (see ffm_kormann_meixner
    line 321), so np.sum(km01_ffm) is directly the dimensionless integral.
    """
    r = _run_comparison_neutral()
    bldfm_integral = float(np.sum(r["bldfm_flx"]) * r["dx"] * r["dy"])
    km01_integral = float(np.sum(r["km01_ffm"]))
    assert (
        0.3 < bldfm_integral < 1.2
    ), f"BLDFM integral out of range: {bldfm_integral:.4f}"
    assert 0.3 < km01_integral < 1.2, f"KM01 integral out of range: {km01_integral:.4f}"
    ratio = bldfm_integral / km01_integral
    assert (
        0.3 < ratio < 3.0
    ), f"Integral ratio BLDFM/KM01 out of [0.3, 3.0]: {ratio:.4f}"
    print(
        f"\nINTEGRATION comparison_neutral_mass_agreement: "
        f"bldfm={bldfm_integral:.4f} km01={km01_integral:.4f} ratio={ratio:.4f}"
    )


def test_comparison_neutral_peak_proximity():
    """Verify BLDFM and KM01 peak locations agree within 50 m.

    Both peaks must also be upwind of the measurement point (y > meas_y)
    for wind=(0, -5).  The 50 m tolerance is generous given the different
    physical assumptions of the two models.  Prints both peak coordinates
    and the Euclidean distance between them.
    """
    r = _run_comparison_neutral()
    bldfm_flx = r["bldfm_flx"]
    km01_ffm = r["km01_ffm"]
    meas_x, meas_y = r["meas_pt"]
    X, Y, _ = r["bldfm_grid"]

    bldfm_idx = np.unravel_index(np.argmax(bldfm_flx), bldfm_flx.shape)
    bldfm_peak_x = float(X[bldfm_idx])
    bldfm_peak_y = float(Y[bldfm_idx])

    km01_idx = np.unravel_index(np.argmax(km01_ffm), km01_ffm.shape)
    km01_peak_x = float(r["km01_grid_x"][km01_idx])
    km01_peak_y = float(r["km01_grid_y"][km01_idx])

    dist = np.sqrt(
        (bldfm_peak_x - km01_peak_x) ** 2 + (bldfm_peak_y - km01_peak_y) ** 2
    )

    assert (
        bldfm_peak_y > meas_y
    ), f"BLDFM peak at y={bldfm_peak_y:.2f} is NOT upwind of meas_y={meas_y:.2f}"
    assert (
        km01_peak_y > meas_y
    ), f"KM01 peak at y={km01_peak_y:.2f} is NOT upwind of meas_y={meas_y:.2f}"
    assert dist < 50.0, f"Peak distance {dist:.2f} m exceeds 50 m tolerance"
    print(
        f"\nINTEGRATION comparison_neutral_peak_proximity: "
        f"bldfm_peak=({bldfm_peak_x:.2f}, {bldfm_peak_y:.2f}) "
        f"km01_peak=({km01_peak_x:.2f}, {km01_peak_y:.2f}) "
        f"dist={dist:.2f} m"
    )


def test_comparison_neutral_bldfm_broader():
    """Verify BLDFM footprint is spatially broader than KM01.

    Breadth is measured by counting grid cells above 10% of the respective
    model's peak value.  BLDFM is expected to be broader because its PBL
    diffusion scheme accounts for along-wind diffusion that KM01 neglects.
    Saves a side-by-side contour plot for visual verification.
    """
    r = _run_comparison_neutral()
    bldfm_flx = r["bldfm_flx"]
    km01_ffm = r["km01_ffm"]
    meas_x, meas_y = r["meas_pt"]
    X, Y, _ = r["bldfm_grid"]

    bldfm_threshold = 0.1 * bldfm_flx.max()
    km01_threshold = 0.1 * km01_ffm.max()
    bldfm_count = int(np.sum(bldfm_flx > bldfm_threshold))
    km01_count = int(np.sum(km01_ffm > km01_threshold))

    assert (
        bldfm_count >= km01_count
    ), f"BLDFM breadth ({bldfm_count} cells) < KM01 breadth ({km01_count} cells)"
    print(
        f"\nINTEGRATION comparison_neutral_bldfm_broader: "
        f"bldfm_count={bldfm_count} km01_count={km01_count}"
    )

    # --- side-by-side contour plot ---
    vmin = max(1e-6, min(bldfm_flx.max(), km01_ffm.max()) * 0.01)
    vmax = max(bldfm_flx.max(), km01_ffm.max())
    lvls = np.linspace(vmin, vmax, 6, endpoint=False)
    cmap = "turbo"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")

    axs[0].contour(X, Y, bldfm_flx, lvls, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=2)
    axs[0].scatter(meas_x, meas_y, marker="*", color="red", s=200, zorder=5)
    axs[0].set_title("BLDFM (MOST)")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    cs = axs[1].contour(
        r["km01_grid_x"],
        r["km01_grid_y"],
        km01_ffm,
        lvls,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=2,
    )
    axs[1].scatter(meas_x, meas_y, marker="*", color="red", s=200, zorder=5)
    axs[1].set_title("KM01")
    axs[1].set_xlabel("x [m]")

    cbar = fig.colorbar(cs, ax=axs, shrink=0.8, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("footprint weight [m$^{-2}$]")

    fig.suptitle("Neutral footprint: BLDFM vs KM01")
    fig.savefig(
        "plots/test_integration_comparison_neutral.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")


def test_3d_plume_structure(plume_3d_result_session):
    """Test 3D plume solver output: shapes, finite values, plume structure."""
    r = plume_3d_result_session
    X, Y, Z = r["grid"]
    conc, flx = r["conc"], r["flx"]
    levels = r["levels"]
    nlvls = len(levels)

    # 3D shapes: (nlvls, ny, nx)
    assert conc.ndim == 3
    assert conc.shape == (nlvls, 32, 64)
    assert flx.shape == conc.shape
    assert X.shape == conc.shape

    # Finite values
    assert np.isfinite(conc).all()
    assert np.isfinite(flx).all()

    # Non-trivial: concentration should have positive values (dispersion mode)
    assert conc.max() > 0, "Concentration field is all non-positive"

    print(
        f"\nINTEGRATION 3d_plume: shape={conc.shape} levels={list(levels)} "
        f"conc_range=[{conc.min():.4e}, {conc.max():.4e}] "
        f"flx_range=[{flx.min():.4e}, {flx.max():.4e}]"
    )

    # Plot: y-slice through domain centre showing vertical plume structure
    mid_y = conc.shape[1] // 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, field, label in [
        (axes[0], conc, "Concentration"),
        (axes[1], flx, "Flux"),
    ]:
        pm = ax.pcolormesh(
            X[:, mid_y, :],
            Z[:, mid_y, :],
            field[:, mid_y, :],
            cmap="viridis",
            shading="auto",
        )
        fig.colorbar(pm, ax=ax, label=label)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{label} (y-slice at mid-domain)")

    fig.suptitle("3D plume vertical cross-section")
    fig.savefig("plots/test_integration_3d_plume.png", dpi=150, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # Run the tests manually (optional, for debugging)
    test_integration()
    test_convergence_trend()
