"""
This module contains unit tests for the :py:func:`src.pbl_model.vertical_profiles` function in the :py:mod:`src.pbl_model` module. The tests validate the behavior of the function under different closure schemes, including:

- `CONSTANT` Closure: Ensures the profiles for velocity and eddy diffusivity are constant and match expected values.
- `MOST` (Monin-Obukhov Similarity Theory) Closure: Verifies that the profiles vary with height and are physically reasonable.

The tests use parameterized inputs to check the function's output shapes, values, and physical consistency across various scenarios.
"""

import pytest

pytestmark = pytest.mark.unit
import numpy as np
from bldfm.pbl_model import vertical_profiles


@pytest.mark.parametrize(
    "n, meas_height, wind, ustar, prsc",
    [
        (10, 5.0, (3.0, 1.0), 0.4, 1.0),
        (20, 10.0, (2.0, 2.0), 0.3, 0.8),
    ],
)
def test_constant_closure(n, meas_height, wind, ustar, prsc):
    """Test the CONSTANT closure for scalar inputs.

    Raises:
        AssertionError: If the output shapes do not match the expected dimensions or if the profiles are not constant.
    """
    closure = "CONSTANT"

    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=n,
        meas_height=meas_height,
        wind=wind,
        ustar=ustar,
        closure=closure,
        prsc=prsc,
    )

    # Check the shape of the outputs
    assert len(u) == len(z)
    assert len(v) == len(z)
    assert len(Kx) == len(z)
    assert len(Ky) == len(z)
    assert len(Kz) == len(z)

    # # Check that the profiles are constant
    np.testing.assert_array_almost_equal(u, np.full(len(z), wind[0]))
    np.testing.assert_array_almost_equal(v, np.full(len(z), wind[1]))
    np.testing.assert_array_almost_equal(
        Kz, np.full(len(z), 0.4 * ustar * meas_height / prsc)
    )


@pytest.mark.parametrize(
    "n, meas_height, wind, ustar, mol",
    [
        (10, 5.0, (3.0, 1.0), 0.4, -100.0),
        (20, 10.0, (2.0, 2.0), 0.3, -50.0),
    ],
)
def test_most_closure(n, meas_height, wind, ustar, mol):
    """Test the MOST closure for scalar inputs.

    Raises:
        AssertionError: If the output shapes do not match the expected dimensions or if the profiles are not physically reasonable.
    """
    closure = "MOST"

    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=n,
        meas_height=meas_height,
        wind=wind,
        ustar=ustar,
        closure=closure,
        mol=mol,
    )

    # Check the shape of the outputs
    assert len(u) == len(z)
    assert len(v) == len(z)
    assert len(Kx) == len(z)
    assert len(Ky) == len(z)
    assert len(Kz) == len(z)

    # Check that the profiles are not constant
    assert not np.isclose(u[0], u[-1])
    assert not np.isclose(v[0], v[-1])
    assert not np.isclose(Kz[0], Kz[-1])

    # Check that the profiles are physically reasonable
    assert np.all(z > 0)  # Heights should be positive
    assert np.all(Kx > 0)  # Eddy diffusivity should be positive
    assert np.all(Ky > 0)  # Eddy diffusivity should be positive
    assert np.all(Kz > 0)  # Eddy diffusivity should be positive


def test_oaahoc_closure():
    """Test the OAAHOC (one-and-a-half order) closure with explicit TKE."""
    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=10,
        meas_height=5.0,
        wind=(3.0, 1.0),
        ustar=0.4,
        closure="OAAHOC",
        tke=2.0,
    )

    assert len(u) == len(z)
    assert len(Kx) == len(z)

    # Profiles should vary with height
    assert not np.isclose(u[0], u[-1])
    assert not np.isclose(Kz[0], Kz[-1])

    # Physically reasonable
    assert np.all(z > 0)
    assert np.all(Kz > 0)

    # OAAHOC has isotropic diffusion: Kx == Ky == Kz
    np.testing.assert_array_equal(Kx, Ky)
    np.testing.assert_array_equal(Kx, Kz)


def test_oaahoc_closure_default_tke():
    """Test that OAAHOC defaults TKE to 1.0 when not provided."""
    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=10,
        meas_height=5.0,
        wind=(3.0, 1.0),
        ustar=0.4,
        closure="OAAHOC",
        tke=None,
    )

    # Should still produce valid output
    assert len(u) == len(z)
    assert np.all(z > 0)
    assert np.all(Kz > 0)


def test_mostm_closure():
    """Test the MOSTM (modified MOST) closure has anisotropic diffusion."""
    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=10,
        meas_height=5.0,
        wind=(3.0, 1.0),
        ustar=0.4,
        closure="MOSTM",
        mol=-100.0,
    )

    assert len(u) == len(z)
    assert np.all(z > 0)
    assert np.all(Kx > 0)
    assert np.all(Ky > 0)
    assert np.all(Kz > 0)

    # MOSTM has anisotropic horizontal diffusion: Kx != Ky
    assert not np.allclose(Kx, Ky)

    # Profiles should vary with height
    assert not np.isclose(u[0], u[-1])


def test_z0_only_path():
    """Test that providing z0 without ustar derives ustar from z0."""
    z, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=10,
        meas_height=5.0,
        wind=(3.0, 1.0),
        ustar=None,
        z0=0.1,
        closure="MOST",
        mol=-100.0,
    )

    assert len(u) == len(z)
    assert np.all(z > 0)
    assert np.all(Kz > 0)


def test_both_z0_and_ustar_raises():
    """Test that providing both z0 and ustar raises ValueError."""
    with pytest.raises(ValueError, match="Either z0 or ustar"):
        vertical_profiles(
            n=10,
            meas_height=5.0,
            wind=(3.0, 1.0),
            ustar=0.4,
            z0=0.1,
            closure="MOST",
        )


def test_invalid_closure_raises():
    """Test that an invalid closure type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid closure type"):
        vertical_profiles(
            n=10,
            meas_height=5.0,
            wind=(3.0, 1.0),
            ustar=0.4,
            closure="INVALID",
        )


def test_stretch_and_domain_height():
    """Test custom stretch and domain_height parameters."""
    custom_stretch = 15.0
    custom_domain_height = 50.0

    z_default, _ = vertical_profiles(
        n=10, meas_height=5.0, wind=(3.0, 1.0), ustar=0.4, closure="MOST"
    )
    z_custom, (u, v, Kx, Ky, Kz) = vertical_profiles(
        n=10,
        meas_height=5.0,
        wind=(3.0, 1.0),
        ustar=0.4,
        closure="MOST",
        stretch=custom_stretch,
        domain_height=custom_domain_height,
    )

    # Custom domain_height should extend the grid higher than default (2 * meas_height = 10)
    assert z_custom[-1] > z_default[-1]
    assert len(u) == len(z_custom)
    assert np.all(Kz > 0)
