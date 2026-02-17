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
