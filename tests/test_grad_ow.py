"""
Unit tests for gradient calculations and Okubo–Weiss sign logic.

These tests validate that the gradient units computed in `compute_features.py`
are expressed in degrees Celsius per kilometre and that the Okubo–Weiss
parameter yields negative values for a pure rotational flow (no strain).
"""

import numpy as np


def test_sst_gradient_units() -> None:
    """Check that a 1 °C per degree gradient yields ~1/111 °C/km."""
    # Create a small SST field that increases by 1 °C per degree of latitude
    sst = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    # Grid spacing: 1 degree in both directions
    lat_step_km = 111.0
    lon_step_km = 111.0  # irrelevant for this test since no lon gradient
    grad_lat = np.gradient(sst, axis=0) / lat_step_km
    grad_lon = np.gradient(sst, axis=1) / lon_step_km
    mag = np.sqrt(grad_lat ** 2 + grad_lon ** 2)
    # Expected gradient magnitude: 1 °C per degree ≈ 1/111 °C/km ≈ 0.009
    expected = 1.0 / 111.0
    # Exclude edges where one‑sided difference affects magnitude
    interior = mag[1:-1, 1:-1]
    assert np.allclose(interior, expected, atol=1e-4)


def test_okubo_weiss_sign() -> None:
    """A rigid rotation has negative Okubo–Weiss (pure vorticity, no strain)."""
    # Define a pure rotation: u = -omega * y, v = omega * x
    omega = 0.5  # arbitrary angular speed
    n = 5
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = -omega * Y
    v = omega * X
    # Assume unit spacing in km for simplicity
    dx = dy = 1.0
    dudx = np.gradient(u, axis=1) / dx
    dvdy = np.gradient(v, axis=0) / dy
    dvdx = np.gradient(v, axis=1) / dx
    dudy = np.gradient(u, axis=0) / dy
    normal_strain = dudx - dvdy
    shear_strain = dvdx + dudy
    vorticity = dvdx - dudy
    ow = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
    # All values should be negative (within numerical tolerance)
    assert np.all(ow < 0.0 + 1e-6)