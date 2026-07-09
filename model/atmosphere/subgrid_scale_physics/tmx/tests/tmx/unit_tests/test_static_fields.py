# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure numpy derivations in static_fields.py.

These tests exercise individual analytic computations on small synthetic
column data, requiring no serialized ICON savepoint or GT4Py backend.

Covered:
  (a) ``compute_wgtfacq1_c`` — top-boundary quadratic extrapolation
      coefficients, compared against the same Fortran formula computed
      independently in the test.
  (b) ``compute_geopot_agl_ifc`` — geopotential above ground level.
  (c) ``compute_inv_reciprocal`` — reciprocal field (used for all
      inv_ddqz_z_half derivations).
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import static_fields
from icon4py.model.common import constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_z_ifc(ncells: int, nlev: int) -> np.ndarray:
    """Build a synthetic z_ifc array: monotonically decreasing from top to bottom.

    Heights (in metres) decrease linearly from 15 km at the top interface (k=0)
    to 0 m at the surface (k=nlev).  The returned array has shape
    (ncells, nlev + 1).
    """
    heights = np.linspace(15_000.0, 0.0, nlev + 1)  # 1D, top to bottom
    z_ifc = np.tile(heights, (ncells, 1))  # broadcast to (ncells, nlev+1)
    return z_ifc


# ---------------------------------------------------------------------------
# (a) wgtfacq1_c — top-boundary quadratic extrapolation coefficients
# ---------------------------------------------------------------------------

class TestWgtfacq1C:
    """Tests for compute_wgtfacq1_c with a hand-built 4-interface column."""

    def _reference_wgtfacq1_c(self, z_ifc: np.ndarray) -> np.ndarray:
        """Independent re-implementation of the Fortran formula (0-based indices).

        Source: mo_vertical_grid.f90 lines 953-967.
        Uses only the first 4 interface levels (k = 0, 1, 2, 3).
        """
        z1 = 0.5 * (z_ifc[:, 1] - z_ifc[:, 0])
        z2 = 0.5 * (z_ifc[:, 1] + z_ifc[:, 2]) - z_ifc[:, 0]
        z3 = 0.5 * (z_ifc[:, 2] + z_ifc[:, 3]) - z_ifc[:, 0]
        w3 = z1 * z2 / ((z2 - z3) * (z1 - z3))
        w2 = (z1 - w3 * (z1 - z3)) / (z1 - z2)
        w1 = 1.0 - (w2 + w3)
        return np.stack([w1, w2, w3], axis=1)  # (ncells, 3), Fortran order

    def test_shape(self) -> None:
        """Result must be (ncells, 3)."""
        ncells, nlev = 5, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_wgtfacq1_c(z_ifc)
        assert result.shape == (ncells, 3)

    def test_values_match_reference(self) -> None:
        """Computed coefficients must match the independently-derived reference."""
        ncells, nlev = 4, 10  # small so we can inspect
        z_ifc = _make_z_ifc(ncells, nlev)

        result = static_fields.compute_wgtfacq1_c(z_ifc)
        ref = self._reference_wgtfacq1_c(z_ifc)

        np.testing.assert_allclose(result, ref, rtol=1.0e-14)

    def test_coefficients_sum_to_one(self) -> None:
        """w1 + w2 + w3 == 1 by construction."""
        ncells, nlev = 8, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_wgtfacq1_c(z_ifc)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1.0e-14)

    def test_uniform_column_uses_linear_weights(self) -> None:
        """For a perfectly uniform vertical spacing, the coefficients must
        still sum to one and the formula must be well-conditioned."""
        ncells = 3
        # Uniform spacing: dz = 200 m, 91 interfaces from 18000 m to 0 m
        nlev = 90
        z_ifc = _make_z_ifc(ncells, nlev)  # already linearly spaced
        result = static_fields.compute_wgtfacq1_c(z_ifc)
        assert np.all(np.isfinite(result)), "NaN or Inf in result for uniform column"
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1.0e-14)

    def test_fortran_order_column_0_multiplies_topmost_level(self) -> None:
        """Column 0 of the result (w1) should match the reference w1."""
        ncells, nlev = 2, 10
        z_ifc = _make_z_ifc(ncells, nlev)
        ref = self._reference_wgtfacq1_c(z_ifc)
        result = static_fields.compute_wgtfacq1_c(z_ifc)
        # In Fortran coefficient order, col 0 = w1 (multiplies topmost full level)
        np.testing.assert_allclose(result[:, 0], ref[:, 0], rtol=1.0e-14)


# ---------------------------------------------------------------------------
# (b) geopot_agl_ifc — geopotential above ground level
# ---------------------------------------------------------------------------

class TestGeopot:
    """Tests for compute_geopot_agl_ifc."""

    def test_shape(self) -> None:
        ncells, nlev = 3, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_geopot_agl_ifc(z_ifc)
        assert result.shape == z_ifc.shape

    def test_values(self) -> None:
        """geopot_agl_ifc == grav * (z_ifc - z_ifc[:, -1:])."""
        ncells, nlev = 5, 20
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_geopot_agl_ifc(z_ifc)
        expected = constants.GRAV * (z_ifc - z_ifc[:, -1:])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_ground_level_is_zero(self) -> None:
        """The surface interface (last column) must be zero."""
        ncells, nlev = 4, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_geopot_agl_ifc(z_ifc)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1.0e-10)

    def test_monotone_from_top(self) -> None:
        """Geopotential above ground must be non-negative and decreasing toward surface."""
        ncells, nlev = 2, 10
        z_ifc = _make_z_ifc(ncells, nlev)
        result = static_fields.compute_geopot_agl_ifc(z_ifc)
        assert np.all(result >= 0.0), "geopot_agl_ifc must be non-negative"

    def test_nonzero_terrain(self) -> None:
        """Surface level (z_ifc[:, -1]) at non-zero height: ground level still 0."""
        ncells, nlev = 2, 4
        # Create z_ifc with non-zero ground level
        z_ifc = np.array([
            [5000.0, 4000.0, 3000.0, 2000.0, 1000.0],
            [4500.0, 3500.0, 2500.0, 1500.0,  500.0],
        ])  # shape (2, 5), ground at z_ifc[:, -1] = [1000, 500]
        result = static_fields.compute_geopot_agl_ifc(z_ifc)
        expected = constants.GRAV * (z_ifc - z_ifc[:, -1:])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1.0e-10)


# ---------------------------------------------------------------------------
# (c) inv_ddqz_z_half — reciprocal field
# ---------------------------------------------------------------------------

class TestInvReciprocal:
    """Tests for compute_inv_reciprocal (used for all inv_ddqz_z_* fields)."""

    def test_values(self) -> None:
        """1 / arr gives the element-wise reciprocal."""
        arr = np.array([[2.0, 4.0, 8.0], [1.0, 0.5, 0.25]])
        result = static_fields.compute_inv_reciprocal(arr)
        expected = np.array([[0.5, 0.25, 0.125], [1.0, 2.0, 4.0]])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_roundtrip(self) -> None:
        """arr * compute_inv_reciprocal(arr) == 1 for positive arrays."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.1, 1000.0, size=(10, 91))
        inv_arr = static_fields.compute_inv_reciprocal(arr)
        np.testing.assert_allclose(arr * inv_arr, 1.0, rtol=1.0e-13)

    def test_ddqz_z_half_reciprocal(self) -> None:
        """Specific case: inv_ddqz_z_half = 1 / ddqz_z_half (element-wise)."""
        ncells, nlev = 3, 5
        ddqz_z_half = np.linspace(50.0, 200.0, (nlev + 1) * ncells).reshape(ncells, nlev + 1)
        inv_ddqz_z_half = static_fields.compute_inv_reciprocal(ddqz_z_half)
        np.testing.assert_allclose(
            inv_ddqz_z_half,
            1.0 / ddqz_z_half,
            rtol=1.0e-14,
        )
