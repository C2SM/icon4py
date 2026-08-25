# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.metrics import compute_weight_factors as weight_factors
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_wgtfac_c(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfac_c = data_alloc.zero_field(
        icon_grid,
        dims.CellDim,
        dims.KDim,
        dtype=ta.wpfloat,
        extend={dims.KDim: 1},
        allocator=backend,
    )
    wgtfac_c_ref = metrics_savepoint.wgtfac_c()
    z_ifc = metrics_savepoint.z_ifc()

    vertical_end = icon_grid.num_levels

    weight_factors.compute_wgtfac_c.with_backend(backend)(
        wgtfac_c,
        z_ifc,
        nlev=vertical_end,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=vertical_end + 1,
        offset_provider={},
    )

    assert test_utils.dallclose(wgtfac_c.asnumpy(), wgtfac_c_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_wgtfacq_e_dsl(
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfacq_e_ref = metrics_savepoint.wgtfacq_e()
    wgtfacq_c_ref = metrics_savepoint.wgtfacq_c()

    wgtfacq_e_dsl = weight_factors.compute_wgtfacq_e_dsl(
        e2c=icon_grid.get_connectivity("E2C").ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        wgtfacq_c_dsl=wgtfacq_c_ref.ndarray,
        c_lin_e=interpolation_savepoint.c_lin_e().ndarray,
        n_edges=icon_grid.num_edges,
        nlev=icon_grid.num_levels,
        exchange=decomposition.SingleNodeExchange(),
    )

    assert test_utils.dallclose(data_alloc.as_numpy(wgtfacq_e_dsl), wgtfacq_e_ref.asnumpy())


@pytest.mark.datatest
def test_compute_wgtfacq_c_dsl(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    wgtfacq_c_ref = metrics_savepoint.wgtfacq_c()

    wgtfacq_c_dsl = weight_factors.compute_wgtfacq_c_dsl(
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        nlev=icon_grid.num_levels,
    )
    assert test_utils.dallclose(data_alloc.as_numpy(wgtfacq_c_dsl), wgtfacq_c_ref.asnumpy())


# ---------------------------------------------------------------------------
# Property tests for the derived-metric formulas (wgtfacq1, inverse layer
# thicknesses, geopotential above ground) — moved from the tmx package when the
# computations were promoted into the metrics factory. Data-free: synthetic
# columns and connectivities only.
# ---------------------------------------------------------------------------


def _cells_to_edges(cell_field: np.ndarray, c_lin_e: np.ndarray, e2c: np.ndarray) -> np.ndarray:
    return weight_factors.compute_inv_ddqz_z_half_e(
        e2c=e2c,
        inv_ddqz_z_half=cell_field,
        c_lin_e=c_lin_e,
        exchange=decomposition.single_node_exchange,
    )


def _cells_to_verts(
    cell_field: np.ndarray, cells_aw_verts: np.ndarray, v2c: np.ndarray
) -> np.ndarray:
    return weight_factors.compute_inv_ddqz_z_half_v(
        v2c=v2c,
        inv_ddqz_z_half=cell_field,
        cells_aw_verts=cells_aw_verts,
        exchange=decomposition.single_node_exchange,
    )


def _wgtfacq1_e(wgtfacq1_c: np.ndarray, c_lin_e: np.ndarray, e2c: np.ndarray) -> np.ndarray:
    return weight_factors.compute_wgtfacq1_e(
        e2c=e2c,
        wgtfacq1_c=wgtfacq1_c,
        c_lin_e=c_lin_e,
        exchange=decomposition.single_node_exchange,
    )


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


def _make_z_ifc_nonuniform(ncells: int, nlev: int, z_top: float = 15_000.0) -> np.ndarray:
    """Power-law stretched z_ifc (non-uniform layer spacing).

    Uses h_k = z_top * ((nlev - k) / nlev)^1.5, giving thicker layers near
    the surface.  Shape (ncells, nlev + 1), monotonically decreasing.
    All cells share the same column profile.
    """
    k = np.arange(nlev + 1)
    heights = z_top * ((nlev - k) / nlev) ** 1.5
    return np.tile(heights, (ncells, 1))


# ---------------------------------------------------------------------------
# (a) wgtfacq1_c — top-boundary quadratic extrapolation coefficients
# ---------------------------------------------------------------------------


class TestWgtfacq1C:
    """Tests for compute_wgtfacq1_c using independent mathematical properties.

    The weights w1, w2, w3 are Lagrange quadratic extrapolation weights:
    given full-level mid-point distances z1, z2, z3 (signed, measured from the
    top interface z_ifc[:, 0]), the weights satisfy
        w1*p(z1) + w2*p(z2) + w3*p(z3) == p(0)
    for ANY quadratic polynomial p.  This property is tested on a
    non-uniformly spaced column so the assertion is non-trivial.
    """

    def test_shape(self) -> None:
        """Result must be (ncells, 3)."""
        ncells, nlev = 5, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_wgtfacq1_c(z_ifc)
        assert result.shape == (ncells, 3)

    def test_quadratic_extrapolation_property(self) -> None:
        """w1*p(z1)+w2*p(z2)+w3*p(z3)==p(0) for random quadratics (fixed seed).

        This is an INDEPENDENT property test — it avoids duplicating the Fortran
        formula.  The x-values z1, z2, z3 (distances below the top interface,
        i.e. negative for a descending height grid) are recovered from the same
        z_ifc definition as in the Fortran source and used purely to evaluate p.
        A non-uniform (power-law stretched) column is used so the test is
        sensitive to formula errors.
        """
        rng = np.random.default_rng(2024)
        ncells, nlev = 6, 30
        z_ifc = _make_z_ifc_nonuniform(ncells, nlev)

        weights = weight_factors.compute_wgtfacq1_c(z_ifc)  # (ncells, 3): [w1, w2, w3]

        # Sample points (signed distances from top interface) — same definitions
        # as in the Fortran source; negative because heights decrease downward.
        z1 = 0.5 * (z_ifc[:, 1] - z_ifc[:, 0])  # midpoint of level 0
        z2 = 0.5 * (z_ifc[:, 1] + z_ifc[:, 2]) - z_ifc[:, 0]  # midpoint of level 1
        z3 = 0.5 * (z_ifc[:, 2] + z_ifc[:, 3]) - z_ifc[:, 0]  # midpoint of level 2

        # Test several random quadratics: p(x) = a*x^2 + b*x + c, p(0) = c
        for _ in range(10):
            a, b, c = rng.uniform(-1.0, 1.0, 3)
            pz1 = a * z1**2 + b * z1 + c
            pz2 = a * z2**2 + b * z2 + c
            pz3 = a * z3**2 + b * z3 + c
            approx = weights[:, 0] * pz1 + weights[:, 1] * pz2 + weights[:, 2] * pz3
            # Tolerance: z-values are O(100-2000 m), so a*z^2 can reach ~10^6 while
            # p(0)=c is O(1).  Catastrophic cancellation limits accuracy to ~1e-10
            # absolute; rtol=1e-8 still catches any formula error by 7+ orders of
            # magnitude while remaining robust across all random seeds.
            np.testing.assert_allclose(
                approx,
                c,
                rtol=1.0e-8,
                atol=1.0e-8,
                err_msg=f"quadratic extrapolation failed for a={a:.4f}, b={b:.4f}, c={c:.4f}",
            )

    def test_coefficients_sum_to_one(self) -> None:
        """w1 + w2 + w3 == 1 by construction."""
        ncells, nlev = 8, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_wgtfacq1_c(z_ifc)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1.0e-14)

    def test_uniform_column_uses_linear_weights(self) -> None:
        """For a perfectly uniform vertical spacing, the coefficients must
        still sum to one and the formula must be well-conditioned."""
        ncells = 3
        # Uniform spacing: dz = 200 m, 91 interfaces from 18000 m to 0 m
        nlev = 90
        z_ifc = _make_z_ifc(ncells, nlev)  # already linearly spaced
        result = weight_factors.compute_wgtfacq1_c(z_ifc)
        assert np.all(np.isfinite(result)), "NaN or Inf in result for uniform column"
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1.0e-14)

    def test_nonuniform_column_coefficients_sum_to_one(self) -> None:
        """Sum-to-one holds for power-law stretched columns too."""
        ncells, nlev = 5, 40
        z_ifc = _make_z_ifc_nonuniform(ncells, nlev)
        result = weight_factors.compute_wgtfacq1_c(z_ifc)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=1.0e-14)


# ---------------------------------------------------------------------------
# (b) geopot_agl_ifc — geopotential above ground level
# ---------------------------------------------------------------------------


class TestGeopot:
    """Tests for compute_geopot_agl_ifc."""

    def test_shape(self) -> None:
        ncells, nlev = 3, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_geopot_agl_ifc(z_ifc)
        assert result.shape == z_ifc.shape

    def test_values(self) -> None:
        """geopot_agl_ifc == grav * (z_ifc - z_ifc[:, -1:])."""
        ncells, nlev = 5, 20
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_geopot_agl_ifc(z_ifc)
        expected = constants.GRAV * (z_ifc - z_ifc[:, -1:])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_ground_level_is_zero(self) -> None:
        """The surface interface (last column) must be zero."""
        ncells, nlev = 4, 90
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_geopot_agl_ifc(z_ifc)
        np.testing.assert_allclose(result[:, -1], 0.0, atol=1.0e-10)

    def test_monotone_from_top(self) -> None:
        """Geopotential above ground must be non-negative and decreasing toward surface."""
        ncells, nlev = 2, 10
        z_ifc = _make_z_ifc(ncells, nlev)
        result = weight_factors.compute_geopot_agl_ifc(z_ifc)
        assert np.all(result >= 0.0), "geopot_agl_ifc must be non-negative"

    def test_nonzero_terrain(self) -> None:
        """Surface level (z_ifc[:, -1]) at non-zero height: ground level still 0."""
        # Create z_ifc with non-zero ground level
        z_ifc = np.array(
            [
                [5000.0, 4000.0, 3000.0, 2000.0, 1000.0],
                [4500.0, 3500.0, 2500.0, 1500.0, 500.0],
            ]
        )  # shape (2, 5), ground at z_ifc[:, -1] = [1000, 500]
        result = weight_factors.compute_geopot_agl_ifc(z_ifc)
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
        result = weight_factors.compute_inv_ddqz_z_half(arr)
        expected = np.array([[0.5, 0.25, 0.125], [1.0, 2.0, 4.0]])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_roundtrip(self) -> None:
        """arr * compute_inv_reciprocal(arr) == 1 for positive arrays."""
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.1, 1000.0, size=(10, 91))
        inv_arr = weight_factors.compute_inv_ddqz_z_half(arr)
        np.testing.assert_allclose(arr * inv_arr, 1.0, rtol=1.0e-13)

    def test_ddqz_z_half_reciprocal(self) -> None:
        """Specific case: inv_ddqz_z_half = 1 / ddqz_z_half (element-wise)."""
        ncells, nlev = 3, 5
        ddqz_z_half = np.linspace(50.0, 200.0, (nlev + 1) * ncells).reshape(ncells, nlev + 1)
        inv_ddqz_z_half = weight_factors.compute_inv_ddqz_z_half(ddqz_z_half)
        test_utils.assert_dallclose(
            np.asarray(inv_ddqz_z_half),
            1.0 / ddqz_z_half,
            rtol=1.0e-14,
        )


# ---------------------------------------------------------------------------
# (d) Interpolation helpers — cells_to_edges, cells_to_verts,
#     compute_wgtfacq1_e, dsl_to_fortran_order
# ---------------------------------------------------------------------------


class TestCellsToEdges:
    """Unit tests for cells_to_edges with hand-crafted connectivity."""

    def test_weighted_average_basic(self) -> None:
        """Result is c_lin_e-weighted average of the two neighboring cell values.

        Expected values are computed by hand to catch wrong einsum subscripts
        or transposed connectivity.
        """
        # 4 cells, 3 edges, 2 levels
        cell_field = np.array(
            [
                [1.0, 2.0],  # cell 0
                [3.0, 4.0],  # cell 1
                [5.0, 6.0],  # cell 2
                [7.0, 8.0],  # cell 3
            ]
        )  # shape (4, 2)

        e2c = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)  # (3, 2)
        c_lin_e = np.array([[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]])  # (3, 2)

        result = _cells_to_edges(cell_field, c_lin_e, e2c)

        # Hand-computed:
        # Edge 0: 0.4*[1,2] + 0.6*[3,4] = [0.4+1.8, 0.8+2.4] = [2.2, 3.2]
        # Edge 1: 0.5*[3,4] + 0.5*[5,6] = [1.5+2.5, 2.0+3.0] = [4.0, 5.0]
        # Edge 2: 0.3*[5,6] + 0.7*[7,8] = [1.5+4.9, 1.8+5.6] = [6.4, 7.4]
        expected = np.array([[2.2, 3.2], [4.0, 5.0], [6.4, 7.4]])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_asymmetric_weights_are_not_equal(self) -> None:
        """Swapping the two neighbors must change the result (catches transposed e2c)."""
        cell_field = np.array([[1.0], [5.0]])  # 2 cells, 1 level
        e2c_forward = np.array([[0, 1]], dtype=int)
        e2c_backward = np.array([[1, 0]], dtype=int)
        c_lin_e = np.array([[0.3, 0.7]])  # asymmetric

        fwd = _cells_to_edges(cell_field, c_lin_e, e2c_forward)
        bwd = _cells_to_edges(cell_field, c_lin_e, e2c_backward)

        # expected 0.3*1 + 0.7*5 = 3.8 (fwd) and 0.3*5 + 0.7*1 = 2.2 (bwd)
        np.testing.assert_allclose(fwd, [[3.8]], rtol=1.0e-14)
        np.testing.assert_allclose(bwd, [[2.2]], rtol=1.0e-14)

    def test_boundary_skip_value_zero_weight(self) -> None:
        """Edge with one boundary neighbor (c_lin_e weight = 0) gives single-cell value."""
        cell_field = np.array([[10.0], [20.0]])  # 2 cells, 1 level
        e2c = np.array([[0, -1]], dtype=int)  # boundary edge; skip neighbor at index -1
        c_lin_e = np.array([[1.0, 0.0]])  # weight 0 on skip neighbor

        result = _cells_to_edges(cell_field, c_lin_e, e2c)
        # 1.0*cell[0] + 0.0*cell[-1] = 10.0  (cell[-1] = cell[1] = 20.0, but weight = 0)
        np.testing.assert_allclose(result, [[10.0]], rtol=1.0e-14)

    def test_shape(self) -> None:
        """Output shape is (nedges, nlev)."""
        ncells, nedges, nlev = 5, 4, 3
        cell_field = np.ones((ncells, nlev))
        e2c = np.zeros((nedges, 2), dtype=int)
        c_lin_e = np.ones((nedges, 2)) * 0.5
        result = _cells_to_edges(cell_field, c_lin_e, e2c)
        assert result.shape == (nedges, nlev)


class TestCellsToVerts:
    """Unit tests for cells_to_verts with hand-crafted connectivity."""

    def test_weighted_average_basic(self) -> None:
        """Result is area-weighted average of neighboring cell values.

        Expected values are hand-computed to catch wrong einsum subscripts
        or transposed connectivity.
        """
        # 4 cells, 3 vertices, 2 levels
        cell_field = np.array(
            [
                [1.0, 2.0],  # cell 0
                [3.0, 4.0],  # cell 1
                [5.0, 6.0],  # cell 2
                [7.0, 8.0],  # cell 3
            ]
        )  # shape (4, 2)

        v2c = np.array([[0, 1, 2], [1, 2, 3], [0, 2, 3]], dtype=int)  # (3 verts, 3 neighbors)
        cells_aw_verts = np.array(
            [
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],  # vertex 0 — equal weights
                [0.50, 0.25, 0.25],  # vertex 1 — asymmetric
                [0.20, 0.30, 0.50],  # vertex 2 — asymmetric
            ]
        )  # (3, 3)

        result = _cells_to_verts(cell_field, cells_aw_verts, v2c)

        # Hand-computed:
        # Vertex 0: 1/3*[1,2] + 1/3*[3,4] + 1/3*[5,6] = [3.0, 4.0]
        # Vertex 1: 0.5*[3,4] + 0.25*[5,6] + 0.25*[7,8] = [1.5+1.25+1.75, 2+1.5+2] = [4.5, 5.5]
        # Vertex 2: 0.2*[1,2] + 0.3*[5,6] + 0.5*[7,8] = [0.2+1.5+3.5, 0.4+1.8+4.0] = [5.2, 6.2]
        expected = np.array([[3.0, 4.0], [4.5, 5.5], [5.2, 6.2]])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_asymmetric_weights_are_not_equal(self) -> None:
        """Swapping connectivity order changes the result (catches transposed v2c)."""
        cell_field = np.array([[2.0], [8.0]])  # 2 cells, 1 level
        v2c_forward = np.array([[0, 1]], dtype=int)
        v2c_backward = np.array([[1, 0]], dtype=int)
        cells_aw_verts = np.array([[0.25, 0.75]])  # asymmetric

        fwd = _cells_to_verts(cell_field, cells_aw_verts, v2c_forward)
        bwd = _cells_to_verts(cell_field, cells_aw_verts, v2c_backward)

        # expected 0.25*2 + 0.75*8 = 6.5 (fwd) and 0.25*8 + 0.75*2 = 3.5 (bwd)
        np.testing.assert_allclose(fwd, [[6.5]], rtol=1.0e-14)
        np.testing.assert_allclose(bwd, [[3.5]], rtol=1.0e-14)

    def test_shape(self) -> None:
        """Output shape is (nverts, nlev)."""
        ncells, nverts, v2c_size, nlev = 6, 4, 3, 5
        cell_field = np.ones((ncells, nlev))
        v2c = np.zeros((nverts, v2c_size), dtype=int)
        cells_aw_verts = np.ones((nverts, v2c_size)) / v2c_size
        result = _cells_to_verts(cell_field, cells_aw_verts, v2c)
        assert result.shape == (nverts, nlev)


class TestWgtfacq1E:
    """Unit tests for compute_wgtfacq1_e with hand-crafted connectivity."""

    def test_basic_interpolation(self) -> None:
        """Interpolate 3-column cell coefficients to edges.

        Expected values are hand-computed to catch wrong einsum subscripts
        or transposed connectivity.
        """
        # 3 cells, 2 edges, each cell has 3 quadratic coefficients
        wgtfacq1_c = np.array(
            [
                [0.6, 0.3, 0.1],  # cell 0
                [0.5, 0.3, 0.2],  # cell 1
                [0.4, 0.4, 0.2],  # cell 2
            ]
        )  # (3, 3)

        e2c = np.array([[0, 1], [1, 2]], dtype=int)  # (2, 2)
        c_lin_e = np.array([[0.4, 0.6], [0.7, 0.3]])  # (2, 2)

        result = _wgtfacq1_e(wgtfacq1_c, c_lin_e, e2c)

        # Edge 0: 0.4*[0.6,0.3,0.1] + 0.6*[0.5,0.3,0.2]
        #       = [0.24+0.30, 0.12+0.18, 0.04+0.12] = [0.54, 0.30, 0.16]
        # Edge 1: 0.7*[0.5,0.3,0.2] + 0.3*[0.4,0.4,0.2]
        #       = [0.35+0.12, 0.21+0.12, 0.14+0.06] = [0.47, 0.33, 0.20]
        expected = np.array([[0.54, 0.30, 0.16], [0.47, 0.33, 0.20]])
        np.testing.assert_allclose(result, expected, rtol=1.0e-14)

    def test_asymmetric_weights_are_not_equal(self) -> None:
        """Swapping cell order changes the result (catches transposed e2c in einsum)."""
        wgtfacq1_c = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (2, 3)
        e2c_forward = np.array([[0, 1]], dtype=int)
        e2c_backward = np.array([[1, 0]], dtype=int)
        c_lin_e = np.array([[0.3, 0.7]])

        fwd = _wgtfacq1_e(wgtfacq1_c, c_lin_e, e2c_forward)
        bwd = _wgtfacq1_e(wgtfacq1_c, c_lin_e, e2c_backward)

        # expected 0.3*[1,0,0] + 0.7*[0,0,1] = [0.3, 0.0, 0.7] (fwd)
        # and 0.3*[0,0,1] + 0.7*[1,0,0] = [0.7, 0.0, 0.3] (bwd)
        np.testing.assert_allclose(fwd, [[0.3, 0.0, 0.7]], rtol=1.0e-14)
        np.testing.assert_allclose(bwd, [[0.7, 0.0, 0.3]], rtol=1.0e-14)

    def test_shape(self) -> None:
        """Output shape is (nedges, 3)."""
        ncells, nedges = 5, 4
        wgtfacq1_c = np.ones((ncells, 3))
        e2c = np.zeros((nedges, 2), dtype=int)
        c_lin_e = np.ones((nedges, 2)) * 0.5
        result = _wgtfacq1_e(wgtfacq1_c, c_lin_e, e2c)
        assert result.shape == (nedges, 3)
