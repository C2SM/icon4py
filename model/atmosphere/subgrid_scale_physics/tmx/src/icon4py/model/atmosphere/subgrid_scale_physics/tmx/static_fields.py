# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Build TmxMetricState and TmxInterpolationState from field-factory sources.

Each scalar derivation is exposed as a pure numpy function (array in, array
out) so it can be unit-tested without a real grid or field factory.

Convention for wgtfacq coefficient order
-----------------------------------------
The metrics factory stores bottom-extrapolation coefficients in *DSL order*
(bottom-up, mirroring ``compute_wgtfacq_c_dsl`` which returns ``[:, -3:]``):
  col 0 → w3 (multiplies the 3rd full level from the bottom = nlev-3)
  col 1 → w2 (multiplies nlev-2)
  col 2 → w1 (multiplies nlev-1, the surface-adjacent level)

``TmxMetricState`` documents *Fortran coefficient order*:
  col 0 (k=0) → w1 (multiplies full level nlev-1-0 = nlev-1)
  col 1 (k=1) → w2
  col 2 (k=2) → w3

The conversion between the two is a column reversal (``[:, ::-1]``),
mirroring ``flip_back`` in the integration-test utilities.

Top-boundary coefficients (``wgtfacq1_c``, ``wgtfacq1_e``) are always built
in Fortran coefficient order (k=0 → w1 for the topmost full level = level 0).
They are NOT available from the metrics factory and must be derived here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory as states_factory


# ---------------------------------------------------------------------------
# Pure numpy derivations (individually testable)
# ---------------------------------------------------------------------------


def compute_inv_reciprocal(arr: np.ndarray) -> np.ndarray:
    """Return element-wise 1/arr (generic reciprocal helper)."""
    return 1.0 / arr


def compute_geopot_agl_ifc(z_ifc: np.ndarray) -> np.ndarray:
    """Geopotential above ground level at cell-center interface levels.

    Args:
        z_ifc: geometric height at cell interface levels, shape (ncells, nlev+1).
               The last column (index nlev) is the ground (z=surface height).

    Returns:
        grav * (z_ifc - z_ifc[:, -1:]), shape (ncells, nlev+1) [m²/s²].
    """
    return constants.GRAV * (z_ifc - z_ifc[:, -1:])


def compute_wgtfacq1_c(z_ifc: np.ndarray) -> np.ndarray:
    """Quadratic extrapolation coefficients to the top interface level (cells).

    Implements mo_vertical_grid.f90:953-967 (0-based numpy indexing).
    The formula uses the three topmost interface heights z_ifc[:, 0..3].

    Returns:
        Array of shape (ncells, 3) in *Fortran coefficient order*:
        col 0 = w1 (multiplies full level 0, the topmost),
        col 1 = w2 (multiplies full level 1),
        col 2 = w3 (multiplies full level 2).
    """
    z1 = 0.5 * (z_ifc[:, 1] - z_ifc[:, 0])
    z2 = 0.5 * (z_ifc[:, 1] + z_ifc[:, 2]) - z_ifc[:, 0]
    z3 = 0.5 * (z_ifc[:, 2] + z_ifc[:, 3]) - z_ifc[:, 0]
    w3 = z1 * z2 / ((z2 - z3) * (z1 - z3))
    w2 = (z1 - w3 * (z1 - z3)) / (z1 - z2)
    w1 = 1.0 - (w2 + w3)
    return np.stack([w1, w2, w3], axis=1)  # (ncells, 3), Fortran order


def compute_wgtfacq1_e(
    wgtfacq1_c: np.ndarray,
    c_lin_e: np.ndarray,
    e2c: np.ndarray,
) -> np.ndarray:
    """Interpolate top-boundary quadratic coefficients from cells to edges.

    Mirrors ``compute_wgtfacq_e_dsl`` (mo_vertical_grid.f90:989-1014)
    for the top-boundary block (z_aux_c[:, 4:6, :] in Fortran 1-indexed).

    Args:
        wgtfacq1_c: (ncells, 3) in Fortran coefficient order (see module doc).
        c_lin_e:    (nedges, 2) linear interpolation weights cell→edge.
        e2c:        (nedges, 2) E2C connectivity table (skip values = -1).

    Returns:
        (nedges, 3) in Fortran coefficient order.

    Note: skip-value neighbors (e2c == -1) have zero c_lin_e weight in the
    standard ICON interpolation setup, so the corresponding row of
    ``wgtfacq1_c[-1]`` is multiplied by 0 and does not affect the result.
    """
    # wgtfacq1_c[e2c]: (nedges, 2, 3)
    # c_lin_e[:, :, newaxis] * wgtfacq1_c[e2c] → sum over local dim → (nedges, 3)
    return np.einsum("ej,ejk->ek", c_lin_e, wgtfacq1_c[e2c])


def cells_to_edges(
    cell_field: np.ndarray,
    c_lin_e: np.ndarray,
    e2c: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a (ncells, nlev) cell field to (nedges, nlev).

    Args:
        cell_field: (ncells, nlev_or_nlev+1)
        c_lin_e:    (nedges, 2)
        e2c:        (nedges, 2) connectivity (skip values = -1, weight = 0)

    Returns:
        (nedges, nlev_or_nlev+1)
    """
    return np.einsum("ej,ejk->ek", c_lin_e, cell_field[e2c])


def cells_to_verts(
    cell_field: np.ndarray,
    cells_aw_verts: np.ndarray,
    v2c: np.ndarray,
) -> np.ndarray:
    """Area-weighted interpolation of a (ncells, nlev) cell field to (nverts, nlev).

    Args:
        cell_field:     (ncells, nlev_or_nlev+1)
        cells_aw_verts: (nverts, v2c_size)  area-weighting coefficients
        v2c:            (nverts, v2c_size) connectivity (skip values = -1, weight = 0)

    Returns:
        (nverts, nlev_or_nlev+1)
    """
    return np.einsum("vj,vjk->vk", cells_aw_verts, cell_field[v2c])


def dsl_to_fortran_order(arr: np.ndarray) -> np.ndarray:
    """Reverse the 3-column DSL wgtfacq array into Fortran coefficient order.

    DSL: col0=w3, col1=w2, col2=w1  →  Fortran: col0=w1, col1=w2, col2=w3.
    """
    assert arr.shape[1] == 3, f"expected 3 K columns, got {arr.shape[1]}"
    return arr[:, ::-1]


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------


def build_tmx_static_states(
    *,
    grid: base_grid.Grid,
    geometry_source: states_factory.FieldSource,
    interpolation_source: states_factory.FieldSource,
    metrics_source: states_factory.FieldSource,
    backend: gtx_typing.Backend | None,
) -> tuple[tmx_states.TmxMetricState, tmx_states.TmxInterpolationState]:
    """Construct TmxMetricState and TmxInterpolationState from field factories.

    All numpy derivations (inv_ddqz_z_half*, geopot_agl_ifc, wgtfacq1_*) are
    performed once at init time; the results are wrapped as GT4Py fields with
    ``gtx.as_field(..., allocator=backend)``.

    Args:
        grid:                  The icon grid (used for E2C / V2C connectivities).
        geometry_source:       GridGeometry factory.
        interpolation_source:  InterpolationFieldsFactory.
        metrics_source:        MetricsFieldsFactory.
        backend:               GT4Py backend (or None for CPU numpy).

    Returns:
        (TmxMetricState, TmxInterpolationState)
    """
    allocator = backend  # gtx.as_field accepts backend as allocator

    # ------------------------------------------------------------------
    # 1. Fetch fields directly from factory sources
    # ------------------------------------------------------------------

    # Metrics: fetched as-is
    ddqz_z_full = metrics_source.get(metrics_attributes.DDQZ_Z_FULL)
    inv_ddqz_z_full = metrics_source.get(metrics_attributes.INV_DDQZ_Z_FULL)
    ddqz_z_half = metrics_source.get(metrics_attributes.DDQZ_Z_HALF)
    ddqz_z_full_e = metrics_source.get(metrics_attributes.DDQZ_Z_FULL_E)
    wgtfac_c = metrics_source.get(metrics_attributes.WGTFAC_C)
    wgtfac_e = metrics_source.get(metrics_attributes.WGTFAC_E)
    z_mc = metrics_source.get(metrics_attributes.Z_MC)
    z_ifc = metrics_source.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL)

    # Factory wgtfacq fields come in DSL order (col0=w3, col1=w2, col2=w1)
    wgtfacq_c_dsl_arr = metrics_source.get(metrics_attributes.WGTFACQ_C).asnumpy()
    wgtfacq_e_dsl_arr = metrics_source.get(metrics_attributes.WGTFACQ_E).asnumpy()

    # Interpolation: fetched as-is
    c_lin_e = interpolation_source.get(interpolation_attributes.C_LIN_E)
    e_bln_c_s = interpolation_source.get(interpolation_attributes.E_BLN_C_S)
    geofac_div = interpolation_source.get(interpolation_attributes.GEOFAC_DIV)
    cells_aw_verts = interpolation_source.get(interpolation_attributes.CELL_AW_VERTS)
    rbf_coeff_v1 = interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_V1)
    rbf_coeff_v2 = interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_V2)
    rbf_coeff_e = interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_E)
    rbf_coeff_c1 = interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C1)
    rbf_coeff_c2 = interpolation_source.get(interpolation_attributes.RBF_VEC_COEFF_C2)

    # Geometry
    edge_cell_length = geometry_source.get(geometry_attributes.EDGE_CELL_DISTANCE)

    # ------------------------------------------------------------------
    # 2. Get connectivity tables (numpy) for interpolation derivations
    # ------------------------------------------------------------------
    e2c_arr = grid.get_connectivity(dims.E2C.value).ndarray   # (nedges, 2)
    v2c_arr = grid.get_connectivity(dims.V2C.value).ndarray   # (nverts, v2c_size)

    c_lin_e_arr = c_lin_e.asnumpy()                    # (nedges, 2)
    cells_aw_verts_arr = cells_aw_verts.asnumpy()       # (nverts, v2c_size)

    # ------------------------------------------------------------------
    # 3. Derived metric fields
    # ------------------------------------------------------------------

    # --- 3a. inv_ddqz_z_half ---
    inv_ddqz_z_half_arr = compute_inv_reciprocal(ddqz_z_half.asnumpy())
    inv_ddqz_z_half = gtx.as_field(
        (dims.CellDim, dims.KDim), inv_ddqz_z_half_arr, allocator=allocator
    )

    # --- 3b. inv_ddqz_z_full_e ---
    inv_ddqz_z_full_e_arr = compute_inv_reciprocal(ddqz_z_full_e.asnumpy())
    inv_ddqz_z_full_e = gtx.as_field(
        (dims.EdgeDim, dims.KDim), inv_ddqz_z_full_e_arr, allocator=allocator
    )

    # --- 3c. inv_ddqz_z_half_e  (cells→edges, E2C, c_lin_e) ---
    # Note: skip-value E2C entries (e2c=-1) contribute 0 because c_lin_e
    # sets coefficient 0 for the missing neighbor at lateral boundaries.
    inv_ddqz_z_half_e_arr = cells_to_edges(inv_ddqz_z_half_arr, c_lin_e_arr, e2c_arr)
    inv_ddqz_z_half_e = gtx.as_field(
        (dims.EdgeDim, dims.KDim), inv_ddqz_z_half_e_arr, allocator=allocator
    )

    # --- 3d. inv_ddqz_z_half_v  (cells→vertices, V2C, cells_aw_verts) ---
    inv_ddqz_z_half_v_arr = cells_to_verts(inv_ddqz_z_half_arr, cells_aw_verts_arr, v2c_arr)
    inv_ddqz_z_half_v = gtx.as_field(
        (dims.VertexDim, dims.KDim), inv_ddqz_z_half_v_arr, allocator=allocator
    )

    # --- 3e. geopot_agl_ifc ---
    z_ifc_arr = z_ifc.asnumpy()
    geopot_agl_ifc_arr = compute_geopot_agl_ifc(z_ifc_arr)
    geopot_agl_ifc = gtx.as_field(
        (dims.CellDim, dims.KDim), geopot_agl_ifc_arr, allocator=allocator
    )

    # --- 3f. wgtfacq_c/wgtfacq_e: DSL → Fortran order ---
    wgtfacq_c = gtx.as_field(
        (dims.CellDim, dims.KDim),
        dsl_to_fortran_order(wgtfacq_c_dsl_arr),
        allocator=allocator,
    )
    wgtfacq_e = gtx.as_field(
        (dims.EdgeDim, dims.KDim),
        dsl_to_fortran_order(wgtfacq_e_dsl_arr),
        allocator=allocator,
    )

    # --- 3g. wgtfacq1_c (top-boundary, derived from z_ifc) ---
    wgtfacq1_c_arr = compute_wgtfacq1_c(z_ifc_arr)  # (ncells, 3) Fortran order
    wgtfacq1_c = gtx.as_field(
        (dims.CellDim, dims.KDim), wgtfacq1_c_arr, allocator=allocator
    )

    # --- 3h. wgtfacq1_e (cell→edge interpolation of wgtfacq1_c) ---
    wgtfacq1_e_arr = compute_wgtfacq1_e(wgtfacq1_c_arr, c_lin_e_arr, e2c_arr)
    wgtfacq1_e = gtx.as_field(
        (dims.EdgeDim, dims.KDim), wgtfacq1_e_arr, allocator=allocator
    )

    # ------------------------------------------------------------------
    # 4. Assemble output states
    # ------------------------------------------------------------------

    metric_state = tmx_states.TmxMetricState(
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        ddqz_z_half=ddqz_z_half,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        inv_ddqz_z_half_e=inv_ddqz_z_half_e,
        inv_ddqz_z_half_v=inv_ddqz_z_half_v,
        wgtfac_c=wgtfac_c,
        wgtfac_e=wgtfac_e,
        wgtfacq_c=wgtfacq_c,
        wgtfacq1_c=wgtfacq1_c,
        wgtfacq_e=wgtfacq_e,
        wgtfacq1_e=wgtfacq1_e,
        geopot_agl_ifc=geopot_agl_ifc,
        z_mc=z_mc,
        z_ifc=z_ifc,
        edge_cell_length=edge_cell_length,
    )

    interp_state = tmx_states.TmxInterpolationState(
        c_lin_e=c_lin_e,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        cells_aw_verts=cells_aw_verts,
        rbf_coeff_v1=rbf_coeff_v1,
        rbf_coeff_v2=rbf_coeff_v2,
        rbf_coeff_e=rbf_coeff_e,
        rbf_coeff_c1=rbf_coeff_c1,
        rbf_coeff_c2=rbf_coeff_c2,
    )

    return metric_state, interp_state
