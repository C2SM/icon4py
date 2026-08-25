# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Build TmxMetricState and TmxInterpolationState from field-factory sources.

The derived metric fields (inv_ddqz_z_half*, geopot_agl_ifc, wgtfacq1_*) are
computed and cached by the metrics factory (numpy formulas in
``common/metrics/compute_weight_factors.py``); this module only fetches and
assembles.

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

Top-boundary coefficients (``wgtfacq1_c``, ``wgtfacq1_e``) are stored by the
metrics factory in Fortran coefficient order already (k=0 → w1 for the topmost
full level = level 0); no conversion is needed for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import factory as states_factory


def dsl_to_fortran_order(arr: np.ndarray) -> np.ndarray:
    """Reverse the 3-column DSL wgtfacq array into Fortran coefficient order.

    DSL: col0=w3, col1=w2, col2=w1  →  Fortran: col0=w1, col1=w2, col2=w3.
    """
    assert arr.shape[1] == 3, f"expected 3 K columns, got {arr.shape[1]}"
    return arr[:, ::-1]


def _half_level_to_granule_kdim(field: gtx.Field, allocator: gtx_typing.Backend) -> gtx.Field:
    """Re-type a factory half-level field from (H, KHalfDim) to the granule's (H, KDim).

    The field factories type half-level fields on KHalfDim; the tmx granule
    (like the serialized ICON reference data) types every vertical axis as KDim
    with nlev+1 entries. Same buffer contents, different vertical dimension tag —
    without this re-wrap the granule's typed gt4py programs reject the field.
    """
    horizontal_dim = field.domain.dims[0]
    return gtx.as_field((horizontal_dim, dims.KDim), field.asnumpy(), allocator=allocator)


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

    Everything is fetched from the factory sources (derived metric fields are
    registered in the metrics factory); the only local work is the wgtfacq
    DSL→Fortran column reorder and re-typing half-level fields from the
    factories' KHalfDim to the granule's KDim convention.

    Args:
        grid:                  The icon grid (used for E2C / V2C connectivities).
        geometry_source:       GridGeometry factory.
        interpolation_source:  InterpolationFieldsFactory.
        metrics_source:        MetricsFieldsFactory.
        backend:               GT4Py backend (or None for CPU numpy).

    Returns:
        (TmxMetricState, TmxInterpolationState)
    """
    allocator = model_backends.get_allocator(backend)

    # ------------------------------------------------------------------
    # 1. Fetch fields directly from factory sources
    # ------------------------------------------------------------------

    # Metrics: full-level fields fetched as-is; half-level fields re-typed from
    # the factories' KHalfDim to the granule's KDim convention
    ddqz_z_full = metrics_source.get(metrics_attributes.DDQZ_Z_FULL)
    inv_ddqz_z_full = metrics_source.get(metrics_attributes.INV_DDQZ_Z_FULL)
    ddqz_z_half = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.DDQZ_Z_HALF), allocator
    )
    wgtfac_c = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.WGTFAC_C), allocator
    )
    wgtfac_e = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.WGTFAC_E), allocator
    )
    z_mc = metrics_source.get(metrics_attributes.Z_MC)
    z_ifc = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL), allocator
    )

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
    # 2. Derived metric fields — computed and cached by the metrics factory
    #    (registered in metrics_factory.py; numpy formulas in
    #    common/metrics/compute_weight_factors.py)
    # ------------------------------------------------------------------
    inv_ddqz_z_half = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF), allocator
    )
    inv_ddqz_z_full_e = metrics_source.get(metrics_attributes.INV_DDQZ_Z_FULL_E)
    inv_ddqz_z_half_e = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF_E), allocator
    )
    inv_ddqz_z_half_v = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.INV_DDQZ_Z_HALF_V), allocator
    )
    geopot_agl_ifc = _half_level_to_granule_kdim(
        metrics_source.get(metrics_attributes.GEOPOT_AGL_IFC), allocator
    )
    wgtfacq1_c = metrics_source.get(metrics_attributes.WGTFACQ1_C)
    wgtfacq1_e = metrics_source.get(metrics_attributes.WGTFACQ1_E)

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
