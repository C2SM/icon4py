# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""States of the tmx turbulent mixing granule."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import geometry_attributes
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes


if TYPE_CHECKING:
    from icon4py.model.common.states import factory as states_factory


@dataclasses.dataclass(frozen=True)
class TmxMetricState:
    """Represents the metric (vertical grid) fields needed by tmx."""

    ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Layer thickness at cell centers on full levels [m]."""
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat]
    """Inverse layer thickness at cell centers on full levels [1/m]."""
    ddqz_z_half: fa.CellKHalfField[ta.wpfloat]
    """Vertical distance between full levels, at cell centers on half levels [m]."""
    inv_ddqz_z_half: fa.CellKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at cell centers on half levels [1/m]."""
    inv_ddqz_z_full_e: fa.EdgeKField[ta.wpfloat]
    """Inverse layer thickness at edge midpoints on full levels [1/m]."""
    inv_ddqz_z_half_e: fa.EdgeKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at edge midpoints on half levels [1/m]."""
    inv_ddqz_z_half_v: fa.VertexKHalfField[ta.wpfloat]
    """Inverse vertical distance between full levels, at vertices on half levels [1/m]."""
    wgtfac_c: fa.CellKHalfField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at cell centers (half levels)."""
    wgtfac_e: fa.EdgeKHalfField[ta.wpfloat]
    """Weighting factor for interpolation from full to half levels at edge midpoints (half levels)."""
    wgtfacq_c: fa.CellKField[ta.wpfloat]
    """Extrapolation coefficients to the bottom surface half level at cell centers.

    Three K rows aligned to the levels they multiply: the row at K index j is the
    weight of full level j, so the field is defined on KDim in [nlev - 3, nlev).
    This is what the metrics factory emits and what the dycore stencils consume."""
    wgtfacq1_c: fa.CellKField[ta.wpfloat]
    """Extrapolation coefficients to the top half level at cell centers.

    Three K rows aligned to the levels they multiply, i.e. KDim in [0, 3)."""
    wgtfacq_e: fa.EdgeKField[ta.wpfloat]
    """Extrapolation coefficients to the bottom surface half level at edges.

    Aligned to the levels they multiply, see :attr:`wgtfacq_c`."""
    wgtfacq1_e: fa.EdgeKField[ta.wpfloat]
    """Extrapolation coefficients to the top half level at edges.

    Aligned to the levels they multiply, see :attr:`wgtfacq1_c`."""
    geopot_agl_ifc: fa.CellKHalfField[ta.wpfloat]
    """Geopotential above ground level at cell centers on half levels [m^2/s^2]."""
    height_above_ground: fa.CellKField[ta.wpfloat]
    """Geometric height of the full levels above the surface [m] (``ghf``)."""
    z_mc: fa.CellKField[ta.wpfloat]
    """Geometric height at cell centers on full levels [m]."""
    z_ifc: fa.CellKHalfField[ta.wpfloat]
    """Geometric height at cell centers on half levels [m]."""
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    """Distance between the edge midpoint and the circumcenters of the two
    adjacent cells [m] (``t_grid_edges%edge_cell_length`` in mo_model_domain.f90).

    A grid-geometry field carried here because it is not part of the common
    ``grid_states.EdgeParams`` (used by the horizontal w diffusion)."""

    @classmethod
    def from_sources(
        cls,
        *,
        metrics: states_factory.FieldSource,
        geometry: states_factory.FieldSource,
    ) -> TmxMetricState:
        """Build the state from the metrics and geometry field factories."""
        return cls(
            ddqz_z_full=metrics.get(metrics_attributes.DDQZ_Z_FULL),
            inv_ddqz_z_full=metrics.get(metrics_attributes.INV_DDQZ_Z_FULL),
            ddqz_z_half=metrics.get(metrics_attributes.DDQZ_Z_HALF),
            inv_ddqz_z_half=metrics.get(metrics_attributes.INV_DDQZ_Z_HALF),
            inv_ddqz_z_full_e=metrics.get(metrics_attributes.INV_DDQZ_Z_FULL_E),
            inv_ddqz_z_half_e=metrics.get(metrics_attributes.INV_DDQZ_Z_HALF_E),
            inv_ddqz_z_half_v=metrics.get(metrics_attributes.INV_DDQZ_Z_HALF_V),
            wgtfac_c=metrics.get(metrics_attributes.WGTFAC_C),
            wgtfac_e=metrics.get(metrics_attributes.WGTFAC_E),
            wgtfacq_c=metrics.get(metrics_attributes.WGTFACQ_C),
            wgtfacq1_c=metrics.get(metrics_attributes.WGTFACQ1_C),
            wgtfacq_e=metrics.get(metrics_attributes.WGTFACQ_E),
            wgtfacq1_e=metrics.get(metrics_attributes.WGTFACQ1_E),
            geopot_agl_ifc=metrics.get(metrics_attributes.GEOPOT_AGL_IFC),
            height_above_ground=metrics.get(metrics_attributes.HEIGHT_ABOVE_GROUND),
            z_mc=metrics.get(metrics_attributes.Z_MC),
            z_ifc=metrics.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL),
            edge_cell_length=geometry.get(geometry_attributes.EDGE_CELL_DISTANCE),
        )


@dataclasses.dataclass(frozen=True)
class TmxInterpolationState:
    """Represents the ICON interpolation coefficients needed by tmx."""

    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat]
    """Coefficients for linear interpolation from cell centers to edge midpoints."""
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat]
    """Coefficients for bilinear interpolation from edge midpoints to cell centers."""
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat]
    """Geometric factors for the cell-centered divergence of an edge-normal vector field."""
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat]
    """Coefficients for area-weighted interpolation from cell centers to vertices."""
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat]
    """RBF coefficients for the zonal wind component at vertices (rbf_vec_coeff_v_1)."""
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat]
    """RBF coefficients for the meridional wind component at vertices (rbf_vec_coeff_v_2)."""
    rbf_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat]
    """RBF coefficients for the tangential wind component at edges (rbf_vec_coeff_e)."""
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]
    """RBF coefficients for the zonal wind component at cell centers (rbf_vec_coeff_c_1)."""
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat]
    """RBF coefficients for the meridional wind component at cell centers (rbf_vec_coeff_c_2)."""

    @classmethod
    def from_sources(cls, *, interpolation: states_factory.FieldSource) -> TmxInterpolationState:
        """Build the state from the interpolation field factory."""
        return cls(
            c_lin_e=interpolation.get(interpolation_attributes.C_LIN_E),
            e_bln_c_s=interpolation.get(interpolation_attributes.E_BLN_C_S),
            geofac_div=interpolation.get(interpolation_attributes.GEOFAC_DIV),
            cells_aw_verts=interpolation.get(interpolation_attributes.CELL_AW_VERTS),
            rbf_coeff_v1=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V1),
            rbf_coeff_v2=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_V2),
            rbf_coeff_e=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_E),
            rbf_coeff_c1=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C1),
            rbf_coeff_c2=interpolation.get(interpolation_attributes.RBF_VEC_COEFF_C2),
        )
