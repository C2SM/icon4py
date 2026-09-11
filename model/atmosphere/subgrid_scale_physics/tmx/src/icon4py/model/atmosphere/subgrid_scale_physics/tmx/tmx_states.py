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

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


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
    """Quadratic extrapolation coefficients to the surface level at cell centers."""
    wgtfacq1_c: fa.CellKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the model top level at cell centers."""
    wgtfacq_e: fa.EdgeKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the surface level at edges."""
    wgtfacq1_e: fa.EdgeKField[ta.wpfloat]
    """Quadratic extrapolation coefficients to the model top level at edges."""
    geopot_agl_ifc: fa.CellKHalfField[ta.wpfloat]
    """Geopotential above ground level at cell centers on half levels [m^2/s^2]."""
    height_above_ground: fa.CellKField[ta.wpfloat]
    """Geometric height of the full levels above the surface [m] (``ghf``)."""
    z_mc: fa.CellKField[ta.wpfloat]
    """Geometric height at cell centers on full levels [m]."""
    z_ifc: fa.CellKHalfField[ta.wpfloat]
    """Geometric height at cell centers on half levels [m]."""


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
