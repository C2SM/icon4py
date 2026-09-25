# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    _calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    _calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_smag_and_turbulence_diagnostics(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    smag_offset: vpfloat,
    compute_diagnostic_quantities: bool,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    kh_smag_e, kh_smag_ec, z_nabla2_e = _calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag=diff_multfac_smag,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        vn=vn,
        smag_limit=smag_limit,
        smag_offset=smag_offset,
    )
    div_ic, hdef_ic = (
        _calculate_diagnostic_quantities_for_turbulence(
            kh_smag_ec=kh_smag_ec,
            vn=vn,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            wgtfac_c=wgtfac_c,
        )
        if compute_diagnostic_quantities
        else (div_ic, hdef_ic)
    )
    return kh_smag_e, z_nabla2_e, div_ic, hdef_ic


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_smag_and_turbulence_diagnostics(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    smag_offset: vpfloat,
    compute_diagnostic_quantities: bool,
    edge_horizontal_start: gtx.int32,
    edge_horizontal_end: gtx.int32,
    edge_vertical_start: gtx.int32,
    edge_vertical_end: gtx.int32,
    cell_horizontal_start: gtx.int32,
    cell_horizontal_end: gtx.int32,
    cell_vertical_start: gtx.int32,
    cell_vertical_end: gtx.int32,
) -> None:
    _calculate_nabla2_smag_and_turbulence_diagnostics(
        diff_multfac_smag=diff_multfac_smag,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        vn=vn,
        smag_limit=smag_limit,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        wgtfac_c=wgtfac_c,
        div_ic=div_ic,
        hdef_ic=hdef_ic,
        smag_offset=smag_offset,
        compute_diagnostic_quantities=compute_diagnostic_quantities,
        out=(kh_smag_e, z_nabla2_e, div_ic, hdef_ic),
        domain=(
            {
                dims.EdgeDim: (edge_horizontal_start, edge_horizontal_end),
                dims.KDim: (edge_vertical_start, edge_vertical_end),
            },
            {
                dims.EdgeDim: (edge_horizontal_start, edge_horizontal_end),
                dims.KDim: (edge_vertical_start, edge_vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KHalfDim: (cell_vertical_start, cell_vertical_end),
            },
            {
                dims.CellDim: (cell_horizontal_start, cell_horizontal_end),
                dims.KHalfDim: (cell_vertical_start, cell_vertical_end),
            },
        ),
    )
