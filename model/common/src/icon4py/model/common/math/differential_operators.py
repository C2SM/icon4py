# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2E2CO, E2C, E2C2V, V2E
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def normal_gradient(
    scalar_field: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    return inv_dual_edge_length * (scalar_field(E2C[1]) - scalar_field(E2C[0]))


@gtx.field_operator
def laplacian_on_cell_khalf(
    scalar_field: fa.CellKHalfField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
) -> fa.CellKHalfField[wpfloat]:
    return neighbor_sum(scalar_field(C2E2CO) * geofac_n2s, axis=dims.C2E2CODim)


@gtx.field_operator
def laplacian_on_edge(  # noqa: PLR0917 [too-many-positional-arguments]
    vec_normal: fa.EdgeKField[wpfloat],
    vec_eastward: fa.VertexKField[wpfloat],
    vec_northward: fa.VertexKField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    normal_component_at_vertices = (
        vec_eastward(E2C2V) * primal_normal_vert_x + vec_northward(E2C2V) * primal_normal_vert_y
    )

    laplacian = (
        normal_component_at_vertices[dims.E2C2VDim(0)]
        + normal_component_at_vertices[dims.E2C2VDim(1)]
        - wpfloat("2.0") * vec_normal
    ) * (inv_primal_edge_length * inv_primal_edge_length) + (
        normal_component_at_vertices[dims.E2C2VDim(2)]
        + normal_component_at_vertices[dims.E2C2VDim(3)]
        - wpfloat("2.0") * vec_normal
    ) * (inv_vert_vert_length * inv_vert_vert_length)
    # The factor of 4 comes from the lengths in the denominator being twice those needed
    # for the diffusion stencil (https://doi.org/10.1002%2Fqj.2378).
    return wpfloat("4.0") * laplacian


@gtx.field_operator
def divergence(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKField[wpfloat]:
    return neighbor_sum(vec_e(C2E) * geofac_div, axis=dims.C2EDim)


@gtx.field_operator
def compute_curl(
    vec_e: fa.EdgeKField[wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
) -> fa.VertexKField[wpfloat]:
    return neighbor_sum(vec_e(V2E) * geofac_rot, axis=dims.V2EDim)


@gtx.field_operator
def cell_horizontal_gradients_by_green_gauss_method(
    scalar_field: fa.CellKField[vpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    scalar_field_wp = astype((scalar_field), wpfloat)

    p_grad_1_u_wp = neighbor_sum(geofac_grg_x * scalar_field_wp(C2E2CO), axis=dims.C2E2CODim)
    p_grad_1_v_wp = neighbor_sum(geofac_grg_y * scalar_field_wp(C2E2CO), axis=dims.C2E2CODim)
    return astype((p_grad_1_u_wp, p_grad_1_v_wp), vpfloat)


@gtx.field_operator
def cell_horizontal_gradients_by_green_gauss_method_on_half_levels(
    scalar_field: fa.CellKHalfField[wpfloat],
    weights_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    weights_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
) -> tuple[
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    gradx_wp = neighbor_sum(weights_x * scalar_field(C2E2CO), axis=dims.C2E2CODim)
    grady_wp = neighbor_sum(weights_y * scalar_field(C2E2CO), axis=dims.C2E2CODim)
    return astype((gradx_wp, grady_wp), vpfloat)
