# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype, broadcast

from icon4py.model.atmosphere.dycore.stencils.compute_advection_in_vertical_momentum_equation import (
    _compute_advective_vertical_wind_tendency,
    _compute_contravariant_corrected_w_and_cfl,
    _interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _interpolate_contravariant_correction_to_cells_on_half_levels(
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    nflatlev: gtx.int32,
) -> fa.CellKField[ta.vpfloat]:
    # TODO (Chia Rui): contravariant_correction_at_edges_on_model_levels is already zero when k < nflatlev, we may not need this concat_where
    contravariant_correction_at_cells_model_levels = concat_where(
        dims.KDim >= nflatlev,
        _interpolate_to_cell_center(contravariant_correction_at_edges_on_model_levels, e_bln_c_s),
        broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )
    contravariant_correction_at_cells_model_levels = astype(
        contravariant_correction_at_cells_model_levels, vpfloat
    )

    contravariant_correction_at_cells_on_half_levels = concat_where(
        dims.KDim >= nflatlev + 1,
        _interpolate_cell_field_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )

    return contravariant_correction_at_cells_on_half_levels


@gtx.field_operator
def _compute_contravariant_correction_and_advection_in_vertical_momentum_equation(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    contravariant_correction_at_cells_on_half_levels = (
        _interpolate_contravariant_correction_to_cells_on_half_levels(
            contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s,
            wgtfac_c,
            nflatlev,
        )
    )

    (
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
    ) = _compute_contravariant_corrected_w_and_cfl(
        w,
        contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half,
        cfl_w_limit,
        dtime,
        nflatlev,
        nlev,
        end_index_of_damping_layer,
    )

    if not skip_compute_predictor_vertical_advection:
        vertical_wind_advective_tendency = _compute_advective_vertical_wind_tendency(
            vertical_wind_advective_tendency,
            w,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping,
            coeff1_dwdz,
            coeff2_dwdz,
            c_intp,
            inv_dual_edge_length,
            inv_primal_edge_length,
            tangent_orientation,
            e_bln_c_s,
            ddqz_z_half,
            area,
            geofac_n2s,
            owner_mask,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
        )

    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels, nlev
        )
    )

    return (
        contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_contravariant_correction_and_advection_in_vertical_momentum_equation(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vertical_cfl: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    This computes the vertical momentum advection in the vertical momentum equation

    Args:
        - contravariant_correction_at_cells_on_half_levels: contravariant correction at cells on model levels
        - vertical_wind_advective_tendency: vertical advection tendency of the vertical wind
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at cells on model levels
        - vertical_cfl: vertical cfl number at cells on half levels
        - w: vertical wind at cells on half levels
        - tangential_wind: tangential wind at edges on model levels
        - vn_on_half_levels: normal wind at edges on half levels
        - contravariant_correction_at_edges_on_model_levels: contravariant correction at edges on model levels
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - c_intp: interpolation field for cell-to-vertex interpolation
        - inv_dual_edge_length: inverse dual edge length
        - inv_primal_edge_length: inverse primal edge length
        - tangent_orientation: orientation of the edge with respect to the grid
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - wgtfac_c: metric coefficient for interpolating a cell variable from full to half levels
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - owner_mask: ownership mask for each cell
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - skip_compute_predictor_vertical_advection: logical flag to skip the vertical advection
        - nflatlev: number of flat levels
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal dimension
        - horizontal_end: end index in the horizontal dimension
        - vertical_start: start index in the vertical dimension at model top
        - vertical_end: end index in the vertical dimension at model bottom (or number of full/model vertical levels)

    Returns:
        - contravariant_correction_at_cells_on_half_levels
        - vertical_wind_advective_tendency
        - contravariant_corrected_w_at_cells_on_model_levels
        - vertical_cfl
    """

    _compute_contravariant_correction_and_advection_in_vertical_momentum_equation(
        vertical_wind_advective_tendency,
        w,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        contravariant_correction_at_edges_on_model_levels,
        coeff1_dwdz,
        coeff2_dwdz,
        c_intp,
        inv_dual_edge_length,
        inv_primal_edge_length,
        tangent_orientation,
        e_bln_c_s,
        wgtfac_c,
        ddqz_z_half,
        area,
        geofac_n2s,
        owner_mask,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        skip_compute_predictor_vertical_advection,
        nflatlev,
        vertical_end,
        end_index_of_damping_layer,
        out=(
            contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
