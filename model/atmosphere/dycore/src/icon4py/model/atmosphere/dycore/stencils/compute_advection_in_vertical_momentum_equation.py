# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.fbuiltins import broadcast, maximum, where

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    _add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_advective_vertical_wind_tendency import (
    _compute_advective_vertical_wind_tendency,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_contravariant_vertical_velocity_to_full_levels import (
    _interpolate_contravariant_vertical_velocity_to_full_levels,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_advective_vertical_wind_tendency_and_apply_diffusion(
    vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
) -> fa.CellKField[vpfloat]:
    k = broadcast(k, (dims.CellDim, dims.KDim))

    vertical_wind_advective_tendency = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (1 <= k),
        _compute_advective_vertical_wind_tendency(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        ),
        vertical_wind_advective_tendency,
    )
    vertical_wind_advective_tendency = where(
        (cell_lower_bound <= cell < cell_upper_bound) & (1 <= k),
        _add_interpolated_horizontal_advection_of_w(
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels,
            vertical_wind_advective_tendency,
        ),
        vertical_wind_advective_tendency,
    )
    vertical_wind_advective_tendency = where(
        (cell_lower_bound <= cell < cell_upper_bound)
        & ((maximum(3, nrdmax - 2) - 1) <= k < nlev - 3),
        _add_extra_diffusion_for_w_con_approaching_cfl(
            levelmask,
            cfl_clipping,
            owner_mask,
            contravariant_corrected_w_at_cells_on_half_levels,
            ddqz_z_half,
            area,
            geofac_n2s,
            w,
            vertical_wind_advective_tendency,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
        ),
        vertical_wind_advective_tendency,
    )

    return vertical_wind_advective_tendency


@gtx.field_operator
def _compute_advection_in_vertical_momentum_equation(
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[wpfloat],
    vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[vpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    end_cell_halo: gtx.int32,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    contravariant_corrected_w_at_cells_on_model_levels = where(
        start_cell_lateral_boundary <= cell < end_cell_halo,
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels
        ),
        contravariant_corrected_w_at_cells_on_model_levels,
    )
    vertical_wind_advective_tendency = (
        _compute_advective_vertical_wind_tendency_and_apply_diffusion(
            vertical_wind_advective_tendency,
            w,
            horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels,
            coeff1_dwdz,
            coeff2_dwdz,
            e_bln_c_s,
            levelmask,
            cfl_clipping,
            owner_mask,
            ddqz_z_half,
            area,
            geofac_n2s,
            cell,
            k,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            cell_lower_bound,
            cell_upper_bound,
            nlev,
            nrdmax,
        )
        if not skip_compute_predictor_vertical_advection
        else vertical_wind_advective_tendency
    )

    return (contravariant_corrected_w_at_cells_on_model_levels, vertical_wind_advective_tendency)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_vertical_momentum_equation(
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[vpfloat],
    vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[vpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    scalfac_exdiff: wpfloat,
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    levelmask: fa.KField[bool],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    cell: fa.CellField[gtx.int32],
    k: fa.KField[gtx.int32],
    cell_lower_bound: gtx.int32,
    cell_upper_bound: gtx.int32,
    nlev: gtx.int32,
    nrdmax: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    end_cell_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_advection_in_vertical_momentum_equation(
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_wind_advective_tendency,
        w,
        contravariant_corrected_w_at_cells_on_half_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
        coeff1_dwdz,
        coeff2_dwdz,
        e_bln_c_s,
        ddqz_z_half,
        area,
        geofac_n2s,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
        skip_compute_predictor_vertical_advection,
        levelmask,
        cfl_clipping,
        owner_mask,
        cell,
        k,
        cell_lower_bound,
        cell_upper_bound,
        nlev,
        nrdmax,
        start_cell_lateral_boundary,
        end_cell_halo,
        out=(contravariant_corrected_w_at_cells_on_model_levels, vertical_wind_advective_tendency),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
