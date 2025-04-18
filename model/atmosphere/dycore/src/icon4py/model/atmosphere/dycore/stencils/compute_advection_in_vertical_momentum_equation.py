# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import maximum

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
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _compute_advective_vertical_wind_tendency_and_apply_diffusion(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> fa.CellKField[ta.vpfloat]:
    # TODO(havogt): we should get rid of the cell_lower_bound and cell_upper_bound,
    # they are only to protect write to halo (if I understand correctly)
    vertical_wind_advective_tendency = concat_where(
        1 <= dims.KDim,
        _compute_advective_vertical_wind_tendency(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        ),
        vertical_wind_advective_tendency,
    )
    vertical_wind_advective_tendency = concat_where(
        1 <= dims.KDim,
        _add_interpolated_horizontal_advection_of_w(
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels,
            vertical_wind_advective_tendency,
        ),
        vertical_wind_advective_tendency,
    )
    vertical_wind_advective_tendency = concat_where(
        ((maximum(3, end_index_of_damping_layer - 2) - 1) <= dims.KDim) & (dims.KDim < (nlev - 3)),
        _add_extra_diffusion_for_w_con_approaching_cfl(
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
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[ta.vpfloat]]:
    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels
        )
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
            cfl_clipping,
            owner_mask,
            ddqz_z_half,
            area,
            geofac_n2s,
            scalfac_exdiff,
            cfl_w_limit,
            dtime,
            nlev,
            end_index_of_damping_layer,
        )
        if not skip_compute_predictor_vertical_advection
        else vertical_wind_advective_tendency
    )

    return (contravariant_corrected_w_at_cells_on_model_levels, vertical_wind_advective_tendency)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_vertical_momentum_equation(
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_velocity_advection_stencil_15_to_18.

    This computes the vertical momentum advection in the vertical momentum equation

    Args:
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at model levels
        - vertical_wind_advective_tendency: vertical advection tendency of the vertical wind
        - w: vertical wind at cell centers
        - contravariant_corrected_w_at_cells_on_half_levels: contravariant-corrected vertical velocity at cells on half levels
        - horizontal_advection_of_w_at_edges_on_half_levels: horizontal advection for vertical velocity at edges on half levels
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - skip_compute_predictor_vertical_advection: logical flag to skip the vertical advection
        - cfl_clipping: boolean field indicating CFL clipping applied per cell and level
        - owner_mask: ownership mask for each cell
        - nlev: number of (full/model) vertical levels
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal dimension
        - horizontal_end: end index in the horizontal dimension
        - vertical_start: start index in the vertical dimension
        - vertical_end: end index in the vertical dimension

    Returns:
        - contravariant_corrected_w_at_cells_on_model_levels
        - vertical_wind_advective_tendency

    """

    _compute_advection_in_vertical_momentum_equation(
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
        cfl_clipping,
        owner_mask,
        nlev,
        end_index_of_damping_layer,
        out=(contravariant_corrected_w_at_cells_on_model_levels, vertical_wind_advective_tendency),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
