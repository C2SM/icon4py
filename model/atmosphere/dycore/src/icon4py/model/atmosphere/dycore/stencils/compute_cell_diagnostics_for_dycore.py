# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Final

import gt4py.next as gtx
from gt4py.next import program
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.ffront.fbuiltins import astype, bool, broadcast, maximum

from icon4py.model.atmosphere.dycore.dycore_states import (
    HorizontalPressureDiscretizationType,
)
from icon4py.model.atmosphere.dycore.solve_nonhydro_stencils import (
    _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures,
)
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_cell_kdim_fields_with_zero_vp import (
    _init_two_cell_kdim_fields_with_zero_vp,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.math.derivative import _compute_first_vertical_derivative_at_cells
from icon4py.model.common.type_alias import vpfloat, wpfloat


horzpres_discr_type: Final = HorizontalPressureDiscretizationType()


@field_operator
def _zero_initialize_for_limited_area_mode(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    limited_area: bool,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels) = (
        _init_two_cell_kdim_fields_with_zero_vp()
        if limited_area
        else (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels)
    )

    return (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels)

@field_operator
def _compute_perturbed_quantities_and_interpolation(
    current_rho: fa.CellKField[ta.wpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.wpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.wpfloat],
    pressure_buoyancy_acceleration_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    """Despite its name this field operator is not directly called by the program `compute_perturbed_quantities_and_interpolation`
    """
    exner_at_cells_on_half_levels = (
        concat_where(
            (maximum(1, nflatlev) <= dims.KDim),
            _interpolate_cell_field_to_half_levels_vp(
                wgtfac_c=wgtfac_c, interpolant=temporal_extrapolation_of_perturbed_exner
            ),
            exner_at_cells_on_half_levels,
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else exner_at_cells_on_half_levels
    )

    (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
        rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        pressure_buoyancy_acceleration_at_cells_on_half_levels,
    ) = _compute_pressure_gradient_and_perturbed_rho_and_potential_temperatures(
        rho=current_rho,
        z_rth_pr_1=perturbed_rho_at_cells_on_model_levels,
        z_rth_pr_2=perturbed_theta_v_at_cells_on_model_levels,
        rho_ref_mc=reference_rho_at_cells_on_model_levels,
        theta_v=current_theta_v,
        theta_ref_mc=reference_theta_at_cells_on_model_levels,
        rho_ic=rho_at_cells_on_half_levels,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        exner_pr=perturbed_exner_at_cells_on_model_levels,
        d_exner_dz_ref_ic=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        z_theta_v_pr_ic=perturbed_theta_v_at_cells_on_half_levels,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        z_th_ddz_exner_c=pressure_buoyancy_acceleration_at_cells_on_half_levels,
    )

    perturbed_theta_v_at_cells_on_half_levels = concat_where(
        dims.KDim == 0,
        broadcast(0.0, (dims.CellDim, dims.KDim)),
        perturbed_theta_v_at_cells_on_half_levels,
    )

    return (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
        temporal_extrapolation_of_perturbed_exner,
        perturbed_exner_at_cells_on_model_levels,
        rho_at_cells_on_half_levels,
        exner_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        pressure_buoyancy_acceleration_at_cells_on_half_levels,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )


@field_operator
def _surface_computations(
    wgtfacq_c: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    igradp_method: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    temporal_extrapolation_of_perturbed_exner = _init_cell_kdim_field_with_zero_wp()

    exner_at_cells_on_half_levels = (
        _interpolate_to_surface(
            wgtfacq_c=wgtfacq_c, interpolant=temporal_extrapolation_of_perturbed_exner
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else exner_at_cells_on_half_levels
    )

    return (
        temporal_extrapolation_of_perturbed_exner,
        exner_at_cells_on_half_levels,
    )


@field_operator
def _compute_first_and_second_vertical_derivative_of_exner(
    exner_at_cells_on_half_levels: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[vpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        concat_where(
            (nflatlev <= dims.KDim),
            _compute_first_vertical_derivative_at_cells(
                exner_at_cells_on_half_levels, inv_ddqz_z_full
            ),
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )

    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        concat_where(
            (nflat_gradp <= dims.KDim),
            -vpfloat("0.5")
            * (
                (
                    perturbed_theta_v_at_cells_on_half_levels
                    - perturbed_theta_v_at_cells_on_half_levels(Koff[1])
                )
                * d2dexdz2_fac1_mc
                + perturbed_theta_v_at_cells_on_model_levels * d2dexdz2_fac2_mc
            ),
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )
        if igradp_method == horzpres_discr_type.TAYLOR_HYDRO
        else d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )

    return (
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )


@field_operator
def _set_theta_v_and_exner_on_surface_level(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat], fa.CellKField[vpfloat]]:
    perturbed_theta_v_at_cells_on_half_levels = _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c, interpolant=perturbed_theta_v_at_cells_on_model_levels
    )
    theta_v_at_cells_on_half_levels = (
        reference_theta_at_cells_on_half_levels + perturbed_theta_v_at_cells_on_half_levels
    )

    exner_at_cells_on_half_levels = _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c, interpolant=temporal_extrapolation_of_perturbed_exner
    )

    return (
        perturbed_theta_v_at_cells_on_half_levels,
        astype(theta_v_at_cells_on_half_levels, wpfloat),
        exner_at_cells_on_half_levels,
    )

@field_operator
def _compute_perturbed_quantities_and_interpolation_wrapper(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    wgtfacq_c: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    pressure_buoyancy_acceleration_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    time_extrapolation_parameter_for_exner: fa.CellKField[ta.vpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[ta.vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[ta.vpfloat],
    limited_area: bool,
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> tuple[
        fa.CellKField[ta.vpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.vpfloat],
        fa.CellKField[ta.vpfloat],
        fa.CellKField[ta.vpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.vpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat]
]:
    """
    This function is a wrapper for everything `compute_perturbed_quantities_and_interpolation` does.
    It can not be named `_compute_perturbed_quantities_and_interpolation` because that name is already
    taken and it does something else.

    Returns:
        #- temporal_extrapolation_of_perturbed_exner
        #- perturbed_exner_at_cells_on_model_levels
        #- exner_at_cells_on_half_levels
        #- perturbed_rho_at_cells_on_model_levels
        #- perturbed_theta_v_at_cells_on_model_levels
        #- rho_at_cells_on_half_levels
        #- perturbed_theta_v_at_cells_on_half_levels
        #- theta_v_at_cells_on_half_levels
        #- pressure_buoyancy_acceleration_at_cells_on_half_levels
        #- ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_level
        #- d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    """

    (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels) = concat_where(
            (start_cell_halo_level_2 <= dims.CellDim) & (dims.CellDim < start_cell_lateral_boundary_level_3) & (vertical_start <= dims.KDim) & (dims.KDim < (vertical_end - 1)),
            _zero_initialize_for_limited_area_mode(
                temporal_extrapolation_of_perturbed_exner,
                perturbed_exner_at_cells_on_model_levels,
                limited_area
            ),
            (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels)
    )

    (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels) = concat_where(
            (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_halo) & (vertical_start <= dims.KDim) & (dims.KDim < (vertical_end - 1)),
            _extrapolate_temporally_exner_pressure(
                time_extrapolation_parameter_for_exner,
                current_exner,
                reference_exner_at_cells_on_model_levels,
                perturbed_exner_at_cells_on_model_levels
            ),
            (temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels)
    )

    (temporal_extrapolation_of_perturbed_exner, exner_at_cells_on_half_levels) = concat_where(
        (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_halo) & ((vertical_end - 1) <= dims.KDim) & (dims.KDim < vertical_end),
        _surface_computations(
            wgtfacq_c,
            exner_at_cells_on_half_levels,
            igradp_method
        ),
        (temporal_extrapolation_of_perturbed_exner, exner_at_cells_on_half_levels)
    )

    (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
        temporal_extrapolation_of_perturbed_exner,
        perturbed_exner_at_cells_on_model_levels,
        rho_at_cells_on_half_levels,
        exner_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        pressure_buoyancy_acceleration_at_cells_on_half_levels,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    ) = concat_where(
            (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_halo) & (vertical_start <= dims.KDim) & (dims.KDim < (vertical_end - 1)),
            _compute_perturbed_quantities_and_interpolation(
                current_rho,
                reference_rho_at_cells_on_model_levels,
                current_theta_v,
                reference_theta_at_cells_on_model_levels,
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_half_levels,
                wgtfac_c,
                vwind_expl_wgt,
                perturbed_exner_at_cells_on_model_levels,
                ddz_of_reference_exner_at_cells_on_half_levels,
                ddqz_z_half,
                pressure_buoyancy_acceleration_at_cells_on_half_levels,
                rho_at_cells_on_half_levels,
                exner_at_cells_on_half_levels,
                temporal_extrapolation_of_perturbed_exner,
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                theta_v_at_cells_on_half_levels,
                igradp_method,
                nflatlev,
            ),
            (
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels,
                temporal_extrapolation_of_perturbed_exner,
                perturbed_exner_at_cells_on_model_levels,
                rho_at_cells_on_half_levels,
                exner_at_cells_on_half_levels,
                perturbed_theta_v_at_cells_on_half_levels,
                theta_v_at_cells_on_half_levels,
                pressure_buoyancy_acceleration_at_cells_on_half_levels,
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            )
    )

    (
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        exner_at_cells_on_half_levels,
    ) = concat_where(
            (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_halo) & ((vertical_end - 1) <= dims.KDim) & (dims.KDim <vertical_end),
            _set_theta_v_and_exner_on_surface_level(
                temporal_extrapolation_of_perturbed_exner,
                wgtfacq_c,
                perturbed_theta_v_at_cells_on_model_levels,
                reference_theta_at_cells_on_half_levels,
            ),
            (
                perturbed_theta_v_at_cells_on_half_levels,
                theta_v_at_cells_on_half_levels,
                exner_at_cells_on_half_levels,
            )
    )

    (
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    ) = concat_where(
            (start_cell_lateral_boundary_level_3 <= dims.CellDim) & (dims.CellDim < end_cell_halo) & (vertical_start <= dims.KDim) & (dims.KDim < (vertical_end - 1)),
            _compute_first_and_second_vertical_derivative_of_exner(
                exner_at_cells_on_half_levels,
                inv_ddqz_z_full,
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                perturbed_theta_v_at_cells_on_half_levels,
                d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc,
                perturbed_theta_v_at_cells_on_model_levels,
                igradp_method,
                nflatlev,
                nflat_gradp,
            ),
            (
                ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
                d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            )
    )

    (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
    ) = concat_where(
            (start_cell_halo_level_2 <= dims.CellDim) & (dims.CellDim < end_cell_halo_level_2) & (vertical_start <= dims.KDim) & (dims.KDim < (vertical_end - 1)),
            _compute_perturbation_of_rho_and_theta(
                rho=current_rho,
                rho_ref_mc=reference_rho_at_cells_on_model_levels,
                theta_v=current_theta_v,
                theta_ref_mc=reference_theta_at_cells_on_model_levels,
            ),
            (
                perturbed_rho_at_cells_on_model_levels,
                perturbed_theta_v_at_cells_on_model_levels,
            )
    )

    return (
            temporal_extrapolation_of_perturbed_exner,
            perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            pressure_buoyancy_acceleration_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )




@program(grid_type=GridType.UNSTRUCTURED)
def compute_perturbed_quantities_and_interpolation(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    wgtfacq_c: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    pressure_buoyancy_acceleration_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    time_extrapolation_parameter_for_exner: fa.CellKField[ta.vpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[ta.vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[ta.vpfloat],
    limited_area: bool,
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_solve_nonhydro_stencil_1_to_13_predictor.

    Returns:
        - temporal_extrapolation_of_perturbed_exner
        - perturbed_exner_at_cells_on_model_levels
        - exner_at_cells_on_half_levels
        - perturbed_rho_at_cells_on_model_levels
        - perturbed_theta_v_at_cells_on_model_levels
        - rho_at_cells_on_half_levels
        - perturbed_theta_v_at_cells_on_half_levels
        - theta_v_at_cells_on_half_levels
        - pressure_buoyancy_acceleration_at_cells_on_half_levels
        - ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_level
        - d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    """

    _compute_perturbed_quantities_and_interpolation_wrapper(
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        current_rho=current_rho,
        reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
        current_theta_v=current_theta_v,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        reference_theta_at_cells_on_half_levels=reference_theta_at_cells_on_half_levels,
        wgtfacq_c=wgtfacq_c,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        pressure_buoyancy_acceleration_at_cells_on_half_levels=pressure_buoyancy_acceleration_at_cells_on_half_levels,
        time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
        current_exner=current_exner,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        inv_ddqz_z_full=inv_ddqz_z_full,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        limited_area=limited_area,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        start_cell_lateral_boundary=start_cell_lateral_boundary,
        start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
        start_cell_halo_level_2=start_cell_halo_level_2,
        end_cell_halo=end_cell_halo,
        end_cell_halo_level_2=end_cell_halo_level_2,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        out=(
            temporal_extrapolation_of_perturbed_exner,
            perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            pressure_buoyancy_acceleration_at_cells_on_half_levels,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )



@field_operator
def _interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration(
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
]:
    (
        contravariant_correction_at_cells_on_half_levels_wp,
        wgtfac_c_wp,
        reference_theta_at_cells_on_model_levels_wp,
        ddqz_z_half_wp,
    ) = astype(
        (
            contravariant_correction_at_cells_on_half_levels,
            wgtfac_c,
            reference_theta_at_cells_on_model_levels,
            ddqz_z_half,
        ),
        wpfloat,
    )

    back_trajectory_w_at_cells_on_half_levels = (
        -(w - contravariant_correction_at_cells_on_half_levels_wp)
        * dtime
        * wpfloat("0.5")
        / ddqz_z_half_wp
    )

    time_averaged_rho = wgt_nnow_rth * current_rho + wgt_nnew_rth * next_rho
    time_averaged_rho_kup = wgt_nnow_rth * current_rho(Koff[-1]) + wgt_nnew_rth * next_rho(Koff[-1])

    time_averaged_theta_v = wgt_nnow_rth * current_theta_v + wgt_nnew_rth * next_theta_v
    time_averaged_theta_v_kup = wgt_nnow_rth * current_theta_v(
        Koff[-1]
    ) + wgt_nnew_rth * next_theta_v(Koff[-1])

    rho_at_cells_on_half_levels = (
        wgtfac_c_wp * time_averaged_rho
        + (wpfloat("1.0") - wgtfac_c_wp) * time_averaged_rho_kup
        + back_trajectory_w_at_cells_on_half_levels * (time_averaged_rho_kup - time_averaged_rho)
    )

    time_averaged_perturbed_theta_v_kup = (
        time_averaged_theta_v_kup - reference_theta_at_cells_on_model_levels_wp(Koff[-1])
    )
    time_averaged_perturbed_theta_v = (
        time_averaged_theta_v - reference_theta_at_cells_on_model_levels_wp
    )

    time_averaged_perturbed_theta_v_vp, time_averaged_perturbed_theta_v_kup_vp = astype(
        (time_averaged_perturbed_theta_v, time_averaged_perturbed_theta_v_kup), vpfloat
    )
    perturbed_theta_v_at_cells_on_half_levels = (
        wgtfac_c * time_averaged_perturbed_theta_v_vp
        + (vpfloat("1.0") - wgtfac_c) * time_averaged_perturbed_theta_v_kup_vp
    )

    theta_v_at_cells_on_half_levels = (
        wgtfac_c_wp * time_averaged_theta_v
        + (wpfloat("1.0") - wgtfac_c_wp) * time_averaged_theta_v_kup
        + back_trajectory_w_at_cells_on_half_levels
        * (time_averaged_theta_v_kup - time_averaged_theta_v)
    )
    pressure_buoyancy_acceleration_at_cells_on_half_levels = (
        vwind_expl_wgt
        * theta_v_at_cells_on_half_levels
        * (
            perturbed_exner_at_cells_on_model_levels(Koff[-1])
            - perturbed_exner_at_cells_on_model_levels
        )
        / astype(ddqz_z_half, wpfloat)
        + astype(
            perturbed_theta_v_at_cells_on_half_levels
            * ddz_of_reference_exner_at_cells_on_half_levels,
            wpfloat,
        )
    )
    return (
        rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        astype(pressure_buoyancy_acceleration_at_cells_on_half_levels, vpfloat),
    )


@program(grid_type=GridType.UNSTRUCTURED)
def interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration(
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    pressure_buoyancy_acceleration_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    vwind_expl_wgt: fa.CellField[ta.wpfloat],
    dtime: ta.wpfloat,
    wgt_nnow_rth: ta.wpfloat,
    wgt_nnew_rth: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Formerly known as fused_solve_nonhydro_stencil_1_to_13_corrector.

    This program calculates the air density, virtual potential temperature and perturberd virtual potential temperature on half levels,
    and also computes the pressure buoyancy acceleration.

    Args:
        - rho_at_cells_on_half_levels: air density at cells on half levels [kg m-3]
        - perturbed_theta_v_at_cells_on_half_levels: perturbed virtual potential temperature (actual virtual potential temperature minus reference virtual potential temperature) at cells on half levels [kg m-3]
        - theta_v_at_cells_on_half_levels: virtual potential temperature at cells on half levels [K]
        - pressure_buoyancy_acceleration_at_cells_on_half_levels: pressure buoyancy acceleration at cells on half levels [m s-2]
        - w: vertical wind at cell centers [m s-1]
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - current_rho: air density at current substep [K]
        - next_rho: air density at next substep [K]
        - current_theta_v: virtual potential temperature at current substep [K]
        - next_theta_v: virtual potential temperature at next substep [K]
        - perturbed_exner_at_cells_on_model_levels: perturbed exner function at model levels
        - reference_theta_at_cells_on_model_levels: reference virtual potential temperature at cells on model levels [K]
        - ddz_of_reference_exner_at_cells_on_half_levels: vertical gradient of reference exner function at cells on half levels [m-1]
        - ddqz_z_half: vertical spacing on half levels (distance between the height of cell centers at k at k-1)  [m]
        - wgtfac_c: metrics field
        - vwind_expl_wgt: external weight field for wind extrapolation
        - dtime: time step
        - wgt_nnow_rth: interpolation coefficient of virtual potential temperature at predictor step
        - wgt_nnew_rth: interpolation coefficient of virtual potential temperature at corrector step
        - horizontal_start: start index of the horizontal domain
        - horizontal_end: end index of the horizontal domain
        - vertical_start: start index of the vertical domain
        - vertical_end: end index of the vertical domain

    Returns:
        - rho_at_cells_on_half_levels
        - perturbed_theta_v_at_cells_on_half_levels
        - theta_v_at_cells_on_half_levels
        - pressure_buoyancy_acceleration_at_cells_on_half_levels
    """
    _interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration(
        w,
        contravariant_correction_at_cells_on_half_levels,
        current_rho,
        next_rho,
        current_theta_v,
        next_theta_v,
        perturbed_exner_at_cells_on_model_levels,
        reference_theta_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half,
        wgtfac_c,
        vwind_expl_wgt,
        dtime,
        wgt_nnow_rth,
        wgt_nnew_rth,
        out=(
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            pressure_buoyancy_acceleration_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
