# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_wp import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels_2: fa.CellKField[ta.vpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_09."""
    wgtfac_c_wp, ddqz_z_half_wp = astype((wgtfac_c, ddqz_z_half), wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_cell_field_to_half_levels_vp(
        wgtfac_c=wgtfac_c, interpolant=perturbed_theta_v_at_cells_on_model_levels_2
    )
    theta_v_ic_wp = _interpolate_cell_field_to_half_levels_wp(
        wgtfac_c=wgtfac_c_wp, interpolant=theta_v
    )
    z_th_ddz_exner_c_wp = exner_w_explicit_weight_parameter * theta_v_ic_wp * (
        perturbed_exner_at_cells_on_model_levels(Koff[-1]) - perturbed_exner_at_cells_on_model_levels
    ) / ddqz_z_half_wp + astype(z_theta_v_pr_ic_vp * d_exner_dz_ref_ic, wpfloat)
    return z_theta_v_pr_ic_vp, theta_v_ic_wp, astype(z_th_ddz_exner_c_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_virtual_potential_temperatures_and_pressure_gradient(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels_2: fa.CellKField[ta.vpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_virtual_potential_temperatures_and_pressure_gradient(
        wgtfac_c,
        perturbed_theta_v_at_cells_on_model_levels_2,
        theta_v,
        exner_w_explicit_weight_parameter,
        perturbed_exner_at_cells_on_model_levels,
        d_exner_dz_ref_ic,
        ddqz_z_half,
        out=(perturbed_theta_v_at_cells_on_half_levels, theta_v_at_cells_on_half_levels, ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_virtual_potential_temperatures(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels_2: fa.CellKField[ta.vpfloat],
    theta_v: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
]:
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    z_theta_v_pr_ic_vp = _interpolate_cell_field_to_half_levels_vp(
        wgtfac_c=wgtfac_c, interpolant=perturbed_theta_v_at_cells_on_model_levels_2
    )
    theta_v_ic_wp = wgtfac_c_wp * theta_v + (wpfloat("1.0") - wgtfac_c_wp) * theta_v(Koff[-1])
    return z_theta_v_pr_ic_vp, theta_v_ic_wp


@gtx.field_operator
def _compute_pressure_gradient(
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.vpfloat]:
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)
    z_th_ddz_exner_c_wp = exner_w_explicit_weight_parameter * theta_v_at_cells_on_half_levels * (
        perturbed_exner_at_cells_on_model_levels(Koff[-1]) - perturbed_exner_at_cells_on_model_levels
    ) / ddqz_z_half_wp + astype(perturbed_theta_v_at_cells_on_half_levels * d_exner_dz_ref_ic, wpfloat)
    return astype(z_th_ddz_exner_c_wp, vpfloat)
