# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_explicit_part_for_rho_and_exner(
    current_rho: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    divergence_of_mass: fa.CellKField[vpfloat],
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[wpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    divergence_of_theta_v: fa.CellKField[vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[vpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_48 or _mo_solve_nonhydro_stencil_49."""
    inv_ddqz_z_full_wp, z_flxdiv_mass_wp, z_beta_wp, z_flxdiv_theta_wp, ddt_exner_phy_wp = astype(
        (inv_ddqz_z_full, divergence_of_mass, tridiagonal_beta_coeff_at_cells_on_model_levels, divergence_of_theta_v, exner_tendency_due_to_slow_physics), wpfloat
    )

    z_rho_expl_wp = current_rho - dtime * inv_ddqz_z_full_wp * (
        z_flxdiv_mass_wp + vertical_mass_flux_at_cells_on_half_levels - vertical_mass_flux_at_cells_on_half_levels(Koff[1])
    )

    z_exner_expl_wp = (
        perturbed_exner_at_cells_on_model_levels
        - z_beta_wp
        * (
            z_flxdiv_theta_wp
            + theta_v_at_cells_on_half_levels * vertical_mass_flux_at_cells_on_half_levels
            - theta_v_at_cells_on_half_levels(Koff[1]) * vertical_mass_flux_at_cells_on_half_levels(Koff[1])
        )
        + dtime * ddt_exner_phy_wp
    )
    return z_rho_expl_wp, z_exner_expl_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_part_for_rho_and_exner(
    rho_explicit_term: fa.CellKField[wpfloat],
    exner_explicit_term: fa.CellKField[wpfloat],
    current_rho: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    divergence_of_mass: fa.CellKField[vpfloat],
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[wpfloat],
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[vpfloat],
    divergence_of_theta_v: fa.CellKField[vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[vpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_explicit_part_for_rho_and_exner(
        current_rho,
        inv_ddqz_z_full,
        divergence_of_mass,
        vertical_mass_flux_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels,
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        divergence_of_theta_v,
        theta_v_at_cells_on_half_levels,
        exner_tendency_due_to_slow_physics,
        dtime,
        out=(rho_explicit_term, exner_explicit_term),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
