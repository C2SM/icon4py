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

import dataclasses
from typing import Any, Callable, Optional

from gt4py import next as gtx

from icon4py.model.atmosphere.dycore.accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields as accumulate_prep_adv_fields_orig,
)
from icon4py.model.atmosphere.dycore.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation as add_analysis_increments_from_data_assimilation_orig,
)
from icon4py.model.atmosphere.dycore.add_analysis_increments_to_vn import (
    add_analysis_increments_to_vn as add_analysis_increments_to_vn_orig,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn import (
    add_temporal_tendencies_to_vn as add_temporal_tendencies_to_vn_orig,
)
from icon4py.model.atmosphere.dycore.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels as add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_orig,
)
from icon4py.model.atmosphere.dycore.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping as add_vertical_wind_derivative_to_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.apply_2nd_order_divergence_damping import (
    apply_2nd_order_divergence_damping as apply_2nd_order_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.apply_4th_order_divergence_damping import (
    apply_4th_order_divergence_damping as apply_4th_order_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure import (
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure as apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_orig,
)
from icon4py.model.atmosphere.dycore.apply_rayleigh_damping_mechanism import (
    apply_rayleigh_damping_mechanism as apply_rayleigh_damping_mechanism_orig,
)
from icon4py.model.atmosphere.dycore.apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping as apply_weighted_2nd_and_4th_order_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.compute_approx_of_2nd_vertical_derivative_of_exner import (
    compute_approx_of_2nd_vertical_derivative_of_exner as compute_approx_of_2nd_vertical_derivative_of_exner_orig,
)
from icon4py.model.atmosphere.dycore.compute_avg_vn import compute_avg_vn as compute_avg_vn_orig
from icon4py.model.atmosphere.dycore.compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt as compute_avg_vn_and_graddiv_vn_and_vt_orig,
)
from icon4py.model.atmosphere.dycore.compute_divergence_of_fluxes_of_rho_and_theta import (
    compute_divergence_of_fluxes_of_rho_and_theta as compute_divergence_of_fluxes_of_rho_and_theta_orig,
)
from icon4py.model.atmosphere.dycore.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping as compute_dwdz_for_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.compute_exner_from_rhotheta import (
    compute_exner_from_rhotheta as compute_exner_from_rhotheta_orig,
)
from icon4py.model.atmosphere.dycore.compute_graddiv2_of_vn import (
    compute_graddiv2_of_vn as compute_graddiv2_of_vn_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates as compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates as compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_extner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_extner_pressure_for_multiple_levels as compute_horizontal_gradient_of_extner_pressure_for_multiple_levels_orig,
)
from icon4py.model.atmosphere.dycore.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term as compute_hydrostatic_correction_term_orig,
)
from icon4py.model.atmosphere.dycore.compute_mass_flux import compute_mass_flux as compute_mass_flux_orig
from icon4py.model.atmosphere.dycore.compute_pertubation_of_rho_and_theta import (
    compute_pertubation_of_rho_and_theta as compute_pertubation_of_rho_and_theta_orig,
)
from icon4py.model.atmosphere.dycore.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables as compute_results_for_thermodynamic_variables_orig,
)
from icon4py.model.atmosphere.dycore.compute_rho_virtual_potential_temperatures_and_pressure_gradient import (
    compute_rho_virtual_potential_temperatures_and_pressure_gradient as compute_rho_virtual_potential_temperatures_and_pressure_gradient_orig,
)
from icon4py.model.atmosphere.dycore.compute_theta_and_exner import (
    compute_theta_and_exner as compute_theta_and_exner_orig,
)
from icon4py.model.atmosphere.dycore.compute_vn_on_lateral_boundary import (
    compute_vn_on_lateral_boundary as compute_vn_on_lateral_boundary_orig,
)
from icon4py.model.atmosphere.dycore.copy_cell_kdim_field_to_vp import (
    copy_cell_kdim_field_to_vp as copy_cell_kdim_field_to_vp_orig,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl as mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig,
)
from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl as mo_math_gradients_grad_green_gauss_cell_dsl_orig,
)
from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_to_zero_vp import (
    set_two_cell_kdim_fields_to_zero_vp as set_two_cell_kdim_fields_to_zero_vp_orig,
)
from icon4py.model.atmosphere.dycore.set_two_cell_kdim_fields_to_zero_wp import (
    set_two_cell_kdim_fields_to_zero_wp as set_two_cell_kdim_fields_to_zero_wp_orig,
)
from icon4py.model.atmosphere.dycore.set_two_edge_kdim_fields_to_zero_wp import (
    set_two_edge_kdim_fields_to_zero_wp as set_two_edge_kdim_fields_to_zero_wp_orig,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution as solve_tridiagonal_matrix_for_w_back_substitution_orig,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep as solve_tridiagonal_matrix_for_w_forward_sweep_orig,
)
from icon4py.model.atmosphere.dycore.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment as update_dynamical_exner_time_increment_orig,
)
from icon4py.model.atmosphere.dycore.update_mass_volume_flux import (
    update_mass_volume_flux as update_mass_volume_flux_orig,
)
from icon4py.model.atmosphere.dycore.update_mass_flux_weighted import (
    update_mass_flux_weighted as update_mass_flux_weighted_orig,
)
from icon4py.model.atmosphere.dycore.update_theta_v import update_theta_v as update_theta_v_orig
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program import (
    predictor_stencils_2_3,
    predictor_stencils_4_5_6,
    predictor_stencils_7_8_9,
    predictor_stencils_11_lower_upper,

)


@dataclasses.dataclass
class CachedProgram:
    program: gtx.ffront.decorator.Program
    _compiled_program: Optional[Callable] = None
    _compiled_args: tuple = dataclasses.field(default_factory=tuple)

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        backend = self.program.backend
        transformer = backend.transformer.replace(
            args=args, kwargs=kwargs | {"offset_provider": offset_provider}
        )
        program_call = transformer(self.program.definition_stage)
        self._compiled_args = program_call.args
        return backend.executor.otf_workflow(program_call)

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )

        size_args = self._compiled_args[len(args) :]
        return self.compiled_program(*args, *size_args, offset_provider=offset_provider)


accumulate_prep_adv_fields = CachedProgram(accumulate_prep_adv_fields_orig)
add_analysis_increments_from_data_assimilation = CachedProgram(add_analysis_increments_from_data_assimilation_orig)
add_analysis_increments_to_vn = CachedProgram(add_analysis_increments_to_vn_orig)
add_temporal_tendencies_to_vn_by_interpolating_between_time_levels = CachedProgram(add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_orig)
add_temporal_tendencies_to_vn = CachedProgram(add_temporal_tendencies_to_vn_orig)
add_vertical_wind_derivative_to_divergence_damping = CachedProgram(add_vertical_wind_derivative_to_divergence_damping_orig)
apply_4th_order_divergence_damping = CachedProgram(apply_4th_order_divergence_damping_orig)
apply_rayleigh_damping_mechanism = CachedProgram(apply_rayleigh_damping_mechanism_orig)
apply_2nd_order_divergence_damping = CachedProgram(apply_2nd_order_divergence_damping_orig)
apply_weighted_2nd_and_4th_order_divergence_damping = CachedProgram(apply_weighted_2nd_and_4th_order_divergence_damping_orig)
apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure = CachedProgram(apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_orig)
compute_approx_of_2nd_vertical_derivative_of_exner = CachedProgram(compute_approx_of_2nd_vertical_derivative_of_exner_orig)
compute_avg_vn = CachedProgram(compute_avg_vn_orig)
compute_vn_on_lateral_boundary = CachedProgram(compute_vn_on_lateral_boundary_orig)
compute_results_for_thermodynamic_variables = CachedProgram(compute_results_for_thermodynamic_variables_orig)
compute_theta_and_exner = CachedProgram(compute_theta_and_exner_orig)
compute_exner_from_rhotheta = CachedProgram(compute_exner_from_rhotheta_orig)
compute_graddiv2_of_vn = CachedProgram(compute_graddiv2_of_vn_orig)
compute_hydrostatic_correction_term = CachedProgram(compute_hydrostatic_correction_term_orig)
compute_mass_flux = CachedProgram(compute_mass_flux_orig)
compute_dwdz_for_divergence_damping = CachedProgram(compute_dwdz_for_divergence_damping_orig)
compute_rho_virtual_potential_temperatures_and_pressure_gradient = CachedProgram(compute_rho_virtual_potential_temperatures_and_pressure_gradient_orig)
compute_pertubation_of_rho_and_theta = CachedProgram(compute_pertubation_of_rho_and_theta_orig)
compute_horizontal_gradient_of_extner_pressure_for_multiple_levels = CachedProgram(compute_horizontal_gradient_of_extner_pressure_for_multiple_levels_orig)
compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates = CachedProgram(compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_orig)
compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates = CachedProgram(compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_orig)
compute_avg_vn_and_graddiv_vn_and_vt = CachedProgram(compute_avg_vn_and_graddiv_vn_and_vt_orig)
compute_divergence_of_fluxes_of_rho_and_theta = CachedProgram(compute_divergence_of_fluxes_of_rho_and_theta_orig)
copy_cell_kdim_field_to_vp = CachedProgram(copy_cell_kdim_field_to_vp_orig)
mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = CachedProgram(mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig)
set_two_cell_kdim_fields_to_zero_vp = CachedProgram(set_two_cell_kdim_fields_to_zero_vp_orig)
set_two_cell_kdim_fields_to_zero_wp = CachedProgram(set_two_cell_kdim_fields_to_zero_wp_orig)
set_two_edge_kdim_fields_to_zero_wp = CachedProgram(set_two_edge_kdim_fields_to_zero_wp_orig)
mo_math_gradients_grad_green_gauss_cell_dsl = CachedProgram(mo_math_gradients_grad_green_gauss_cell_dsl_orig)
solve_tridiagonal_matrix_for_w_back_substitution = CachedProgram(solve_tridiagonal_matrix_for_w_back_substitution_orig)
solve_tridiagonal_matrix_for_w_forward_sweep = CachedProgram(solve_tridiagonal_matrix_for_w_forward_sweep_orig)
update_theta_v = CachedProgram(update_theta_v_orig)
update_dynamical_exner_time_increment = CachedProgram(update_dynamical_exner_time_increment_orig)
update_mass_volume_flux = CachedProgram(update_mass_volume_flux_orig)
update_mass_flux_weighted = CachedProgram(update_mass_flux_weighted_orig)
