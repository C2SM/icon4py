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

import numpy as np
from gt4py import next as gtx
from gt4py.next.otf import workflow


try:
    import cupy as cp
    from gt4py.next.embedded.nd_array_field import CuPyArrayField
except ImportError:
    cp: Optional = None  # type:ignore[no-redef]

from gt4py.next.embedded.nd_array_field import NumPyArrayField
from gt4py.next.program_processors.runners.gtfn import extract_connectivity_args

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
from icon4py.model.atmosphere.dycore.apply_4th_order_divergence_damping_nonmeancell import (
    apply_4th_order_divergence_damping_nonmeancell as apply_4th_order_divergence_damping_nonmeancell_orig,
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

# New divergence stencils
from icon4py.model.atmosphere.dycore.compute_divergence_of_flux_of_normal_wind import (
    compute_divergence_of_flux_of_normal_wind as compute_divergence_of_flux_of_normal_wind_orig,
)
from icon4py.model.atmosphere.dycore.add_dwdz_to_divergence_of_flux_of_normal_wind import (
    add_dwdz_to_divergence_of_flux_of_normal_wind as add_dwdz_to_divergence_of_flux_of_normal_wind_orig,
)
from icon4py.model.atmosphere.dycore.compute_full3d_graddiv_normal import (
    compute_full3d_graddiv_normal as compute_full3d_graddiv_normal_orig,
)
from icon4py.model.atmosphere.dycore.compute_full3d_graddiv_vertical import (
    compute_full3d_graddiv_vertical as compute_full3d_graddiv_vertical_orig,
)
from icon4py.model.atmosphere.dycore.compute_dgraddiv_dz_for_full3d_divergence_damping import (
    compute_dgraddiv_dz_for_full3d_divergence_damping as compute_dgraddiv_dz_for_full3d_divergence_damping_orig,
)
from icon4py.model.atmosphere.dycore.compute_divergence_of_flux_of_full3d_graddiv import (
    compute_divergence_of_flux_of_full3d_graddiv as compute_divergence_of_flux_of_full3d_graddiv_orig,
)
from icon4py.model.atmosphere.dycore.add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv import (
    add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv as add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv_orig,
)
from icon4py.model.atmosphere.dycore.compute_full3d_graddiv2_normal import (
    compute_full3d_graddiv2_normal as compute_full3d_graddiv2_normal_orig,
)
from icon4py.model.atmosphere.dycore.compute_full3d_graddiv2_vertical import (
    compute_full3d_graddiv2_vertical as compute_full3d_graddiv2_vertical_orig,
)
from icon4py.model.atmosphere.dycore.apply_4th_order_3d_divergence_damping_to_vn import (
    apply_4th_order_3d_divergence_damping_to_vn as apply_4th_order_3d_divergence_damping_to_vn_orig,
)
from icon4py.model.atmosphere.dycore.apply_4th_order_3d_divergence_damping_to_w import (
    apply_4th_order_3d_divergence_damping_to_w as apply_4th_order_3d_divergence_damping_to_w_orig,
)
from icon4py.model.atmosphere.dycore.compute_2nd_order_divergence_of_flux_of_normal_wind import (
    compute_2nd_order_divergence_of_flux_of_normal_wind as compute_2nd_order_divergence_of_flux_of_normal_wind_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell import (
    interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell as interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell_orig,
)
from icon4py.model.atmosphere.dycore.compute_2nd_order_divergence_of_flux_of_full3d_graddiv import (
    compute_2nd_order_divergence_of_flux_of_full3d_graddiv as compute_2nd_order_divergence_of_flux_of_full3d_graddiv_orig,
)
from icon4py.model.atmosphere.dycore.interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell import (
    interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell as interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell_orig,
)
# end of new divergence stencils

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
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels as compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_orig,
)
from icon4py.model.atmosphere.dycore.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates as compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_orig,
)
from icon4py.model.atmosphere.dycore.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term as compute_hydrostatic_correction_term_orig,
)
from icon4py.model.atmosphere.dycore.compute_mass_flux import (
    compute_mass_flux as compute_mass_flux_orig,
)
from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta import (
    compute_perturbation_of_rho_and_theta as compute_perturbation_of_rho_and_theta_orig,
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
from icon4py.model.atmosphere.dycore.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp as init_cell_kdim_field_with_zero_wp_orig,
)
from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_with_zero_vp import (
    init_two_cell_kdim_fields_with_zero_vp as init_two_cell_kdim_fields_with_zero_vp_orig,
)
from icon4py.model.atmosphere.dycore.init_two_cell_kdim_fields_with_zero_wp import (
    init_two_cell_kdim_fields_with_zero_wp as init_two_cell_kdim_fields_with_zero_wp_orig,
)
from icon4py.model.atmosphere.dycore.init_two_edge_kdim_fields_with_zero_wp import (
    init_two_edge_kdim_fields_with_zero_wp as init_two_edge_kdim_fields_with_zero_wp_orig,
)
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl as mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig,
)
from icon4py.model.atmosphere.dycore.mo_math_gradients_grad_green_gauss_cell_dsl import (
    mo_math_gradients_grad_green_gauss_cell_dsl as mo_math_gradients_grad_green_gauss_cell_dsl_orig,
)
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro_program import (
    compute_horizontal_advection_of_rho_and_theta as compute_horizontal_advection_of_rho_and_theta_orig,
    init_test_fields as init_test_fields_orig,
    predictor_stencils_2_3 as predictor_stencils_2_3_orig,
    predictor_stencils_4_5_6 as predictor_stencils_4_5_6_orig,
    predictor_stencils_7_8_9 as predictor_stencils_7_8_9_orig,
    compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels as compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels_orig,
    predictor_stencils_7_8_9_firststep as predictor_stencils_7_8_9_firststep_orig,
    predictor_stencils_7_8_9_secondstep as predictor_stencils_7_8_9_secondstep_orig,
    predictor_stencils_11_lower_upper as predictor_stencils_11_lower_upper_orig,
    predictor_stencils_35_36 as predictor_stencils_35_36_orig,
    predictor_stencils_37_38 as predictor_stencils_37_38_orig,
    stencils_39_40 as stencils_39_40_orig,
    stencils_42_44_45_45b as stencils_42_44_45_45b_orig,
    stencils_43_44_45_45b as stencils_43_44_45_45b_orig,
    stencils_47_48_49 as stencils_47_48_49_orig,
    stencils_61_62 as stencils_61_62_orig,
)

from icon4py.model.atmosphere.dycore.compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_pressure_gradient as compute_pressure_gradient_orig,
)

from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_back_substitution import (
    solve_tridiagonal_matrix_for_w_back_substitution as solve_tridiagonal_matrix_for_w_back_substitution_orig,
)
from icon4py.model.atmosphere.dycore.solve_tridiagonal_matrix_for_w_forward_sweep import (
    solve_tridiagonal_matrix_for_w_forward_sweep as solve_tridiagonal_matrix_for_w_forward_sweep_orig,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import (
    compute_z_raylfac as compute_z_raylfac_orig,
    calculate_scal_divdamp_half as calculate_scal_divdamp_half_orig,
    calculate_divdamp_fields as calculate_divdamp_fields_orig,
)
from icon4py.model.atmosphere.dycore.update_dynamical_exner_time_increment import (
    update_dynamical_exner_time_increment as update_dynamical_exner_time_increment_orig,
)
from icon4py.model.atmosphere.dycore.update_mass_flux_weighted import (
    update_mass_flux_weighted as update_mass_flux_weighted_orig,
)
from icon4py.model.atmosphere.dycore.update_mass_volume_flux import (
    update_mass_volume_flux as update_mass_volume_flux_orig,
)
from icon4py.model.atmosphere.dycore.update_theta_v import update_theta_v as update_theta_v_orig
from icon4py.model.common.math.smagorinsky import (
    en_smag_fac_for_zero_nshift as en_smag_fac_for_zero_nshift_orig,
)
from icon4py.model.common.settings import device


def handle_numpy_integer(value):
    return int(value)


def handle_common_field(value, sizes):
    sizes.extend(value.shape)
    return value  # Return the value unmodified, but side-effect on sizes


def handle_default(value):
    return value  # Return the value unchanged


if cp:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
        CuPyArrayField: handle_common_field,
    }
else:
    type_handlers = {
        np.integer: handle_numpy_integer,
        NumPyArrayField: handle_common_field,
    }


def process_arg(value, sizes):
    handler = type_handlers.get(type(value), handle_default)
    return handler(value, sizes) if handler == handle_common_field else handler(value)


@dataclasses.dataclass
class CachedProgram:
    program: gtx.ffront.decorator.Program
    with_domain: bool = True
    _compiled_program: Optional[Callable] = None
    _conn_args: Any = None
    _compiled_args: tuple = dataclasses.field(default_factory=tuple)

    @property
    def compiled_program(self) -> Callable:
        return self._compiled_program

    @property
    def conn_args(self) -> Callable:
        return self._conn_args

    def compile_the_program(
        self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any
    ) -> Callable:
        backend = self.program.backend
        program_call = backend.transforms_prog(
            workflow.InputWithArgs(
                data=self.program.definition_stage,
                args=args,
                kwargs=kwargs | {"offset_provider": offset_provider},
            )
        )
        self._compiled_args = program_call.args
        return backend.executor.otf_workflow(program_call)

    def __call__(self, *args, offset_provider: dict[str, gtx.Dimension], **kwargs: Any) -> None:
        if not self.compiled_program:
            self._compiled_program = self.compile_the_program(
                *args, offset_provider=offset_provider, **kwargs
            )
            self._conn_args = extract_connectivity_args(offset_provider, device)

        kwargs_as_tuples = tuple(kwargs.values())
        program_args = list(args) + list(kwargs_as_tuples)
        sizes = []

        # Convert numpy integers in args to int and handle gtx.common.Field
        for i in range(len(program_args)):
            program_args[i] = process_arg(program_args[i], sizes)

        if not self.with_domain:
            program_args.extend(sizes)

        # todo(samkellerhals): if we merge gt4py PR we can also pass connectivity args here conn_args=self.conn_args
        return self.compiled_program(*program_args, offset_provider=offset_provider)


accumulate_prep_adv_fields = CachedProgram(accumulate_prep_adv_fields_orig)

add_analysis_increments_from_data_assimilation = CachedProgram(
    add_analysis_increments_from_data_assimilation_orig
)

add_analysis_increments_to_vn = CachedProgram(add_analysis_increments_to_vn_orig)

add_temporal_tendencies_to_vn = CachedProgram(add_temporal_tendencies_to_vn_orig)

add_temporal_tendencies_to_vn_by_interpolating_between_time_levels = CachedProgram(
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_orig
)

add_vertical_wind_derivative_to_divergence_damping = CachedProgram(
    add_vertical_wind_derivative_to_divergence_damping_orig
)

apply_2nd_order_divergence_damping = CachedProgram(apply_2nd_order_divergence_damping_orig)

apply_4th_order_divergence_damping = CachedProgram(apply_4th_order_divergence_damping_orig)

apply_4th_order_divergence_damping_nonmeancell = CachedProgram(apply_4th_order_divergence_damping_nonmeancell_orig)

apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure = CachedProgram(
    apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure_orig
)

apply_rayleigh_damping_mechanism = CachedProgram(apply_rayleigh_damping_mechanism_orig)

apply_weighted_2nd_and_4th_order_divergence_damping = CachedProgram(
    apply_weighted_2nd_and_4th_order_divergence_damping_orig
)

compute_approx_of_2nd_vertical_derivative_of_exner = CachedProgram(
    compute_approx_of_2nd_vertical_derivative_of_exner_orig
)

compute_avg_vn = CachedProgram(compute_avg_vn_orig)

compute_avg_vn_and_graddiv_vn_and_vt = CachedProgram(compute_avg_vn_and_graddiv_vn_and_vt_orig)

compute_divergence_of_fluxes_of_rho_and_theta = CachedProgram(
    compute_divergence_of_fluxes_of_rho_and_theta_orig
)

# New divergence stencils
compute_divergence_of_flux_of_normal_wind = CachedProgram(
    compute_divergence_of_flux_of_normal_wind_orig
)

add_dwdz_to_divergence_of_flux_of_normal_wind = CachedProgram(
    add_dwdz_to_divergence_of_flux_of_normal_wind_orig
)

compute_full3d_graddiv_normal = CachedProgram(
    compute_full3d_graddiv_normal_orig
)

compute_full3d_graddiv_vertical = CachedProgram(
    compute_full3d_graddiv_vertical_orig
)

compute_dgraddiv_dz_for_full3d_divergence_damping = CachedProgram(
    compute_dgraddiv_dz_for_full3d_divergence_damping_orig
)

compute_divergence_of_flux_of_full3d_graddiv = CachedProgram(
    compute_divergence_of_flux_of_full3d_graddiv_orig
)

add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv = CachedProgram(
    add_dgraddiv_dz_to_full3d_divergence_flux_of_graddiv_orig
)

compute_full3d_graddiv2_normal = CachedProgram(
    compute_full3d_graddiv2_normal_orig
)

compute_full3d_graddiv2_vertical = CachedProgram(
    compute_full3d_graddiv2_vertical_orig
)

apply_4th_order_3d_divergence_damping_to_vn = CachedProgram(
    apply_4th_order_3d_divergence_damping_to_vn_orig
)

apply_4th_order_3d_divergence_damping_to_w = CachedProgram(
    apply_4th_order_3d_divergence_damping_to_w_orig
)
compute_2nd_order_divergence_of_flux_of_normal_wind = CachedProgram(
    compute_2nd_order_divergence_of_flux_of_normal_wind_orig
)
interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell = CachedProgram(
    interpolate_2nd_order_divergence_of_flux_of_normal_wind_to_cell_orig
)
compute_2nd_order_divergence_of_flux_of_full3d_graddiv = CachedProgram(
    compute_2nd_order_divergence_of_flux_of_full3d_graddiv_orig
)
interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell = CachedProgram(
    interpolate_2nd_order_divergence_of_flux_of_full3d_graddiv_to_cell_orig
)
# end of new divergence stencils

compute_dwdz_for_divergence_damping = CachedProgram(compute_dwdz_for_divergence_damping_orig)

compute_exner_from_rhotheta = CachedProgram(compute_exner_from_rhotheta_orig)

compute_graddiv2_of_vn = CachedProgram(compute_graddiv2_of_vn_orig)

compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates = CachedProgram(
    compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates_orig
)

compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates = CachedProgram(
    compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates_orig
)

compute_horizontal_gradient_of_exner_pressure_for_multiple_levels = CachedProgram(
    compute_horizontal_gradient_of_exner_pressure_for_multiple_levels_orig
)

compute_hydrostatic_correction_term = CachedProgram(compute_hydrostatic_correction_term_orig)

compute_mass_flux = CachedProgram(compute_mass_flux_orig)

compute_perturbation_of_rho_and_theta = CachedProgram(compute_perturbation_of_rho_and_theta_orig)

compute_results_for_thermodynamic_variables = CachedProgram(
    compute_results_for_thermodynamic_variables_orig
)

compute_rho_virtual_potential_temperatures_and_pressure_gradient = CachedProgram(
    compute_rho_virtual_potential_temperatures_and_pressure_gradient_orig
)

compute_theta_and_exner = CachedProgram(compute_theta_and_exner_orig)

compute_vn_on_lateral_boundary = CachedProgram(compute_vn_on_lateral_boundary_orig)

copy_cell_kdim_field_to_vp = CachedProgram(copy_cell_kdim_field_to_vp_orig)

mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = CachedProgram(
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_orig
)

mo_math_gradients_grad_green_gauss_cell_dsl = CachedProgram(
    mo_math_gradients_grad_green_gauss_cell_dsl_orig
)

solve_tridiagonal_matrix_for_w_back_substitution = CachedProgram(
    solve_tridiagonal_matrix_for_w_back_substitution_orig
)

solve_tridiagonal_matrix_for_w_forward_sweep = CachedProgram(
    solve_tridiagonal_matrix_for_w_forward_sweep_orig
)

update_dynamical_exner_time_increment = CachedProgram(update_dynamical_exner_time_increment_orig)

update_mass_volume_flux = CachedProgram(update_mass_volume_flux_orig)

update_mass_flux_weighted = CachedProgram(update_mass_flux_weighted_orig)

update_theta_v = CachedProgram(update_theta_v_orig)

en_smag_fac_for_zero_nshift = CachedProgram(en_smag_fac_for_zero_nshift_orig, with_domain=False)

init_cell_kdim_field_with_zero_wp = CachedProgram(init_cell_kdim_field_with_zero_wp_orig)

init_two_cell_kdim_fields_with_zero_vp = CachedProgram(init_two_cell_kdim_fields_with_zero_vp_orig)

init_two_cell_kdim_fields_with_zero_wp = CachedProgram(init_two_cell_kdim_fields_with_zero_wp_orig)

init_two_edge_kdim_fields_with_zero_wp = CachedProgram(init_two_edge_kdim_fields_with_zero_wp_orig)

init_test_fields = CachedProgram(init_test_fields_orig)

predictor_stencils_2_3 = CachedProgram(predictor_stencils_2_3_orig)

predictor_stencils_4_5_6 = CachedProgram(predictor_stencils_4_5_6_orig)

predictor_stencils_7_8_9 = CachedProgram(predictor_stencils_7_8_9_orig)

compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels = CachedProgram(compute_perturbed_rho_and_potential_temperatures_at_half_and_full_levels_orig)

compute_pressure_gradient = CachedProgram(compute_pressure_gradient_orig)

predictor_stencils_7_8_9_firststep = CachedProgram(predictor_stencils_7_8_9_firststep_orig)

predictor_stencils_7_8_9_secondstep = CachedProgram(predictor_stencils_7_8_9_secondstep_orig)

predictor_stencils_11_lower_upper = CachedProgram(predictor_stencils_11_lower_upper_orig)

compute_horizontal_advection_of_rho_and_theta = CachedProgram(
    compute_horizontal_advection_of_rho_and_theta_orig
)

predictor_stencils_35_36 = CachedProgram(predictor_stencils_35_36_orig)

predictor_stencils_37_38 = CachedProgram(predictor_stencils_37_38_orig)

stencils_39_40 = CachedProgram(stencils_39_40_orig)

stencils_43_44_45_45b = CachedProgram(stencils_43_44_45_45b_orig)

stencils_47_48_49 = CachedProgram(stencils_47_48_49_orig)

stencils_61_62 = CachedProgram(stencils_61_62_orig)

stencils_42_44_45_45b = CachedProgram(stencils_42_44_45_45b_orig)

compute_z_raylfac = CachedProgram(compute_z_raylfac_orig, with_domain=False)

calculate_scal_divdamp_half = CachedProgram(calculate_scal_divdamp_half_orig)

calculate_divdamp_fields = CachedProgram(calculate_divdamp_fields_orig, with_domain=False)
