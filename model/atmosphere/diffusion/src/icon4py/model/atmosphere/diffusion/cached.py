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

from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    copy_field as copy_field_orig,
    init_diffusion_local_fields_for_regular_timestep as init_diffusion_local_fields_for_regular_timestep_orig,
    scale_k as scale_k_orig,
    setup_fields_for_initial_step as setup_fields_for_initial_step_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import (
    apply_diffusion_to_vn as apply_diffusion_to_vn_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence as apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence as calculate_diagnostic_quantities_for_turbulence_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools as calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn as calculate_nabla2_and_smag_coefficients_for_vn_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_theta import (
    calculate_nabla2_for_theta as calculate_nabla2_for_theta_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points as truly_horizontal_diffusion_nabla_of_theta_over_steep_points_orig,
)
from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    update_theta_and_exner as update_theta_and_exner_orig,
)
from icon4py.model.common.caching import CachedProgram
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex as mo_intp_rbf_rbf_vec_interpol_vertex_orig,
)
from icon4py.model.common.orchestration.decorator import dace_orchestration


if dace_orchestration():
    # Skip caching since there is a caching mechanism in DaCe orchestration
    CachedProgram = lambda stencil, with_domain=True: stencil  # noqa: E731

# diffusion run stencils
apply_diffusion_to_vn = CachedProgram(apply_diffusion_to_vn_orig)
apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence = CachedProgram(
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence_orig
)
calculate_diagnostic_quantities_for_turbulence = CachedProgram(
    calculate_diagnostic_quantities_for_turbulence_orig
)
calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools = CachedProgram(
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools_orig
)
calculate_nabla2_and_smag_coefficients_for_vn = CachedProgram(
    calculate_nabla2_and_smag_coefficients_for_vn_orig
)
calculate_nabla2_for_theta = CachedProgram(calculate_nabla2_for_theta_orig)
truly_horizontal_diffusion_nabla_of_theta_over_steep_points = CachedProgram(
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points_orig
)
update_theta_and_exner = CachedProgram(update_theta_and_exner_orig)

mo_intp_rbf_rbf_vec_interpol_vertex = CachedProgram(mo_intp_rbf_rbf_vec_interpol_vertex_orig)


# model init stencils
setup_fields_for_initial_step = CachedProgram(setup_fields_for_initial_step_orig, with_domain=False)
copy_field = CachedProgram(copy_field_orig, with_domain=False)
init_diffusion_local_fields_for_regular_timestep = CachedProgram(
    init_diffusion_local_fields_for_regular_timestep_orig, with_domain=False
)
scale_k = CachedProgram(scale_k_orig, with_domain=False)
