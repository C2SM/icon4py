# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


gt4py_data_key = "compute"
fortran_data_key = "value"

experiment = "mch_icon-ch1_medium_stencils"
output_file = "bench_blueline_stencil_compute"
input_openacc = "/Users/epaone/repo/icon4py/scripts/bencher=exp.mch_icon-ch1_medium_stencils=0.362198=ACC.json"
input_gt4py = {
    # "gtfn_gpu": "/Users/epaone/repo/icon4py/scripts/gtfn_gt4py_timers_compute.json",
    "dace_gpu": "/Users/epaone/repo/icon4py/scripts/gt4py_dace_timers_202051001.json",
}


fortran_to_icon4py = {
    # -- nh_solve.veltend
    "compute_derived_horizontal_winds_and_ke_and_contravariant_correction": "compute_derived_horizontal_winds_and_ke_and_contravariant_correction",
    # "compute_derived_horizontal_winds_and_ke_and_contravariant_correction_skip": "compute_derived_horizontal_winds_and_ke_and_contravariant_correction",
    "compute_contravariant_correction_and_advection_in_vertical_momentum_equation": "compute_contravariant_correction_and_advection_in_vertical_momentum_equation",
    # "compute_contravariant_correction_and_advection_in_vertical_momentum_equation_ski": "compute_contravariant_correction_and_advection_in_vertical_momentum_equation",
    "compute_advection_in_vertical_momentum_equation": "compute_advection_in_vertical_momentum_equation",
    "compute_advection_in_horizontal_momentum_equation": "compute_advection_in_horizontal_momentum_equation",
    # -- nh_solve.cellcomp
    "compute_perturbed_quantities_and_interpolation": "compute_perturbed_quantities_and_interpolation",
    "interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleratio": "interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration",
    # -- nh_solve.edgecomp
    "compute_horizontal_velocity_quantities_and_fluxes": "compute_horizontal_velocity_quantities_and_fluxes",
    "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection": "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection",
    # "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection_first": "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection",
    # -- nh_solve.vnupd
    "compute_theta_rho_face_values_and_pressure_gradient_and_update_vn": "compute_theta_rho_face_values_and_pressure_gradient_and_update_vn",
    "apply_divergence_damping_and_update_vn": "apply_divergence_damping_and_update_vn",
    # -- nh_solve.vimpl
    "compute_dwdz_and_boundary_update_rho_theta_w": "compute_dwdz_and_boundary_update_rho_theta_w",
    "update_mass_flux_weighted": "update_mass_flux_weighted",
    # "update_mass_flux_weighted_first": "update_mass_flux_weighted",
    # -- not categorized
    "vertically_implicit_solver_at_predictor_step": "vertically_implicit_solver_at_predictor_step",
    # "vertically_implicit_solver_at_predictor_step_first": "vertically_implicit_solver_at_predictor_step",
    "vertically_implicit_solver_at_corrector_step": "vertically_implicit_solver_at_corrector_step",
    # "vertically_implicit_solver_at_corrector_step_first": "vertically_implicit_solver_at_corrector_step",
    # "vertically_implicit_solver_at_corrector_step_last": "vertically_implicit_solver_at_corrector_step",
    "rbf_vector_interpolation_of_u_v_vert_before_nabla2": "rbf_vector_interpolation_of_u_v_vert_before_nabla2",
    "calculate_nabla2_and_smag_coefficients_for_vn": "calculate_nabla2_and_smag_coefficients_for_vn",
    "calculate_diagnostic_quantities_for_turbulence": "calculate_diagnostic_quantities_for_turbulence",
    "rbf_vector_interpolation_of_u_v_vert_before_nabla4": "rbf_vector_interpolation_of_u_v_vert_before_nabla4",
    "apply_diffusion_to_vn": "apply_diffusion_to_vn",
    "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence": "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence",
    "calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools": "calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools",
    "apply_diffusion_to_theta_and_exner": "apply_diffusion_to_theta_and_exner",
}


def load_bencher_log(filename: str, t_meas_key: str, experiment: str) -> dict:
    data = {}
    with open(filename) as f:
        d = json.load(f)
        # assert experiment in d
        for stencil, meas in d.items():
            if stencil in fortran_to_icon4py:
                icon4py_stencil = fortran_to_icon4py[stencil]
                t = meas["latency"][t_meas_key]
                data[icon4py_stencil] = t / 1000.0
            else:
                print(f"skipping openacc meas for {stencil}")
    return data


def load_gt4py_timers(filename: str) -> dict:
    with open(filename) as f:
        data = json.load(f)
    return data


columns = ["openacc"]
data = {
    stencil: [t]
    for stencil, t in load_bencher_log(input_openacc, fortran_data_key, experiment).items()
}
data_err = {stencil: [0.0] for stencil, _ in data.items()}

# regex to remove the backend from the stencil name
re_stencil = re.compile("(\S+)\[.+\]")

for backend, filename in input_gt4py.items():
    columns.append(f"{backend}_{gt4py_data_key}")
    d = load_gt4py_timers(filename)
    for name, meas in d.items():
        m = re_stencil.match(name)
        assert m
        stencil = m[1]
        if stencil in data:
            # note that we drop the first measurement
            t = meas[gt4py_data_key][1:]
            data[stencil].append(np.median(t))
            data_err[stencil].append(np.std(t))
        else:
            print(f"skipping gt4py meas for {stencil}")

# keep only stencils that exists both in openacc and gt4py report
data = {k: v for k, v in data.items() if len(v) == len(columns)}
data_err = {k: v for k, v in data_err.items() if len(v) == len(columns)}

df = pd.DataFrame.from_dict(data, orient="index", columns=columns)
err = pd.DataFrame.from_dict(data_err, orient="index", columns=columns)
print(df)

# create a horizontal bar plot
ax = df.plot.barh(
    y=columns,
    # yerr=err,
    capsize=2,  # length of the little caps on the errorbars
    legend=True,
    figsize=(16, 10),
)
plt.title("Average stencil compute time [s] on mch_icon-ch1_medium (1GPU A100)")
plt.tight_layout()
plt.savefig(output_file)

print("")
print(f"Plot figure saved to {output_file}.png")
