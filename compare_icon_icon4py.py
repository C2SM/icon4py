import json
import pathlib
from collections import defaultdict

import numpy as np


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


def load_bencher_log(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_gt4py_timers(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def main():
    fortran_data = load_bencher_log(
        pathlib.Path("bencher_mch_icon-ch1_medium_stencils_3275369_OPENACC.json")  # TODO
    )
    gt4py_data = load_gt4py_timers("gt4py_timers.json")  # TODO
    backend = "[run_gtfn_gpu_cached]"
    gt4py_data_key = "compute"
    print(gt4py_data.keys())
    results = defaultdict(dict)
    notfound = set()
    for fortran_name, icon4py_name in fortran_to_icon4py.items():
        if fortran_name in fortran_data["mch_icon-ch1_medium_stencils"]:
            fortran_values = fortran_data["mch_icon-ch1_medium_stencils"][fortran_name]
            icon4py_name = icon4py_name + backend
            if icon4py_name not in gt4py_data:
                notfound.add(icon4py_name)
            else:
                data = np.asarray(gt4py_data[icon4py_name][gt4py_data_key])
                mean = np.mean(data)
                min = np.min(data)
                max = np.max(data)
                # std = np.std(data)
                # results[fortran_name]["acc"] = fortran_values[""]
                print(f"{fortran_name:100} {min:10.6f} {mean:10.6f} {max:10.6f}")
                print(
                    f"{(fortran_name + '[acc]'):100} {fortran_values['lower_value']:10.6f} {fortran_values['value']:10.6f} {fortran_values['upper_value']:10.6f}"
                )
                print("----------------")

    # print("Remaining keys:")
    # print(list(fortran_data["mch_icon-ch1_medium_stencils"].keys()))
    print("Not found in gt4py data:")
    print(notfound)


if __name__ == "__main__":
    main()
