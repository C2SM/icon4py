# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


experiment = "mch_icon-ch1_medium"
target = "GH200 1-rank"

openacc_backend = "openacc"
output_filename = "bench_blueline_stencil_compute"

# the default 'file_prefix' assumes that the json files are in the script folder
file_prefix = pathlib.Path(__file__).parent
openacc_input = file_prefix / "bencher=exp.mch_icon-ch1_medium_stencils=0.373574=ACC.json"
gt4py_input = {
    "gtfn_gpu": file_prefix / "gt4py_gtfn_timers_202051008_G-89541209_I-dda0d1872.json",
    "dace_gpu": file_prefix / "gt4py_dace_timers_202051008_G-89541209_I-dda0d1872.json",
}
gt4py_metrics = ["compute"]  # here we can add other metrics, e.g. 'total'

# Mapping from fortran stencil metric to gt4py stencils. The mapped value is a list,
# because one stencil in Fortran could correspond to multiple stencils in ICON4Py,
# if they are not fused in a combined stencil.
fortran_to_icon4py = {
    # -- total
    # -- integrate_nh
    # -- nh_solve
    # -- nh_solve.veltend
    "compute_derived_horizontal_winds_and_ke_and_contravariant_correction": [
        "compute_derived_horizontal_winds_and_ke_and_contravariant_correction"
    ],
    # -- compute_derived_horizontal_winds_and_ke_and_contravariant_correction_skip
    "compute_contravariant_correction_and_advection_in_vertical_momentum_equation": [
        "compute_contravariant_correction_and_advection_in_vertical_momentum_equation"
    ],
    # -- compute_contravariant_correction_and_advection_in_vertical_momentum_equation_ski
    "compute_advection_in_vertical_momentum_equation": [
        "compute_advection_in_vertical_momentum_equation"
    ],
    "compute_advection_in_horizontal_momentum_equation": [
        "compute_advection_in_horizontal_momentum_equation"
    ],
    # -- nh_solve.cellcomp
    "compute_perturbed_quantities_and_interpolation": [
        "compute_perturbed_quantities_and_interpolation"
    ],
    # -- interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleratio
    # -- nh_solve.edgecomp
    "compute_horizontal_velocity_quantities_and_fluxes": [
        "compute_horizontal_velocity_quantities_and_fluxes"
    ],
    "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection": [
        "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection"
    ],
    # -- compute_averaged_vn_and_fluxes_and_prepare_tracer_advection_first
    # -- nh_solve.vnupd
    "compute_theta_rho_face_values_and_pressure_gradient_and_update_vn": [
        "compute_theta_rho_face_values_and_pressure_gradient_and_update_vn"
    ],
    "apply_divergence_damping_and_update_vn": ["apply_divergence_damping_and_update_vn"],
    # -- nh_solve.vimpl
    # -- compute_dwdz_and_boundary_update_rho_theta_w
    "update_mass_flux_weighted": ["update_mass_flux_weighted"],
    # -- update_mass_flux_weighted_first
    # -- nh_solve.exch
    "boundary_halo_cleanup": [
        "compute_exner_from_rhotheta",
        "compute_theta_and_exner",
        "update_theta_v",
    ],
    # -- nh_hdiff_initial_run
    # -- nh_hdiff
    # -- transport
    # -- adv_horiz
    # -- adv_hflx
    # -- back_traj
    # -- adv_vert
    # -- adv_vflx
    # -- action
    # -- global_sum
    # -- wrt_output
    # -- wait_for_async_io
    "vertically_implicit_solver_at_predictor_step": [
        "vertically_implicit_solver_at_predictor_step"
    ],
    # -- vertically_implicit_solver_at_predictor_step_first
    "vertically_implicit_solver_at_corrector_step": [
        "vertically_implicit_solver_at_corrector_step"
    ],
    # -- vertically_implicit_solver_at_corrector_step_first
    # -- vertically_implicit_solver_at_corrector_step_last
    # -- rbf_vector_interpolation_of_u_v_vert_before_nabla2
    "calculate_nabla2_and_smag_coefficients_for_vn": [
        "calculate_nabla2_and_smag_coefficients_for_vn"
    ],
    "calculate_diagnostic_quantities_for_turbulence": [
        "calculate_diagnostic_quantities_for_turbulence"
    ],
    # -- rbf_vector_interpolation_of_u_v_vert_before_nabla4
    "apply_diffusion_to_vn": ["apply_diffusion_to_vn"],
    "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence": [
        "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence"
    ],
    "calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools": [
        "calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools"
    ],
    "apply_diffusion_to_theta_and_exner": ["apply_diffusion_to_theta_and_exner"],
    # -- physics
    # -- nwp_radiation
    # -- preradiaton
    # -- phys_acc_sync
    # -- ordglb_sum
    # -- satad
    # -- phys_u_v
    # -- nwp_turbulence
    # -- nwp_turbtrans
    # -- nwp_turbdiff
    # -- nwp_surface
    # -- nwp_microphysics
    # -- rediag_prog_vars
    # -- sso
    # -- cloud_cover
    # -- radheat
    # -- nh_diagnostics
    # -- diagnose_pres_temp
    # -- model_init
    # -- compute_domain_decomp
    # -- compute_intp_coeffs
    # -- init_ext_data
    # -- init_icon
    # -- init_latbc
    # -- init_nwp_phy
    # -- upper_atmosphere
    # -- upatmo_construction
    # -- upatmo_destruction
    # -- write_restart
    # -- write_restart_io
    # -- write_restart_communication
    # -- optional_diagnostics_atmosphere
}
# The mapping below allows to add the time spent in different variants to the time spent in the main stencil.
# Note that this just a workaround until GT4Py can report metrics for each variant.
fortran_combined_metrics = {
    "compute_derived_horizontal_winds_and_ke_and_contravariant_correction_skip": "compute_derived_horizontal_winds_and_ke_and_contravariant_correction",
    "compute_contravariant_correction_and_advection_in_vertical_momentum_equation_ski": "compute_contravariant_correction_and_advection_in_vertical_momentum_equation",
    "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection_first": "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection",
    "vertically_implicit_solver_at_predictor_step_first": "vertically_implicit_solver_at_predictor_step",
    "vertically_implicit_solver_at_corrector_step_first": "vertically_implicit_solver_at_corrector_step",
    "vertically_implicit_solver_at_corrector_step_last": "vertically_implicit_solver_at_corrector_step",
}

icon4py_stencils_ = []
for v in fortran_to_icon4py.values():
    icon4py_stencils_.extend(v)
mapped_icon4py_stencils = set(icon4py_stencils_)
assert len(icon4py_stencils_) == len(mapped_icon4py_stencils)

log = logging.getLogger(__name__)


def load_openacc_log(filename: pathlib.Path) -> dict:
    with filename.open("r") as f:
        j = json.load(f)
    data = {}
    count = {}
    for stencil, meas in j.items():
        t = meas["latency_total"]["value"] / 1000.0  # milliseconds to seconds
        ncalls = meas["num_calls"]["value"]
        if stencil in fortran_to_icon4py:
            data[stencil] = t
            count[stencil] = ncalls
        elif stencil in fortran_combined_metrics:
            main_stencil = fortran_combined_metrics[stencil]
            assert main_stencil in data  # main stencil should be processed first
            data[main_stencil] += t
            count[main_stencil] += ncalls
        else:
            log.warning(f"skipping openacc meas for {stencil}")
    return data, count


def load_gt4py_timers(filename: pathlib.Path, metric: str) -> dict:
    with filename.open("r") as f:
        j = json.load(f)

    # regex to remove the backend from the stencil name
    re_stencil = re.compile("(\S+)\[.+\]")

    jj = {}
    for k, v in j.items():
        # remove the backend form the stencil name
        m = re_stencil.match(k)
        assert m is not None
        stencil = m[1]
        if stencil in mapped_icon4py_stencils:
            jj[stencil] = v[metric]
        else:
            log.warning(f"skipping gt4py meas for {stencil}")

    data = {}
    for stencil, icon4py_val in fortran_to_icon4py.items():
        assert isinstance(icon4py_val, list)
        assert len(icon4py_val) > 0
        if len(icon4py_val) == 1:
            # 1-to-1 mapping from fortran to gt4py stencil
            s = icon4py_val[0]
            metric_data = jj[s]
        else:
            # multiple gt4py stencils are summed into the same fortran stencil
            combined_stencil_data = zip(
                *[jj[s] for s in icon4py_val],
                strict=True,
            )
            metric_data = [np.sum(v) for v in combined_stencil_data]
        # we replace the first measurement with the median value
        metric_data[0] = np.median(metric_data)
        data[stencil] = metric_data
    return data


openacc_meas, openacc_count = load_openacc_log(openacc_input)

# Sort stencil names in descendent order of openacc total time.
stencil_names = [v[0] for v in sorted(openacc_meas.items(), key=lambda x: x[1], reverse=True)]

backends = [openacc_backend]
data = {openacc_backend: [openacc_meas[stencil] for stencil in stencil_names]}
for backend, filename in gt4py_input.items():
    for metric in gt4py_metrics:
        # create a unique name for the combination of backend and metric
        name = f"{backend}_{metric}" if len(gt4py_metrics) > 1 else backend
        backends.append(name)
        gt4py_meas = load_gt4py_timers(filename, metric)
        values = []
        for stencil in stencil_names:
            tvalues = gt4py_meas[stencil]
            if len(tvalues) != openacc_count[stencil]:
                log.error(
                    f"Mismatch number of calls on {stencil} {openacc_backend}={openacc_count[stencil]} {name}={len(tvalues)}."
                )
            values.append(np.sum(tvalues))
        data[name] = values


# Combine all bar plots in a single plot
fig, ax = plt.subplots(figsize=(20, 12))
fig.subplots_adjust(left=0.3, right=0.98)
bar_width = 0.5
spacing = 1.0  # Additional spacing between stencil names
index = np.arange(len(stencil_names)) * (bar_width * len(backends) + spacing)

# Define base RGB colors for different backends
base_colors = [
    (0.1, 0.2, 0.5),  # Example RGB color 1
    (0.2, 0.6, 0.3),  # Example RGB color 2
    (0.8, 0.4, 0.1),  # Example RGB color 3
    (0.5, 0.1, 0.7),  # Example RGB color 4
    (0.3, 0.3, 0.3),  # Example RGB color 5
]

if len(base_colors) < len(backends):
    raise ValueError("Not enough base colors defined for the different backends.")

for i, backend in enumerate(backends):
    color = base_colors[i]
    values = data[backend]
    ax.barh(index + i * bar_width, values, bar_width, label=backend, color=color)
    if i > 0:  # Only annotate bars for gt4py backends
        ratios = [
            val / openacc_meas[stencil] for stencil, val in zip(stencil_names, values, strict=True)
        ]
        for k, (val, ratio) in enumerate(zip(values, ratios)):
            ax.text(
                val + 0.05,  # Position slightly above the bar
                index[k] + (i - 0.5) * bar_width,
                f"{ratio:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

ax.set_title(f"Backend comparison on {experiment} ({target})")
ax.set_xlabel("Total compute time [s]")
ax.set_ylabel(f"Stencil name (speedup w.r.t. {openacc_backend} next to the bars)")
ax.set_yticks(index + (len(backends) * bar_width) / 2 - bar_width / 2)
ax.set_yticklabels(stencil_names, rotation=0)

ax.legend(loc="upper right")

# Save the plot to a file
output_dir = pathlib.Path.cwd() / "plots"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"{output_filename}.png"
plt.savefig(output_file, bbox_inches="tight")
plt.close()

print("")
print(f"Plot figure saved to {output_file}")
