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

import datetime
import logging
import math
import os
from cmath import sqrt as c_sqrt
from pathlib import Path

import numpy as np

from icon4py.model.common.config import Device
from icon4py.model.common.decomposition.definitions import (
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.settings import device, xp
from icon4py.model.driver import divergence_utils as div_utils
from icon4py.model.driver.dycore_driver import initialize
from icon4py.model.driver.initialization_utils import (
    WAVETYPE,
    DivWave,
    ExperimentType,
    SerializationType,
    configure_logging,
)
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm


log = logging.getLogger(__name__)


def test_stability_divergence_first_order():

    resolutions = "1000m"

    delta_t = np.array([0.5, 0.999], dtype=float)

    dtime_seconds_1 = 1.0 # for delta_t = 1
    end_time_seconds_1 = 200.0 # 200 time steps for delta_t = 1
    dtime_seconds = delta_t * dtime_seconds_1
    end_time_seconds = np.array(end_time_seconds_1 / delta_t + 1, dtype=int)
    k_cri = (1.0 / 3.0 / math.sqrt(3.0), 4.0 / math.sqrt(3.0)) # 6.0 * math.sqrt(3.0) / 5.0

    font_size = 18

    x_wavenumber_factor = np.array([
        #50.0, 40.0, 30.0, 20.0, 10.0, 5.0,
        #50.0, 48.0, 46.0, 44.0, 42.0, 40.0,
        10.0, 30.0, 50.0,
        # 30.0, 50.0, 70.0,
        #44.4, 33.33, 22.2
        # 40.0, 30.0, 20.0
    ], dtype=float)

    y_wavenumber_factor = np.array([
        10.0, 30.0, 50.0,
        # 30.0, 50.0, 70.0,
    ], dtype=float)

    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + resolutions + ".nc"
    mask = div_utils.create_mask(grid_filename)

    ############# X DIVERGENCE #############
    total_divergence = [ [ [[] for _ in range(x_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    total_windspeed = [ [ [[] for _ in range(x_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    inital_total_divergence = np.zeros((delta_t.shape[0], x_wavenumber_factor.shape[0]), dtype = float)
    inital_total_windspeed = np.zeros((delta_t.shape[0], x_wavenumber_factor.shape[0]), dtype = float)

    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(x_wavenumber_factor.shape[0]):
                base_input_path = "/scratch/mch/cong/data/div_converge_res"
                run_path = "./"
                experiment_type = ExperimentType.DIVCONVERGE
                serialization_type = SerializationType.SB
                div_wave = DivWave(wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=x_wavenumber_factor[x], y_wavenumber_factor=0.0)
                log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
                div_utils.print_config(div_wave)

                disable_logging = True
                mpi = False
                profile = False
                grid_root = 0
                grid_level = 2
                enable_output = False
                enable_debug_message = False

                initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
                input_path = base_input_path + resolutions + "/ser_data/"

                parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
                configure_logging(run_path, experiment_type, parallel_props, disable_logging)

                end_date = initial_date + datetime.timedelta(seconds=int(end_time_seconds[t]))

                log.info(
                    f" running delta_t = {t}, x_wavenumber = {x_wavenumber_factor[x]}, end_date = {end_date}"
                )

                do_3d_divergence_damping = True if order == 1 else False
                (
                    timeloop,
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    output_state,
                    prep_adv,
                    inital_divdamp_fac_o2,
                ) = initialize(
                    Path(input_path),
                    parallel_props,
                    serialization_type,
                    experiment_type,
                    grid_root,
                    grid_level,
                    enable_output,
                    enable_debug_message,
                    div_wave,
                    dtime_seconds=1.0,
                    end_date=end_date,
                    output_seconds_interval=100000.0,
                    do_o2_divdamp=True,
                    do_3d_divergence_damping=do_3d_divergence_damping,
                    divergence_order=order + 1,
                    divdamp_fac=k_cri[order] * delta_t[t],
                )

                cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
                cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
                edge_length = timeloop.solve_nonhydro.edge_geometry.primal_edge_lengths.asnumpy()
                cell_area = timeloop.solve_nonhydro.cell_params.area.asnumpy()

                initial_vn = np.array(prognostic_state_list[0].vn.asnumpy(), copy=True)
                
                analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
                    cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
                )

                if device == Device.GPU:
                    analytic_divergence = analytic_divergence.get()

                inital_total_divergence[t, x] = np.sum(np.abs(analytic_divergence) * cell_area) / np.sum(cell_area)
                inital_total_windspeed[t, x] = np.sum(np.abs(initial_vn[:, 0]) * edge_length) / np.sum(edge_length)
        
                timeloop.time_integration(
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    prep_adv,
                    inital_divdamp_fac_o2,
                    do_prep_adv=False,
                    output_state=output_state,
                    profile=profile,
                )

                if order == 1:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence) # / 3000.0 # height of the domain
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed) # / 3000.0 # height of the domain
                        )
                    )
                else:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)
                        )
                    )
                
                # div_utils.plot_tridata(
                #     grid_filename,
                #     diagnostic_state.u.ndarray[:, 0],
                #     mask,
                #     f"U at order={order}, dt={delta_t[t]}, xwave={x_wavenumber_factor[x]}",
                #     f"plot_u_order_{order}_dt_{t}_xwave_{int(x_wavenumber_factor[x])}",
                # )
                # div_utils.plot_tridata(
                #     grid_filename,
                #     timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0],
                #     mask,
                #     f"Div at order={order}, dt={delta_t[t]}, xwave={x_wavenumber_factor[x]}",
                #     f"plot_div_order_{order}_dt_{t}_xwave_{int(x_wavenumber_factor[x])}",
                # )


    time_xaxis = [np.linspace(0.0, dtime_seconds[t] * end_time_seconds[t], end_time_seconds[t] + 1) for t in range(delta_t.shape[0])]

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # line_label = ["K_cri/5", "K_cri/4", "K_cri/3", "K_cri/2", "K_cri"]
    line_label = [
        f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 0.5$", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 0.5$", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 0.5$",
        f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 1$", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 1$", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 1$",
        f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 0.5$", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 0.5$", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 0.5$",
        f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 1$", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 1$", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 1$",
    ]
    dashdotdotted = ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5)))
    line_style = [ ["solid", "dashed"], ["solid", "dashed"] ]
    line_color = ["green", "red", "blue"]
    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(x_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_divergence[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # labels[0] = ""
        # for i in range(max_xaxis_value):
        #     labels[i+1] = axis_labels[i]
        # ax.set_xticks(dx, axis_labels)
        # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean divergence residual (s$^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        # ax.set_title("Mean error in divergence")
        ax[order].legend(fontsize=font_size-5)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_divergence_x.pdf", format="pdf", dpi=500)

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(x_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean windspeed residual ($m s^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        ax[order].set_ylim(1.e-17, 1.e-1)
        ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # ax.set_title("Mean error in windspeed")
        ax[order].legend(fontsize=font_size-5)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_windspeed_x.pdf", format="pdf", dpi=500)



    ############# Y DIVERGENCE #############
    total_divergence = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    total_windspeed = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    inital_total_divergence = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)
    inital_total_windspeed = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)

    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                base_input_path = "/scratch/mch/cong/data/div_converge_res"
                run_path = "./"
                experiment_type = ExperimentType.DIVCONVERGE
                serialization_type = SerializationType.SB
                div_wave = DivWave(wave_type=WAVETYPE.Y_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=y_wavenumber_factor[x])
                log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
                div_utils.print_config(div_wave)

                disable_logging = True
                mpi = False
                profile = False
                grid_root = 0
                grid_level = 2
                enable_output = False
                enable_debug_message = False

                initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
                input_path = base_input_path + resolutions + "/ser_data/"

                parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
                configure_logging(run_path, experiment_type, parallel_props, disable_logging)

                end_date = initial_date + datetime.timedelta(seconds=int(end_time_seconds[t]))

                log.info(
                    f" running delta_t = {t}, y_wavenumber = {y_wavenumber_factor[x]}, end_date = {end_date}"
                )

                do_3d_divergence_damping = True if order == 1 else False
                (
                    timeloop,
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    output_state,
                    prep_adv,
                    inital_divdamp_fac_o2,
                ) = initialize(
                    Path(input_path),
                    parallel_props,
                    serialization_type,
                    experiment_type,
                    grid_root,
                    grid_level,
                    enable_output,
                    enable_debug_message,
                    div_wave,
                    dtime_seconds=1.0,
                    end_date=end_date,
                    output_seconds_interval=100000.0,
                    do_o2_divdamp=True,
                    do_3d_divergence_damping=do_3d_divergence_damping,
                    divergence_order=order + 1,
                    divdamp_fac=k_cri[order] * delta_t[t],
                )

                cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
                cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
                edge_length = timeloop.solve_nonhydro.edge_geometry.primal_edge_lengths.asnumpy()
                cell_area = timeloop.solve_nonhydro.cell_params.area.asnumpy()

                initial_vn = np.array(prognostic_state_list[0].vn.asnumpy(), copy=True)
                
                analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
                    cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
                )

                if device == Device.GPU:
                    analytic_divergence = analytic_divergence.get()

                inital_total_divergence[t, x] = np.sum(np.abs(analytic_divergence) * cell_area) / np.sum(cell_area)
                inital_total_windspeed[t, x] = np.sum(np.abs(initial_vn[:, 0]) * edge_length) / np.sum(edge_length)
        
                timeloop.time_integration(
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    prep_adv,
                    inital_divdamp_fac_o2,
                    do_prep_adv=False,
                    output_state=output_state,
                    profile=profile,
                )

                if order == 1:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)# / 3000.0 # height of the domain
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)# / 3000.0 # height of the domain
                        )
                    )
                else:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)
                        )
                    )
                
                # div_utils.plot_tridata(
                #     grid_filename,
                #     diagnostic_state.u.ndarray[:, 0],
                #     mask,
                #     f"U at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_u_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )
                # div_utils.plot_tridata(
                #     grid_filename,
                #     timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0],
                #     mask,
                #     f"Div at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_div_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )


    time_xaxis = [np.linspace(0.0, dtime_seconds[t] * end_time_seconds[t], end_time_seconds[t] + 1) for t in range(delta_t.shape[0])]

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # line_label = ["K_cri/5", "K_cri/4", "K_cri/3", "K_cri/2", "K_cri"]
    line_label = [
        f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
        f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
        f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
        f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    ]
    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_divergence[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # labels[0] = ""
        # for i in range(max_xaxis_value):
        #     labels[i+1] = axis_labels[i]
        # ax.set_xticks(dx, axis_labels)
        # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean divergence residual (s$^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        # ax.set_title("Mean error in divergence")
        ax[order].legend(fontsize=font_size-5)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_divergence_y.pdf", format="pdf", dpi=500)

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean windspeed residual ($m s^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        ax[order].set_ylim(1.e-17, 1.e-1)
        ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # ax.set_title("Mean error in windspeed")
        ax[order].legend(fontsize=font_size-5)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_windspeed_y.pdf", format="pdf", dpi=500)


    ############# X AND Y DIVERGENCE #############
    total_divergence = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    total_windspeed = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    inital_total_divergence = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)
    inital_total_windspeed = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)

    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                base_input_path = "/scratch/mch/cong/data/div_converge_res"
                run_path = "./"
                experiment_type = ExperimentType.DIVCONVERGE
                serialization_type = SerializationType.SB
                div_wave = DivWave(wave_type=WAVETYPE.X_AND_Y_WAVE, x_wavenumber_factor=x_wavenumber_factor[x], y_wavenumber_factor=y_wavenumber_factor[x])
                log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
                div_utils.print_config(div_wave)

                disable_logging = True
                mpi = False
                profile = False
                grid_root = 0
                grid_level = 2
                enable_output = False
                enable_debug_message = False

                initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
                input_path = base_input_path + resolutions + "/ser_data/"

                parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
                configure_logging(run_path, experiment_type, parallel_props, disable_logging)

                end_date = initial_date + datetime.timedelta(seconds=int(end_time_seconds[t]))

                log.info(
                    f" running delta_t = {t}, y_wavenumber = {y_wavenumber_factor[x]}, end_date = {end_date}"
                )

                do_3d_divergence_damping = True if order == 1 else False
                (
                    timeloop,
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    output_state,
                    prep_adv,
                    inital_divdamp_fac_o2,
                ) = initialize(
                    Path(input_path),
                    parallel_props,
                    serialization_type,
                    experiment_type,
                    grid_root,
                    grid_level,
                    enable_output,
                    enable_debug_message,
                    div_wave,
                    dtime_seconds=1.0,
                    end_date=end_date,
                    output_seconds_interval=100000.0,
                    do_o2_divdamp=True,
                    do_3d_divergence_damping=do_3d_divergence_damping,
                    divergence_order=order + 1,
                    divdamp_fac=k_cri[order] * delta_t[t],
                )

                cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
                cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
                edge_length = timeloop.solve_nonhydro.edge_geometry.primal_edge_lengths.asnumpy()
                cell_area = timeloop.solve_nonhydro.cell_params.area.asnumpy()

                initial_vn = np.array(prognostic_state_list[0].vn.asnumpy(), copy=True)
                
                analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
                    cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
                )

                if device == Device.GPU:
                    analytic_divergence = analytic_divergence.get()

                inital_total_divergence[t, x] = np.sum(np.abs(analytic_divergence) * cell_area) / np.sum(cell_area)
                inital_total_windspeed[t, x] = np.sum(np.abs(initial_vn[:, 0]) * edge_length) / np.sum(edge_length)
        
                timeloop.time_integration(
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    prep_adv,
                    inital_divdamp_fac_o2,
                    do_prep_adv=False,
                    output_state=output_state,
                    profile=profile,
                )

                if order == 1:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)# / 3000.0 # height of the domain
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)# / 3000.0 # height of the domain
                        )
                    )
                else:
                    total_divergence[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)
                        )
                    )
                    total_windspeed[order][t][x] = np.concatenate(
                        (
                            [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)
                        )
                    )
                
                # div_utils.plot_tridata(
                #     grid_filename,
                #     diagnostic_state.u.ndarray[:, 0],
                #     mask,
                #     f"U at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_u_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )
                # div_utils.plot_tridata(
                #     grid_filename,
                #     timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0],
                #     mask,
                #     f"Div at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_div_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )


    time_xaxis = [np.linspace(0.0, dtime_seconds[t] * end_time_seconds[t], end_time_seconds[t] + 1) for t in range(delta_t.shape[0])]

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # line_label = ["K_cri/5", "K_cri/4", "K_cri/3", "K_cri/2", "K_cri"]
    line_label = [
        f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
        f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
        f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
        f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    ]
    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_divergence[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
        # labels = [item.get_text() for item in ax.get_xticklabels()]
        # labels[0] = ""
        # for i in range(max_xaxis_value):
        #     labels[i+1] = axis_labels[i]
        # ax.set_xticks(dx, axis_labels)
        # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean divergence residual (s$^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        # ax.set_title("Mean error in divergence")
        ax[order].legend(fontsize=font_size)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_divergence_xy.pdf", format="pdf", dpi=500)

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    counter = 0
    for order in range(2):
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
                counter = counter + 1
    for order in range(2):
        ax[order].set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean windspeed residual ($m s^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        ax[order].set_ylim(1.e-17, 1.e-1)
        ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # ax.set_title("Mean error in windspeed")
        ax[order].legend(fontsize=font_size)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_windspeed_xy.pdf", format="pdf", dpi=500)


def test_order2_div_y_damping():
    resolutions = "1000m"

    delta_t = np.array([0.5], dtype=float)

    dtime_seconds_1 = 1.0 # for delta_t = 1
    # end_time_seconds_1 = 9.0 # 200 time steps for delta_t = 1
    # dtime_seconds = delta_t * dtime_seconds_1
    # end_time_seconds = np.array(end_time_seconds_1 / delta_t + 1, dtype=int)
    k_cri = (1.0 / 3.0 / math.sqrt(3.0), 4.0 / math.sqrt(3.0)) # 6.0 * math.sqrt(3.0) / 5.0

    font_size = 18

    y_wavenumber_factor = np.array([
        10.0,
    ], dtype=float)

    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + resolutions + ".nc"
    mask = div_utils.create_mask(grid_filename)

    total_divergence = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    total_windspeed = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    inital_total_divergence = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)
    inital_total_windspeed = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)

    for end_second in range(15):
        # if end_second != 99:
        #     continue
        end_time_seconds_1 = end_second + 1.0 # 200 time steps for delta_t = 1
        dtime_seconds = delta_t * dtime_seconds_1
        end_time_seconds = np.array(end_time_seconds_1 / delta_t + 1, dtype=int)
        for t in range(delta_t.shape[0]):
            for x in range(y_wavenumber_factor.shape[0]):
                base_input_path = "/scratch/mch/cong/data/div_converge_res"
                run_path = "./"
                experiment_type = ExperimentType.DIVCONVERGE
                serialization_type = SerializationType.SB
                div_wave = DivWave(wave_type=WAVETYPE.Y_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=y_wavenumber_factor[x])
                log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
                div_utils.print_config(div_wave)

                disable_logging = True
                mpi = False
                profile = False
                grid_root = 0
                grid_level = 2
                enable_output = False
                enable_debug_message = False

                initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
                input_path = base_input_path + resolutions + "/ser_data/"

                parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
                configure_logging(run_path, experiment_type, parallel_props, disable_logging)

                end_date = initial_date + datetime.timedelta(seconds=int(end_time_seconds[t]))

                log.info(
                    f" running delta_t = {t}, y_wavenumber = {y_wavenumber_factor[x]}, end_date = {end_date}"
                )

                do_3d_divergence_damping = True
                (
                    timeloop,
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    output_state,
                    prep_adv,
                    inital_divdamp_fac_o2,
                ) = initialize(
                    Path(input_path),
                    parallel_props,
                    serialization_type,
                    experiment_type,
                    grid_root,
                    grid_level,
                    enable_output,
                    enable_debug_message,
                    div_wave,
                    dtime_seconds=1.0,
                    end_date=end_date,
                    output_seconds_interval=100000.0,
                    do_o2_divdamp=True,
                    do_3d_divergence_damping=do_3d_divergence_damping,
                    divergence_order=1,
                    divdamp_fac=k_cri[0] * delta_t[t],
                )

                cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
                cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
                edge_length = timeloop.solve_nonhydro.edge_geometry.primal_edge_lengths.asnumpy()
                cell_area = timeloop.solve_nonhydro.cell_params.area.asnumpy()

                initial_vn = np.array(prognostic_state_list[0].vn.asnumpy(), copy=True)
                
                analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
                    cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
                )

                if device == Device.GPU:
                    analytic_divergence = analytic_divergence.get()

                inital_total_divergence[t, x] = np.sum(np.abs(analytic_divergence) * cell_area) / np.sum(cell_area)
                inital_total_windspeed[t, x] = np.sum(np.abs(initial_vn[:, 0]) * edge_length) / np.sum(edge_length)

                eigen_value, eigen_vector, eigen_vector_cartesian = div_utils.eigen_divergence(
                    divergence_factor=0.0,
                    order=timeloop.solve_nonhydro.config.divergence_order,
                    grid_space=1000.0,
                    div_wave=div_wave,
                )
                damping_eigen_vector = eigen_vector[np.argmax(np.abs(eigen_value))]
        
                div_utils.plot_triedgedata(
                    grid_filename,
                    prognostic_state_list[0].vn.ndarray,
                    f"VN",
                    "plot_finalfirst"+str(end_second+1)+"_vn",
                    div_wave,
                    eigen_vector=damping_eigen_vector,
                    plot_analytic=False,
                )
                timeloop.time_integration(
                    diffusion_diagnostic_state,
                    solve_nonhydro_diagnostic_state,
                    diagnostic_metric_state,
                    diagnostic_state,
                    prognostic_state_list,
                    prep_adv,
                    inital_divdamp_fac_o2,
                    do_prep_adv=False,
                    output_state=output_state,
                    profile=profile,
                )

                div_utils.plot_triedgedata(
                    grid_filename,
                    prognostic_state_list[0].vn.ndarray,
                    f"VN",
                    "plot_final"+str(end_second+1)+"_vn",
                    div_wave,
                    eigen_vector=damping_eigen_vector,
                    plot_analytic=False,
                )
                div_utils.plot_triedgedata(
                    grid_filename,
                    prognostic_state_list[1].vn.ndarray,
                    f"VN",
                    "plot_finalold"+str(end_second+1)+"_vn",
                    div_wave,
                    eigen_vector=damping_eigen_vector,
                    plot_analytic=False,
                )
                div_utils.plot_tridata(
                    grid_filename,
                    timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray,
                    mask,
                    f"Computed divergence",
                    "plot_computed_divergence"+str(end_second+1),
                )
                # div_utils.plot_tridata(
                #     grid_filename,
                #     diagnostic_state.u.ndarray[:, 0],
                #     mask,
                #     f"U at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_u_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )
                # div_utils.plot_tridata(
                #     grid_filename,
                #     timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0],
                #     mask,
                #     f"Div at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
                #     f"plot_div_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
                # )



def test_debug():
    resolutions = "1000m"

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = False
    enable_debug_message = False

    dtime_seconds = 1.0
    initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_time = 1000 * dtime_seconds
    mean_error = np.zeros(len(resolutions), dtype=float)
    input_path = base_input_path + resolutions + "/ser_data/"
    grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + resolutions + ".nc"

    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    configure_logging(run_path, experiment_type, parallel_props, disable_logging)

    end_date = initial_date + datetime.timedelta(seconds=end_time)

    div_wave1 = DivWave(wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=10.0, y_wavenumber_factor=0.0)
    (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        diagnostic_state1,
        prognostic_state_list1,
        output_state,
        prep_adv,
        inital_divdamp_fac_o2,
    ) = initialize(
        Path(input_path),
        parallel_props,
        serialization_type,
        experiment_type,
        grid_root,
        grid_level,
        enable_output,
        enable_debug_message,
        div_wave1,
        dtime_seconds=dtime_seconds,
        end_date=end_date,
        output_seconds_interval=1000 * dtime_seconds,
        do_o2_divdamp=True,
        do_3d_divergence_damping=False,
        divergence_order=1,
        divdamp_fac=1.0 / 3.0 / math.sqrt(3.0) / 2.0,
    )
    div_wave2 = DivWave(wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=20.0, y_wavenumber_factor=0.0)
    (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        diagnostic_state2,
        prognostic_state_list2,
        output_state,
        prep_adv,
        inital_divdamp_fac_o2,
    ) = initialize(
        Path(input_path),
        parallel_props,
        serialization_type,
        experiment_type,
        grid_root,
        grid_level,
        enable_output,
        enable_debug_message,
        div_wave2,
        dtime_seconds=dtime_seconds,
        end_date=end_date,
        output_seconds_interval=1000 * dtime_seconds,
        do_o2_divdamp=True,
        do_3d_divergence_damping=False,
        divergence_order=1,
        divdamp_fac=1.0 / 3.0 / math.sqrt(3.0) / 2.0,
    )

    cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
    cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
    analytic_divergence1 = div_utils.determine_divergence_in_div_converge_experiment(
        cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave1
    )
    analytic_divergence2 = div_utils.determine_divergence_in_div_converge_experiment(
        cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave2
    )

    diff1_divergence = analytic_divergence1 - analytic_divergence2

    mask = div_utils.create_mask(grid_filename)

    initial_vn1 = xp.array(prognostic_state_list1[0].vn.ndarray, copy=True)
    initial_vn2 = xp.array(prognostic_state_list2[0].vn.ndarray, copy=True)

    div_utils.plot_tridata(
        grid_filename,
        diff1_divergence,
        mask,
        f"Analytic divergence",
        "plot_analytic_divergence",
    )

    div_utils.plot_triedgedata(
        grid_filename,
        initial_vn1 - initial_vn2,
        f"VN",
        "plot_initial_vn",
        div_wave2,
        eigen_vector=None,
        plot_analytic=True,
    )
    div_utils.plot_tridata(
        grid_filename,
        diagnostic_state1.u.ndarray - diagnostic_state2.u.ndarray,
        mask,
        f"U",
        "plot_initial_U",
    )
    div_utils.plot_tridata(
        grid_filename,
        diagnostic_state1.u.ndarray - float(xp.abs(diagnostic_state1.u.ndarray).max()),
        mask,
        f"U1",
        "plot_initial_U1",
    )
    div_utils.plot_tridata(
        grid_filename,
        diagnostic_state2.u.ndarray - float(xp.abs(diagnostic_state2.u.ndarray).max()),
        mask,
        f"U1",
        "plot_initial_U2",
    )
    div_utils.plot_tridata(
        grid_filename,
        diagnostic_state1.v.ndarray - diagnostic_state2.v.ndarray,
        mask,
        f"V",
        "plot_initial_V",
    )
    div_utils.plot_tridata(
        grid_filename,
        cell_lon,
        mask,
        f"LON",
        "plot_initial_lon",
    )

if __name__ == "__main__":
    test_stability_divergence_first_order()
    # test_order2_div_y_damping()
    # test_debug()