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

    #delta_t = np.array([0.5, 0.999], dtype=float)
    delta_t = np.array([0.999,], dtype=float) # 0.642857142857143
    # delta_t = np.array([0.5, 0.5 * 3.0 / 2.0], dtype=float)

    dtime_seconds_1 = 1.0 # for delta_t = 1
    end_time_seconds_1 = 200.0 # 200 time steps for delta_t = 1
    dtime_seconds = delta_t * dtime_seconds_1
    end_time_seconds = np.array(end_time_seconds_1 / delta_t + 1, dtype=int)
    # k_cri = (1.0 / 3.0 / math.sqrt(3.0), 4.0 / math.sqrt(3.0)) # 6.0 * math.sqrt(3.0) / 5.0
    k_cri = (0.75 * 1.0 / 3.0 / math.sqrt(3.0), 1.0 / 4.0 / 2.0 * 4.5 * 4.0 / math.sqrt(3.0))

    damping_lambda_order1_x = (
        -0.048943483704848,
        -0.2928932188134524,
        -0.412214747707527,
        -0.4909585842496289,
        -1.2181432413965425,
        -1.48175367410172,
        -1.9995065603657316
    ) # 25: -0.2928932188134524 -2.7071067811865475, 57: -1.7818567586034575 -1.2181432413965425
    damping_lambda_order1_y = (-0.048940761789931, -0.29215196867478643, -0.36105867943011793, -0.409855104627015,-0.9992009155125331, -0.908869280230764) # 25: -0.29215196867478643 -2.707848031325214, 57: -2.000799084487467 -0.9992009155125331
    damping_lambda_order2_x = (-0.824429495415054, -3.414213562373095, -3.90211303259031, -3.999013120731463, -0.7741858926940477, -0.003946543143457, -3.99112392920616)
    damping_lambda_order2_y = (-0.824844156194135, -3.49493046454979, -3.9011720480508587, -4.11558300969176, -4.0015981669331415, -4.17909620731023)
    theoretical_k_order1 = 0.5 #2.0 / 3.0
    theoretical_k_order2 = 1.0 / 4.0 # 2.0 / 4.5
    font_size = 18

    # minimum x wavelength: dx, 3dx, minimum y wavelength: sqrt(3) dx, 2 sqrt(3)

    theory_t = np.linspace(0.0, 100.0, 101)
    theory_order1_x_10 = 0.1 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[0], theory_t) # theory_t**2 if dt = 0.5
    theory_order1_x_25 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[1], theory_t)
    theory_order1_x_30 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[2], theory_t)
    theory_order1_x_33 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[3], theory_t)
    theory_order1_x_57 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[4], theory_t)
    theory_order1_x_66 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[5], theory_t)
    theory_order1_x_99 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_x[6], theory_t)
    theory_order1_y_10 = 0.1 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[0], theory_t)
    theory_order1_y_25 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[1], theory_t)
    theory_order1_y_28 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[2], theory_t)
    theory_order1_y_30 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[3], theory_t)
    theory_order1_y_57 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[4], theory_t)
    theory_order1_y_66 = 1.e-9 * np.power(1.0 + theoretical_k_order1 * damping_lambda_order1_y[5], theory_t)
    theory_order2_x_10 = 1.e-9 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[0], theory_t)
    theory_order2_x_25 = 1.e-9 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[1], theory_t)
    theory_order2_x_30 = 1.e-9 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[2], theory_t)
    theory_order2_x_33 = 1.e-9 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[3], theory_t)
    theory_order2_x_57 = 0.01 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[4], theory_t)
    theory_order2_x_66 = 0.01 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[5], theory_t)
    theory_order2_x_99 = 0.01 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_x[6], theory_t)
    # for item_i, item in enumerate(theory_order1_x_99):
    #     print(item, theory_order1_y_57[item_i])
    # return
    theory_order2_y_10 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[0], theory_t)
    theory_order2_y_25 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[1], theory_t)
    theory_order2_y_28 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[2], theory_t)
    theory_order2_y_30 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[3], theory_t)
    theory_order2_y_57 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[4], theory_t)
    theory_order2_y_66 = 0.1 * np.power(1.0 + theoretical_k_order2 * damping_lambda_order2_y[5], theory_t)

    x_wavenumber_factor = np.array([
        #50.0, 40.0, 30.0, 20.0, 10.0, 5.0,
        #50.0, 48.0, 46.0, 44.0, 42.0, 40.0,
        10.0, 33.0, 66.0, 99.0,
        # 30.0, 50.0, 70.0,
        #44.4, 33.33, 22.2
        # 40.0, 30.0, 20.0
    ], dtype=float)

    y_wavenumber_factor = np.array([
        10.0, 28.0, 57.0,
        # 53.0, 58.0, 63.0,
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

                # do_3d_divergence_damping = True if order == 1 else False
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

                # if order == 1:
                if not do_3d_divergence_damping:
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


    def force_aspect(ax,aspect=1):
        # im = ax.get_images()
        # extent =  im.get_extent()
        # extent = ax.get_window_extent()
        # ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        ax.set_aspect(abs((x_limits[1]-x_limits[0])/(y_limits[1]-y_limits[0]))/aspect)
        # ax.set_aspect('equal')

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # line_label = ["K_cri/5", "K_cri/4", "K_cri/3", "K_cri/2", "K_cri"]
    # line_label = [
    #     f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 1$ s", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 1$ s", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 1$ s",
    #     f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_x = ${int(x_wavenumber_factor[0])}, $dt = 1$ s", f"$k_x = ${int(x_wavenumber_factor[1])}, $dt = 1$ s", f"$k_x = ${int(x_wavenumber_factor[2])}, $dt = 1$ s",
    # ]
    line_label = [
        f"$k_x = ${int(x_wavenumber_factor[0])}", f"$k_x = ${int(x_wavenumber_factor[1])}", f"$k_x = ${int(x_wavenumber_factor[2])}", f"$k_x = ${int(x_wavenumber_factor[3])}",
        f"$k_x = ${int(x_wavenumber_factor[0])}", f"$k_x = ${int(x_wavenumber_factor[1])}", f"$k_x = ${int(x_wavenumber_factor[2])}", f"$k_x = ${int(x_wavenumber_factor[3])}",
        f"$k_x = ${int(x_wavenumber_factor[0])}", f"$k_x = ${int(x_wavenumber_factor[1])}", f"$k_x = ${int(x_wavenumber_factor[2])}", f"$k_x = ${int(x_wavenumber_factor[3])}",
        f"$k_x = ${int(x_wavenumber_factor[0])}", f"$k_x = ${int(x_wavenumber_factor[1])}", f"$k_x = ${int(x_wavenumber_factor[2])}", f"$k_x = ${int(x_wavenumber_factor[3])}",
    ]
    dashdotdotted = ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5)))
    line_style = [ ["solid", "dashed"], ["solid", "dashed"] ]
    line_color = ["green", "red", "blue", "orange"]
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
        ax[order].legend(fontsize=font_size)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    plt.savefig("plot_mean_divergence_x.pdf", format="pdf", dpi=500)

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    counter = 0
    order = 0
    for t in range(delta_t.shape[0]):
        for x in range(x_wavenumber_factor.shape[0]):
            ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
            counter = counter + 1
    order = 1
    for t in range(delta_t.shape[0]):
        for x in range(x_wavenumber_factor.shape[0] - 1):
            ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
            counter = counter + 1
    ax[0].plot(theory_t, theory_order1_x_10, color="black", linestyle="dashed")
    ax[0].plot(70.0 + theory_t[:45], theory_order1_x_33[:45], color="black", linestyle="dashed")
    ax[0].plot(18.0 + theory_t[:11], theory_order1_x_66[:11], color="black", linestyle="dashed")
    if np.any(theory_order1_x_99 < 0.0):
        ax[0].plot(7.5 + theory_t[:3], np.abs(theory_order1_x_99[:3]), color="black", linestyle="dotted")
    else:
        ax[0].plot(7.5 + theory_t[:3], theory_order1_x_99[:3], color="black", linestyle="dashed")

    if np.any(theory_order2_x_10 < 0.0):
        ax[1].plot(100.0 + theory_t[:70], np.abs(theory_order2_x_10[:70]), color="black", linestyle="dotted")
    else:
        ax[1].plot(100.0 + theory_t[:70], theory_order2_x_10[:70], color="black", linestyle="dashed")
    if np.any(theory_order2_x_66 < 0.0):
        ax[1].plot(60.0 + theory_t[:120], np.abs(theory_order2_x_66[:120]), color="black", linestyle="dotted")
    else:
        ax[1].plot(60.0 + theory_t[:120], theory_order2_x_66[:120], color="black", linestyle="dashed")
    if np.any(theory_order2_x_33 < 0.0):
        ax[1].plot(15.0 + theory_t[:30], np.abs(theory_order2_x_33[:30]), color="black", linestyle="dotted")
    else:
        ax[1].plot(15.0 + theory_t[:3], theory_order2_x_33[:3], color="black", linestyle="dashed")
    # if np.any(theory_order2_x_99 < 0.0):
    #     ax[1].plot(theory_t, np.abs(theory_order2_x_99), color="black", linestyle="dotted")
    # else:
    #     ax[1].plot(theory_t, theory_order2_x_99, color="black", linestyle="dashed")
    for order in range(2):
        ax[order].set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean windspeed residual (m s$^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        ax[order].set_ylim(1.e-17, 1.)
        ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # ax.set_title("Mean error in windspeed")
        ax[order].legend(fontsize=font_size)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    # force_aspect(ax[0], aspect=1)
    # force_aspect(ax[1], aspect=1)
    # f.tight_layout(w_pad=2)
    plt.savefig("figure_mean_windspeed_x.pdf", format="pdf", dpi=500)



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

                # do_3d_divergence_damping = True if order == 1 else False
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

                # if order == 1:
                if not do_3d_divergence_damping:
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
    # line_label = [
    #     f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    #     f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_y = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_y = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    # ]
    line_label = [
        f"$k_y = ${int(y_wavenumber_factor[0])}", f"$k_y = ${int(y_wavenumber_factor[1])}", f"$k_y = ${int(y_wavenumber_factor[2])}",
        f"$k_y = ${int(y_wavenumber_factor[0])}", f"$k_y = ${int(y_wavenumber_factor[1])}", f"$k_y = ${int(y_wavenumber_factor[2])}",
        f"$k_y = ${int(y_wavenumber_factor[0])}", f"$k_y = ${int(y_wavenumber_factor[1])}", f"$k_y = ${int(y_wavenumber_factor[2])}",
        f"$k_y = ${int(y_wavenumber_factor[0])}", f"$k_y = ${int(y_wavenumber_factor[1])}", f"$k_y = ${int(y_wavenumber_factor[2])}",
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
    plt.savefig("plot_mean_divergence_y.pdf", format="pdf", dpi=500)

    plt.close()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    counter = 0
    order = 0
    for t in range(delta_t.shape[0]):
        for x in range(y_wavenumber_factor.shape[0]):
            ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
            counter = counter + 1
    order = 1
    for t in range(delta_t.shape[0]):
        for x in range(y_wavenumber_factor.shape[0] - 1):
            ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
            counter = counter + 1
    ax[0].plot(theory_t, theory_order1_y_10, color="black", linestyle="dashed")
    ax[0].plot(70.0 + theory_t[:60], theory_order1_y_28[:60], color="black", linestyle="dashed")
    ax[0].plot(5.0 + theory_t[:20], theory_order1_y_57[:20], color="black", linestyle="dashed")
    if np.any(theory_order2_y_10 < 0.0):
        ax[1].plot(theory_t[:180], np.abs(theory_order2_y_10[:180]), color="black", linestyle="dotted")
    else:
        ax[1].plot(theory_t[:180], theory_order2_y_10[:180], color="black", linestyle="dashed")
    if np.any(theory_order2_y_28 < 0.0):
        ax[1].plot(theory_t[:10], np.abs(theory_order2_y_28[:10]), color="black", linestyle="dotted")
    else:
        ax[1].plot(theory_t[:10], theory_order2_y_28[:10], color="black", linestyle="dashed")
    # if np.any(theory_order2_y_57 < 0.0):
    #     ax[1].plot(theory_t[:4], np.abs(theory_order2_y_57[:4]), color="black", linestyle="dotted")
    # else:
    #     ax[1].plot(theory_t[:4], theory_order2_y_57[:4], color="black", linestyle="dashed")
    for order in range(2):
        ax[order].set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
        ax[order].set_ylabel("Mean windspeed residual (m s$^{-1}$)", fontsize=font_size)
        ax[order].set_xlabel("Time (s)", fontsize=font_size)
        ax[order].tick_params(axis='both', which='major', labelsize=font_size)
        ax[order].set_ylim(1.e-17, 1.)
        ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # ax.set_title("Mean error in windspeed")
        ax[order].legend(fontsize=font_size)
    ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    # force_aspect(ax[0], aspect=1)
    # force_aspect(ax[1], aspect=1)
    # f.tight_layout(w_pad=2)
    plt.savefig("figure_mean_windspeed_y.pdf", format="pdf", dpi=500)


    ############# X AND Y DIVERGENCE #############
    # total_divergence = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    # total_windspeed = [ [ [[] for _ in range(y_wavenumber_factor.shape[0]) ] for _ in range(delta_t.shape[0]) ] for _ in range(2) ]
    # inital_total_divergence = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)
    # inital_total_windspeed = np.zeros((delta_t.shape[0], y_wavenumber_factor.shape[0]), dtype = float)

    # for order in range(2):
    #     for t in range(delta_t.shape[0]):
    #         for x in range(y_wavenumber_factor.shape[0]):
    #             base_input_path = "/scratch/mch/cong/data/div_converge_res"
    #             run_path = "./"
    #             experiment_type = ExperimentType.DIVCONVERGE
    #             serialization_type = SerializationType.SB
    #             div_wave = DivWave(wave_type=WAVETYPE.X_AND_Y_WAVE, x_wavenumber_factor=x_wavenumber_factor[x], y_wavenumber_factor=y_wavenumber_factor[x])
    #             log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    #             div_utils.print_config(div_wave)

    #             disable_logging = True
    #             mpi = False
    #             profile = False
    #             grid_root = 0
    #             grid_level = 2
    #             enable_output = False
    #             enable_debug_message = False

    #             initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    #             input_path = base_input_path + resolutions + "/ser_data/"

    #             parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    #             configure_logging(run_path, experiment_type, parallel_props, disable_logging)

    #             end_date = initial_date + datetime.timedelta(seconds=int(end_time_seconds[t]))

    #             log.info(
    #                 f" running delta_t = {t}, y_wavenumber = {y_wavenumber_factor[x]}, end_date = {end_date}"
    #             )

    #             do_3d_divergence_damping = True if order == 1 else False
    #             (
    #                 timeloop,
    #                 diffusion_diagnostic_state,
    #                 solve_nonhydro_diagnostic_state,
    #                 diagnostic_metric_state,
    #                 diagnostic_state,
    #                 prognostic_state_list,
    #                 output_state,
    #                 prep_adv,
    #                 inital_divdamp_fac_o2,
    #             ) = initialize(
    #                 Path(input_path),
    #                 parallel_props,
    #                 serialization_type,
    #                 experiment_type,
    #                 grid_root,
    #                 grid_level,
    #                 enable_output,
    #                 enable_debug_message,
    #                 div_wave,
    #                 dtime_seconds=1.0,
    #                 end_date=end_date,
    #                 output_seconds_interval=100000.0,
    #                 do_o2_divdamp=True,
    #                 do_3d_divergence_damping=do_3d_divergence_damping,
    #                 divergence_order=order + 1,
    #                 divdamp_fac=k_cri[order] * delta_t[t],
    #             )

    #             cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
    #             cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
    #             edge_length = timeloop.solve_nonhydro.edge_geometry.primal_edge_lengths.asnumpy()
    #             cell_area = timeloop.solve_nonhydro.cell_params.area.asnumpy()

    #             initial_vn = np.array(prognostic_state_list[0].vn.asnumpy(), copy=True)
                
    #             analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
    #                 cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
    #             )

    #             if device == Device.GPU:
    #                 analytic_divergence = analytic_divergence.get()

    #             inital_total_divergence[t, x] = np.sum(np.abs(analytic_divergence) * cell_area) / np.sum(cell_area)
    #             inital_total_windspeed[t, x] = np.sum(np.abs(initial_vn[:, 0]) * edge_length) / np.sum(edge_length)
        
    #             timeloop.time_integration(
    #                 diffusion_diagnostic_state,
    #                 solve_nonhydro_diagnostic_state,
    #                 diagnostic_metric_state,
    #                 diagnostic_state,
    #                 prognostic_state_list,
    #                 prep_adv,
    #                 inital_divdamp_fac_o2,
    #                 do_prep_adv=False,
    #                 output_state=output_state,
    #                 profile=profile,
    #             )

    #             if order == 1:
    #                 total_divergence[order][t][x] = np.concatenate(
    #                     (
    #                         [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)# / 3000.0 # height of the domain
    #                     )
    #                 )
    #                 total_windspeed[order][t][x] = np.concatenate(
    #                     (
    #                         [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)# / 3000.0 # height of the domain
    #                     )
    #                 )
    #             else:
    #                 total_divergence[order][t][x] = np.concatenate(
    #                     (
    #                         [float(inital_total_divergence[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_divergence)
    #                     )
    #                 )
    #                 total_windspeed[order][t][x] = np.concatenate(
    #                     (
    #                         [float(inital_total_windspeed[t, x])], np.array(timeloop.solve_nonhydro.intermediate_fields.total_windspeed)
    #                     )
    #                 )
                
    #             # div_utils.plot_tridata(
    #             #     grid_filename,
    #             #     diagnostic_state.u.ndarray[:, 0],
    #             #     mask,
    #             #     f"U at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
    #             #     f"plot_u_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
    #             # )
    #             # div_utils.plot_tridata(
    #             #     grid_filename,
    #             #     timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0],
    #             #     mask,
    #             #     f"Div at order={order}, dt={delta_t[t]}, ywave={y_wavenumber_factor[x]}",
    #             #     f"plot_div_order_{order}_dt_{t}_ywave_{int(y_wavenumber_factor[x])}",
    #             # )


    # time_xaxis = [np.linspace(0.0, dtime_seconds[t] * end_time_seconds[t], end_time_seconds[t] + 1) for t in range(delta_t.shape[0])]

    # plt.close()

    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # # line_label = ["K_cri/5", "K_cri/4", "K_cri/3", "K_cri/2", "K_cri"]
    # line_label = [
    #     f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    #     f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 0.5$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 0.5$ s",
    #     f"$k_xy = ${int(y_wavenumber_factor[0])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[1])}, $dt = 1$ s", f"$k_xy = ${int(y_wavenumber_factor[2])}, $dt = 1$ s",
    # ]
    # counter = 0
    # for order in range(2):
    #     for t in range(delta_t.shape[0]):
    #         for x in range(y_wavenumber_factor.shape[0]):
    #             ax[order].plot(time_xaxis[t], total_divergence[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
    #             counter = counter + 1
    # for order in range(2):
    #     ax[order].set_yscale("log")
    #     # labels = [item.get_text() for item in ax.get_xticklabels()]
    #     # labels[0] = ""
    #     # for i in range(max_xaxis_value):
    #     #     labels[i+1] = axis_labels[i]
    #     # ax.set_xticks(dx, axis_labels)
    #     # ax.set_xticklabels(axis_labels)
    #     ax[order].set_ylabel("Mean divergence residual (s$^{-1}$)", fontsize=font_size)
    #     ax[order].set_xlabel("Time (s)", fontsize=font_size)
    #     ax[order].tick_params(axis='both', which='major', labelsize=font_size)
    #     # ax.set_title("Mean error in divergence")
    #     ax[order].legend(fontsize=font_size)
    # ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    # ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    # plt.savefig("plot_mean_divergence_xy.pdf", format="pdf", dpi=500)

    # plt.close()

    # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9), constrained_layout=True)

    # counter = 0
    # for order in range(2):
    #     for t in range(delta_t.shape[0]):
    #         for x in range(y_wavenumber_factor.shape[0]):
    #             ax[order].plot(time_xaxis[t], total_windspeed[order][t][x], linestyle = line_style[order][t], color=line_color[x], label=line_label[counter])
    #             counter = counter + 1
    # for order in range(2):
    #     ax[order].set_yscale("log")
    # # labels = [item.get_text() for item in ax.get_xticklabels()]
    # # labels[0] = ""
    # # for i in range(max_xaxis_value):
    # #     labels[i+1] = axis_labels[i]
    # # ax.set_xticks(dx, axis_labels)
    # # ax.set_xticklabels(axis_labels)
    #     ax[order].set_ylabel("Mean windspeed residual ($m s^{-1}$)", fontsize=font_size)
    #     ax[order].set_xlabel("Time (s)", fontsize=font_size)
    #     ax[order].tick_params(axis='both', which='major', labelsize=font_size)
    #     ax[order].set_ylim(1.e-17, 1.e-1)
    #     ax[order].set_yticks([1.e-17, 1.e-13, 1.e-9, 1.e-5, 1.e-1])
    # # ax.set_title("Mean error in windspeed")
    #     ax[order].legend(fontsize=font_size)
    # ax[0].set_title("(a) First-order divergence", fontsize=font_size+2)
    # ax[1].set_title("(b) Second-order divergence", fontsize=font_size+2)
    # plt.savefig("plot_mean_windspeed_xy.pdf", format="pdf", dpi=500)


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

    for end_second in range(200):
        if end_second != 199:
            continue
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
                    divergence_order=2,
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
                    "figure_final"+str(end_second+1)+"_vn",
                    div_wave,
                    eigen_vector=damping_eigen_vector,
                    plot_analytic=False,
                )
                div_utils.plot_triedgedata(
                    grid_filename,
                    prognostic_state_list[1].vn.ndarray,
                    f"VN",
                    "figure_finalold"+str(end_second+1)+"_vn",
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
    # test_stability_divergence_first_order()
    test_order2_div_y_damping()
    # test_debug()