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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import colors

from icon4py.model.common.config import Device
from icon4py.model.common.decomposition.definitions import (
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.settings import device, xp
from icon4py.model.driver.dycore_driver import initialize
from icon4py.model.driver.initialization_utils import (
    WAVETYPE,
    DivWave,
    ExperimentType,
    SerializationType,
    configure_logging,
    determine_u_v_w_in_div_converge_experiment,
    read_geometry_fields,
    read_static_fields,
)
from icon4py.model.driver import divergence_utils as div_utils


log = logging.getLogger(__name__)



def test_divergence_single_time_step():
    resolutions = (
        "1000m",
        "500m",
        # '250m',
        # '125m',
    )
    dx = np.array(
        (
            1000.0,
            500.0,
            250.0,
            125.0,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    # div_wave = DivWave(wave_type=WAVETYPE.SPHERICAL_HARMONICS, x_wavenumber_factor=1.0, y_wavenumber_factor=1.0)
    div_wave = DivWave(wave_type=WAVETYPE.Y_WAVE, y_wavenumber_factor=50.0)
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    div_utils.print_config(div_wave)

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=dtime_seconds)
    mean_error1 = np.zeros(len(resolutions), dtype=float)
    mean_error2 = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + res + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

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
            dtime_seconds,
            end_date,
            output_interval,
            div_wave,
        )

        mask = div_utils.create_mask(grid_filename)
        div_utils.plot_tridata(
            grid_filename,
            diagnostic_state.u.ndarray,
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            diagnostic_state.v.ndarray,
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

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

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        # analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
            cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
        )

        mean_error1[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)
        mean_error2[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error1 = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.ndarray[
                :, 0
            ]
            - analytic_divergence
        )
        divergence_error2 = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.ndarray[
                :, 0
            ]
            - analytic_divergence
        )

        # if device == Device.GPU:
        #     analytic_divergence = analytic_divergence.get()
        #     divergence_error1 = divergence_error1.get()
        #     divergence_error2 = divergence_error2.get()

        div_utils.plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.ndarray,
            mask,
            f"Computed 1st order divergence at {res}",
            "plot_computed_1st_order_divergence_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.ndarray,
            mask,
            f"Computed 2nd order divergence at {res}",
            "plot_computed_2nd_order_divergence_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            divergence_error1,
            mask,
            f"First order divergence error at DX = {res}",
            "plot_error_1st_order_divergence_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            divergence_error2,
            mask,
            f"Second order divergence error at DX = {res}",
            "plot_error_2nd_order_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error1[i]} {mean_error2[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    div_utils.plot_error(mean_error1, dx[0 : len(resolutions)], resolutions, 1)
    div_utils.plot_error(mean_error2, dx[0 : len(resolutions)], resolutions, 2)


def test_divergence_multiple_time_step():
    resolutions = (
        "1000m",
        # "500m",
        # '250m',
        # '125m',
    )
    dx = np.array(
        (
            1000.0,
            500.0,
            250.0,
            125.0,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    div_wave = DivWave(wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=17.1, y_wavenumber_factor=0.0)
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    div_utils.print_config(div_wave)

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_time = 1000 * dtime_seconds
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + res + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        end_date = initial_date + datetime.timedelta(seconds=end_time)

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
            dtime_seconds=dtime_seconds,
            end_date=end_date,
            output_seconds_interval=1000 * dtime_seconds,
            do_o2_divdamp=True,
            do_3d_divergence_damping=False,
            divergence_order=1,
            divdamp_fac=1.0 / 3.0 / math.sqrt(3.0) / 2.0,
        )

        eigen_value, eigen_vector, eigen_vector_cartesian = div_utils.eigen_divergence(
            divergence_factor=0.0,
            order=timeloop.solve_nonhydro.config.divergence_order,
            grid_space=1000.0,
            div_wave=div_wave,
        )
        damping_eigen_vector = eigen_vector[np.argmax(np.abs(eigen_value))]

        mask = div_utils.create_mask(grid_filename)

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

        div_utils.plot_triedgedata(
            grid_filename,
            initial_vn,
            f"VN at {res}",
            "plot_initial_vn" + res,
            div_wave,
            eigen_vector=damping_eigen_vector,
            plot_analytic=True,
        )
        div_utils.plot_tridata(
            grid_filename,
            diagnostic_state.u.ndarray,
            mask,
            f"U at {res}",
            "plot_initial_U" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            diagnostic_state.v.ndarray,
            mask,
            f"V at {res}",
            "plot_initial_V" + res,
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

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        v_scale = 90.0 / 7.5
        sphere_radius = 6371229.0
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
            cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
        )

        diff1_divergence = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0]
            - analytic_divergence
        )
        diff2_divergence = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.ndarray[:, 0]
            - analytic_divergence
        )

        div_utils.plot_tridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        div_utils.plot_triedgedata(
            grid_filename,
            prognostic_state_list[0].vn.ndarray,
            f"VN at {res}",
            "plot_final_vn" + res,
            div_wave,
            eigen_vector=damping_eigen_vector,
            plot_analytic=False,
        )
        div_utils.plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray,
            mask,
            f"Computed divergence order 1 at {res}",
            "plot_computed_divergence1_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.ndarray,
            mask,
            f"Computed divergence order 2 at {res}",
            "plot_computed_divergence2_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            diff1_divergence,
            mask,
            f"Difference in divergence with order 1 at {res}",
            "plot_diff1_divergence_" + res,
        )
        div_utils.plot_tridata(
            grid_filename,
            diff2_divergence,
            mask,
            f"Difference in divergence with order 2 at {res}",
            "plot_diff2_divergence_" + res,
        )

        log.info(
            f"Sum of divergence1: {xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:,0]))}"
        )
        log.info(
            f"Sum of divergence2: {xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.ndarray[:,0]))}"
        )
        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

        end_time = end_time * 2


def test_divergence_single_time_step_on_globe():
    resolutions = (
        "r2b4",
        "r2b5",
        "r2b6",
        # 'r2b7',
    )
    grid_name = (
        "icon_grid_0010_R02B04_G",
        "icon_grid_0008_R02B05_G",
        "icon_grid_0002_R02B06_G",
        "icon_grid_0004_R02B07_G",
    )
    dx = np.array(
        (
            220,
            110,
            55,
            27.5,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/flat_earth_"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.GLOBEDIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=dtime_seconds)
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + grid_name[i] + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

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
            dtime_seconds,
            end_date,
            output_interval,
        )

        mask = div_utils.create_globe_mask(grid_filename)
        div_utils.plot_globetridata(
            grid_filename,
            diagnostic_state.u.ndarray,
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            diagnostic_state.v.ndarray,
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        z_ifc = diagnostic_metric_state.z_ifc.asnumpy()

        log.info(
            f"Surface max min:  {z_ifc[:,timeloop.grid.num_levels].max()}, {z_ifc[:,timeloop.grid.num_levels].min()}"
        )
        log.info(
            f"First level max min:  {z_ifc[:,timeloop.grid.num_levels-1].max()}, {z_ifc[:,timeloop.grid.num_levels-1].min()}"
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

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

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        sphere_radius = 6371229.0
        theta_shift = 0.5 * math.pi
        lon_shift = math.pi
        cell_lon = cell_lon + lon_shift
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        # u_cell = u_factor * xp.cos(2.0 * cell_lon) * xp.cos(theta_shift - cell_lat) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        # v_cell = v_factor * xp.cos(cell_lon) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        analytic_divergence = (
            -u_factor * xp.sin(2.0 * cell_lon) * xp.sin(2.0 * (theta_shift - cell_lat))
            + 0.25
            * v_factor
            * xp.cos(cell_lon)
            * (3.0 * xp.sin(3.0 * (theta_shift - cell_lat)) / xp.sin(theta_shift - cell_lat) - 1.0)
        ) / sphere_radius

        mean_error[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:, 0]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        div_utils.plot_globetridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray,
            mask,
            f"Computed divergence at {res}",
            "plot_computed_divergence_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            divergence_error,
            mask,
            f"Divergence error at {res}",
            "plot_error_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    div_utils.plot_error(mean_error, dx[0 : len(resolutions)], resolutions)


def test_divergence_multiple_time_step_on_globe():
    resolutions = (
        "r2b4",
        # 'r2b5',
        # 'r2b6',
        # 'r2b7',
    )
    grid_name = (
        "icon_grid_0010_R02B04_G",
        "icon_grid_0008_R02B05_G",
        "icon_grid_0002_R02B06_G",
        "icon_grid_0004_R02B07_G",
    )
    dx = np.array(
        (
            220,
            110,
            55,
            27.5,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/flat_earth_"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.GLOBEDIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = 100 * dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(
        seconds=1000 * dtime_seconds
    )
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + grid_name[i] + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

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
            dtime_seconds,
            end_date,
            output_interval,
        )

        mask = div_utils.create_globe_mask(grid_filename)
        div_utils.plot_globetridata(
            grid_filename,
            diagnostic_state.u.ndarray,
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            diagnostic_state.v.ndarray,
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

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

        # os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        theta_shift = 0.5 * math.pi
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        u_cell = (
            u_factor
            * xp.cos(2.0 * cell_lon)
            * xp.cos(theta_shift - cell_lat)
            * xp.cos(theta_shift - cell_lat)
            * xp.sin(theta_shift - cell_lat)
        )
        v_cell = (
            v_factor
            * xp.cos(cell_lon)
            * xp.cos(theta_shift - cell_lat)
            * xp.sin(theta_shift - cell_lat)
        )
        analytic_divergence = (
            -0.5
            / xp.sqrt(2.0 * math.pi)
            * (
                1.0e-5
                * 2.0
                * math.pi
                * xp.sqrt(105.0)
                * xp.sin(2.0 * cell_lon)
                * xp.cos(theta_shift - cell_lat)
                * xp.cos(theta_shift - cell_lat)
                * xp.sin(theta_shift - cell_lat)
                + 1.0e-5
                * math.pi
                * xp.sqrt(15.0)
                * xp.cos(cell_lon)
                * xp.cos(2.0 * (theta_shift - cell_lat))
            )
        )

        mean_error[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:, 0]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        div_utils.plot_globetridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray,
            mask,
            f"Computed divergence at {res}",
            "plot_computed_divergence_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        div_utils.plot_globetridata(
            grid_filename,
            divergence_error,
            mask,
            f"Difference in divergence at {res}",
            "plot_diff_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        # os.system(f"mv plot_* dummy* data_output* data_{res}")


def test_eigen_divergence():

    import xarray as xr
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res1000m.nc"
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    eoc = grid.edge_of_cell.T.values - 1
    random_number = 7000
    normal_y = grid.edge_primal_normal_cartesian_y.values[eoc[random_number]]
    normal_x = grid.edge_primal_normal_cartesian_x.values[eoc[random_number]]

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB

    disable_logging = False
    mpi = False
    grid_root = 0
    grid_level = 2
    input_path = base_input_path + "1000m" + "/ser_data/"

    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    configure_logging(run_path, experiment_type, parallel_props, disable_logging)
    (edge_geometry, cell_geometry, vertical_geometry, c_owner_mask) = read_geometry_fields(
        Path(input_path),
        damping_height=3000.0,
        rank=parallel_props.rank,
        ser_type=serialization_type,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    ) = read_static_fields(
        enable_debug_message=False,
        path=Path(input_path),
        rank=parallel_props.rank,
        ser_type=serialization_type,
        grid_root=grid_root,
        grid_level=grid_level,
    )

    edge_lat = edge_geometry.edge_center[0].ndarray
    edge_lon = edge_geometry.edge_center[1].ndarray
    z_ifc = diagnostic_metric_state.z_ifc.ndarray
    num_levels = z_ifc.shape[1] - 1

    primal_normal_x = edge_geometry.primal_normal[0].ndarray
    primal_normal_y = edge_geometry.primal_normal[1].ndarray

    for y in range(0, 1):
        for x in range(0, 80):
            print("WAVE NUMBERS (X Y):", x, y)
            if x == 0 and y == 0:
                continue
            if x == 0:
                div_wave = DivWave(
                    wave_type=WAVETYPE.Y_WAVE,
                    x_wavenumber_factor=float(x),
                    y_wavenumber_factor=float(y),
                )
            elif y == 0:
                div_wave = DivWave(
                    wave_type=WAVETYPE.X_WAVE,
                    x_wavenumber_factor=float(x),
                    y_wavenumber_factor=float(y),
                )
            else:
                div_wave = DivWave(
                    wave_type=WAVETYPE.X_AND_Y_WAVE,
                    x_wavenumber_factor=float(x),
                    y_wavenumber_factor=float(y),
                )
            u_edge, v_edge, w_ndarray = determine_u_v_w_in_div_converge_experiment(edge_lat, edge_lon, z_ifc, num_levels, div_wave)
            vn_ndarray = xp.zeros(edge_lat.shape[0], dtype=float)
            vn_ndarray[:] = u_edge[:] * primal_normal_x[:] + v_edge[:] * primal_normal_y[:]
            print("REAL VN SPACE: ", vn_ndarray[eoc[random_number]])
            print("CHECKING REAL SPACE x: ", primal_normal_x[eoc[random_number]])
            print("CHECKING REAL SPACE y: ", primal_normal_y[eoc[random_number]])
            print()

            eigen_value, eigen_vector, eigen_vector_cartesian = div_utils.eigen_divergence(
                divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave, debug=True
            )
            print()
            # eigen_vector = xp.asarray(eigen_vector)
            # # print("DOT BETWEEN VN AND EIGENVECTOR1: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[0]))
            # # print("DOT BETWEEN VN AND EIGENVECTOR2: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[1]))
            # # print("DOT BETWEEN VN AND EIGENVECTOR3: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[2]))
            # print("DIFF EIGENVECTOR1: ", np.abs(eigen_vector[0, 0] - 0.5 * np.sum(eigen_vector[0, 1:])), np.abs(eigen_vector[0, 1] + eigen_vector[0, 2]))
            # print("DIFF EIGENVECTOR2: ", np.abs(eigen_vector[1, 0] - 0.5 * np.sum(eigen_vector[1, 1:])), np.abs(eigen_vector[1, 1] + eigen_vector[1, 2]))
            # print("DIFF EIGENVECTOR3: ", np.abs(eigen_vector[2, 0] - 0.5 * np.sum(eigen_vector[2, 1:])), np.abs(eigen_vector[2, 1] + eigen_vector[2, 2]))
            # print()
            # print()
            # print()

            # eigen_value, eigen_vector, eigen_vector_cartesian = div_utils.eigen_divergence(
            #     divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave, debug=True
            # )
            # print()
            # eigen_vector = xp.asarray(eigen_vector)
            # print("DOT BETWEEN VN AND EIGENVECTOR1: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[0]))
            # print("DOT BETWEEN VN AND EIGENVECTOR2: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[1]))
            # print("DOT BETWEEN VN AND EIGENVECTOR3: ", xp.dot(vn_ndarray[eoc[random_number]], eigen_vector[2]))

            # non_zero_eigen_vector, non_zero_eigen_value = [], []
            # for eigen in range(3):
            #     if np.abs(eigen_value[eigen]) > 1.e-10:
            #         non_zero_eigen_vector.append(eigen_vector[eigen])
            #         non_zero_eigen_value.append(eigen_value[eigen])
            # assert len(non_zero_eigen_vector) == 2
            # eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave, threeD=False, debug=True)
            # print("eigenvalue component: ", eigen_vector_wave_component)
            # if eigen_vector_wave_component[0] > eigen_vector_wave_component[1]:
            #     import sys
            #     print("irregular eigen value spotted!!!: ")
            #     # sys.exit(0)
            print()
            # print()
            # print()

            # eigen_vorticity(divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave)    


def plot_eigen_divergence():
    x_grid = np.arange(0.5, 100.0, 0.5)
    y_grid = np.arange(0.5, 100.0, 0.5)
    x_wave = 2.0 * math.pi / 100000.0 * x_grid * 1000.0
    y_wave = 2.0 * math.pi / 100000.0 * y_grid * 1000.0
    xy_xwave, xy_ywave = np.meshgrid(x_wave, y_wave, indexing="ij")

    xdamp1_max = np.zeros(x_grid.shape, dtype=float)
    xdamp1_actual = np.zeros(x_grid.shape, dtype=float)
    xdamp2 = np.zeros(x_grid.shape, dtype=float)
    ydamp1_max = np.zeros(y_grid.shape, dtype=float)
    ydamp1_actual = np.zeros(y_grid.shape, dtype=float)
    ydamp2 = np.zeros(y_grid.shape, dtype=float)
    xydamp1_max = np.zeros(xy_xwave.shape, dtype=float)
    xydamp1_actual = np.zeros(xy_xwave.shape, dtype=float)
    xydamp2 = np.zeros(xy_xwave.shape, dtype=float)

    proj_xdamp1_max = np.zeros(x_grid.shape, dtype=float)
    proj_xdamp1_actual = np.zeros(x_grid.shape, dtype=float)
    proj_xdamp2 = np.zeros(x_grid.shape, dtype=float)
    proj_ydamp1_max = np.zeros(y_grid.shape, dtype=float)
    proj_ydamp1_actual = np.zeros(y_grid.shape, dtype=float)
    proj_ydamp2 = np.zeros(y_grid.shape, dtype=float)
    proj_xydamp1_max = np.zeros(xy_xwave.shape, dtype=float)
    proj_xydamp1_actual = np.zeros(xy_xwave.shape, dtype=float)
    proj_xydamp2 = np.zeros(xy_xwave.shape, dtype=float)

    for i, x_wavenumber in enumerate(x_grid):
        if x_wavenumber == 0.0:
            continue
        div_wave = DivWave(
            # wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=x_wavenumber, y_wavenumber_factor=0.0
            wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=x_wavenumber, y_wavenumber_factor=0.0,
        )
        eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_divergence(
            divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
        )
        eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_divergence(
            divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
        )
        non_zero_eigen_vector, non_zero_eigen_value = [], []
        # for eigen in range(3):
        #     if np.abs(eigen_value1[eigen]) > 1.e-10:
        #         non_zero_eigen_vector.append(eigen_vector1[eigen])
        #         non_zero_eigen_value.append(eigen_value1[eigen])
        # assert len(non_zero_eigen_vector) == 2
        # eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
        # if np.abs(eigen_vector_wave_component[0]) > np.abs(eigen_vector_wave_component[1]):
        #     xdamp1_actual[i] = non_zero_eigen_value[0]
        #     xdamp1_max[i] = non_zero_eigen_value[1]
        #     proj_xdamp1_actual[i] = eigen_vector_wave_component[0]
        #     proj_xdamp1_max[i] = eigen_vector_wave_component[1]
        # else:
        #     xdamp1_max[i] = non_zero_eigen_value[0]
        #     xdamp1_actual[i] = non_zero_eigen_value[1]
        #     proj_xdamp1_max[i] = eigen_vector_wave_component[0]
        #     proj_xdamp1_actual[i] = eigen_vector_wave_component[1]
        # non_zero_eigen_vector.append(eigen_vector1[np.argmax(np.abs(eigen_value1))])
        # xdamp1_max[i] = eigen_value1[np.argmax(np.abs(eigen_value1))]
        # # eigen_value1 = np.sort(eigen_value1)
        # argsort = eigen_value1.argsort()
        # eigen_value1 = eigen_value1[argsort]
        # eigen_vector1 = eigen_vector1[argsort]
        # non_zero_eigen_vector.append(eigen_vector1[1])
        # xdamp1_actual[i] = eigen_value1[1]
        # eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
        # proj_xdamp1_max[i] = eigen_vector_wave_component[0]
        # proj_xdamp1_actual[i] = eigen_vector_wave_component[1]
        xdamp1_actual[i] = eigen_value1[1]
        xdamp1_max[i] = eigen_value1[2]
        non_zero_eigen_vector.append(eigen_vector1[1])
        non_zero_eigen_vector.append(eigen_vector1[2])
        eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
        proj_xdamp1_actual[i] = eigen_vector_wave_component[0]
        proj_xdamp1_max[i] = eigen_vector_wave_component[1]

        xdamp2[i] = eigen_value2[np.argmax(np.abs(eigen_value2))]
        
    for i, y_wavenumber in enumerate(y_grid):
        if y_wavenumber == 0.0:
            continue
        div_wave = DivWave(
            wave_type=WAVETYPE.Y_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=y_wavenumber
        )
        eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_divergence(
            divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
        )
        eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_divergence(
            divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
        )
        non_zero_eigen_vector, non_zero_eigen_value = [], []
        ydamp1_actual[i] = eigen_value1[1]
        ydamp1_max[i] = eigen_value1[2]
        non_zero_eigen_vector.append(eigen_vector1[1])
        non_zero_eigen_vector.append(eigen_vector1[2])
        eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
        proj_ydamp1_actual[i] = eigen_vector_wave_component[0]
        proj_ydamp1_max[i] = eigen_vector_wave_component[1]

        ydamp2[i] = eigen_value2[np.argmax(np.abs(eigen_value2))]


    # xtick = [0, math.pi / 4.0, math.pi / 2.0, math.pi * 3.0 / 4.0, math.pi]
    # xtick_label = (
    #     "$2\\pi/8\\Delta x$",
    #     "$2\\pi/4\\Delta x$",
    #     "$3\\pi/4\\Delta x$",
    #     "$2\\pi/2\\Delta x$",
    # )
    xtick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, 2.0 * math.pi]
    xtick_label = (
        "0",
        "$\\pi/2h$",
        "$2\\pi/2h$",
        "$3\\pi/2h$",
        "$2\\pi/h$",
    )

    scale_to_wavenumber = 2.0 * math.pi / 100000.0 * 1000.0
    hexagon_min_x_wavelength = scale_to_wavenumber * 100.0 / 3.0 # 3h
    triangle_min_x_wavelength = scale_to_wavenumber * 100.0 # h
    hexagon_min_y_wavelength = scale_to_wavenumber * 100.0 / (2.0 * math.sqrt(3.0)) # 2sqrt(3)h 
    triangle_min_y_wavelength = scale_to_wavenumber * 100.0 / math.sqrt(3.0) # sqrt(3)h

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

    f, ax = plt.subplots(nrows=1, ncols=2)
    # for i, item in enumerate(xdamp1_max):
    #     print("DE: ", f"{x_grid[i]:.5f}, {item:.5f}, {xdamp2[i]:.5f}")
    ax[0].plot(x_wave, xdamp1_max, label="order 1 ($\\lambda_1$)")
    ax[0].plot(x_wave, xdamp1_actual, label="order 1 ($\\lambda_2$)")
    ax[0].plot(x_wave, xdamp2, label="order 2")
    ax[0].plot(np.full(11, fill_value=hexagon_min_x_wavelength), np.linspace(-6.0, 0.0, 11), c='black', linestyle='dashed')
    ax[0].plot(np.full(11, fill_value=triangle_min_x_wavelength), np.linspace(-6.0, 0.0, 11), c='black', linestyle='solid')
    ax[0].set_ylabel("eigenvalue")
    ax[0].set_xlabel("x wavenumber")
    ax[0].set_xticks(xtick)
    ax[0].set_ylim(-6.0, 0.0)
    ax[0].set_title("(a) Eigenvalue in x-axis")
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_xticklabels(labels)

    ax[1].plot(y_wave, ydamp1_max, label="order 1 ($\\lambda_1$)")
    ax[1].plot(y_wave, ydamp1_actual, label="order 1 ($\\lambda_2$)")
    ax[1].plot(y_wave, ydamp2, label="order 2")
    ax[1].plot(np.full(11, fill_value=hexagon_min_y_wavelength), np.linspace(-6.0, 0.0, 11), c='black', linestyle='dashed')
    ax[1].plot(np.full(11, fill_value=triangle_min_y_wavelength), np.linspace(-6.0, 0.0, 11), c='black', linestyle='solid')
    ax[1].set_ylabel("eigenvalue")
    ax[1].set_xlabel("y wavenumber")
    ax[1].set_xticks(xtick)
    ax[1].set_title("(b) Eigenvalue in y-axis")
    ax[1].set_ylim(-6.0, 0.0)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_xticklabels(labels)

    ax[0].legend(prop={'size': 7.1}, loc='lower left', bbox_to_anchor=(0.46,0.0))
    ax[1].legend(prop={'size': 7.1}, loc='lower left', bbox_to_anchor=(0.57,0.0))
    force_aspect(ax[0], aspect=1)
    force_aspect(ax[1], aspect=1)
    f.tight_layout(w_pad=2)
    plt.savefig("figure_damp_in_x_and_y_axis.pdf", format="pdf", dpi=500)

    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(x_wave, proj_xdamp1_max, label="order 1 ($\\lambda_1$)")
    ax[0].plot(x_wave, proj_xdamp1_actual, label="order 1 ($\\lambda_2$)")
    ax[0].set_ylabel("Projection")
    ax[0].set_xlabel("x wavenumber")
    ax[0].set_xticks(xtick)
    ax[0].set_title("(a) Projection in x-axis")
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_xticklabels(labels)

    ax[1].plot(y_wave, proj_ydamp1_max, label="order 1 ($\\lambda_1$)")
    ax[1].plot(y_wave, proj_ydamp1_actual, label="order 1 ($\\lambda_2$)")
    ax[1].set_ylabel("Projection")
    ax[1].set_xlabel("y wavenumber")
    ax[1].set_xticks(xtick)
    ax[1].set_title("(b) Projection in y-axis")
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_xticklabels(labels)
    
    ax[0].legend(prop={'size': 8})
    ax[1].legend(prop={'size': 8})
    force_aspect(ax[0], aspect=1)
    force_aspect(ax[1], aspect=1)
    f.tight_layout(w_pad=2)
    plt.savefig("figure_proj_in_x_and_y_axis.pdf", format="pdf", dpi=500)

    for i, x_wavenumber in enumerate(x_grid):
        for j, y_wavenumber in enumerate(y_grid):
            if x_wavenumber == 0.0 and y_wavenumber == 0.0:
                continue
            div_wave = DivWave(
                wave_type=WAVETYPE.X_AND_Y_WAVE,
                x_wavenumber_factor=x_wavenumber,
                y_wavenumber_factor=y_wavenumber,
            )
            eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_divergence(
                divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
            )
            eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_divergence(
                divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
            )

            non_zero_eigen_vector, non_zero_eigen_value = [], []
            # for eigen in range(3):
            #     if np.abs(eigen_value1[eigen]) > 1.e-10:
            #         non_zero_eigen_vector.append(eigen_vector1[eigen])
            #         non_zero_eigen_value.append(eigen_value1[eigen])
            # assert len(non_zero_eigen_vector) == 2
            # eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
            # if np.abs(eigen_vector_wave_component[0]) > np.abs(eigen_vector_wave_component[1]):
            #     xydamp1_actual[i, j] = non_zero_eigen_value[0]
            #     xydamp1_max[i, j] = non_zero_eigen_value[1]
            # else:
            #     xydamp1_max[i, j] = non_zero_eigen_value[0]
            #     xydamp1_actual[i, j] = non_zero_eigen_value[1]
            # non_zero_eigen_vector.append(eigen_vector1[np.argmax(np.abs(eigen_value1))])
            # xydamp1_max[i, j] = eigen_value1[np.argmax(np.abs(eigen_value1))]
            # # eigen_value1 = np.sort(eigen_value1)
            # argsort = eigen_value1.argsort()
            # eigen_value1 = eigen_value1[argsort]
            # eigen_vector1 = eigen_vector1[argsort]
            # non_zero_eigen_vector.append(eigen_vector1[1])
            # xydamp1_actual[i, j] = eigen_value1[1]
            # eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
            # proj_xydamp1_max[i, j] = eigen_vector_wave_component[0]
            # proj_xydamp1_actual[i, j] = eigen_vector_wave_component[1]
            xydamp1_actual[i, j] = eigen_value1[1]
            xydamp1_max[i, j] = eigen_value1[2]
            non_zero_eigen_vector.append(eigen_vector1[1])
            non_zero_eigen_vector.append(eigen_vector1[2])
            eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
            proj_xydamp1_actual[i, j] = eigen_vector_wave_component[0]
            proj_xydamp1_max[i, j] = eigen_vector_wave_component[1]

            xydamp2[i, j] = eigen_value2[np.argmax(np.abs(eigen_value2))]

    xydamp1_max_ind = np.unravel_index(np.argmax(xydamp1_max, axis=None), xydamp1_max.shape)
    print("MAX MAX: ", xydamp1_max[xydamp1_max_ind], xydamp1_max_ind)


    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.66666))
    # cmap = plt.get_cmap('plasma')
    cmap = plt.get_cmap("gist_rainbow")
    cmap = cmap.reversed()
    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax[0].contourf(xy_xwave, xy_ywave, xydamp1_max, cmap=cmap, levels=boundaries, norm=lnorm)
    # cax = f.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
    # cb1 = plt.colorbar(cp, ax=cax) #, location="right"
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(cp, cax=cax, format='%.2f')
    

    ax[0].set_xlabel("x wavenumber")
    ax[0].set_ylabel("y wavenumber")
    ax[0].set_xticks(xtick)
    ax[0].set_yticks(xtick)
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[0].get_yticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_yticklabels(labels)

    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax[1].contourf(xy_xwave, xy_ywave, xydamp1_actual, cmap=cmap, levels=boundaries, norm=lnorm)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(cp, cax=cax, format='%.2f')
    # cb1 = plt.colorbar(cp, ax=ax[1], location="right")

    ax[1].set_xlabel("x wavenumber")
    ax[1].set_ylabel("y wavenumber")
    ax[1].set_xticks(xtick)
    ax[1].set_yticks(xtick)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[1].get_yticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_yticklabels(labels)

    ax[0].set_title("(a) First eigenvalue")
    ax[1].set_title("(b) Second eigenvalue")
    force_aspect(ax[0], aspect=1)
    force_aspect(ax[1], aspect=1)
    f.tight_layout(w_pad=2)
    plt.savefig("figure_order1_2D_damp.pdf", format="pdf", dpi=400)

    plt.clf()
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.66666))
    cmap = plt.get_cmap("gist_rainbow")
    cmap = cmap.reversed()
    boundaries = np.linspace(0.0, 1.5, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax[0].contourf(xy_xwave, xy_ywave, proj_xydamp1_max, cmap=cmap, levels=boundaries, norm=lnorm)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(cp, cax=cax, format='%.2f')
    # cb1 = plt.colorbar(cp, ax=ax[0], location="right")

    ax[0].set_xlabel("x wavenumber")
    ax[0].set_ylabel("y wavenumber")
    ax[0].set_xticks(xtick)
    ax[0].set_yticks(xtick)
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[0].get_yticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[0].set_yticklabels(labels)

    boundaries = np.linspace(0.0, 1.5, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax[1].contourf(xy_xwave, xy_ywave, proj_xydamp1_actual, cmap=cmap, levels=boundaries, norm=lnorm)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(cp, cax=cax, format='%.2f')
    # cb1 = plt.colorbar(cp, ax=ax[1], location="right")

    ax[1].set_xlabel("x wavenumber")
    ax[1].set_ylabel("y wavenumber")
    ax[1].set_xticks(xtick)
    ax[1].set_yticks(xtick)
    labels = [item.get_text() for item in ax[1].get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_xticklabels(labels)
    labels = [item.get_text() for item in ax[1].get_yticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax[1].set_yticklabels(labels)

    ax[0].set_title("(a) First projection coefficient")
    ax[1].set_title("(b) Second projection coefficient")
    force_aspect(ax[0], aspect=1)
    force_aspect(ax[1], aspect=1)
    f.tight_layout(w_pad=2)
    plt.savefig("figure_order1_2D_proj.pdf", format="pdf", dpi=400)


    plt.clf()
    f, ax = plt.subplots()
    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax.contourf(xy_xwave, xy_ywave, xydamp2, cmap=cmap, levels=boundaries, norm=lnorm)
    cb1 = plt.colorbar(cp, location="right", format='%.2f')

    ax.set_xlabel("x wavenumber")
    ax.set_ylabel("y wavenumber")
    ax.set_xticks(xtick)
    ax.set_yticks(xtick)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = xtick_label[0]
    labels[1] = xtick_label[1]
    labels[2] = xtick_label[2]
    labels[3] = xtick_label[3]
    labels[4] = xtick_label[4]
    ax.set_yticklabels(labels)

    ax.set_title("Eigenvalue")
    plt.savefig("figure_order2_2D_damp.pdf", format="pdf", dpi=400)

    print("MEAN VALUE OF DIV1: ", np.mean(xydamp1_max), np.mean(xydamp1_actual))
    print("MEAN VALUE OF DIV2: ", np.mean(xydamp2))


def test_debug():
    x_grid = np.arange(1.0, 51.0, 1.0)
    y_grid = math.sqrt(3.0) * x_grid
    x_wave = 2.0 * math.pi / 100000.0 * x_grid * 1000.0
    y_wave = math.sqrt(3.0) * x_wave

    x_grid = np.arange(1.0, 51.0, 1.0)
    y_grid = math.sqrt(3.0) * x_grid
    x_wave = 2.0 * math.pi / 100000.0 * x_grid * 1000.0
    y_wave = math.sqrt(3.0) * x_wave

    # for i, x_wavenumber in enumerate(x_grid):
    #     if x_wavenumber == 0.0:
    #         continue
    #     div_wave = DivWave(
    #         wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=x_wavenumber, y_wavenumber_factor=0.0
    #     )
    #     eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_divergence(
    #         divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
    #     )
    #     non_zero_eigen_vector, non_zero_eigen_value = [], []
    #     non_zero_eigen_vector.append(eigen_vector1[1])
    #     non_zero_eigen_vector.append(eigen_vector1[2])
    #     eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
    #     print(x_wavenumber, ' === ', non_zero_eigen_vector, '  -----  ', eigen_vector_wave_component)

    print('')
    print('////////////////////////////////////////////////')
    print('')

    for i, x_wavenumber in enumerate(x_grid):
        if x_wavenumber == 0.0:
            continue
        div_wave = DivWave(
            wave_type=WAVETYPE.X_AND_Y_WAVE, x_wavenumber_factor=x_wavenumber, y_wavenumber_factor=y_grid[i]
        )
        eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_divergence(
            divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
        )
        non_zero_eigen_vector, non_zero_eigen_value = [], []
        non_zero_eigen_vector.append(eigen_vector1[1])
        non_zero_eigen_vector.append(eigen_vector1[2])
        eigen_vector_wave_component = div_utils.compute_wave_vector_component([non_zero_eigen_vector[0], non_zero_eigen_vector[1]], div_wave)
        print(x_wavenumber, ' === ', non_zero_eigen_vector, '  -----  ', eigen_vector_wave_component)

    
if __name__ == "__main__":
    # test_divergence_single_time_step()
    # test_divergence_multiple_time_step()

    # test_divergence_single_time_step_on_globe()
    # test_divergence_multiple_time_step_on_globe()

    test_eigen_divergence()
    # plot_eigen_divergence()

    # test_debug()