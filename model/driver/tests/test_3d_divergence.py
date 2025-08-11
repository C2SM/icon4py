import datetime
import logging
import math
import os
from cmath import sqrt as c_sqrt, exp as c_exp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import xarray as xr
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm

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
)
from icon4py.model.driver import divergence_utils as div_utils
import scipy


log = logging.getLogger(__name__)


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

    base_input_path = "/scratch/mch/cong/data/div_converge_3d_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    div_wave = DivWave(wave_type=WAVETYPE.Y_AND_Z_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=30.0, z_wavenumber_factor=30.0)
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    div_utils.print_config(div_wave)

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = False
    enable_debug_message = False

    dtime_seconds = 1.0
    initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    mean_error = np.zeros(len(resolutions), dtype=float)
    end_time = (dtime_seconds, 5 * dtime_seconds, 10 * dtime_seconds, 15 * dtime_seconds, 20 * dtime_seconds ) # 
    for end_t in end_time:
        for i, res in enumerate(resolutions):
            input_path = base_input_path + res + "/ser_data/"
            grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + res + ".nc"

            parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
            configure_logging(run_path, experiment_type, parallel_props, disable_logging)

            end_date = initial_date + datetime.timedelta(seconds=end_t)

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
                output_seconds_interval=dtime_seconds,
                do_o2_divdamp=True,
                do_3d_divergence_damping=True,
                divergence_order=1,
                divdamp_fac=1.0 / 3.0 / math.sqrt(3.0) * 0.5,
                # divdamp_fac=4.0 / math.sqrt(3.0) / 17.0 * 4.5 * 1.2,
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

            if end_t == end_time[0]:
                w_full = 0.5 * (prognostic_state_list[0].w.ndarray[:, 1:] + prognostic_state_list[0].w.ndarray[:, :-1])
                div_utils.plot_vertical_section(
                    grid_filename,
                    0.5 * (prognostic_state_list[0].w.ndarray[:, 1:] + prognostic_state_list[0].w.ndarray[:, :-1]),
                    diagnostic_metric_state.z_ifc.ndarray,
                    "w",
                    "plot_initial_w"
                )
                
                for z_ind in range(5):
                    div_utils.plot_triedgedata(
                        grid_filename,
                        initial_vn,
                        f"VN at {int(end_t)} at z={z_ind}",
                        "plot_initial_vn"+"_"+str(z_ind)+"z",
                        div_wave,
                        eigen_vector=damping_eigen_vector,
                        plot_analytic=False,
                        z_ind=z_ind,
                    )
                div_utils.plot_tridata(
                    grid_filename,
                    diagnostic_state.u.ndarray[:,0:5],
                    mask,
                    f"U",
                    "plot_initial_U",
                )
                div_utils.plot_tridata(
                    grid_filename,
                    diagnostic_state.v.ndarray[:,0:5],
                    mask,
                    f"V",
                    "plot_initial_V",
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

            # os.system(f"rm -rf data_output*")

            cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
            cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
            cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

            v_scale = 90.0 / 7.5
            sphere_radius = 6371229.0
            u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
            v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
            # analytic_divergence = div_utils.determine_divergence_in_div_converge_experiment(
            #     cell_lat, cell_lon, diagnostic_metric_state.z_ifc.ndarray, timeloop.grid.num_levels, div_wave
            # )

            # div_utils.plot_tridata(
            #     grid_filename,
            #     analytic_divergence[:,0:5],
            #     mask,
            #     f"Analytic divergence",
            #     "plot_analytic_divergence",
            # )
            for z_ind in range(5):
                div_utils.plot_triedgedata(
                    grid_filename,
                    prognostic_state_list[0].vn.ndarray,
                    f"VN at {int(end_t)} at z={z_ind}",
                    "plot_final_vn" + str(int(end_t))+"_"+str(z_ind)+"z",
                    div_wave,
                    eigen_vector=damping_eigen_vector,
                    plot_analytic=False,
                    z_ind=z_ind,
                )
            div_utils.plot_tridata(
                grid_filename,
                timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:,0:5],
                mask,
                f"Computed divergence order 1 at {int(end_t)}",
                "plot_computed_divergence1_" + str(int(end_t)),
            )
            div_utils.plot_vertical_section(
                grid_filename,
                0.5 * (prognostic_state_list[0].w.ndarray[:, 1:] + prognostic_state_list[0].w.ndarray[:, :-1]),
                diagnostic_metric_state.z_ifc.ndarray,
                "w",
                "plot_final_w" + str(int(end_t)),
            )
            div_utils.plot_vertical_section_lat(
                grid_filename,
                timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray,
                diagnostic_metric_state.z_ifc.ndarray,
                "div",
                "plot_final_div" + str(int(end_t)),
            )

            # os.system(f"mv plot_* dummy* data_output* data_{res}")


def test_plot_3d_divergence():
    # x_grid = np.arange(0.1, 99.9, 0.1)
    # y_grid = np.arange(0.1, 99.9, 0.1)
    # z_grid = np.arange(0.1, 99.9, 0.1)
    x_grid = np.arange(0.1, 50.1, 0.1)
    y_grid = np.arange(0.1, 50.1, 0.1)
    z_grid = np.arange(0.1, 50.1, 0.1)
    x_wave = 2.0 * math.pi / 10000.0 * x_grid * 100.0
    y_wave = 2.0 * math.pi / 10000.0 * y_grid * 100.0
    z_wave = 2.0 * math.pi / 10000.0 * z_grid * 100.0
    xz_xwave, xz_zwave = np.meshgrid(x_wave, z_wave, indexing="ij")
    yz_ywave, yz_zwave = np.meshgrid(y_wave, z_wave, indexing="ij")

    order1_xzdamp1_real = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp2_real = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp3_real = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp4_real = np.zeros(xz_xwave.shape, dtype=float)
    order1_yzdamp1_real = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp2_real = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp3_real = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp4_real = np.zeros(yz_ywave.shape, dtype=float)
    order1_xzdamp1_imag = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp2_imag = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp3_imag = np.zeros(xz_xwave.shape, dtype=float)
    order1_xzdamp4_imag = np.zeros(xz_xwave.shape, dtype=float)
    order1_yzdamp1_imag = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp2_imag = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp3_imag = np.zeros(yz_ywave.shape, dtype=float)
    order1_yzdamp4_imag = np.zeros(yz_ywave.shape, dtype=float)

    order2_xzdamp1 = np.zeros(xz_xwave.shape, dtype=float)
    order2_xzdamp2 = np.zeros(xz_xwave.shape, dtype=float)
    order2_yzdamp1 = np.zeros(yz_ywave.shape, dtype=float)
    order2_yzdamp2 = np.zeros(yz_ywave.shape, dtype=float)

    for j, z_wavenumber in enumerate(z_grid):
        print("PROGRESS: ", j, z_wavenumber)
        for i, x_wavenumber in enumerate(x_grid):
            div_wave = DivWave(
                wave_type=WAVETYPE.X_AND_Z_WAVE,
                x_wavenumber_factor=x_wavenumber,
                y_wavenumber_factor=0.0,
                z_wavenumber_factor=z_wavenumber,
            )
            eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_3d_divergence(
                order=1, grid_space=1000.0, div_wave=div_wave, full_return=True, method=1
            )
            eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_3d_divergence(
                order=2, grid_space=1000.0, div_wave=div_wave, full_return=False, method=1
            )

            non_zero_eigen_vector, non_zero_eigen_value = [], []
            for eigen in range(4):
                if np.abs(eigen_value1[eigen]) > 1.e-12:
                    non_zero_eigen_vector.append(eigen_vector1[eigen])
                    non_zero_eigen_value.append(eigen_value1[eigen])
            # assert len(non_zero_eigen_vector) == 3, f"Len of non zero: {z_wavenumber} {x_wavenumber} {len(non_zero_eigen_vector)} {eigen_value1}"
            if len(non_zero_eigen_vector) == 3:
                eigen_vector_wave_component = div_utils.compute_wave_vector_component(
                    [non_zero_eigen_vector[0], non_zero_eigen_vector[1], non_zero_eigen_vector[2]],
                    div_wave,
                    threeD=True,
                )
                argsort = np.abs(eigen_vector_wave_component).argsort()
                eigen_vector_wave_component = eigen_vector_wave_component[argsort[::-1]]
                non_zero_eigen_value = np.array(non_zero_eigen_value, dtype=complex)
                non_zero_eigen_value = non_zero_eigen_value[argsort[::-1]]
                order1_xzdamp1_real[i, j] = non_zero_eigen_value[0].real
                order1_xzdamp1_imag[i, j] = non_zero_eigen_value[0].imag
                order1_xzdamp2_real[i, j] = non_zero_eigen_value[1].real
                order1_xzdamp2_imag[i, j] = non_zero_eigen_value[1].imag
                order1_xzdamp3_real[i, j] = non_zero_eigen_value[2].real
                order1_xzdamp3_imag[i, j] = non_zero_eigen_value[2].imag
            else:
                eigen_vector_wave_component = div_utils.compute_wave_vector_component(
                    [non_zero_eigen_vector[0], non_zero_eigen_vector[1]],
                    div_wave,
                    threeD=True,
                )
                argsort = np.abs(eigen_vector_wave_component).argsort()
                eigen_vector_wave_component = eigen_vector_wave_component[argsort[::-1]]
                non_zero_eigen_value = np.array(non_zero_eigen_value, dtype=complex)
                non_zero_eigen_value = non_zero_eigen_value[argsort[::-1]]
                order1_xzdamp1_real[i, j] = non_zero_eigen_value[0].real
                order1_xzdamp1_imag[i, j] = non_zero_eigen_value[0].imag
                order1_xzdamp2_real[i, j] = non_zero_eigen_value[1].real
                order1_xzdamp2_imag[i, j] = non_zero_eigen_value[1].imag

            order2_xzdamp1[i, j] = eigen_value2[np.argmax(np.abs(eigen_value2))]
            eigen_value2 = np.sort(eigen_value2)
            order2_xzdamp2[i, j] = eigen_value2[2] # minus, zero, plus

        for i, y_wavenumber in enumerate(y_grid):
            div_wave = DivWave(
                wave_type=WAVETYPE.Y_AND_Z_WAVE,
                x_wavenumber_factor=0.0,
                y_wavenumber_factor=y_wavenumber,
                z_wavenumber_factor=z_wavenumber,
            )
            eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_3d_divergence(
                order=1, grid_space=1000.0, div_wave=div_wave, full_return=True, method=1
            )
            eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_3d_divergence(
                order=2, grid_space=1000.0, div_wave=div_wave, full_return=False, method=1
            )

            non_zero_eigen_vector, non_zero_eigen_value = [], []
            for eigen in range(4):
                if np.abs(eigen_value1[eigen]) > 1.e-10:
                    non_zero_eigen_vector.append(eigen_vector1[eigen])
                    non_zero_eigen_value.append(eigen_value1[eigen])
            assert len(non_zero_eigen_vector) == 3, f"Len of non zero: {z_wavenumber} {y_wavenumber} {len(non_zero_eigen_vector)}"
            eigen_vector_wave_component = div_utils.compute_wave_vector_component(
                [non_zero_eigen_vector[0], non_zero_eigen_vector[1], non_zero_eigen_vector[2]],
                div_wave,
                threeD=True,
            )
            argsort = np.abs(eigen_vector_wave_component).argsort()
            eigen_vector_wave_component = eigen_vector_wave_component[argsort[::-1]]
            non_zero_eigen_value = np.array(non_zero_eigen_value, dtype=complex)
            non_zero_eigen_value = non_zero_eigen_value[argsort[::-1]]
            order1_yzdamp1_real[i, j] = non_zero_eigen_value[0].real
            order1_yzdamp1_imag[i, j] = non_zero_eigen_value[0].imag
            order1_yzdamp2_real[i, j] = non_zero_eigen_value[1].real
            order1_yzdamp2_imag[i, j] = non_zero_eigen_value[1].imag
            order1_yzdamp3_real[i, j] = non_zero_eigen_value[2].real
            order1_yzdamp3_imag[i, j] = non_zero_eigen_value[2].imag

            order2_yzdamp1[i, j] = eigen_value2[np.argmax(np.abs(eigen_value2))]
            eigen_value2 = np.sort(eigen_value2)
            order2_yzdamp2[i, j] = eigen_value2[2] # minus, zero, plus
            # print(i, eigen_value2)

    # post processing:
    for j in range(1, z_grid.shape[0]):
        for i in range(1, x_grid.shape[0]):
            if j > z_grid.shape[0] / 2 and np.abs(order1_xzdamp1_real[i-1, j] - order1_xzdamp2_real[i, j]) < np.abs(order1_xzdamp1_real[i-1, j] - order1_xzdamp1_real[i, j]):
            # if np.abs(
            #     0.25 * (
            #         order1_xzdamp1_real[i-1, j] + order1_xzdamp1_real[i+1, j] + order1_xzdamp1_real[i, j-1] + order1_xzdamp1_real[i, j+1]
            #     ) - order1_xzdamp2_real[i, j]
            # ) < np.abs(
            #     0.25 * (
            #         order1_xzdamp1_real[i-1, j] + order1_xzdamp1_real[i+1, j] + order1_xzdamp1_real[i, j-1] + order1_xzdamp1_real[i, j+1]
            #     ) - order1_xzdamp1_real[i, j]
            # ):
                temp = order1_xzdamp2_real[i, j]
                order1_xzdamp2_real[i, j] = order1_xzdamp1_real[i, j]
                order1_xzdamp1_real[i, j] = temp
                temp = order1_xzdamp2_imag[i, j]
                order1_xzdamp2_imag[i, j] = order1_xzdamp1_imag[i, j]
                order1_xzdamp1_imag[i, j] = temp
            if np.abs(order1_yzdamp1_real[i-1, j] - order1_yzdamp2_real[i, j]) < np.abs(order1_yzdamp1_real[i-1, j] - order1_yzdamp1_real[i, j]):
            # if np.abs(
            #     0.25 * (
            #         order1_yzdamp1_real[i-1, j] + order1_yzdamp1_real[i+1, j] + order1_yzdamp1_real[i, j-1] + order1_yzdamp1_real[i, j+1]
            #     ) - order1_yzdamp2_real[i, j]
            # ) < np.abs(
            #     0.25 * (
            #         order1_yzdamp1_real[i-1, j] + order1_yzdamp1_real[i+1, j] + order1_yzdamp1_real[i, j-1] + order1_yzdamp1_real[i, j+1]
            #     ) - order1_yzdamp1_real[i, j]
            # ):
                temp = order1_yzdamp2_real[i, j]
                order1_yzdamp2_real[i, j] = order1_yzdamp1_real[i, j]
                order1_yzdamp1_real[i, j] = temp
                temp = order1_yzdamp2_imag[i, j]
                order1_yzdamp2_imag[i, j] = order1_yzdamp1_imag[i, j]
                order1_yzdamp1_imag[i, j] = temp
            if np.abs(order1_yzdamp2_real[i-1, j] - order1_yzdamp3_real[i, j]) < np.abs(order1_yzdamp2_real[i-1, j] - order1_yzdamp2_real[i, j]):
                temp = order1_yzdamp3_real[i, j]
                order1_yzdamp3_real[i, j] = order1_yzdamp2_real[i, j]
                order1_yzdamp2_real[i, j] = temp
                temp = order1_yzdamp3_imag[i, j]
                order1_yzdamp3_imag[i, j] = order1_yzdamp2_imag[i, j]
                order1_yzdamp2_imag[i, j] = temp
        for i in range(x_grid.shape[0] - 2, 1, -1):
            if np.abs(order1_yzdamp1_real[i+1, j] - order1_yzdamp2_real[i, j]) < np.abs(order1_yzdamp1_real[i+1, j] - order1_yzdamp1_real[i, j]):
                temp = order1_yzdamp2_real[i, j]
                order1_yzdamp2_real[i, j] = order1_yzdamp1_real[i, j]
                order1_yzdamp1_real[i, j] = temp
                temp = order1_yzdamp2_imag[i, j]
                order1_yzdamp2_imag[i, j] = order1_yzdamp1_imag[i, j]
                order1_yzdamp1_imag[i, j] = temp
    for i in range(1, x_grid.shape[0]):
        if np.abs(order1_yzdamp2_real[i-1, 0] - order1_yzdamp3_real[i, 0]) < np.abs(order1_yzdamp2_real[i-1, 0] - order1_yzdamp2_real[i, 0]):
            temp = order1_yzdamp3_real[i, 0]
            order1_yzdamp3_real[i, 0] = order1_yzdamp2_real[i, 0]
            order1_yzdamp2_real[i, 0] = temp
            temp = order1_yzdamp3_imag[i, 0]
            order1_yzdamp3_imag[i, 0] = order1_yzdamp2_imag[i, 0]
            order1_yzdamp2_imag[i, 0] = temp
        last_ind = z_grid.shape[0] - 1
        if np.abs(order1_yzdamp2_real[i-1, last_ind] - order1_yzdamp3_real[i, last_ind]) < np.abs(order1_yzdamp2_real[i-1, last_ind] - order1_yzdamp2_real[i, last_ind]):
            temp = order1_yzdamp3_real[i, last_ind]
            order1_yzdamp3_real[i, last_ind] = order1_yzdamp2_real[i, last_ind]
            order1_yzdamp2_real[i, last_ind] = temp
            temp = order1_yzdamp3_imag[i, last_ind]
            order1_yzdamp3_imag[i, last_ind] = order1_yzdamp2_imag[i, last_ind]
            order1_yzdamp2_imag[i, last_ind] = temp

    order1_xzdamp1_real = np.sqrt(order1_xzdamp1_real**2 + order1_xzdamp1_imag**2)
    order1_xzdamp2_real = np.sqrt(order1_xzdamp2_real**2 + order1_xzdamp2_imag**2)
    order1_xzdamp3_real = np.sqrt(order1_xzdamp3_real**2 + order1_xzdamp3_imag**2)
    order1_yzdamp1_real = np.sqrt(order1_yzdamp1_real**2 + order1_yzdamp1_imag**2)
    order1_yzdamp2_real = np.sqrt(order1_yzdamp2_real**2 + order1_yzdamp2_imag**2)
    order1_yzdamp3_real = np.sqrt(order1_yzdamp3_real**2 + order1_yzdamp3_imag**2)

    def _plot_damp(x_or_y, damping, title: str):
        plt.close()
        f, ax = plt.subplots(constrained_layout=True)
        # cmap = plt.get_cmap('plasma')
        cmap = plt.get_cmap("gist_rainbow")
        cmap = cmap.reversed() 
        boundaries = np.linspace(damping.min(), damping.max(), 101)
        print(title, ' --- ', damping.min(), damping.max())
        lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
        if x_or_y == 1:
            cp = ax.contourf(xz_xwave, xz_zwave, damping, cmap=cmap, levels=boundaries, norm=lnorm)
            ax.set_xlabel("x wavenumber")
        elif x_or_y == 2:
            cp = ax.contourf(yz_ywave, yz_zwave, damping, cmap=cmap, levels=boundaries, norm=lnorm)
            ax.set_xlabel("y wavenumber")
        else:
            NotImplementedError()
        cb1 = f.colorbar(cp, location="right")
        ax.set_ylabel("z wavenumber")
        tick = [0, math.pi / 4.0, math.pi / 2.0, math.pi * 3.0 / 4.0, math.pi]
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[1] = "$2\\pi/8\\Delta x$"
        labels[2] = "$2\\pi/4\\Delta x$"
        labels[3] = "$3\\pi/4\\Delta x$"
        labels[4] = "$2\\pi/2\\Delta x$"
        ax.set_xticklabels(labels)
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[1] = "$2\\pi/8\\Delta x$"
        labels[2] = "$2\\pi/4\\Delta x$"
        labels[3] = "$3\\pi/4\\Delta x$"
        labels[4] = "$2\\pi/2\\Delta x$"
        ax.set_yticklabels(labels)
        plt.savefig(title+".pdf", format="pdf", dpi=400)

    _plot_damp(1, order2_xzdamp1, "fig_order2_xzdamp1")
    _plot_damp(1, order2_xzdamp2, "fig_order2_xzdamp2")
    _plot_damp(2, order2_yzdamp1, "fig_order2_yzdamp1")
    _plot_damp(2, order2_yzdamp2, "fig_order2_yzdamp2")

    _plot_damp(1, order1_xzdamp1_real, "fig_order1_xzdamp1_real")
    _plot_damp(1, order1_xzdamp2_real, "fig_order1_xzdamp2_real")
    _plot_damp(1, order1_xzdamp3_real, "fig_order1_xzdamp3_real")
    # _plot_damp(1, order1_xzdamp4_real, "fig_order1_xzdamp4_real")
    _plot_damp(1, order1_xzdamp1_imag, "fig_order1_xzdamp1_imag")
    _plot_damp(1, order1_xzdamp2_imag, "fig_order1_xzdamp2_imag")
    _plot_damp(1, order1_xzdamp3_imag, "fig_order1_xzdamp3_imag")
    # _plot_damp(1, order1_xzdamp4_imag, "fig_order1_xzdamp4_imag")
    _plot_damp(1, order1_yzdamp1_real, "fig_order1_yzdamp1_real")
    _plot_damp(1, order1_yzdamp2_real, "fig_order1_yzdamp2_real")
    _plot_damp(1, order1_yzdamp3_real, "fig_order1_yzdamp3_real")
    # _plot_damp(1, order1_yzdamp4_real, "fig_order1_yzdamp4_real")
    _plot_damp(1, order1_yzdamp1_imag, "fig_order1_yzdamp1_imag")
    _plot_damp(1, order1_yzdamp2_imag, "fig_order1_yzdamp2_imag")
    _plot_damp(1, order1_yzdamp3_imag, "fig_order1_yzdamp3_imag")
    # _plot_damp(1, order1_yzdamp4_imag, "fig_order1_yzdamp4_imag")

    # diff = 0.0
    # _plot_damp(1, order1_yzdamp2_real[1:,:] - order1_yzdamp2_real[:-1,:], "fig_order1_yzdamp2_real_diff")
    # _plot_damp(1, order1_yzdamp3_real[1:,:] - order1_yzdamp3_real[:-1,:], "fig_order1_yzdamp3_real_diff")
    

def test_3d_eigenvalue():
    x_grid = np.arange(1.0, 99.0, 1.0)
    y_grid = np.arange(5.0, 55.0, 5.0)
    z_grid = np.array([10.0, 30.0, 50.0])#np.arange(1.0, 99.0, 1.0)

    print("########################################################")
    print("########################################################")
    print("########################################################")
    order1_yzdamp1_real = np.zeros((z_grid.shape[0], y_grid.shape[0]), dtype=float)
    order1_yzdamp2_real = np.zeros((z_grid.shape[0], y_grid.shape[0]), dtype=float)
    order1_yzdamp3_real = np.zeros((z_grid.shape[0], y_grid.shape[0]), dtype=float)
    for j, z_wavenumber in enumerate(z_grid):
        print()
        print("PROGRESS: ", j, z_wavenumber)
        print()
        for i, y_wavenumber in enumerate(y_grid):
            div_wave = DivWave(
                wave_type=WAVETYPE.X_AND_Z_WAVE,
                y_wavenumber_factor=0.0,
                x_wavenumber_factor=y_wavenumber,
                z_wavenumber_factor=z_wavenumber,
            )
            eigen_value1, eigen_vector1, eigen_vector_cartesian1 = div_utils.eigen_3d_divergence(
                order=1, grid_space=1000.0, div_wave=div_wave, full_return=True, method=1
            )
            eigen_value2, eigen_vector2, eigen_vector_cartesian2 = div_utils.eigen_3d_divergence(
                order=2, grid_space=1000.0, div_wave=div_wave, full_return=False, method=1
            )
            non_zero_eigen_vector, non_zero_eigen_value = [], []
            for eigen in range(4):
                if np.abs(eigen_value1[eigen]) > 1.e-10:
                    non_zero_eigen_vector.append(eigen_vector1[eigen])
                    non_zero_eigen_value.append(eigen_value1[eigen])
            # assert len(non_zero_eigen_vector) == 3, f"Len of non zero: {z_wavenumber} {y_wavenumber} {len(non_zero_eigen_vector)}"
            if len(non_zero_eigen_vector) == 3:
                eigen_vector_wave_component = div_utils.compute_wave_vector_component(
                    [non_zero_eigen_vector[0], non_zero_eigen_vector[1], non_zero_eigen_vector[2]],
                    div_wave,
                    threeD=True,
                )
                _non_zero_eigen_vector = np.zeros((3, 4))
                argsort = np.abs(eigen_vector_wave_component).argsort()
                eigen_vector_wave_component = eigen_vector_wave_component[argsort[::-1]]
                non_zero_eigen_value = np.array(non_zero_eigen_value)
                non_zero_eigen_value = non_zero_eigen_value[argsort[::-1]]
                # print(eigen_vector_wave_component)
                for eigen in range(3):
                    _non_zero_eigen_vector[eigen] = non_zero_eigen_vector[eigen]
                non_zero_eigen_vector = _non_zero_eigen_vector[argsort[::-1]]
                non_zero_eigen_vector = non_zero_eigen_vector[argsort[::-1]]
                order1_yzdamp1_real[j, i] = non_zero_eigen_value[0].imag
                order1_yzdamp2_real[j, i] = non_zero_eigen_value[1].imag
                order1_yzdamp3_real[j, i] = non_zero_eigen_value[2].imag
            else:
                eigen_vector_wave_component = div_utils.compute_wave_vector_component(
                    [non_zero_eigen_vector[0], non_zero_eigen_vector[1]],
                    div_wave,
                    threeD=True,
                )
                _non_zero_eigen_vector = np.zeros((2, 4))
                argsort = np.abs(eigen_vector_wave_component).argsort()
                eigen_vector_wave_component = eigen_vector_wave_component[argsort[::-1]]
                non_zero_eigen_value = np.array(non_zero_eigen_value)
                non_zero_eigen_value = non_zero_eigen_value[argsort[::-1]]
                # print(eigen_vector_wave_component)
                for eigen in range(2):
                    _non_zero_eigen_vector[eigen] = non_zero_eigen_vector[eigen]
                non_zero_eigen_vector = _non_zero_eigen_vector[argsort[::-1]]
                non_zero_eigen_vector = non_zero_eigen_vector[argsort[::-1]]
                order1_yzdamp1_real[j, i] = non_zero_eigen_value[0].imag
                order1_yzdamp2_real[j, i] = non_zero_eigen_value[1].imag
            print("DEBUG", f"{order1_yzdamp1_real[j, i]:.5e}, {order1_yzdamp2_real[j, i]:.5e}, {order1_yzdamp3_real[j, i]:.5e}")
            # if np.abs(eigen_vector_wave_component[0]) > np.abs(eigen_vector_wave_component[1]):
            # print(y_wavenumber, ": ", eigen_vector_wave_component[0], " == ", non_zero_eigen_vector[0].real, " == ", non_zero_eigen_value[0])
            # print(y_wavenumber, ": ", eigen_vector_wave_component[1], " == ", non_zero_eigen_vector[1].real, " == ", non_zero_eigen_value[1])
            # print(y_wavenumber, ": ", eigen_vector_wave_component[2], " == ", non_zero_eigen_vector[2].real, " == ", non_zero_eigen_value[2])
            # print(i, eigen_value1)
            # for item in range(4):
            #     print("eigen value ", item, f"{eigen_vector1[item, 0]} {eigen_vector1[item, 1]} {eigen_vector1[item, 2]} {eigen_vector1[item, 3]}")

    for j, z_wavenumber in enumerate(z_grid):
        print()
        for i, y_wavenumber in enumerate(y_grid):
            print(j, i, f"{order1_yzdamp1_real[j, i]:.5e} {order1_yzdamp2_real[j, i]:.5e} {order1_yzdamp3_real[j, i]:.5e}")
    

def test_debug():
    div_wave = DivWave(
        # wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=30.0, y_wavenumber_factor=0.0
        wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=70.0
    )
    z_wavenumber_factor = 50.0

    print()
    print("x, y, z wave numbers: ", div_wave.x_wavenumber_factor, div_wave.y_wavenumber_factor, z_wavenumber_factor)
    print()

    grid_space = 1000.0
    k = 2.0 * math.pi * div_wave.x_wavenumber_factor * 1.0e-5
    l = 2.0 * math.pi * div_wave.y_wavenumber_factor * 1.0e-5
    a = 0.25 * 3.0 * grid_space * k
    b = 0.25 * math.sqrt(3.0) * grid_space * l
    nn = 0.5 * grid_space * math.pi * z_wavenumber_factor * 1.0e-5 * 2.0 # n detla_z / 2, the extra factor of 2 at the end may be needed for complete Fourier space

    h = 1000.0
    delta_z = 1000.0
    d_tri = h / math.sqrt(3.0)
    d_hex = h * math.sqrt(3.0)
    constant_2d = 1.0 # 8.0 * KD / h**2
    constant_hw = -math.sqrt(3) / 2.0 # - 4.0 * math.sqrt(3.0) * KD / h / delta_z
    constant_w = 1.j / math.sqrt(3.0) # 8.0 / math.sqrt(3.0) * 1.j * KD / h / delta_z

    diagonal_element = -1.0
    h_2d_11 = constant_2d * diagonal_element
    h_2d_12 = constant_2d * math.cos(a + b)
    h_2d_13 = constant_2d * math.cos(a - b)
    h_2d_21 = h_2d_12
    h_2d_22 = constant_2d * diagonal_element
    h_2d_23 = constant_2d * -math.cos(2.0 * a)
    h_2d_31 = h_2d_13
    h_2d_32 = h_2d_23
    h_2d_33 = constant_2d * diagonal_element

    h_1 = constant_hw * math.sin(0.5 * d_tri * l) * math.sin(nn)
    h_2 = constant_hw * math.sin(0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l) * math.sin(nn)
    h_3 = constant_hw * math.sin(-0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l) * math.sin(nn)
    w1 = constant_w * c_exp(-0.5j * d_tri * l) * math.sin(nn)
    w2 = constant_w * c_exp(1.0j * (0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l)) * math.sin(nn)
    w3 = constant_w * c_exp(1.0j * (-0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l)) * math.sin(nn)
    w4 = -0.5 * math.sin(nn)**2 # - 4.0 * KD / delta_z**2 * math.sin(nn)**2

    matrix = np.array(
        [
            [h_2d_11, h_2d_12, h_2d_13, h_1],
            [h_2d_21, h_2d_22, h_2d_23, h_2],
            [h_2d_31, h_2d_32, h_2d_33, h_3],
            [w1, w2, w3, w4],
        ], dtype = complex,
    )

    for i in range(matrix.shape[0]):
        print([f"{matrix[i, j]:.5e}" for j in range(matrix.shape[1])])

    for i in range(3):
        print()
    print("UPPER TRIANGULAR MATRIX: ")

    T2, Z2 = scipy.linalg.schur(matrix, output='complex')

    for i in range(T2.shape[0]):
        print([f"{T2[i, j]:.5e}" for j in range(T2.shape[1])])

    for i in range(3):
        print()
    print("UNITARY MATRIX: ")

    for i in range(Z2.shape[0]):
        print([f"{Z2[i, j]:.5e}" for j in range(Z2.shape[1])])

    print()
    test_vector = np.array([1.0, 4.0, 4.0, 0.0], dtype=float)
    print("testing normal dot product: ")
    print(np.dot(Z2[:,0], Z2[:,1]))
    print(np.dot(Z2[:,0], Z2[:,2]))
    print(np.dot(Z2[:,1], Z2[:,2]))

    print()
    print("testing complex dot product: ")
    print(np.vdot(Z2[:,0], Z2[:,1]))
    print(np.vdot(Z2[:,0], Z2[:,2]))
    print(np.vdot(Z2[:,1], Z2[:,2]))

    print()
    print("coefficient of test vector: ", test_vector)
    print(np.vdot(Z2[:,0], test_vector))
    print(np.vdot(Z2[:,1], test_vector))
    print(np.vdot(Z2[:,2], test_vector))

    # second order divergence
    constant_2d = 1.0 # 4.0 * KD / 9.0 / h**2
    constant_hw = -9.0 / math.sqrt(3.0) # -4.0 * KD / h / delta_z / math.sqrt(3.0)
    constant_w = -18.0 / math.sqrt(3.0) # -8.0 / math.sqrt(3.0) * KD / h / delta_z

    diagonal_element = -1.0
    h_2d_11 = constant_2d * (math.cos(4.0*b) - 1.0)
    h_2d_12 = constant_2d * (math.cos(a - 3.0*b) - math.cos(a + b))
    h_2d_13 = constant_2d * (math.cos(a + 3.0*b) - math.cos(a - b))
    h_2d_21 = h_2d_12
    h_2d_22 = constant_2d * (math.cos(2.0*a - 2.0*b) - 1.0)
    h_2d_23 = constant_2d * (math.cos(2.0*b) - math.cos(2.0*a))
    h_2d_31 = h_2d_13
    h_2d_32 = h_2d_23
    h_2d_33 = constant_2d * (math.cos(2.0*a + 2.0*b) - 1.0)

    hw_1 = math.sin(0.5 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
    hw_2 = math.sin(-0.25 * 3.0 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
    hw_3 = math.sin(0.25 * 3.0 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
    h_1 = constant_hw * hw_1
    h_2 = constant_hw * hw_2
    h_3 = constant_hw * hw_3
    w1 = constant_w * hw_1
    w2 = constant_w * hw_2
    w3 = constant_w * hw_3
    w4 = -9.0 * math.sin(nn)**2 # - 4.0 * KD / delta_z**2 * math.sin(nn)**2

    matrix = np.array(
        [
            [h_2d_11, h_2d_12, h_2d_13, h_1],
            [h_2d_21, h_2d_22, h_2d_23, h_2],
            [h_2d_31, h_2d_32, h_2d_33, h_3],
            [w1, w2, w3, w4],
        ], dtype = float,
    )

    T2, Z2 = scipy.linalg.schur(matrix, output='complex')

    print()
    print()
    print()

    for i in range(T2.shape[0]):
        print([f"{T2[i, j]:.5e}" for j in range(T2.shape[1])])

    for i in range(3):
        print()
    print("UNITARY MATRIX: ")

    for i in range(Z2.shape[0]):
        print([f"{Z2[i, j]:.5e}" for j in range(Z2.shape[1])])


if __name__ == "__main__":
    test_divergence_multiple_time_step()
    # test_plot_3d_divergence()
    # test_3d_eigenvalue()
    # test_debug()
    