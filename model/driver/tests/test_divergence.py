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

import os

import logging
import numpy as np
import math
import datetime

from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.driver.dycore_driver import TimeLoop, initialize
from icon4py.model.common.settings import xp
from icon4py.model.driver.initialization_utils import (
    ExperimentType,
    SerializationType,
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)
from pathlib import Path
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import xarray as xr


log = logging.getLogger(__name__)


def create_mask(grid_filename: str) -> np.ndarray:

    grid = xr.open_dataset(grid_filename, engine='netcdf4')

    mask = (
        (grid.clat.values > np.deg2rad(-7.3))
        & (grid.clat.values < np.deg2rad(7.3))
    )
    # mask = xp.ones(grid.clat.shape[0], dtype = bool)
    voc = grid.vertex_of_cell.T.values - 1
    vx = grid.cartesian_x_vertices.values
    vy = grid.cartesian_y_vertices.values
    vx_c = vx[voc]
    vx_c1 = np.roll(vx_c, shift=1, axis=1)
    vy_c = vy[voc]
    vy_c1 = np.roll(vy_c, shift=1, axis=1)
    vv_distance = np.sum(np.abs(vx_c - vx_c1), axis=1) + np.sum(np.abs(vy_c - vy_c1), axis=1)
    neighbor_cell = grid.neighbor_cell_index.T.values - 1
    threshold = 2.0
    interior_mask_ = (vv_distance < threshold * 2.0 * grid.mean_edge_length)
    interior_mask = np.array(interior_mask_, copy=True)
    interior_mask = np.where(np.sum(interior_mask[neighbor_cell],axis=1) < 3, False, interior_mask)
    mask = (mask & interior_mask)

    return mask


def plot_tridata(grid_filename: str, data: xp.ndarray, mask:np.ndarray, title: str, output_filename: str):

    grid = xr.open_dataset(grid_filename, engine='netcdf4')

    voc = grid.vertex_of_cell.T[mask].values - 1
    if len(data.shape) == 1:
        number_of_layers = 1
    else:
        number_of_layers = data.shape[1]
    log.info(f"plotting {title} with levels {number_of_layers}, {len(data.shape)}, {data.shape}")
    
    for k in range(number_of_layers):
        if len(data.shape) == 1:
            cell_data = np.asarray(data[mask])
        else:
            cell_data = np.asarray(data[mask,k])
        data_max, data_min = cell_data.max(), cell_data.min()
        log.info(f"data max min: {data_max}, {data_min}")

        used_vertices = np.unique(voc)
        lat_min = grid.vlat[used_vertices].min().values
        lat_max = grid.vlat[used_vertices].max().values
        lon_min = grid.vlon[used_vertices].min().values
        lon_max = grid.vlon[used_vertices].max().values

        # N = 20
        # cmap = plt.get_cmap('jet', N)
        #norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        # creating ScalarMappable 
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
        # sm.set_array([]) 
        
        cmap = plt.get_cmap('seismic')
        norm = TwoSlopeNorm(vmin=data_min-1.e-8, vcenter=0.0, vmax=data_max+1.e-8)

        plt.tripcolor(grid.vlon, grid.vlat, voc, cell_data, cmap=cmap, norm=norm)
        plt.title(title)
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        
        plt.colorbar() # sm, ticks=np.linspace(0, 2, N)
        
        plt.savefig(output_filename+'_at_'+str(k)+'_level.png', dpi=300)
        plt.clf()
        
    return

def plot_error(error: np.ndarray, dx: np.ndarray, axis_labels: list):
    
    if not isinstance(error, np.ndarray):
        raise TypeError(f"error is not a numpy array, instead it is {type(error)}")
    if len(error.shape) != 1:
        raise ValueError(f"error is not one dimensional, instead it is {error.shape}")
    
    assert error.shape[0] == len(axis_labels)

    max_xaxis_value = error.shape[0]

    plt.close()

    f, ax = plt.subplots()

    x = np.arange(max_xaxis_value, dtype=float)
    x = x + 1.0
    first_order = 2.0 * error[0] * dx / dx[0]
    second_order = 0.5 * error[0] * dx**2 / dx[0]**2
    ax.plot(dx, error, label="error")
    ax.plot(dx, first_order, label="O(dx)")
    ax.plot(dx, second_order, label="O(dx^2)")
    # ax.set_xlim(0.0, max_xaxis_value + 0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
    ax.set_ylabel("l1 error")
    ax.set_xlabel("edge length (m)")
    ax.set_title('Mean error in divergence')
    plt.legend()
    plt.savefig('plot_div_error.png', dpi=300)

def test_divergence_single_time_step():
    
    resolutions = (
        '1000m',
        '500m',
        '250m',
        '125m',
    )
    dx = np.array(
        (
        1000.0,
        500.0,
        250.0,
        125.0,
        ), dtype=float
    )
    
    base_input_path = '/scratch/mch/cong/data/div_converge_res'
    grid_file_folder = '/scratch/mch/cong/grid-generator/grids/'
    run_path = './'
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = True
    
    dtime_seconds = 1.0
    output_interval = dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=dtime_seconds)
    mean_error = np.zeros(len(resolutions), dtype = float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + '/ser_data/'
        grid_filename = grid_file_folder + 'Torus_Triangles_100km_x_100km_res' + res + '.nc'

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

        mask = create_mask(grid_filename)
        plot_tridata(grid_filename, diagnostic_state.u.ndarray, mask, f'Initial u at {res}', 'plot_initial_u_'+res)
        plot_tridata(grid_filename, diagnostic_state.v.ndarray, mask, f'Initial v at {res}', 'plot_initial_v_'+res)

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
        # analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (1.e-5 * 2.0 * math.pi * xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + 1.e-5 * math.pi * xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        
        mean_error[i] = float(xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence)*cell_area) ) # / xp.sum(cell_area)

        divergence_error = timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence

        plot_tridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray, mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        plot_tridata(grid_filename, analytic_divergence, mask, f'Analytic divergence at {res}', 'plot_analytic_divergence_'+res)
        plot_tridata(grid_filename, divergence_error, mask, f'Divergence error at {res}', 'plot_error_divergence_'+res)

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error, dx[0:len(resolutions)], resolutions)


def test_divergence_multiple_time_step():
    
    resolutions = (
        '1000m',
        # '500m',
        # '250m',
        # '125m',
    )
    dx = np.array(
        (
        1000.0,
        500.0,
        250.0,
        125.0,
        ), dtype=float
    )
    
    base_input_path = '/scratch/mch/cong/data/div_converge_res'
    grid_file_folder = '/scratch/mch/cong/grid-generator/grids/'
    run_path = './'
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = True
    
    dtime_seconds = 1.0
    output_interval = 1000*dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=10000*dtime_seconds)
    mean_error = np.zeros(len(resolutions), dtype = float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + '/ser_data/'
        grid_filename = grid_file_folder + 'Torus_Triangles_100km_x_100km_res' + res + '.nc'

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

        mask = create_mask(grid_filename)
        
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

        plot_tridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv_vn.ndarray, mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        
        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error, dx[0:len(resolutions)], resolutions)


if __name__ == "__main__":
    # test_divergence_single_time_step()
    test_divergence_multiple_time_step()