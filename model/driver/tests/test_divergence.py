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
from icon4py.model.common.settings import xp, device
from icon4py.model.common.config import Device
from icon4py.model.driver.initialization_utils import (
    ExperimentType,
    SerializationType,
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
    U_AMPLIFICATION_FACTOR,
    WAVENUMBER_FACTOR,
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
        (grid.clat.values > np.deg2rad(-7.4))
        & (grid.clat.values < np.deg2rad(7.4))
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


def plot_tridata(grid_filename: str, data: xp.ndarray, mask: np.ndarray, title: str, output_filename: str):

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
        
        plt.savefig(output_filename+'_at_'+str(k)+'_level.pdf', format='pdf', dpi=500)
        plt.clf()
        
    return


def create_globe_mask(grid_filename: str) -> np.ndarray:

    grid = xr.open_dataset(grid_filename, engine='netcdf4')

    mask = (
        (grid.clat.values > np.deg2rad(-85))
        & (grid.clat.values < np.deg2rad(85))
    )
    voc = grid.vertex_of_cell.T.values - 1
    v_lon = grid.vlon.values # pi unit
    v_lat = grid.vlat.values # pi unit
    sphere_radius = 6371229.0
    # vx = sphere_radius * np.cos(v_lat) * np.cos(v_lon)
    # vy = sphere_radius * np.cos(v_lat) * np.cos(v_lat)
    # vz = sphere_radius * np.sin(v_lat)
    vlon_c = v_lon[voc]
    vlon_c1 = np.roll(vlon_c, shift=1, axis=1)
    vlat_c = v_lat[voc]
    vlat_c1 = np.roll(vlat_c, shift=1, axis=1)
    # vv_distance = np.sum(np.abs(vx_c - vx_c1)**2, axis=1) + np.sum(np.abs(vy_c - vy_c1)**2, axis=1) + np.sum(np.abs(vz_c - vz_c1)**2, axis=1)
    # vv_distance = np.sum((vx_c - vx_c1)**2 + (vy_c - vy_c1)**2 + (vz_c - vz_c1)**2, axis=1)
    vv_distance = np.sum(np.abs(vlon_c - vlon_c1) + np.abs(vlat_c - vlat_c1), axis=1)
    # mean_edge_length = np.sum(vv_distance)/float(vv_distance.shape[0])
    mean_edge_length = math.pi
    neighbor_cell = grid.neighbor_cell_index.T.values - 1
    threshold = 1.0
    interior_mask_ = (vv_distance < threshold * mean_edge_length)
    interior_mask = np.array(interior_mask_, copy=True)
    interior_mask = np.where(np.sum(interior_mask[neighbor_cell],axis=1) < 3, False, interior_mask)
    mask = (mask & interior_mask)

    return mask

def plot_globetridata(grid_filename: str, data: xp.ndarray, mask: np.ndarray, title: str, output_filename: str):

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
        
        plt.savefig(output_filename+'_at_'+str(k)+'_level.pdf', format='pdf', dpi=500)
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
    plt.savefig('plot_div_error.pdf', format='pdf', dpi=500)

def test_divergence_single_time_step():
    
    resolutions = (
        '1000m',
        '500m',
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
    enable_debug_message = False
    
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
        plot_tridata(grid_filename, diagnostic_state.u.asnumpy(), mask, f'Initial u at {res}', 'plot_initial_u_'+res)
        plot_tridata(grid_filename, diagnostic_state.v.asnumpy(), mask, f'Initial v at {res}', 'plot_initial_v_'+res)

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

        v_scale = 90.0 / 7.5
        sphere_radius = 6371229.0
        # analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * WAVENUMBER_FACTOR * (U_AMPLIFICATION_FACTOR * 1.e-5 * 2.0 * math.pi * xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + 1.e-5 * math.pi * xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        
        mean_error[i] = float(xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence)*cell_area) ) # / xp.sum(cell_area)

        divergence_error = timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        plot_tridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.asnumpy(), mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        plot_tridata(grid_filename, analytic_divergence, mask, f'Analytic divergence at {res}', 'plot_analytic_divergence_'+res)
        plot_tridata(grid_filename, divergence_error, mask, f'Divergence error at DX = {res}', 'plot_error_divergence_'+res)

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error, dx[0:len(resolutions)], resolutions)


def test_divergence_multiple_time_step():
    
    resolutions = (
        '1000m',
        '500m',
        '250m',
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
    enable_debug_message = False
    
    dtime_seconds = 1.0
    output_interval = 100*dtime_seconds
    initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_time = 1000 * dtime_seconds
    mean_error = np.zeros(len(resolutions), dtype = float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + '/ser_data/'
        grid_filename = grid_file_folder + 'Torus_Triangles_100km_x_100km_res' + res + '.nc'

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
            dtime_seconds,
            end_date,
            output_interval,
        )

        mask = create_mask(grid_filename)
        
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

        v_scale = 90.0 / 7.5
        sphere_radius = 6371229.0
        # analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * WAVENUMBER_FACTOR * (U_AMPLIFICATION_FACTOR * 1.e-5 * 2.0 * math.pi * xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + 1.e-5 * math.pi * xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))

        diff_divergence = timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv_vn.ndarray[:, 0] - analytic_divergence
        
        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            diff_divergence = diff_divergence.get()

        plot_tridata(grid_filename, analytic_divergence, mask, f'Analytic divergence at {res}', 'plot_analytic_divergence_'+res)
        plot_tridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv_vn.asnumpy(), mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        plot_tridata(grid_filename, diff_divergence, mask, f'Difference in divergence at {res}', 'plot_diff_divergence_'+res)

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

        end_time = end_time * 2
    


def test_divergence_single_time_step_on_globe():
    
    resolutions = (
        'r2b4',
        'r2b5',
        'r2b6',
        'r2b7',
    )
    grid_name = (
        'icon_grid_0010_R02B04_G',
        'icon_grid_0008_R02B05_G',
        'icon_grid_0002_R02B06_G',
        'icon_grid_0004_R02B07_G'
    )
    dx = np.array(
        (
        220,
        110,
        55,
        27.5,
        ), dtype=float
    )
    
    base_input_path = '/scratch/mch/cong/data/flat_earth_'
    grid_file_folder = '/scratch/mch/cong/grid-generator/grids/'
    run_path = './'
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
    mean_error = np.zeros(len(resolutions), dtype = float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + '/ser_data/'
        grid_filename = grid_file_folder + grid_name[i] + '.nc'

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

        mask = create_globe_mask(grid_filename)
        plot_globetridata(grid_filename, diagnostic_state.u.asnumpy(), mask, f'Initial u at {res}', 'plot_initial_u_'+res)
        plot_globetridata(grid_filename, diagnostic_state.v.asnumpy(), mask, f'Initial v at {res}', 'plot_initial_v_'+res)

        z_ifc = diagnostic_metric_state.z_ifc.asnumpy()

        log.info(f"Surface max min:  {z_ifc[:,timeloop.grid.num_levels].max()}, {z_ifc[:,timeloop.grid.num_levels].min()}")
        log.info(f"First level max min:  {z_ifc[:,timeloop.grid.num_levels-1].max()}, {z_ifc[:,timeloop.grid.num_levels-1].min()}")

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
        u_factor = 0.25 * xp.sqrt(0.5*105.0/math.pi) * U_AMPLIFICATION_FACTOR
        v_factor = -0.5 * xp.sqrt(0.5*15.0/math.pi)
        # u_cell = u_factor * xp.cos(2.0 * cell_lon) * xp.cos(theta_shift - cell_lat) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        # v_cell = v_factor * xp.cos(cell_lon) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        analytic_divergence = (- u_factor * xp.sin(2.0 * cell_lon) * xp.sin(2.0 * (theta_shift - cell_lat)) + 0.25 * v_factor * xp.cos(cell_lon)*(3.0 * xp.sin(3.0 * (theta_shift - cell_lat)) / xp.sin(theta_shift - cell_lat) - 1.0)) / sphere_radius
        
        mean_error[i] = float(xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence)*cell_area) ) # / xp.sum(cell_area)

        divergence_error = timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        plot_globetridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.asnumpy(), mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        plot_globetridata(grid_filename, analytic_divergence, mask, f'Analytic divergence at {res}', 'plot_analytic_divergence_'+res)
        plot_globetridata(grid_filename, divergence_error, mask, f'Divergence error at {res}', 'plot_error_divergence_'+res)
        
        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error, dx[0:len(resolutions)], resolutions)



def test_divergence_multiple_time_step_on_globe():
    
    resolutions = (
        'r2b4',
        # 'r2b5',
        # 'r2b6',
        # 'r2b7',
    )
    grid_name = (
        'icon_grid_0010_R02B04_G',
        'icon_grid_0008_R02B05_G',
        'icon_grid_0002_R02B06_G',
        'icon_grid_0004_R02B07_G'
    )
    dx = np.array(
        (
        220,
        110,
        55,
        27.5,
        ), dtype=float
    )
    
    base_input_path = '/scratch/mch/cong/data/flat_earth_'
    grid_file_folder = '/scratch/mch/cong/grid-generator/grids/'
    run_path = './'
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
    output_interval = 100*dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=1000*dtime_seconds)
    mean_error = np.zeros(len(resolutions), dtype = float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + '/ser_data/'
        grid_filename = grid_file_folder + grid_name[i] + '.nc'

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

        mask = create_globe_mask(grid_filename)
        plot_globetridata(grid_filename, diagnostic_state.u.asnumpy(), mask, f'Initial u at {res}', 'plot_initial_u_'+res)
        plot_globetridata(grid_filename, diagnostic_state.v.asnumpy(), mask, f'Initial v at {res}', 'plot_initial_v_'+res)

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
        u_factor = 0.25 * xp.sqrt(0.5*105.0/math.pi) * U_AMPLIFICATION_FACTOR
        v_factor = -0.5 * xp.sqrt(0.5*15.0/math.pi)
        u_cell = u_factor * xp.cos(2.0 * cell_lon) * xp.cos(theta_shift - cell_lat) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        v_cell = v_factor * xp.cos(cell_lon) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (1.e-5 * 2.0 * math.pi * xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(theta_shift - cell_lat) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat) + 1.e-5 * math.pi * xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * (theta_shift - cell_lat)))
        
        mean_error[i] = float(xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence)*cell_area) ) # / xp.sum(cell_area)

        divergence_error = timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:,0] - analytic_divergence

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        plot_globetridata(grid_filename, timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.asnumpy(), mask, f'Computed divergence at {res}', 'plot_computed_divergence_'+res)
        plot_globetridata(grid_filename, analytic_divergence, mask, f'Analytic divergence at {res}', 'plot_analytic_divergence_'+res)
        plot_globetridata(grid_filename, divergence_error, mask, f'Difference in divergence at {res}', 'plot_diff_divergence_'+res)

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        # os.system(f"mv plot_* dummy* data_output* data_{res}")


if __name__ == "__main__":
    # test_divergence_single_time_step()
    # test_divergence_multiple_time_step()

    test_divergence_single_time_step_on_globe()
    # test_divergence_multiple_time_step_on_globe()