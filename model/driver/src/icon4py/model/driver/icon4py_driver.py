# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import logging
import math
from datetime import datetime, timedelta
import pathlib
import uuid
from typing import Callable, Optional

import click
from gt4py.next import as_field
import netCDF4 as nf4
import numpy as np
from cftime import date2num, num2date
from devtools import Timer

from icon4py.model.atmosphere.diffusion import (
    diffusion,
    diffusion_states,
)
from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_pressure import (
    diagnose_pressure,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_surface_pressure import (
    diagnose_surface_pressure,
)
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import (
    diagnose_temperature,
)
from icon4py.model.common.dimension import (
    C2E2C2EDim,
    C2VDim,
    CellDim,
    E2C2VDim,
    EdgeDim,
    KDim,
    V2C2VDim,
)
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.grid import icon as icon_grid, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.states.diagnostic_state import DiagnosticMetricState, DiagnosticState
from icon4py.model.common.states import prognostic_state as prognostics, diagnostic_state as diagnostics
from icon4py.model.driver import (
    icon4py_configuration as driver_config,
    initialization_utils as driver_init,
)
from icon4py.model.common import constants as phy_const
from icon4py.model.common.settings import device
from icon4py.model.common.config import Device


log = logging.getLogger(__name__)


def retract_data(input_data):
    if device == Device.GPU:
        return input_data.ndarray.get()
    return input_data.ndarray

def retract_data_array(input_data):
    if device == Device.GPU:
        return input_data.get()
    return input_data


class NewOutputState:

    def __init__(
        self,
        output_config: driver_config.IconOutputConfig,
        start_date: datetime,
        end_date: datetime,
        grid: icon_grid.IconGrid,
        cell_geometry: h_grid.CellParams,
        edge_geometry: h_grid.EdgeParams,
        diagnostic_metric_state: DiagnosticMetricState,
        data_dict: dict,
    ):
        self.config: driver_config.IconOutputConfig = output_config
        self._variable_list: driver_config.OutputVariableList = output_config.output_variable_list
        self._check_list: dict = {}
        for var_name in self._variable_list.variable_name_list:
            self._check_list[var_name] = 0
        self._output_date = start_date
        self._first_date_in_this_ncfile = start_date
        # compute number of output files
        self._number_of_files = (
            int(math.ceil((end_date - start_date) / self.config.output_file_time_interval))
        )
        if self.config.output_initial_condition_as_a_separate_file:
            self._number_of_files += 1
            self._enforce_new_ncfile = True
        else:
            self._enforce_new_ncfile = False
        log.info(f"Number of files: {self._number_of_files}")

        # TODO (Chia Rui or others): this is only a tentative output method, use a proper netcdf output infrastructure in the future
        self._nf4_basegrp = [
            nf4.Dataset(
                str(self.config.output_path.absolute()) + "/data_output_" + str(i) + ".nc",
                "w",
                format="NETCDF4",
            )
            for i in range(self._number_of_files)
        ]
        for i in range(self._number_of_files):
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.CELL_DIM, grid.num_cells)
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.EDGE_DIM, grid.num_edges)
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.VERTEX_DIM, grid.num_vertices)
            self._nf4_basegrp[i].createDimension("vertices", 3)  # neighboring vertices of a cell
            self._nf4_basegrp[i].createDimension("vertices_2", 4)  # neighboring vertices of an edge
            self._nf4_basegrp[i].createDimension(
                "vertices_3", 6
            )  # neighboring vertices of a vertex
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.FULL_LEVEL, grid.num_levels)  # full level height
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.HALF_LEVEL, grid.num_levels + 1)  # half level height
            self._nf4_basegrp[i].createDimension("bnds", 2)  # boundary points for full level height
            self._nf4_basegrp[i].createDimension(driver_config.OutputDimension.TIME, None)

        self._current_write_step: int = 0
        self._current_file_number: int = 0

        self._create_variables()
        self._write_dimension(grid, diagnostic_metric_state)
        times = self._nf4_basegrp[self._current_file_number].variables[driver_config.OutputDimension.TIME]
        times[self._current_write_step] = date2num(
            start_date, units=times.units, calendar=times.calendar
        )

        self.write_to_netcdf(start_date, data_dict)
        self._grid_to_netcdf(cell_geometry, edge_geometry)

    def _create_variables(self):
        for i in range(self._number_of_files):
            """
            grid information
            """
            times: nf4.Variable = self._nf4_basegrp[i].createVariable(driver_config.OutputDimension.TIME, "f8", (driver_config.OutputDimension.TIME,))
            levels: nf4.Variable = self._nf4_basegrp[i].createVariable(
                driver_config.OutputDimension.FULL_LEVEL, "f8", (driver_config.OutputDimension.FULL_LEVEL,)
            )
            half_levels: nf4.Variable = self._nf4_basegrp[i].createVariable(
                driver_config.OutputDimension.HALF_LEVEL, "f8", (driver_config.OutputDimension.HALF_LEVEL,)
            )
            self._nf4_basegrp[i].createVariable(
                "height_bnds",
                "f8",
                (
                    driver_config.OutputDimension.FULL_LEVEL,
                    "bnds",
                ),
            )
            cell_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clat", "f8", (driver_config.OutputDimension.CELL_DIM,)
            )
            cell_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clon", "f8", (driver_config.OutputDimension.CELL_DIM,)
            )
            cell_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clat_bnds",
                "f8",
                (
                    driver_config.OutputDimension.CELL_DIM,
                    "vertices",
                ),
            )
            cell_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clon_bnds",
                "f8",
                (
                    driver_config.OutputDimension.CELL_DIM,
                    "vertices",
                ),
            )
            edge_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elat", "f8", (driver_config.OutputDimension.EDGE_DIM,)
            )
            edge_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elon", "f8", (driver_config.OutputDimension.EDGE_DIM,)
            )
            edge_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elat_bnds",
                "f8",
                (
                    driver_config.OutputDimension.EDGE_DIM,
                    "vertices_2",
                ),
            )
            edge_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elon_bnds",
                "f8",
                (
                    driver_config.OutputDimension.EDGE_DIM,
                    "vertices_2",
                ),
            )
            vertex_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlat", "f8", (driver_config.OutputDimension.VERTEX_DIM,)
            )
            vertex_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlon", "f8", (driver_config.OutputDimension.VERTEX_DIM,)
            )
            vertex_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlat_bnds",
                "f8",
                (
                    driver_config.OutputDimension.VERTEX_DIM,
                    "vertices_3",
                ),
            )
            vertex_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlon_bnds",
                "f8",
                (
                    driver_config.OutputDimension.VERTEX_DIM,
                    "vertices_3",
                ),
            )

            cell_latitudes.units = "radian"
            cell_longitudes.units = "radian"
            cell_lat_bounds.units = "radian"
            cell_lon_bounds.units = "radian"
            edge_latitudes.units = "radian"
            edge_longitudes.units = "radian"
            edge_lat_bounds.units = "radian"
            edge_lon_bounds.units = "radian"
            vertex_latitudes.units = "radian"
            vertex_longitudes.units = "radian"
            vertex_lat_bounds.units = "radian"
            vertex_lon_bounds.units = "radian"

            levels.units = "m"
            half_levels.units = "m"
            levels.axis = "Z"
            half_levels.axis = "Z"
            times.units = "seconds since 0001-01-01 00:00:00.0"
            times.calendar = "gregorian"
            times.axis = "T"

            times.standard_name = "time"
            cell_latitudes.standard_name = "latitude"
            cell_longitudes.standard_name = "longitude"
            edge_latitudes.standard_name = "latitude"
            edge_longitudes.standard_name = "longitude"
            vertex_latitudes.standard_name = "latitude"
            vertex_longitudes.standard_name = "longitude"
            levels.standard_name = "full height"
            half_levels.standard_name = "half height"

            times.long_name = "time"
            cell_latitudes.long_name = "center latitude"
            cell_longitudes.long_name = "center longitude"
            edge_latitudes.long_name = "edge midpoint latitude"
            edge_longitudes.long_name = "edge midpoint longitude"
            vertex_latitudes.long_name = "vertex latitude"
            vertex_longitudes.long_name = "vertex longitude"
            levels.long_name = "generalized_full_height"
            half_levels.long_name = "generalized_half_height"

            cell_latitudes.bounds = "clat_bnds"
            cell_longitudes.bounds = "clon_bnds"
            edge_latitudes.bounds = "elat_bnds"
            edge_longitudes.bounds = "elon_bnds"
            vertex_latitudes.bounds = "vlon_bnds"
            vertex_longitudes.bounds = "vlat_bnds"
            levels.bounds = "height_bnds"

            """
            output variables
            """
            for var_name in self._variable_list.variable_name_list:
                var_dimension = self._variable_list.variable_dim_list[var_name]
                if var_dimension.vertical_dimension is None:
                    netcdf_dimension = (var_dimension.time_dimension, var_dimension.horizon_dimension,)
                elif var_dimension.horizon_dimension is None:
                    netcdf_dimension = (var_dimension.time_dimension, var_dimension.vertical_dimension,)
                else:
                    netcdf_dimension = (var_dimension.time_dimension, var_dimension.vertical_dimension, var_dimension.horizon_dimension,)
                if i == 0:
                    log.info(f"Creating {var_name} with dimension {netcdf_dimension} in netcdf files.")
                var = self._nf4_basegrp[i].createVariable(
                    var_name,
                    "f8",
                    netcdf_dimension,
                    fill_value=np.nan,
                )
                var_attribute = self._variable_list.variable_attr_list[var_name]
                var.units = var_attribute.units
                var.standard_name = var_attribute.standard_name
                var.long_name = var_attribute.long_name
                var.CDI_grid_type = var_attribute.CDI_grid_type
                var.number_of_grid_in_reference = var_attribute.number_of_grid_in_reference
                var.coordinates = var_attribute.coordinates

    def _write_dimension(self, grid: icon_grid.IconGrid, diagnostic_metric_state: DiagnosticMetricState):
        for i in range(self._number_of_files):
            self._nf4_basegrp[i].variables["clat"][
                :
            ] = retract_data(diagnostic_metric_state.cell_center_lat)
            self._nf4_basegrp[i].variables["clon"][
                :
            ] = retract_data(diagnostic_metric_state.cell_center_lon)
            self._nf4_basegrp[i].variables["clat_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lat)[retract_data_array(grid.connectivities[C2VDim])]
            self._nf4_basegrp[i].variables["clon_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lon)[retract_data_array(grid.connectivities[C2VDim])]

            self._nf4_basegrp[i].variables["elat"][:] = retract_data(diagnostic_metric_state.e_lat)
            self._nf4_basegrp[i].variables["elon"][:] = retract_data(diagnostic_metric_state.e_lon)
            self._nf4_basegrp[i].variables["elat_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lat)[retract_data_array(grid.connectivities[E2C2VDim])]
            self._nf4_basegrp[i].variables["elon_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lon)[retract_data_array(grid.connectivities[E2C2VDim])]
            log.info(
                f"E2C2VDim dimension: {retract_data(diagnostic_metric_state.v_lon)[retract_data_array(grid.connectivities[E2C2VDim])].shape}"
            )
            log.info(
                f"V2C2VDim dimension: {retract_data(diagnostic_metric_state.v_lon)[retract_data_array(grid.connectivities[V2C2VDim])].shape}"
            )

            self._nf4_basegrp[i].variables["vlat"][:] = retract_data(diagnostic_metric_state.v_lat)
            self._nf4_basegrp[i].variables["vlon"][:] = retract_data(diagnostic_metric_state.v_lon)
            self._nf4_basegrp[i].variables["vlat_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lat)[retract_data_array(grid.connectivities[V2C2VDim])]
            self._nf4_basegrp[i].variables["vlon_bnds"][
                :, :
            ] = retract_data(diagnostic_metric_state.v_lon)[retract_data_array(grid.connectivities[V2C2VDim])]

            full_height = np.zeros(grid.num_levels, dtype=float)
            half_height = retract_data(diagnostic_metric_state.vct_a)
            full_height_bnds = np.zeros((grid.num_levels, 2), dtype=float)
            for k in range(grid.num_levels):
                full_height[k] = 0.5 * (half_height[k] + half_height[k + 1])
                full_height_bnds[k, 0] = half_height[k]
                full_height_bnds[k, 1] = half_height[k + 1]
            self._nf4_basegrp[i].variables["height"][:] = full_height
            self._nf4_basegrp[i].variables["height_2"][:] = half_height
            self._nf4_basegrp[i].variables["height_bnds"][:, :] = full_height_bnds

    def _grid_to_netcdf(self, cell_geometry: h_grid.CellParams, edge_geometry: h_grid.EdgeParams):
        # the grid details are only write to the first netCDF file to save memory
        cell_areas: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "cell_area", "f8", (driver_config.OutputDimension.CELL_DIM,)
        )
        edge_areas: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "edge_area", "f8", (driver_config.OutputDimension.EDGE_DIM,)
        )
        primal_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "primal_edge_length", "f8", (driver_config.OutputDimension.EDGE_DIM,)
        )
        vert_vert_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "vert_vert_edge_length", "f8", (driver_config.OutputDimension.EDGE_DIM,)
        )
        dual_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "dual_edge_length", "f8", (driver_config.OutputDimension.EDGE_DIM,)
        )

        cell_areas.units = "m2"
        edge_areas.units = "m2"
        primal_edge_lengths.units = "m"
        vert_vert_edge_lengths.units = "m"
        dual_edge_lengths.units = "m"

        # TODO (Chia Rui or others): check the param for these variables?
        cell_areas.param = "99.0.1"
        edge_areas.param = "99.0.2"
        primal_edge_lengths.param = "99.0.3"
        vert_vert_edge_lengths.param = "99.0.4"
        dual_edge_lengths.param = "99.0.5"

        cell_areas.standard_name = "cell area"
        edge_areas.standard_name = "edge area"
        primal_edge_lengths.standard_name = "edge length"
        vert_vert_edge_lengths.standard_name = "vertex-vertex edge length"
        dual_edge_lengths.standard_name = "dual edge length"

        cell_areas.long_name = "cell area"
        edge_areas.long_name = "edge area"
        primal_edge_lengths.long_name = "edge length"
        vert_vert_edge_lengths.long_name = "vertex-vertex edge length"
        dual_edge_lengths.long_name = "dual edge length"

        cell_areas.CDI_grid_type = "unstructured"
        edge_areas.CDI_grid_type = "unstructured"
        primal_edge_lengths.CDI_grid_type = "unstructured"
        vert_vert_edge_lengths.CDI_grid_type = "unstructured"
        dual_edge_lengths.CDI_grid_type = "unstructured"

        cell_areas.number_of_grid_in_reference = 1
        edge_areas.number_of_grid_in_reference = 1
        primal_edge_lengths.number_of_grid_in_reference = 1
        vert_vert_edge_lengths.number_of_grid_in_reference = 1
        dual_edge_lengths.number_of_grid_in_reference = 1

        cell_areas.coordinates = "clat clon"
        edge_areas.coordinates = "elat elon"
        primal_edge_lengths.coordinates = "elat elon"
        vert_vert_edge_lengths.coordinates = "elat elon"
        dual_edge_lengths.coordinates = "elat elon"

        cell_areas[:] = retract_data(cell_geometry.area)
        edge_areas[:] = retract_data(edge_geometry.edge_areas)
        primal_edge_lengths[:] = retract_data(edge_geometry.primal_edge_lengths)
        vert_vert_edge_lengths[:] = retract_data(edge_geometry.vertex_vertex_lengths)
        dual_edge_lengths[:] = retract_data(edge_geometry.dual_edge_lengths)

    def write_to_netcdf(
        self,
        current_date: datetime,
        output_dict: dict,
    ):
        log.info(
            f"Writing output at {current_date} at {self._current_write_step} in file no. {self._current_file_number}"
        )
        for var_name in output_dict.keys():
            if var_name in self._variable_list.variable_name_list:
                if self._check_list[var_name] == 0:
                    self._nf4_basegrp[self._current_file_number].variables[var_name][self._current_write_step] = retract_data(output_dict[var_name]).transpose()
                    self._check_list[var_name] = 1
                else:
                    log.warning(f"Data {var_name} already existed in file no. {self._current_file_number} at {self._current_write_step}")
                    #raise ValueError


    def advance_time(
        self,
        current_date: datetime,
    ):
        times = self._nf4_basegrp[self._current_file_number].variables["time"]
        time_elapsed_since_last_output = current_date - self._output_date
        time_elapsed_in_this_ncfile = current_date - self._first_date_in_this_ncfile
        log.info(
            f"first date in currect nc file: {self._first_date_in_this_ncfile}, previous output date: {self._output_date}"
        )
        log.info(
            f"time elapsed since last output: {time_elapsed_since_last_output}, time elapsed in this file: {time_elapsed_in_this_ncfile}"
        )
        if time_elapsed_since_last_output >= self.config.output_time_interval:
            if self._enforce_new_ncfile or time_elapsed_in_this_ncfile > self.config.output_file_time_interval:
                for var_name in self._variable_list.variable_name_list:
                    if self._check_list[var_name] == 0:
                        log.warning(f"Data {var_name} in file no. {self._current_file_number} at {self._current_write_step} is not recorded.")
                self._enforce_new_ncfile = False
                self._first_date_in_this_ncfile = self._output_date
                self._current_write_step = 0
                self._nf4_basegrp[self._current_file_number].close()
                self._current_file_number += 1
            else:
                self._current_write_step += 1
            for var_name in self._variable_list.variable_name_list:
                self._check_list[var_name] = 0
            self._output_date = current_date
            times = self._nf4_basegrp[self._current_file_number].variables[driver_config.OutputDimension.TIME]
            times[self._current_write_step] = date2num(
                current_date, units=times.units, calendar=times.calendar
            )
            log.info(f"Current times are  {times[:]}")


class TimeLoop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: driver_config.Icon4pyRunConfig,
        grid: Optional[icon_grid.IconGrid],
        diffusion_granule: diffusion.Diffusion,
        solve_nonhydro_granule: solve_nh.SolveNonhydro,
    ):
        self.run_config: driver_config.Icon4pyRunConfig = run_config
        self.grid: Optional[icon_grid.IconGrid] = grid
        self.diffusion = diffusion_granule
        self.solve_nonhydro = solve_nonhydro_granule

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date) / self.run_config.dtime
        )
        self.dtime_in_seconds: float = self.run_config.dtime.total_seconds()
        self._n_substeps_var: int = self.run_config.n_substeps
        self._substep_timestep: float = float(self.dtime_in_seconds / self._n_substeps_var)

        self._validate_config()

        # current simulation date
        self._simulation_date: datetime.datetime = self.run_config.start_date

        self._is_first_step_in_simulation: bool = not self.run_config.restart_mode

        self._now: int = 0  # TODO (Chia Rui): move to PrognosticState
        self._next: int = 1  # TODO (Chia Rui): move to PrognosticState

    def re_init(self):
        self._simulation_date = self.run_config.start_date
        self._is_first_step_in_simulation = True
        self._n_substeps_var = self.run_config.n_substeps
        self._now: int = 0  # TODO (Chia Rui): move to PrognosticState
        self._next: int = 1  # TODO (Chia Rui): move to PrognosticState

    def _validate_config(self):
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")
        if not self.diffusion.initialized:
            raise Exception("diffusion is not initialized before time loop")
        if not self.solve_nonhydro.initialized:
            raise Exception("nonhydro solver is not initialized before time loop")

    @property
    def first_step_in_simulation(self):
        return self._is_first_step_in_simulation

    def _is_last_substep(self, step_nr: int):
        return step_nr == (self.n_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int):
        return step_nr == 0

    def _next_simulation_date(self):
        self._simulation_date += self.run_config.dtime

    @property
    def n_substeps_var(self):
        return self._n_substeps_var

    @property
    def simulation_date(self):
        return self._simulation_date

    @property
    def prognostic_now(self):
        return self._now

    @property
    def prognostic_next(self):
        return self._next

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @property
    def substep_timestep(self):
        return self._substep_timestep

    def _swap(self):
        time_n_swap = self._next
        self._next = self._now
        self._now = time_n_swap

    def _full_name(self, func: Callable):
        return ":".join((self.__class__.__name__, func.__name__))

    def _diagnose_for_output_and_physics(
        self,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        diagnostic_metric_state: DiagnosticMetricState,
    ):
        edge_2_cell_vector_rbf_interpolation(
            prognostic_state.vn,
            diagnostic_metric_state.rbf_vec_coeff_c1,
            diagnostic_metric_state.rbf_vec_coeff_c2,
            diagnostic_state.u,
            diagnostic_state.v,
            horizontal_start=self.grid.get_start_index(CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim) + 1),
            horizontal_end=self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim)),
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider=self.grid.offset_providers,
        )
        log.debug(
            f"max min v: {diagnostic_state.v.asnumpy().max()} {diagnostic_state.v.asnumpy().min()}"
        )

        diagnose_temperature(
            prognostic_state.theta_v,
            prognostic_state.exner,
            diagnostic_state.temperature,
            horizontal_start=self.grid.get_start_index(CellDim, h_grid.HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim)),
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        diagnose_surface_pressure(
            prognostic_state.exner,
            diagnostic_state.temperature,
            diagnostic_metric_state.ddqz_z_full,
            diagnostic_state.pressure_ifc,
            phy_const.CPD_O_RD,
            phy_const.P0REF,
            phy_const.GRAV_O_RD,
            horizontal_start=self.grid.get_start_index(CellDim, h_grid.HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim)),
            vertical_start=self.grid.num_levels,
            vertical_end=self.grid.num_levels + 1,
            offset_provider=self.grid.offset_providers,
        )

        diagnose_pressure(
            diagnostic_metric_state.ddqz_z_full,
            diagnostic_state.temperature,
            diagnostic_state.pressure_sfc,
            diagnostic_state.pressure,
            diagnostic_state.pressure_ifc,
            phy_const.GRAV_O_RD,
            horizontal_start=self.grid.get_start_index(CellDim, h_grid.HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, h_grid.HorizontalMarkerIndex.end(CellDim)),
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

    def time_integration(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        diagnostic_metric_state: DiagnosticMetricState,
        # TODO (Chia Rui): expand the PrognosticState to include indices of now and next, now it is always assumed that now = 0, next = 1 at the beginning
        prognostic_state_list: list[prognostics.PrognosticState],
        diagnostic_state: DiagnosticState,
        # below is a long list of arguments for dycore time_step that many can be moved to initialization of SolveNonhydro)
        prep_adv: solve_nh_states.PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
        output_state: NewOutputState = None,
    ):
        log.info(
            f"starting time loop for dtime={self.dtime_in_seconds} s and n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self.run_config.apply_initial_stabilization} dtime={self.dtime_in_seconds} s, substep_timestep={self._substep_timestep}"
        )

        # TODO (Chia Rui): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        # TODO (Chia Rui): Compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

        # TODO (Chia Rui): Initialize exner_pr used in solve_nh (compute_exner_pert subroutine)

        if (
            self.diffusion.config.apply_to_horizontal_wind
            and self.run_config.apply_initial_stabilization
            and self._is_first_step_in_simulation
        ):
            log.info("running initial step to diffuse fields before timeloop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_state_list[self._now],
                self.dtime_in_seconds,
            )
        log.info(
            f"starting real time loop for dtime={self.dtime_in_seconds} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(f"simulation date : {self._simulation_date} run timestep : {time_step}")
            log.info(
                f" MAX VN: {prognostic_state_list[self._now].vn.ndarray.max():.15e} , MAX W: {prognostic_state_list[self._now].w.ndarray.max():.15e}"
            )
            log.info(
                f" MAX RHO: {prognostic_state_list[self._now].rho.ndarray.max():.15e} , MAX THETA_V: {prognostic_state_list[self._now].theta_v.ndarray.max():.15e}"
            )
            # TODO (Chia Rui): check with Anurag about printing of max and min of variables.

            self._next_simulation_date()

            # update boundary condition

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv,
                inital_divdamp_fac_o2,
                do_prep_adv,
            )
            timer.capture()

            # TODO (Chia Rui): modify n_substeps_var if cfl condition is not met. (set_dyn_substeps subroutine)

            # TODO (Chia Rui): compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

            # TODO (Chia Rui): simple IO enough for JW test

            if output_state is not None:
                self._diagnose_for_output_and_physics(
                    prognostic_state_list[self._now], diagnostic_state, diagnostic_metric_state
                )

                log.info(f"Debugging U (after diffusion): {np.max(diagnostic_state.u.asnumpy())}")

                output_state.advance_time(self._simulation_date)
                output_data = {}
                output_data['vn'] = prognostic_state_list[self._now].vn
                output_data['rho'] = prognostic_state_list[self._now].rho
                output_data['theta_v'] = prognostic_state_list[self._now].theta_v
                output_data['w'] = prognostic_state_list[self._now].w
                output_data['exner'] = prognostic_state_list[self._now].exner
                output_data['u'] = diagnostic_state.u
                output_data['v'] = diagnostic_state.v
                output_data['pressure'] = diagnostic_state.pressure
                output_data['temperature'] = diagnostic_state.temperature
                output_data['pressure_sfc'] = diagnostic_state.pressure_sfc
                output_state.write_to_netcdf(self._simulation_date, output_data)


        timer.summary(True)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state_list: list[prognostics.PrognosticState],
        prep_adv: solve_nh_states.PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): Add update_spinup_damping here to compute divdamp_fac_o2

        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self.diffusion.run(
                diffusion_diagnostic_state, prognostic_state_list[self._next], self.dtime_in_seconds
            )

        self._swap()

        # TODO (Chia Rui): add tracer advection here

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: solve_nh_states.DiagnosticStateNonHydro,
        prognostic_state_list: list[prognostics.PrognosticState],
        prep_adv: solve_nh_states.PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): compute airmass for prognostic_state here

        do_recompute = True
        do_clean_mflx = True
        for dyn_substep in range(self._n_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} substep / n_substeps : {dyn_substep} / "
                f"{self.n_substeps_var} , is_first_step_in_simulation : {self._is_first_step_in_simulation}, "
                f"nnow: {self._now}, nnew : {self._next}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                divdamp_fac_o2=inital_divdamp_fac_o2,
                dtime=self._substep_timestep,
                l_recompute=do_recompute,
                l_init=self._is_first_step_in_simulation,
                nnew=self._next,
                nnow=self._now,
                lclean_mflx=do_clean_mflx,
                lprep_adv=do_prep_adv,
                at_first_substep=self._is_first_substep(dyn_substep),
                at_last_substep=self._is_last_substep(dyn_substep),
            )

            do_recompute = False
            do_clean_mflx = False

            if not self._is_last_substep(dyn_substep):
                self._swap()

            self._is_first_step_in_simulation = False

        # TODO (Chia Rui): compute airmass for prognostic_state here


def initialize(
    file_path: pathlib.Path,
    props: decomposition.ProcessProperties,
    serialization_type: driver_init.SerializationType,
    experiment_type: driver_init.ExperimentType,
    grid_id: uuid.UUID,
    grid_root,
    grid_level,
    enable_output: bool,
):
    """
    Inititalize the driver run.

    "reads" in
        - load configuration

        - load grid information

        - initialize components: diffusion and solve_nh

        - load diagnostic and prognostic variables (serialized data)

        - setup the time loop

     Returns:
         tl: configured timeloop,
         diffusion_diagnostic_state: initial state for diffusion diagnostic variables
         nonhydro_diagnostic_state: initial state for solve_nonhydro diagnostic variables
         prognostic_state: initial state for prognostic variables
         diagnostic_state: initial state for global diagnostic variables
         prep_advection: fields collecting data for advection during the solve nonhydro timestep
         inital_divdamp_fac_o2: initial divergence damping factor

    """
    log.info("initialize parallel runtime")
    log.info(f"reading configuration: experiment {experiment_type}")
    config = driver_config.read_config(experiment_type)

    decomp_info = driver_init.read_decomp_info(
        file_path, props, serialization_type, grid_id, grid_root, grid_level
    )

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = driver_init.read_icon_grid(
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
        grid_id=grid_id,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    log.info(f"reading input fields from '{file_path}'")
    (
        edge_geometry,
        cell_geometry,
        vertical_geometry,
        c_owner_mask,
    ) = driver_init.read_geometry_fields(
        file_path,
        vertical_grid_config=config.vertical_grid_config,
        rank=props.rank,
        ser_type=serialization_type,
        grid_id=grid_id,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    ) = driver_init.read_static_fields(
        icon_grid,
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
        grid_id=grid_id,
        grid_root=grid_root,
        grid_level=grid_level,
    )

    log.info("initializing diffusion")
    diffusion_params = diffusion.DiffusionParams(config.diffusion_config)
    exchange = decomposition.create_exchange(props, decomp_info)
    diffusion_granule = diffusion.Diffusion(exchange)
    diffusion_granule.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        diffusion_metric_state,
        diffusion_interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config.solve_nonhydro_config)

    solve_nonhydro_granule = solve_nh.SolveNonhydro()
    solve_nonhydro_granule.init(
        grid=icon_grid,
        config=config.solve_nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_geometry,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=c_owner_mask,
    )

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        inital_divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = driver_init.read_initial_state(
        icon_grid,
        cell_geometry,
        edge_geometry,
        file_path,
        rank=props.rank,
        experiment_type=experiment_type,
    )
    prognostic_state_list = [prognostic_state_now, prognostic_state_next]

    log.info("initializing netCDF4 output state")
    output_data = {}
    output_data['vn'] = prognostic_state_list[0].vn
    output_data['rho'] = prognostic_state_list[0].rho
    output_data['theta_v'] = prognostic_state_list[0].theta_v
    output_data['w'] = prognostic_state_list[0].w
    output_data['exner'] = prognostic_state_list[0].exner
    output_data['u'] = diagnostic_state.u
    output_data['v'] = diagnostic_state.v
    output_data['pressure'] = diagnostic_state.pressure
    output_data['temperature'] = diagnostic_state.temperature
    output_data['pressure_sfc'] = diagnostic_state.pressure_sfc
    output_state = NewOutputState(
        config.output_config,
        config.run_config.start_date,
        config.run_config.end_date,
        icon_grid,
        cell_geometry,
        edge_geometry,
        diagnostic_metric_state,
        output_data,
    )

    timeloop = TimeLoop(
        run_config=config.run_config,
        grid=icon_grid,
        diffusion_granule=diffusion_granule,
        solve_nonhydro_granule=solve_nonhydro_granule,
    )
    return (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        prognostic_state_list,
        diagnostic_state,
        output_state,
        prep_adv,
        inital_divdamp_fac_o2,
    )


@click.command()
@click.argument("input_path")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
@click.option(
    "--serialization_type",
    default="serialbox",
    help="serialization type for grid info and static fields",
)
@click.option("--experiment_type", default="any", help="experiment selection")
@click.option("--grid_root", default=2, help="experiment selection")
@click.option("--grid_level", default=4, help="experiment selection")
@click.option(
    "--grid_id",
    default="af122aca-1dd2-11b2-a7f8-c7bf6bc21eba",
    help="uuid of the horizontal grid ('uuidOfHGrid' from gridfile)",
)
@click.option("--enable_output", is_flag=True, help="Enable output.")
def main(
    input_path, run_path, mpi, serialization_type, experiment_type, grid_id, grid_root, grid_level, enable_output
):
    """
    Run the driver.

    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data

    steps:
    1. initialize model from serialized data:

        a) load config of icon and components: diffusion and solve_nh

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) load local, diagnostic and prognostic variables

        e) setup the time loop

    2. run time loop
    """
    parallel_props = decomposition.get_processor_properties(decomposition.get_runtype(with_mpi=mpi))
    grid_id = uuid.UUID(grid_id)
    driver_init.configure_logging(run_path, experiment_type, parallel_props)
    (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        prognostic_state_list,
        diagnostic_state,
        output_state,
        prep_adv,
        inital_divdamp_fac_o2,
    ) = initialize(
        pathlib.Path(input_path),
        parallel_props,
        serialization_type,
        experiment_type,
        grid_id,
        grid_root,
        grid_level,
        enable_output,
    )
    log.info(f"Starting ICON dycore run: {timeloop.simulation_date.isoformat()}")
    log.info(
        f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}, ending date={timeloop.run_config.end_date}"
    )

    log.info(f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    timeloop.time_integration(
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        prognostic_state_list,
        diagnostic_state,
        prep_adv,
        inital_divdamp_fac_o2,
        do_prep_adv=False,
        output_state=output_state,
    )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
