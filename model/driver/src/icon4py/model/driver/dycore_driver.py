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
import cProfile
import logging
import math
from datetime import datetime, timedelta
import pstats
from pathlib import Path
from typing import Callable, Optional

import click
from gt4py.next import as_field
import netCDF4 as nf4
import numpy as np
from cftime import date2num, num2date
from devtools import Timer

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.states import (
    DiagnosticStateNonHydro,
    PrepAdvection,
)
from icon4py.model.common.constants import CPD_O_RD, GRAV_O_RD, P0REF
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
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
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.states.diagnostic_state import DiagnosticMetricState, DiagnosticState
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.driver.icon_configuration import IconOutputConfig, IconRunConfig, read_config, OutputVariableList, OutputDimension, OutputScope
from icon4py.model.common.settings import device
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
)
from icon4py.model.driver.testcase_functions import mo_rbf_vec_interpol_cell_numpy


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
        output_config: IconOutputConfig,
        start_date: datetime,
        end_date: datetime,
        grid: IconGrid,
        cell_geometry: CellParams,
        edge_geometry: EdgeParams,
        diagnostic_metric_state: DiagnosticMetricState,
        data_dict: dict,
    ):
        self.config: IconOutputConfig = output_config
        self._variable_list: OutputVariableList = output_config.output_variable_list
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
            self._nf4_basegrp[i].createDimension(OutputDimension.CELL_DIM, grid.num_cells)
            self._nf4_basegrp[i].createDimension(OutputDimension.EDGE_DIM, grid.num_edges)
            self._nf4_basegrp[i].createDimension(OutputDimension.VERTEX_DIM, grid.num_vertices)
            self._nf4_basegrp[i].createDimension("vertices", 3)  # neighboring vertices of a cell
            self._nf4_basegrp[i].createDimension("vertices_2", 4)  # neighboring vertices of an edge
            self._nf4_basegrp[i].createDimension(
                "vertices_3", 6
            )  # neighboring vertices of a vertex
            self._nf4_basegrp[i].createDimension(OutputDimension.FULL_LEVEL, grid.num_levels)  # full level height
            self._nf4_basegrp[i].createDimension(OutputDimension.HALF_LEVEL, grid.num_levels + 1)  # half level height
            self._nf4_basegrp[i].createDimension("bnds", 2)  # boundary points for full level height
            self._nf4_basegrp[i].createDimension(OutputDimension.TIME, None)

        self._current_write_step: int = 0
        self._current_file_number: int = 0

        self._create_variables()
        self._write_dimension(grid, diagnostic_metric_state)
        times = self._nf4_basegrp[self._current_file_number].variables[OutputDimension.TIME]
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
            times: nf4.Variable = self._nf4_basegrp[i].createVariable(OutputDimension.TIME, "f8", (OutputDimension.TIME,))
            levels: nf4.Variable = self._nf4_basegrp[i].createVariable(
                OutputDimension.FULL_LEVEL, "f8", (OutputDimension.FULL_LEVEL,)
            )
            half_levels: nf4.Variable = self._nf4_basegrp[i].createVariable(
                OutputDimension.HALF_LEVEL, "f8", (OutputDimension.HALF_LEVEL,)
            )
            self._nf4_basegrp[i].createVariable(
                "height_bnds",
                "f8",
                (
                    OutputDimension.FULL_LEVEL,
                    "bnds",
                ),
            )
            cell_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clat", "f8", (OutputDimension.CELL_DIM,)
            )
            cell_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clon", "f8", (OutputDimension.CELL_DIM,)
            )
            cell_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clat_bnds",
                "f8",
                (
                    OutputDimension.CELL_DIM,
                    "vertices",
                ),
            )
            cell_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "clon_bnds",
                "f8",
                (
                    OutputDimension.CELL_DIM,
                    "vertices",
                ),
            )
            edge_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elat", "f8", (OutputDimension.EDGE_DIM,)
            )
            edge_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elon", "f8", (OutputDimension.EDGE_DIM,)
            )
            edge_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elat_bnds",
                "f8",
                (
                    OutputDimension.EDGE_DIM,
                    "vertices_2",
                ),
            )
            edge_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "elon_bnds",
                "f8",
                (
                    OutputDimension.EDGE_DIM,
                    "vertices_2",
                ),
            )
            vertex_latitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlat", "f8", (OutputDimension.VERTEX_DIM,)
            )
            vertex_longitudes: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlon", "f8", (OutputDimension.VERTEX_DIM,)
            )
            vertex_lat_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlat_bnds",
                "f8",
                (
                    OutputDimension.VERTEX_DIM,
                    "vertices_3",
                ),
            )
            vertex_lon_bounds: nf4.Variable = self._nf4_basegrp[i].createVariable(
                "vlon_bnds",
                "f8",
                (
                    OutputDimension.VERTEX_DIM,
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
                netcdf_dimension = (var_dimension.time_dimension, var_dimension.horizon_dimension,) if var_dimension.vertical_dimension is None else (var_dimension.time_dimension, var_dimension.vertical_dimension, var_dimension.horizon_dimension,)
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

    def _write_dimension(self, grid: IconGrid, diagnostic_metric_state: DiagnosticMetricState):
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

    def _grid_to_netcdf(self, cell_geometry: CellParams, edge_geometry: EdgeParams):
        # the grid details are only write to the first netCDF file to save memory
        cell_areas: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "cell_area", "f8", (OutputDimension.CELL_DIM,)
        )
        edge_areas: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "edge_area", "f8", (OutputDimension.EDGE_DIM,)
        )
        primal_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "primal_edge_length", "f8", (OutputDimension.EDGE_DIM,)
        )
        vert_vert_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "vert_vert_edge_length", "f8", (OutputDimension.EDGE_DIM,)
        )
        dual_edge_lengths: nf4.Variable = self._nf4_basegrp[0].createVariable(
            "dual_edge_length", "f8", (OutputDimension.EDGE_DIM,)
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
            times = self._nf4_basegrp[self._current_file_number].variables[OutputDimension.TIME]
            times[self._current_write_step] = date2num(
                current_date, units=times.units, calendar=times.calendar
            )
            log.info(f"Current times are  {times[:]}")






            """
            debugging variables
            """
            '''
            exner_grad = self._nf4_basegrp[i].createVariable(
                "exner_gradient",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            graddiv_vn = self._nf4_basegrp[i].createVariable(
                "Laplacian_vn",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            graddiv2_vn = self._nf4_basegrp[i].createVariable(
                "Laplacian2_vn",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            ddt_vn_apc_1 = self._nf4_basegrp[i].createVariable(
                "ddt_vn_apc_1",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            ddt_vn_apc_2 = self._nf4_basegrp[i].createVariable(
                "ddt_vn_apc_2",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            theta_v_e = self._nf4_basegrp[i].createVariable(
                "theta_v_e",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            ddt_vn_phy = self._nf4_basegrp[i].createVariable(
                "ddt_vn_phy",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )

            diff_multfac_vn = self._nf4_basegrp[i].createVariable(
                "diff_multfac_vn",
                "f8",
                (
                    "time",
                    "height",
                ),
            )
            kh_smag_e = self._nf4_basegrp[i].createVariable(
                "kh_smag_e",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            nabla2_vn_e = self._nf4_basegrp[i].createVariable(
                "nabla2_vn_e",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            intermediate_predictor_theta_v_e = self._nf4_basegrp[i].createVariable(
                "intermediate_predictor_theta_v_e",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            intermediate_predictor_gradh_exner = self._nf4_basegrp[i].createVariable(
                "intermediate_predictor_gradh_exner",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )
            intermediate_predictor_ddt_vn_apc_ntl1 = self._nf4_basegrp[i].createVariable(
                "intermediate_predictor_ddt_vn_apc_ntl1",
                "f8",
                (
                    "time",
                    "height",
                    "ncells_2",
                ),
            )

            exner_grad.units = "m-1"
            graddiv_vn.units = "m-1 s-1"
            graddiv2_vn.units = "m-3 s-1"
            ddt_vn_apc_1.units = "m-1 s-2"
            ddt_vn_apc_2.units = "m-1 s-2"
            theta_v_e.units = "K"
            ddt_vn_phy.units = "m-1 s-2"
            diff_multfac_vn.units = "m-1 s"
            kh_smag_e.units = "m-1 s"
            nabla2_vn_e.units = "m-1 s-1"

            exner_grad.standard_name = "exner_gradient"
            graddiv_vn.standard_name = "DelDivergence_normal_velocity"
            graddiv2_vn.standard_name = "DelDivergence2_normal_velocity"
            ddt_vn_apc_1.standard_name = "vn_tendency_now"
            ddt_vn_apc_2.standard_name = "vn_tendency_next"
            theta_v_e.standard_name = "virtual_potential_temperature_at_edge"
            ddt_vn_phy.standard_name = "vn_tendency_physics"
            diff_multfac_vn.standard_name = "diffusion_multfac_vn"
            kh_smag_e.standard_name = "kh_smag_vn"
            nabla2_vn_e.standard_name = "Laplacian_normal_velocity"

            exner_grad.long_name = "gradient of exner"
            graddiv_vn.long_name = "Directional derivative of Divergence of normal velocity"
            graddiv2_vn.long_name = "double directional derivative of Divergence of normal velocity"
            ddt_vn_apc_1.long_name = "tendency of normal velocity now"
            ddt_vn_apc_2.long_name = "tendency of normal velocity next"
            theta_v_e.long_name = "virtual potential temperature at edge"
            ddt_vn_phy.long_name = "tendency of normal velocity due to physics"
            diff_multfac_vn.long_name = "Difussion multiplication factor for normal velocity"
            kh_smag_e.long_name = "Fourth order Diffusion Smagorinski factor for normal velocity"
            nabla2_vn_e.long_name = "Laplacian of normal velocity"
            '''


# global profiler object
profiler = cProfile.Profile()


def profile_enable():
    profiler.enable()


def profile_disable():
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(f"{__name__}.profile")


class TimeLoop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: IconRunConfig,
        grid: Optional[IconGrid],
        diffusion: Diffusion,
        solve_nonhydro: SolveNonhydro,
    ):
        self.run_config: IconRunConfig = run_config
        self.grid: Optional[IconGrid] = grid
        self.diffusion = diffusion
        self.solve_nonhydro = solve_nonhydro

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date) / self.run_config.dtime
        )
        self.dtime_in_seconds: float = self.run_config.dtime.total_seconds()
        self._n_substeps_var: int = self.run_config.n_substeps
        self._substep_timestep: float = float(self.dtime_in_seconds / self._n_substeps_var)

        self._validate_config()

        # current simulation date
        self._simulation_date: datetime = self.run_config.start_date

        self._is_first_step_in_simulation: bool = not self.run_config.restart_mode

        self._now: int = 0  # TODO (Chia Rui): move to PrognosticState
        self._next: int = 1  # TODO (Chia Rui): move to PrognosticState

        self._timer1 = Timer(self._full_name(self._integrate_one_time_step), dp=5)
        self._timer2 = Timer(self._full_name(self._do_dyn_substepping), dp=5)
        self._timer3 = Timer("Diffusion", dp=5)

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
        prognostic_state: PrognosticState,
        diagnostic_state: DiagnosticState,
        diagnostic_metric_state: DiagnosticMetricState,
    ):
        edge_2_cell_vector_rbf_interpolation(
            prognostic_state.vn,
            diagnostic_metric_state.rbf_vec_coeff_c1,
            diagnostic_metric_state.rbf_vec_coeff_c2,
            diagnostic_state.u,
            diagnostic_state.v,
            horizontal_start=self.grid.get_start_index(CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1),
            horizontal_end=self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
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
            horizontal_start=self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        diagnose_surface_pressure(
            prognostic_state.exner,
            diagnostic_state.temperature,
            diagnostic_metric_state.ddqz_z_full,
            diagnostic_state.pressure_ifc,
            CPD_O_RD,
            P0REF,
            GRAV_O_RD,
            horizontal_start=self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
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
            GRAV_O_RD,
            horizontal_start=self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            horizontal_end=self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

    def time_integration(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        diagnostic_metric_state: DiagnosticMetricState,
        diagnostic_state: DiagnosticState,
        # TODO (Chia Rui): expand the PrognosticState to include indices of now and next, now it is always assumed that now = 0, next = 1 at the beginning
        prognostic_state_list: list[PrognosticState],
        # below is a long list of arguments for dycore time_step that many can be moved to initialization of SolveNonhydro)
        prep_adv: PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
        output_state: NewOutputState = None,
        profile: bool = False,
    ):
        log.info(
            f"starting time loop for dtime={self.dtime_in_seconds} s and n_timesteps={self._n_time_steps}"
        )
        log.info("Initialization of diagnostic variables for output.")
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self.run_config.apply_initial_stabilization} dtime={self.dtime_in_seconds} s, substep_timestep={self._substep_timestep}"
        )

        log.info("Initialization of diagnostic variables for output.")
        if output_state is not None:
            self._diagnose_for_output_and_physics(
                prognostic_state_list[self._now], diagnostic_state, diagnostic_metric_state
            )

            log.info(f"Debugging U (before diffusion): {np.max(diagnostic_state.u.asnumpy())}")
        
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
        if profile:
            profile_enable()

        for time_step in range(self._n_time_steps):
            log.info(f"simulation date : {self._simulation_date} run timestep : {time_step}")
            log.info(
                f" MAX VN: {prognostic_state_list[self._now].vn.ndarray.max():.15e} , MAX W: {prognostic_state_list[self._now].w.ndarray.max():.15e}"
            )
            log.info(
                f" MAX RHO: {prognostic_state_list[self._now].rho.ndarray.max():.15e} , MAX THETA_V: {prognostic_state_list[self._now].theta_v.ndarray.max():.15e}"
            )
            # TODO (Chia Rui): check with Anurag about printing of max and min of variables.

            """
            if not self.is_run_from_serializedData:
                self.stencil_mo_init_ddt_cell_zero(
                    solve_nonhydro_diagnostic_state.ddt_exner_phy,
                    solve_nonhydro_diagnostic_state.ddt_w_adv_ntl1,
                    solve_nonhydro_diagnostic_state.ddt_w_adv_ntl2,
                    self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                    self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                    0,
                    self.grid.num_levels,
                    offset_provider={}
                )
                self.stencil_mo_init_ddt_edge_zero(
                    solve_nonhydro_diagnostic_state.ddt_vn_phy,
                    solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl1,
                    solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl2,
                    self.grid.get_start_index(EdgeDim, HorizontalMarkerIndex.interior(EdgeDim)),
                    self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim)),
                    0,
                    self.grid.num_levels,
                    offset_provider={},
                )
            """

            self._next_simulation_date()

            # update boundary condition

            self._timer1.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv,
                inital_divdamp_fac_o2,
                do_prep_adv,
            )
            self._timer1.capture()

            # TODO (Chia Rui): modify n_substeps_var if cfl condition is not met. (set_dyn_substeps subroutine)

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

        self._timer1.summary(True)
        self._timer2.summary(True)
        self._timer3.summary(True)

        if profile:
            profile_disable()


    def time_integration_speed_test(
        self,
        diagnostic_metric_state: DiagnosticMetricState,
        diagnostic_state: DiagnosticState,
        prognostic_state_list: list[PrognosticState],
    ):
        log.info(f"LEARNING: {self.solve_nonhydro.interpolation_state.e_bln_c_s.asnumpy().shape}")
        for i in range(self.solve_nonhydro.interpolation_state.e_bln_c_s.asnumpy().shape[0]):
            log.info(f"LEARNING: {i} {self.solve_nonhydro.interpolation_state.e_bln_c_s.asnumpy()[i]}")
        ##### TESTING SPEED
        test_vn = prognostic_state_list[self._now].vn
        test_u = diagnostic_state.u
        test_v = diagnostic_state.v
        test_c1 = diagnostic_metric_state.rbf_vec_coeff_c1
        test_c2 = diagnostic_metric_state.rbf_vec_coeff_c2
        test_vn_np = prognostic_state_list[self._now].vn.asnumpy()
        test_c2e2c2e_np = self.grid.connectivities[C2E2C2EDim]
        test_c1_np = diagnostic_metric_state.rbf_vec_coeff_c1.asnumpy()
        test_c2_np = diagnostic_metric_state.rbf_vec_coeff_c2.asnumpy()
        test_h1 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1
        )
        test_h2 = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim))
        test_v1 = 0
        test_v2 = self.grid.num_levels
        offset_provider = {
            "C2E2C2E": self.grid.get_offset_provider("C2E2C2E"),
        }
        log.info(
            f"starting speed-test time loop for gt4py-version rbf interpolation with n_timesteps={self._n_time_steps}"
        )
        fo = edge_2_cell_vector_rbf_interpolation
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(f"run timestep : {time_step}")

            timer.start()
            fo(
                # mo_rbf_vec_interpol_cell.with_backend(backend)(
                test_vn,
                test_c1,
                test_c2,
                test_u,
                test_v,
                test_h1,
                test_h2,
                test_v1,
                test_v2,
                offset_provider=offset_provider,
            )
            timer.capture()

        timer.summary(True)

        log.info(
            f"starting speed-test time loop for numpy-version rbf interpolation with n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(f"run timestep : {time_step}")

            timer.start()
            test_u_np, test_v_np = mo_rbf_vec_interpol_cell_numpy(
                test_vn_np,
                test_c1_np,
                test_c2_np,
                test_c2e2c2e_np,
                test_h1,
                test_h2,
                test_v1,
                test_v2,
            )
            timer.capture()

            diagnostic_state.u = as_field((CellDim, KDim), test_u_np)
            diagnostic_state.v = as_field((CellDim, KDim), test_v_np)

        timer.summary(True)


    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): Add update_spinup_damping here to compute divdamp_fac_o2

        self._timer2.start()
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv,
        )
        self._timer2.capture()

        if self.diffusion.config.apply_to_horizontal_wind:
            self._timer3.start()
            self.diffusion.run(
                diffusion_diagnostic_state, prognostic_state_list[self._next], self.dtime_in_seconds
            )
            self._timer3.capture()

        self._swap()

        # TODO (Chia Rui): add tracer advection here

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
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
    file_path: Path,
    props: ProcessProperties,
    serialization_type: SerializationType,
    experiment_type: ExperimentType,
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
    config = read_config(experiment_type)

    decomp_info = read_decomp_info(file_path, props, serialization_type, grid_root, grid_level)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
        grid_root=grid_root,
        grid_level=grid_level,
    )
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry, c_owner_mask) = read_geometry_fields(
        file_path,
        damping_height=config.run_config.damping_height,
        rank=props.rank,
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
        file_path,
        rank=props.rank,
        ser_type=serialization_type,
        grid_root=grid_root,
        grid_level=grid_level,
    )

    log.info("initializing diffusion")
    diffusion_params = DiffusionParams(config.diffusion_config)
    exchange = create_exchange(props, decomp_info)
    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        diffusion_metric_state,
        diffusion_interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    nonhydro_params = NonHydrostaticParams(config.solve_nonhydro_config)

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
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
    ) = read_initial_state(
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
        diffusion=diffusion,
        solve_nonhydro=solve_nonhydro,
    )
    return (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        diagnostic_state,
        prognostic_state_list,
        output_state,
        prep_adv,
        inital_divdamp_fac_o2,
    )


@click.command()
@click.argument("input_path")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
@click.option("--speed_test", default=False)
@click.option(
    "--serialization_type",
    default="serialbox",
    help="serialization type for grid info and static fields",
)
@click.option("--experiment_type", default="any", help="experiment selection")
@click.option("--grid_root", default=2, help="experiment selection")
@click.option("--grid_level", default=4, help="experiment selection")
@click.option("--profile", default=False, help="Whether to profile code using cProfile.")
@click.option("--disable-logging", is_flag=True, help="Disable all logging output.")
@click.option("--enable_output", is_flag=True, help="Enable output.")
def main(input_path, run_path, mpi, speed_test, serialization_type, experiment_type, grid_root, grid_level, profile, disable_logging, enable_output):
    """
    Run the driver.

    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/jw_node1_nproma50000/ --experiment_type=any --serialization_type=serialbox
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data --experiment_type=jabw  --serialization_type=serialbox

    steps:
    1. initialize model from serialized data:

        a) load config of icon and components: diffusion and solve_nh

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) load local, diagnostic and prognostic variables

        e) setup the time loop

    2. run time loop
    """
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
        Path(input_path), parallel_props, serialization_type, experiment_type, grid_root, grid_level, enable_output
    )
    log.info(f"Starting ICON dycore run: {timeloop.simulation_date.isoformat()}")
    log.info(
        f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}, ending date={timeloop.run_config.end_date}"
    )

    log.info(f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    if speed_test:
        timeloop.time_integration_speed_test(
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
        )
    else:
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

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
