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
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
import numpy as np
from cftime import date2num, num2date
import netCDF4 as nf4

import click
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
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.states.diagnostic_state import DiagnosticState, DiagnosticMetricState
from icon4py.model.common.diagnostic_calculations.mo_init_exner_pr import mo_init_exner_pr
from icon4py.model.common.diagnostic_calculations.mo_init_zero import mo_init_ddt_cell_zero, mo_init_ddt_edge_zero
from icon4py.model.common.diagnostic_calculations.mo_diagnose_temperature_pressure import mo_diagnose_temperature, mo_diagnose_pressure_sfc, mo_diagnose_pressure
from icon4py.model.common.interpolation.stencils.mo_rbf_vec_interpol_cell import mo_rbf_vec_interpol_cell
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config
from icon4py.model.driver.io_utils import (
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
    SerializationType,
    InitializationType,
)
from icon4py.model.common.constants import CPD_O_RD, P0REF, GRAV_O_RD
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.dimension import C2VDim, V2C2VDim, E2C2VDim, CellDim, EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from gt4py.next.program_processors.runners.gtfn import run_gtfn

backend = run_gtfn
log = logging.getLogger(__name__)


class OutputState:
    def __init__(self,path: Path, grid: IconGrid, diagnostic_metric_state: DiagnosticMetricState):
        # TODO (Chia Rui or others): this is only a tentative output method, use a proper netcdf output infrastructure in the future
        nf4_basegrp = nf4.Dataset(str(path.absolute()) + "/data_output.nc", "w", format="NETCDF4")
        nf4_basegrp.createDimension("ncells", grid.num_cells)
        nf4_basegrp.createDimension("ncells_2", grid.num_edges)
        nf4_basegrp.createDimension("ncells_3", grid.num_vertices)
        nf4_basegrp.createDimension("vertices", 3) # neighboring vertices of a cell
        nf4_basegrp.createDimension("vertices_2", 4) # neighboring vertices of an edge
        nf4_basegrp.createDimension("vertices_3", 6) # neighboring vertices of a vertex
        nf4_basegrp.createDimension("height_2", grid.num_levels) # full level height
        nf4_basegrp.createDimension("height", grid.num_levels + 1) # half level height
        nf4_basegrp.createDimension("bnds", 2) # boundary points for full level height
        nf4_basegrp.createDimension("time", None)

        self._nf4_basegrp = nf4_basegrp

        self._create_variables(grid)

        self._write_dimension(grid, diagnostic_metric_state)

    def _create_variables(self, grid: IconGrid):
        """
        grid information
        """
        self.times: nf4.Variable = self._nf4_basegrp.createVariable("time", "f8", ("time",))
        self.levels: nf4.Variable = self._nf4_basegrp.createVariable("height_2", "f8", ("height_2",))
        self.half_levels: nf4.Variable = self._nf4_basegrp.createVariable("height", "f8", ("height",))
        self.level_bounds: nf4.Variable = self._nf4_basegrp.createVariable("height_2_bnds", "f8", ("bnds",))
        #self.cells: nf4.Variable = self._nf4_basegrp.createVariable("cells", "i4", ("ncells",))
        self.cell_latitudes: nf4.Variable = self._nf4_basegrp.createVariable("clat", "f8", ("ncells",))
        self.cell_longitudes: nf4.Variable = self._nf4_basegrp.createVariable("clon", "f8", ("ncells",))
        self.cell_lat_bounds: nf4.Variable = self._nf4_basegrp.createVariable("clat_bnds", "f8", ("ncells", "vertices",))
        self.cell_lon_bounds: nf4.Variable = self._nf4_basegrp.createVariable("clon_bnds", "f8", ("ncells", "vertices",))
        self.edge_latitudes: nf4.Variable = self._nf4_basegrp.createVariable("elat", "f8", ("ncells_2",))
        self.edge_longitudes: nf4.Variable = self._nf4_basegrp.createVariable("elon", "f8", ("ncells_2",))
        self.edge_lat_bounds: nf4.Variable = self._nf4_basegrp.createVariable("elat_bnds", "f8", ("ncells_2", "vertices_2",))
        self.edge_lon_bounds: nf4.Variable = self._nf4_basegrp.createVariable("elon_bnds", "f8", ("ncells_2", "vertices_2",))
        self.vertex_latitudes: nf4.Variable = self._nf4_basegrp.createVariable("vlat", "f8", ("ncells_3",))
        self.vertex_longitudes: nf4.Variable = self._nf4_basegrp.createVariable("vlon", "f8", ("ncells_3",))
        self.vertex_lat_bounds: nf4.Variable = self._nf4_basegrp.createVariable("vlat_bnds", "f8", ("ncells_3", "vertices_3",))
        self.vertex_lon_bounds: nf4.Variable = self._nf4_basegrp.createVariable("vlon_bnds", "f8", ("ncells_3", "vertices_3",))
        """
        output variables
        """
        self.u = self._nf4_basegrp.createVariable("u", "f8", ("time", "height_2", "ncells",))
        self.v = self._nf4_basegrp.createVariable("v", "f8", ("time", "height_2", "ncells",))
        self.temperature = self._nf4_basegrp.createVariable("temperature", "f8", ("time", "height_2", "ncells",))
        self.pressure = self._nf4_basegrp.createVariable("pressure", "f8", ("time", "height_2", "ncells",))
        self.pressure_sfc = self._nf4_basegrp.createVariable("pressure_sfc", "f8", ("time", "ncells",))
        self.exner = self._nf4_basegrp.createVariable("exner", "f8", ("time", "height_2", "ncells",))
        self.theta_v = self._nf4_basegrp.createVariable("theta_v", "f8", ("time", "height_2", "ncells",))
        self.rho = self._nf4_basegrp.createVariable("rho", "f8", ("time", "height_2", "ncells",))
        self.w = self._nf4_basegrp.createVariable("w", "f8", ("time", "height", "ncells",))

        #self.cells[:] = np.arange(grid.num_cells, dtype=int)
        self.cell_latitudes.units = "radian"
        self.cell_longitudes.units = "radian"
        self.cell_lat_bounds.units = "radian"
        self.cell_lon_bounds.units = "radian"
        self.edge_latitudes.units = "radian"
        self.edge_longitudes.units = "radian"
        self.edge_lat_bounds.units = "radian"
        self.edge_lon_bounds.units = "radian"
        self.vertex_latitudes.units = "radian"
        self.vertex_longitudes.units = "radian"
        self.vertex_lat_bounds.units = "radian"
        self.vertex_lon_bounds.units = "radian"

        self.levels.units = "m"
        self.half_levels.units = "m"
        self.levels.axis = "Z"
        self.half_levels.axis = "Z"
        self.times.units = "seconds since 0001-01-01 00:00:00.0"
        self.times.calendar = "gregorian"
        self.times.axis = "T"

        self.u.units = "m s-1"
        self.v.units = "m s-1"
        self.w.units = "m s-1"
        self.temperature.units = "K"
        self.pressure.units = "Pa"
        self.pressure_sfc.units = "Pa"
        self.rho.units = "kg m-3"
        self.theta_v.units = "K"
        self.exner.units = ""

        self.u.units = "m s-1"
        self.v.units = "m s-1"
        self.w.units = "m s-1"
        self.temperature.units = "K"
        self.pressure.units = "Pa"
        self.pressure_sfc.units = "Pa"
        self.rho.units = "kg m-3"
        self.theta_v.units = "K"
        self.exner.units = ""

        self.times.standard_name = "time"
        self.cell_latitudes.standard_name = "latitude"
        self.cell_longitudes.standard_name = "longitude"
        self.edge_latitudes.standard_name = "latitude"
        self.edge_longitudes.standard_name = "longitude"
        self.vertex_latitudes.standard_name = "latitude"
        self.vertex_longitudes.standard_name = "longitude"
        self.levels.standard_name = "height"
        self.half_levels.standard_name = "height"

        self.times.long_name = "time"
        self.cell_latitudes.long_name = "center latitude"
        self.cell_longitudes.long_name = "center longitude"
        self.edge_latitudes.long_name = "edge midpoint latitude"
        self.edge_longitudes.long_name = "edge midpoint longitude"
        self.vertex_latitudes.long_name = "vertex latitude"
        self.vertex_longitudes.long_name = "vertex longitude"
        self.levels.long_name = "generalized_height"
        self.half_levels.long_name = "generalized_height"

        self.cell_latitudes.bounds = "clat_bnds"
        self.cell_longitudes.bounds = "clon_bnds"
        self.edge_latitudes.bounds = "elat_bnds"
        self.edge_longitudes.bounds = "elon_bnds"
        self.vertex_latitudes.bounds = "vlon_bnds"
        self.vertex_longitudes.bounds = "vlat_bnds"
        self.levels.bounds = "height_2_bnds"

        self.u.standard_name = "eastward_wind"
        self.v.standard_name = "northward_wind"
        self.w.standard_name = "upward_air_velocity"
        self.temperature.standard_name = "air_temperature"
        self.pressure.standard_name = "air_pressure"
        self.pressure_sfc.standard_name = "surface_air_pressure"
        self.rho.standard_name = "air_density"
        self.theta_v.standard_name = "virtual_potential_temperature"
        self.exner.standard_name = "exner_pressure"
        self.u.long_name = "Zonal wind"
        self.v.long_name = "Meridional wind"
        self.w.long_name = "Vertical velocity"
        self.temperature.long_name = "Temperature"
        self.pressure.long_name = "Pressure"
        self.pressure_sfc.long_name = "Surface pressure"
        self.rho.long_name = "Density"
        self.theta_v.long_name = "Virtual potential temperature"
        self.exner.long_name = "Exner pressure"

        self.u.CDI_grid_type = "unstructured"
        self.v.CDI_grid_type = "unstructured"
        self.w.CDI_grid_type = "unstructured"
        self.temperature.CDI_grid_type = "unstructured"
        self.pressure.CDI_grid_type = "unstructured"
        self.pressure_sfc.CDI_grid_type = "unstructured"
        self.rho.CDI_grid_type = "unstructured"
        self.theta_v.CDI_grid_type = "unstructured"
        self.exner.CDI_grid_type = "unstructured"

        self.u.number_of_grid_in_reference = 1
        self.v.number_of_grid_in_reference = 1
        self.w.number_of_grid_in_reference = 1
        self.temperature.number_of_grid_in_reference = 1
        self.pressure.number_of_grid_in_reference = 1
        self.pressure_sfc.number_of_grid_in_reference = 1
        self.rho.number_of_grid_in_reference = 1
        self.theta_v.number_of_grid_in_reference = 1
        self.exner.number_of_grid_in_reference = 1

        self.u.coordinates = "clat clon"
        self.v.coordinates = "clat clon"
        self.w.coordinates = "clat clon"
        self.temperature.coordinates = "clat clon"
        self.pressure.coordinates = "clat clon"
        self.pressure_sfc.coordinates = "clat clon"
        self.rho.coordinates = "clat clon"
        self.theta_v.coordinates = "clat clon"
        self.exner.coordinates = "clat clon"

    def _write_dimension(self, grid: IconGrid, diagnostic_metric_state: DiagnosticMetricState):
        self.cell_latitudes[:] = diagnostic_metric_state.cell_center_lat.asnumpy()
        self.cell_longitudes[:] = diagnostic_metric_state.cell_center_lon.asnumpy()
        self.cell_lat_bounds[:, :] = diagnostic_metric_state.v_lat.asnumpy()[grid.connectivities[C2VDim]]
        self.cell_lon_bounds[:, :] = diagnostic_metric_state.v_lon.asnumpy()[grid.connectivities[C2VDim]]

        self.edge_latitudes[:] = diagnostic_metric_state.e_lat.asnumpy()
        self.edge_longitudes[:] = diagnostic_metric_state.e_lon.asnumpy()
        self.edge_lat_bounds[:, :] = diagnostic_metric_state.v_lat.asnumpy()[grid.connectivities[E2C2VDim]]
        self.edge_lon_bounds[:, :] = diagnostic_metric_state.v_lon.asnumpy()[grid.connectivities[E2C2VDim]]
        log.info(f"E2C2VDim dimension: {diagnostic_metric_state.v_lon.asnumpy()[grid.connectivities[E2C2VDim]].shape}")
        log.info(f"V2C2VDim dimension: {diagnostic_metric_state.v_lon.asnumpy()[grid.connectivities[V2C2VDim]].shape}")

        self.vertex_latitudes[:] = diagnostic_metric_state.v_lat.asnumpy()
        self.vertex_longitudes[:] = diagnostic_metric_state.v_lon.asnumpy()
        self.vertex_lat_bounds[:, :] = diagnostic_metric_state.v_lat.asnumpy()[grid.connectivities[V2C2VDim]]
        self.vertex_lon_bounds[:, :] = diagnostic_metric_state.v_lon.asnumpy()[grid.connectivities[V2C2VDim]]

        full_height = np.zeros(grid.num_levels, dtype=float)
        half_height = diagnostic_metric_state.vct_a.asnumpy()
        for k in range(grid.num_levels):
            full_height[k] = 0.5 * (half_height[k] + half_height[k + 1])
        self.levels[:] = full_height
        self.half_levels[:] = half_height

    def write_first_date(self, first_date: datetime):
        self.times[0] = date2num(first_date, units=self.times.units, calendar=self.times.calendar)
        # dates = num2date(times[:], units=times.units, calendar=times.calendar)

    def write_data(self, timestep: int, prognostic_state: PrognosticState, diagnostic_state: DiagnosticState):
        self.u[timestep, :, :] = diagnostic_state.u.asnumpy().transpose()
        self.v[timestep, :, :] = diagnostic_state.v.asnumpy().transpose()
        self.w[timestep, :, :] = prognostic_state.w.asnumpy().transpose()
        self.temperature[timestep, :, :] = diagnostic_state.temperature.asnumpy().transpose()
        self.pressure[timestep, :, :] = diagnostic_state.pressure.asnumpy().transpose()
        self.pressure_sfc[timestep, :] = diagnostic_state.pressure_sfc.asnumpy().transpose()
        self.rho[timestep, :, :] = prognostic_state.rho.asnumpy().transpose()
        self.exner[timestep, :, :] = prognostic_state.exner.asnumpy().transpose()
        self.theta_v[timestep, :, :] = prognostic_state.theta_v.asnumpy().transpose()


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
        is_run_from_serializedData: bool = False,
    ):
        self.run_config: IconRunConfig = run_config
        self.grid: Optional[IconGrid] = grid
        self.diffusion = diffusion
        self.solve_nonhydro = solve_nonhydro
        # TODO (Chia Rui): find a more elegant way to determine whether this timeloop is run for comparison with serialized data
        self.is_run_from_serializedData = is_run_from_serializedData

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date)
            / timedelta(seconds=self.run_config.dtime)
        )
        self._n_substeps_var: int = self.run_config.n_substeps
        self._substep_timestep: float = float(self.run_config.dtime / self._n_substeps_var)

        self._validate_config()

        # current simulation date
        self._simulation_date: datetime = self.run_config.start_date

        self._do_initial_stabilization: bool = self.run_config.apply_initial_stabilization

        self._now: int = 0  # TODO (Chia Rui): move to PrognosticState
        self._next: int = 1  # TODO (Chia Rui): move to PrognosticState

    def re_init(self):
        self._simulation_date = self.run_config.start_date
        self._do_initial_stabilization = self.run_config.apply_initial_stabilization
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

    def _not_first_step(self):
        self._do_initial_stabilization = False

    def _next_simulation_date(self):
        self._simulation_date += timedelta(seconds=self.run_config.dtime)

    @property
    def do_initial_stabilization(self):
        return self._do_initial_stabilization

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
        z_fields: ZFields,  # local constants in solve_nh
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
        output_state: OutputState = None
    ):


        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"Initialization of diagnostic variables for output."
        )
        # TODO (Chia Rui): Move computation diagnostic variables to a module (diag_for_output_dyn subroutine)
        mo_rbf_vec_interpol_cell.with_backend(backend)(
            prognostic_state_list[self._now].vn,
            diagnostic_metric_state.rbf_vec_coeff_c1,
            diagnostic_metric_state.rbf_vec_coeff_c2,
            diagnostic_state.u,
            diagnostic_state.v,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            0,
            self.grid.num_levels,
            offset_provider={
                "C2E2C2E": self.grid.get_offset_provider("C2E2C2E"),
            },
        )

        mo_diagnose_temperature.with_backend(backend)(
            prognostic_state_list[self._now].theta_v,
            prognostic_state_list[self._now].exner,
            diagnostic_state.temperature,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            0,
            self.grid.num_levels,
            offset_provider={}
        )

        exner_nlev_minus2 = prognostic_state_list[self._now].exner[:, self.grid.num_levels - 3]
        temperature_nlev = diagnostic_state.temperature[:, self.grid.num_levels - 1]
        temperature_nlev_minus1 = diagnostic_state.temperature[:, self.grid.num_levels - 2]
        temperature_nlev_minus2 = diagnostic_state.temperature[:, self.grid.num_levels - 3]
        # TODO (Chia Rui): ddqz_z_full is constant, move slicing to initialization
        ddqz_z_full_nlev = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 1]
        ddqz_z_full_nlev_minus1 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 2]
        ddqz_z_full_nlev_minus2 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 3]
        mo_diagnose_pressure_sfc.with_backend(backend)(
            exner_nlev_minus2,
            temperature_nlev,
            temperature_nlev_minus1,
            temperature_nlev_minus2,
            ddqz_z_full_nlev,
            ddqz_z_full_nlev_minus1,
            ddqz_z_full_nlev_minus2,
            diagnostic_state.pressure_sfc,
            CPD_O_RD,
            P0REF,
            GRAV_O_RD,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            offset_provider={}
        )

        '''
        mo_diagnose_pressure.with_backend(backend)(
            diagnostic_state.temperature,
            diagnostic_state.pressure,
            diagnostic_state.pressure_ifc,
            diagnostic_state.pressure_sfc,
            diagnostic_metric_state.ddqz_z_full,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            0,
            self.grid.num_levels,
            offset_provider={}
        )
        '''

        if not self.is_run_from_serializedData:
            output_state.write_data(0, prognostic_state_list[self._now], diagnostic_state)

        if not self.is_run_from_serializedData:
            mo_init_exner_pr.with_backend(backend)(
                prognostic_state_list[self._now].exner,
                self.solve_nonhydro.metric_state_nonhydro.exner_ref_mc,
                solve_nonhydro_diagnostic_state.exner_pr,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                0,
                self.grid.num_levels,
                offset_provider={}
            )

        log.info(f"Debugging U (before diffusion): {np.max(diagnostic_state.u.asnumpy())}")

        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self._do_initial_stabilization} dtime={self.run_config.dtime} substep_timestep={self._substep_timestep}"
        )
        if self.diffusion.config.apply_to_horizontal_wind and self._do_initial_stabilization:
            log.info("running initial step to diffuse fields before timeloop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_state_list[self._now],
                self.run_config.dtime,
            )
        log.info(
            f"starting real time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(
                f"simulation date : {self._simulation_date} run timestep : {time_step} initial_stabilization : {self._do_initial_stabilization}"
            )

            if not self.is_run_from_serializedData:
                mo_init_ddt_cell_zero.with_backend(backend)(
                    solve_nonhydro_diagnostic_state.ddt_exner_phy,
                    solve_nonhydro_diagnostic_state.ddt_w_adv_ntl1,
                    solve_nonhydro_diagnostic_state.ddt_w_adv_ntl2,
                    self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                    self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                    0,
                    self.grid.num_levels,
                    offset_provider={}
                )
                mo_init_ddt_edge_zero.with_backend(backend)(
                    solve_nonhydro_diagnostic_state.ddt_vn_phy,
                    solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl1,
                    solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl2,
                    self.grid.get_start_index(EdgeDim, HorizontalMarkerIndex.interior(EdgeDim)),
                    self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.end(EdgeDim)),
                    0,
                    self.grid.num_levels,
                    offset_provider={}
                )

            self._next_simulation_date()

            # put boundary condition update here

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv,
                z_fields,
                inital_divdamp_fac_o2,
                do_prep_adv,
            )
            timer.capture()

            # TODO (Chia Rui): modify n_substeps_var if cfl condition is not met. (set_dyn_substeps subroutine)

            # TODO (Chia Rui): Move computation diagnostic variables to a module (diag_for_output_dyn subroutine)
            mo_rbf_vec_interpol_cell.with_backend(backend)(
                prognostic_state_list[self._now].vn,
                diagnostic_metric_state.rbf_vec_coeff_c1,
                diagnostic_metric_state.rbf_vec_coeff_c2,
                diagnostic_state.u,
                diagnostic_state.v,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                0,
                self.grid.num_levels,
                offset_provider={
                    "C2E2C2E": self.grid.get_offset_provider("C2E2C2E"),
                },
            )

            mo_diagnose_temperature.with_backend(backend)(
                prognostic_state_list[self._now].theta_v,
                prognostic_state_list[self._now].exner,
                diagnostic_state.temperature,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                0,
                self.grid.num_levels,
                offset_provider={}
            )

            exner_nlev_minus2 = prognostic_state_list[self._now].exner[:, self.grid.num_levels - 3]
            temperature_nlev = diagnostic_state.temperature[:, self.grid.num_levels - 1]
            temperature_nlev_minus1 = diagnostic_state.temperature[:, self.grid.num_levels - 2]
            temperature_nlev_minus2 = diagnostic_state.temperature[:, self.grid.num_levels - 3]
            # TODO (Chia Rui): below are constant, move slicing to initialization
            ddqz_z_full_nlev = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 1]
            ddqz_z_full_nlev_minus1 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 2]
            ddqz_z_full_nlev_minus2 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 3]
            mo_diagnose_pressure_sfc.with_backend(backend)(
                exner_nlev_minus2,
                temperature_nlev,
                temperature_nlev_minus1,
                temperature_nlev_minus2,
                ddqz_z_full_nlev,
                ddqz_z_full_nlev_minus1,
                ddqz_z_full_nlev_minus2,
                diagnostic_state.pressure_sfc,
                CPD_O_RD,
                P0REF,
                GRAV_O_RD,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                offset_provider={}
            )

            '''
            mo_diagnose_pressure.with_backend(backend)(
                diagnostic_state.temperature,
                diagnostic_state.pressure,
                diagnostic_state.pressure_ifc,
                diagnostic_state.pressure_sfc,
                diagnostic_metric_state.ddqz_z_full,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                0,
                self.grid.num_levels,
                offset_provider={}
            )
            '''

            # TODO (Chia Rui): simple IO enough for JW test
            log.info(f"Debugging U (after diffusion): {np.max(diagnostic_state.u.asnumpy())}")

            if not self.is_run_from_serializedData:
                output_state.write_data(time_step + 1, prognostic_state_list[self._now], diagnostic_state)

        timer.summary(True)


    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
        z_fields: ZFields,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):

        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_state_list,
            prep_adv,
            z_fields,
            inital_divdamp_fac_o2,
            do_prep_adv,
        )

        if self.diffusion.config.apply_to_horizontal_wind:
            self.diffusion.run(
                diffusion_diagnostic_state, prognostic_state_list[self._next], self.run_config.dtime
            )

        self._swap()

        # TODO (Chia Rui): add tracer advection here

    def _do_dyn_substepping(
        self,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
        z_fields: ZFields,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): compute airmass for prognostic_state here

        do_recompute = True
        do_clean_mflx = True
        for dyn_substep in range(self._n_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} sub timestep : {dyn_substep}, initial_stabilization : {self._do_initial_stabilization}, nnow: {self._now}, nnew : {self._next}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                z_fields=z_fields,
                divdamp_fac_o2=inital_divdamp_fac_o2,
                dtime=self._substep_timestep,
                idyn_timestep=dyn_substep,
                l_recompute=do_recompute,
                l_init=self._do_initial_stabilization,
                nnew=self._next,
                nnow=self._now,
                lclean_mflx=do_clean_mflx,
                lprep_adv=do_prep_adv,
            )

            do_recompute = False
            do_clean_mflx = False

            if dyn_substep != self._n_substeps_var - 1:
                self._swap()

            self._not_first_step()

        # TODO (Chia Rui): compute airmass for prognostic_state here

# "icon_pydycore"

def initialize(experiment_name: str, fname_prefix: str, ser_type: SerializationType, init_type: InitializationType, run_path: Path, file_path: Path, props: ProcessProperties):
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
         other temporary fields: to be removed in the future
    """
    log.info("initialize parallel runtime")
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name)

    decomp_info = read_decomp_info(fname_prefix, file_path, props, ser_type=ser_type)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(fname_prefix, file_path, rank=props.rank, ser_type=ser_type)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry, c_owner_mask) = read_geometry_fields(
        fname_prefix, file_path, rank=props.rank, ser_type=ser_type
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    ) = read_static_fields(fname_prefix, file_path, ser_type=ser_type, init_type=init_type)

    log.info("initializing netCDF4 output state")
    output_state = OutputState(run_path, icon_grid, diagnostic_metric_state)

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
        z_fields,
        prep_adv,
        inital_divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    ) = read_initial_state(
        icon_grid,
        cell_geometry,
        edge_geometry,
        config.run_config.time_discretization_veladv_offctr,
        config.run_config.time_discretization_rhotheta_offctr,
        file_path,
        rank=props.rank,
        initialization_type=init_type
    )
    prognostic_state_list = [prognostic_state_now, prognostic_state_next]

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
        z_fields,
        prep_adv,
        inital_divdamp_fac_o2,
    )


@click.command()
@click.argument("input_path")
@click.argument("fname_prefix")
@click.option("--experiment_name", default="mch_ch_r04b09_dsl")
@click.option("--ser_type", default="serialbox")
@click.option("--init_type", default="serialbox")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
def main(input_path, fname_prefix, experiment_name, ser_type, init_type, run_path, mpi):
    """
    Run the driver.

    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/jw_node1_nproma50000/ jabw --experiment_name=jabw --ser_type=serialbox --init_type=jabw
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data icon_pydycore --ser_type=serialbox --init_type=serialbox

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
    if ser_type == SerializationType.SB:
        print("1", ser_type)
    elif ser_type == SerializationType.NC:
        print("2", ser_type)
    if init_type == InitializationType.SB:
        print("3", init_type)
    elif init_type == InitializationType.JABW:
        print("4", init_type)

    configure_logging(run_path, experiment_name, parallel_props)

    (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        diagnostic_metric_state,
        diagnostic_state,
        prognostic_state_list,
        output_state,
        z_fields,
        prep_adv,
        inital_divdamp_fac_o2,
    ) = initialize(experiment_name, fname_prefix, ser_type, init_type, Path(run_path), Path(input_path), parallel_props)

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
        diagnostic_state,
        prognostic_state_list,
        prep_adv,
        z_fields,
        inital_divdamp_fac_o2,
        do_prep_adv=False,
        output_state=output_state,
    )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
