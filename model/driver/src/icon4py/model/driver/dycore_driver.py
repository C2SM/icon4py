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
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import click
import netCDF4 as nf4
import numpy as np
from cftime import date2num, num2date
from devtools import Timer
from gt4py.next import as_field
from gt4py.next.program_processors.runners.gtfn import run_gtfn, run_gtfn_cached

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
from icon4py.model.common.diagnostic_calculations.stencils.mo_diagnose_temperature_pressure import (
    mo_diagnose_pressure,
    mo_diagnose_pressure_sfc,
    mo_diagnose_temperature,
)
from icon4py.model.common.diagnostic_calculations.stencils.mo_init_exner_pr import mo_init_exner_pr
from icon4py.model.common.diagnostic_calculations.stencils.mo_init_zero import (
    mo_init_ddt_cell_zero,
    mo_init_ddt_edge_zero,
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
from icon4py.model.common.interpolation.stencils.mo_rbf_vec_interpol_cell import (
    mo_rbf_vec_interpol_cell,
)
from icon4py.model.common.states.diagnostic_state import DiagnosticMetricState, DiagnosticState
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.driver.icon_configuration import IconOutputConfig, IconRunConfig, read_config
from icon4py.model.driver.initialization_utils import (
    SerializationType,
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)
from icon4py.model.driver.testcase_functions import mo_rbf_vec_interpol_cell_numpy


compiler_backend = run_gtfn
compiler_cached_backend = run_gtfn_cached
backend = compiler_cached_backend

log = logging.getLogger(__name__)


class OutputState:
    def __init__(
        self,
        output_config: IconOutputConfig,
        start_date: datetime,
        end_date: datetime,
        grid: IconGrid,
        diagnostic_metric_state: DiagnosticMetricState,
        prognostic_state: PrognosticState,
        diagnostic_state: DiagnosticState,
    ):
        self.config: IconOutputConfig = output_config
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
            self._nf4_basegrp[i].createDimension("ncells", grid.num_cells)
            self._nf4_basegrp[i].createDimension("height", grid.num_levels)  # full level height
            self._nf4_basegrp[i].createDimension("height_2", grid.num_levels + 1)  # half level height
            self._nf4_basegrp[i].createDimension("bnds", 2)  # boundary points for full level height
            self._nf4_basegrp[i].createDimension("time", None)

        self._current_write_step: int = 0
        self._current_file_number: int = 0

        self._create_variables()
        self._write_dimension(grid, diagnostic_metric_state)

        self._write_to_netcdf(start_date, prognostic_state, diagnostic_state)

    @property
    def current_time_step(self):
        return self._current_write_step

    def _create_variables(self):
        for i in range(self._number_of_files):
            """
            grid information
            """
            times: nf4.Variable = self._nf4_basegrp[i].createVariable("time", "f8", ("time",))
            levels: nf4.Variable = self._nf4_basegrp[i].createVariable("height", "f8", ("height",))
            half_levels: nf4.Variable = self._nf4_basegrp[i].createVariable("height_2", "f8", ("height_2",))
            self._nf4_basegrp[i].createVariable(
                "height_bnds",
                "f8",
                (
                    "height",
                    "bnds",
                ),
            )
            """
            output variables
            """
            u = self._nf4_basegrp[i].createVariable(
                "u",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            v = self._nf4_basegrp[i].createVariable(
                "v",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            temp = self._nf4_basegrp[i].createVariable(
                "temp",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            pres = self._nf4_basegrp[i].createVariable(
                "pres",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            pres_sfc = self._nf4_basegrp[i].createVariable(
                "pres_sfc",
                "f8",
                (
                    "time",
                    "ncells",
                ),
            )
            theta_v = self._nf4_basegrp[i].createVariable(
                "theta_v",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            rho = self._nf4_basegrp[i].createVariable(
                "rho",
                "f8",
                (
                    "time",
                    "height",
                    "ncells",
                ),
            )
            w = self._nf4_basegrp[i].createVariable(
                "w",
                "f8",
                (
                    "time",
                    "height_2",
                    "ncells",
                ),
            )

            levels.units = "m"
            half_levels.units = "m"
            levels.axis = "Z"
            half_levels.axis = "Z"
            times.units = "seconds since 0001-01-01 00:00:00.0"
            times.calendar = "gregorian"
            times.axis = "T"

            u.units = "m s-1"
            v.units = "m s-1"
            w.units = "m s-1"
            temp.units = "K"
            pres.units = "Pa"
            pres_sfc.units = "Pa"
            theta_v.units = "K"
            rho.units = "kg m-3"

            u.param = "2.2.0"
            v.param = "3.2.0"
            w.param = "9.2.0"
            temp.param = "0.0.0"
            pres.param = "0.3.0"
            pres_sfc.param = "0.3.0"
            theta_v.param = "15.0.0"
            rho.param = "10.3.0"

            times.standard_name = "time"
            levels.standard_name = "full height"
            half_levels.standard_name = "half height"

            times.long_name = "time"
            levels.long_name = "generalized_full_height"
            half_levels.long_name = "generalized_half_height"

            levels.bounds = "height_bnds"

            u.standard_name = "eastward_wind"
            v.standard_name = "northward_wind"
            w.standard_name = "upward_air_velocity"
            temp.standard_name = "air_temperature"
            pres.standard_name = "air_pressure"
            pres_sfc.standard_name = "surface_air_pressure"
            theta_v.standard_name = "virtual_potential_temperature"
            rho.standard_name = "air_density"

            u.long_name = "Zonal wind"
            v.long_name = "Meridional wind"
            w.long_name = "Vertical velocity"
            temp.long_name = "Temperature"
            pres.long_name = "Pressure"
            pres_sfc.long_name = "Surface pressure"
            theta_v.long_name = "Virtual potential temperature"
            rho.long_name = "Density"

            u.CDI_grid_type = "unstructured"
            v.CDI_grid_type = "unstructured"
            w.CDI_grid_type = "unstructured"
            temp.CDI_grid_type = "unstructured"
            pres.CDI_grid_type = "unstructured"
            pres_sfc.CDI_grid_type = "unstructured"
            theta_v.CDI_grid_type = "unstructured"
            rho.CDI_grid_type = "unstructured"

            u.number_of_grid_in_reference = 1
            v.number_of_grid_in_reference = 1
            w.number_of_grid_in_reference = 1
            temp.number_of_grid_in_reference = 1
            pres.number_of_grid_in_reference = 1
            pres_sfc.number_of_grid_in_reference = 1
            theta_v.number_of_grid_in_reference = 1
            rho.number_of_grid_in_reference = 1

            u.coordinates = "clat clon"
            v.coordinates = "clat clon"
            w.coordinates = "clat clon"
            temp.coordinates = "clat clon"
            pres.coordinates = "clat clon"
            pres_sfc.coordinates = "clat clon"
            theta_v.coordinates = "clat clon"
            rho.coordinates = "clat clon"

    def _write_dimension(self, grid: IconGrid, diagnostic_metric_state: DiagnosticMetricState):
        for i in range(self._number_of_files):

            full_height = np.zeros(grid.num_levels, dtype=float)
            half_height = diagnostic_metric_state.vct_a.asnumpy()
            full_height_bnds = np.zeros((grid.num_levels, 2), dtype=float)
            for k in range(grid.num_levels):
                full_height[k] = 0.5 * (half_height[k] + half_height[k + 1])
                full_height_bnds[k, 0] = half_height[k]
                full_height_bnds[k, 1] = half_height[k + 1]
            self._nf4_basegrp[i].variables["height"][:] = full_height
            self._nf4_basegrp[i].variables["height_2"][:] = half_height
            self._nf4_basegrp[i].variables["height_bnds"][:, :] = full_height_bnds

    def _write_to_netcdf(
        self,
        current_date: datetime,
        prognostic_state: PrognosticState,
        diagnostic_state: DiagnosticState,
    ):
        log.info( f"Writing output at {current_date} at {self._current_write_step} in file no. {self._current_file_number}")

        times = self._nf4_basegrp[self._current_file_number].variables["time"]
        times[self._current_write_step] = date2num(current_date, units=times.units, calendar=times.calendar)
        log.info(f"Times are  {times[:]}")

        self._nf4_basegrp[self._current_file_number].variables["u"][self._current_write_step, :, :] = diagnostic_state.u.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["v"][self._current_write_step, :, :] = diagnostic_state.v.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["w"][self._current_write_step, :, :] = prognostic_state.w.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["temp"][self._current_write_step, :, :] = diagnostic_state.temperature.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["pres"][self._current_write_step, :, :] = diagnostic_state.pressure.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["pres_sfc"][self._current_write_step, :] = diagnostic_state.pressure_sfc.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["theta_v"][self._current_write_step, :, :] = prognostic_state.theta_v.asnumpy().transpose()
        self._nf4_basegrp[self._current_file_number].variables["rho"][self._current_write_step, :, :] = prognostic_state.rho.asnumpy().transpose()

    def output_data(
        self,
        current_date: datetime,
        prognostic_state: PrognosticState,
        diagnostic_state: DiagnosticState,
    ):
        time_elapsed_since_last_output = current_date - self._output_date
        time_elapsed_in_this_ncfile = current_date - self._first_date_in_this_ncfile
        log.info(f"first date in currect nc file: {self._first_date_in_this_ncfile}, previous output date: {self._output_date}")
        log.info(f"time elapsed since last output: {time_elapsed_since_last_output}, time elapsed in this file: {time_elapsed_in_this_ncfile}")

        if time_elapsed_in_this_ncfile > self.config.output_file_time_interval:
            log.info("CLOSING {:} / {:} ||| {:} / {:}".format(self._current_file_number+1, self._number_of_files, time_elapsed_in_this_ncfile, self.config.output_file_time_interval))
            self._nf4_basegrp[self._current_file_number].close()

        if time_elapsed_since_last_output >= self.config.output_time_interval:
            if self._enforce_new_ncfile or time_elapsed_in_this_ncfile > self.config.output_file_time_interval:
                self._enforce_new_ncfile = False
                self._first_date_in_this_ncfile =  self._output_date
                self._current_write_step = 0
                self._current_file_number += 1
            else:
                self._current_write_step += 1
            self._write_to_netcdf(
                current_date,
                prognostic_state,
                diagnostic_state,
            )
            self._output_date = current_date
        else:
            log.info(f"SKIP writing output at {current_date} at {self._current_write_step}")


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

        self.stencil_mo_rbf_vec_interpol_cell = mo_rbf_vec_interpol_cell.with_backend(backend)
        self.stencil_mo_diagnose_temperature = mo_diagnose_temperature.with_backend(backend)
        self.stencil_mo_diagnose_pressure_sfc = mo_diagnose_pressure_sfc.with_backend(backend)
        self.stencil_mo_diagnose_pressure = mo_diagnose_pressure.with_backend(backend)
        self.stencil_mo_init_exner_pr = mo_init_exner_pr.with_backend(backend)
        self.stencil_mo_init_ddt_cell_zero = mo_init_ddt_cell_zero.with_backend(backend)
        self.stencil_mo_init_ddt_edge_zero = mo_init_ddt_edge_zero.with_backend(backend)

        self.offset_provider_c2e2c2e = {
            "C2E2C2E": self.grid.get_offset_provider("C2E2C2E"),
        }

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

    def _is_last_substep(self, step_nr: int):
        return step_nr == (self.n_substeps_var - 1)

    @staticmethod
    def _is_first_substep(step_nr: int):
        return step_nr == 0

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

    def _diagnose_for_output_and_physics(
        self,
        prognostic_state: PrognosticState,
        diagnostic_state: DiagnosticState,
        diagnostic_metric_state: DiagnosticMetricState,
    ):
        self.stencil_mo_rbf_vec_interpol_cell(
            prognostic_state.vn,
            diagnostic_metric_state.rbf_vec_coeff_c1,
            diagnostic_metric_state.rbf_vec_coeff_c2,
            diagnostic_state.u,
            diagnostic_state.v,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 1),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            0,
            self.grid.num_levels,
            offset_provider=self.offset_provider_c2e2c2e,
        )
        log.debug(
            f"max min v: {diagnostic_state.v.asnumpy().max()} {diagnostic_state.v.asnumpy().min()}"
        )

        self.stencil_mo_diagnose_temperature(
            prognostic_state.theta_v,
            prognostic_state.exner,
            diagnostic_state.temperature,
            self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
            self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
            0,
            self.grid.num_levels,
            offset_provider={},
        )

        exner_nlev_minus2 = prognostic_state.exner[:, self.grid.num_levels - 3]
        temperature_nlev = diagnostic_state.temperature[:, self.grid.num_levels - 1]
        temperature_nlev_minus1 = diagnostic_state.temperature[:, self.grid.num_levels - 2]
        temperature_nlev_minus2 = diagnostic_state.temperature[:, self.grid.num_levels - 3]
        ddqz_z_full_nlev = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 1]
        ddqz_z_full_nlev_minus1 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 2]
        ddqz_z_full_nlev_minus2 = diagnostic_metric_state.ddqz_z_full[:, self.grid.num_levels - 3]
        self.stencil_mo_diagnose_pressure_sfc(
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
            offset_provider={},
        )

        # TODO (Chia Rui): to add computation of pressure

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
        output_state: OutputState = None,
    ):
        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        log.info("Initialization of diagnostic variables for output.")

        self._diagnose_for_output_and_physics(
            prognostic_state_list[self._now], diagnostic_state, diagnostic_metric_state
        )

        if not self.is_run_from_serializedData:
            self.stencil_mo_init_exner_pr(
                prognostic_state_list[self._now].exner,
                self.solve_nonhydro.metric_state_nonhydro.exner_ref_mc,
                solve_nonhydro_diagnostic_state.exner_pr,
                self.grid.get_start_index(CellDim, HorizontalMarkerIndex.interior(CellDim)),
                self.grid.get_end_index(CellDim, HorizontalMarkerIndex.end(CellDim)),
                0,
                self.grid.num_levels,
                offset_provider={},
            )

        log.info(f"Debugging U (before diffusion): {np.max(diagnostic_state.u.asnumpy())}")

        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self._do_initial_stabilization} dtime={self.run_config.dtime} substep_timestep={self._substep_timestep}"
        )
        if self.diffusion.config.apply_to_horizontal_wind and self._do_initial_stabilization and not self.run_config.run_testcase:
            log.info("running initial step to diffuse fields before timeloop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_state_list[self._now],
                self.run_config.dtime,
            )

        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):

            self._next_simulation_date()

            log.info(
                f"simulation date : {self._simulation_date} run timestep : {time_step} initial_stabilization : {self._do_initial_stabilization}"
            )

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

            # put boundary condition update here

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

            self._diagnose_for_output_and_physics(
                prognostic_state_list[self._now], diagnostic_state, diagnostic_metric_state
            )

            log.info(f"Debugging U (after diffusion): {np.max(diagnostic_state.u.asnumpy())}")

            if not self.is_run_from_serializedData:
                output_state.output_data(
                    self._simulation_date,
                    prognostic_state_list[self._now],
                    diagnostic_state,
                )

        log.info("CLOSING {:} / {:}".format(output_state._current_file_number+1, output_state._number_of_files))
        output_state._nf4_basegrp[-1].close()

        timer.summary(True)

    def time_integration_speed_test(
        self,
        diagnostic_metric_state: DiagnosticMetricState,
        diagnostic_state: DiagnosticState,
        prognostic_state_list: list[PrognosticState],
    ):
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
        fo = mo_rbf_vec_interpol_cell.with_backend(backend)
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
        self._do_dyn_substepping(
            solve_nonhydro_diagnostic_state,
            prognostic_state_list,
            prep_adv,
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
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        # TODO (Chia Rui): compute airmass for prognostic_state here

        do_recompute = True
        do_clean_mflx = True
        for dyn_substep in range(self._n_substeps_var):
            log.info(
                f"simulation date : {self._simulation_date} substep / n_substeps : {dyn_substep} / "
                f"{self.n_substeps_var} , initial_stabilization : {self._do_initial_stabilization}, "
                f"nnow: {self._now}, nnew : {self._next}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                divdamp_fac_o2=inital_divdamp_fac_o2,
                dtime=self._substep_timestep,
                l_recompute=do_recompute,
                l_init=self._do_initial_stabilization,
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

            self._not_first_step()

        # TODO (Chia Rui): compute airmass for prognostic_state here


# "icon_pydycore"


def initialize(
    experiment_name: str,
    fname_prefix: str,
    ser_type: SerializationType,
    file_path: Path,
    props: ProcessProperties,
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
         prep_advection: fields collecting data for advection during the solve nonhydro timestep
         inital_divdamp_fac_o2: initial divergence damping factor

    """
    log.info("initialize parallel runtime")
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name)

    decomp_info = read_decomp_info(fname_prefix, file_path, props, ser_type=ser_type)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(fname_prefix, file_path, rank=props.rank, ser_type=ser_type)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry, c_owner_mask) = read_geometry_fields(
        fname_prefix, file_path, config.run_config.damping_height, rank=props.rank, ser_type=ser_type
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
        diagnostic_metric_state,
    ) = read_static_fields(experiment_name, fname_prefix, file_path, ser_type=ser_type)

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
        experiment_name,
        fname_prefix,
        icon_grid,
        cell_geometry,
        edge_geometry,
        file_path,
        rank=props.rank,
    )
    prognostic_state_list = [prognostic_state_now, prognostic_state_next]

    log.info("initializing netCDF4 output state")
    output_state = OutputState(
        config.output_config,
        config.run_config.start_date,
        config.run_config.end_date,
        icon_grid,
        diagnostic_metric_state,
        prognostic_state_list[0],
        diagnostic_state,
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
@click.argument("fname_prefix")
@click.option("--experiment_name", default="mch_ch_r04b09_dsl")
@click.option("--ser_type", default="serialbox")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
@click.option("--speed_test", default=False)
def main(input_path, fname_prefix, experiment_name, ser_type, run_path, mpi, speed_test):
    """
    Run the driver.

    usage: python dycore_driver.py abs_path_to_icon4py/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/jw_node1_nproma50000/ jabw --experiment_name=jabw --ser_type=serialbox
    python driver/src/icon4py/model/driver/dycore_driver.py ~/PycharmProjects/main/testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data icon_pydycore --ser_type=serialbox

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

    configure_logging(run_path, experiment_name, parallel_props)

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
    ) = initialize(experiment_name, fname_prefix, ser_type, Path(input_path), parallel_props)

    log.info(f"Starting ICON dycore run: {timeloop.simulation_date.isoformat()}")
    log.info(
        f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}, ending date={timeloop.run_config.end_date}"
    )

    log.info(f"input args: input_path={input_path}, n_time_steps={timeloop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    if speed_test:
        timeloop.time_integration_speed_test(
            diagnostic_metric_state, diagnostic_state, prognostic_state_list
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
        )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
