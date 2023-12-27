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
from icon4py.model.common.dimension import C2VDim, CellDim, EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from gt4py.next.program_processors.runners.gtfn import run_gtfn

backend = run_gtfn
log = logging.getLogger(__name__)


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
    ):
        # TODO (Chia Rui): this is only a tentative output method, use a proper netcdf output infrastructure in the future
        nf4_basegrp = nf4.Dataset(self.run_config.output_path+"data_output.nc", "w", format="NETCDF4")
        nf4_basegrp.createDimension("ncells", self.grid.num_cells)
        nf4_basegrp.createDimension("vertices", 3)
        nf4_basegrp.createDimension("levels", self.grid.num_levels)
        nf4_basegrp.createDimension("half_levels", self.grid.num_levels+1)
        nf4_basegrp.createDimension("time", self._n_time_steps+1)

        nf4_times = nf4_basegrp.createVariable("time", "f8", ("time",))
        nf4_levels = nf4_basegrp.createVariable("levels", "f8", ("levels",))
        nf4_halflevels = nf4_basegrp.createVariable("half_levels", "f8", ("half_levels",))
        nf4_cells = nf4_basegrp.createVariable("cells", "i4", ("ncells",))
        nf4_latitudes = nf4_basegrp.createVariable("clat", "f8", ("ncells",))
        nf4_longitudes = nf4_basegrp.createVariable("clon", "f8", ("ncells",))
        nf4_lat_bounds = nf4_basegrp.createVariable("clat_bnds", "f8", ("ncells", "vertices",))
        nf4_lon_bounds = nf4_basegrp.createVariable("clon_bnds", "f8", ("ncells", "vertices",))

        nf4_cells[:] = np.arange(self.grid.num_cells, dtype=int)
        nf4_cells.units = ""
        nf4_latitudes.units = "rad"
        nf4_longitudes.units = "rad"
        nf4_lat_bounds.units = "rad"
        nf4_lon_bounds.units = "rad"
        nf4_levels.units = "m"
        nf4_halflevels.units = "m"
        nf4_times.units = "seconds since 0001-01-01 00:00:00.0"
        nf4_times.calendar = "gregorian"

        nf4_u = nf4_basegrp.createVariable("u", "f8", ("time", "ncells", "levels",))
        nf4_v = nf4_basegrp.createVariable("v", "f8", ("time", "ncells", "levels",))
        nf4_temperature = nf4_basegrp.createVariable("temperature", "f8", ("time", "ncells", "levels",))
        nf4_pressure = nf4_basegrp.createVariable("pressure", "f8", ("time", "ncells", "levels",))
        nf4_pressure_sfc = nf4_basegrp.createVariable("pressure_sfc", "f8", ("time", "ncells",))
        nf4_exner = nf4_basegrp.createVariable("exner", "f8", ("time", "ncells", "levels",))
        nf4_theta_v = nf4_basegrp.createVariable("theta_v", "f8", ("time", "ncells", "levels",))
        nf4_rho = nf4_basegrp.createVariable("rho", "f8", ("time", "ncells", "levels",))
        nf4_w = nf4_basegrp.createVariable("w", "f8", ("time", "ncells", "half_levels",))

        nf4_u.units = "m s-1"
        nf4_v.units = "m s-1"
        nf4_w.units = "m s-1"
        nf4_temperature.units = "K"
        nf4_pressure.units = "Pa"
        nf4_pressure_sfc.units = "Pa"
        nf4_rho.units = "kg m-3"
        nf4_theta_v.units = "K"
        nf4_exner.units = ""

        nf4_latitudes[:] = diagnostic_metric_state.cell_center_lat.asnumpy()
        nf4_longitudes[:] = diagnostic_metric_state.cell_center_lon.asnumpy()
        nf4_lat_bounds[:, :] = diagnostic_metric_state.v_lat.asnumpy()[self.grid.connectivities[C2VDim]]
        nf4_lon_bounds[:, :] = diagnostic_metric_state.v_lon.asnumpy()[self.grid.connectivities[C2VDim]]
        nf4_times[0] = date2num(self._simulation_date, units=nf4_times.units, calendar=nf4_times.calendar)
        # dates = num2date(times[:], units=times.units, calendar=times.calendar)

        def printing_data(data, title: str, first_write: bool = False):
            if first_write:
                write_mode = "w"
            else:
                write_mode = "a"
            with open(self.run_config.output_path + "jw_data_" + title + ".dat", write_mode) as f:
                no_dim = len(data.shape)
                f.write("{0:7d}\n".format(self._n_time_steps + 1))
                if no_dim == 2:
                    cell_size = data.shape[0]
                    k_size = data.shape[1]
                    f.write("{0:7d} {1:7d}\n".format(cell_size, k_size))
                    log.info(f"Writing {title} with sizes of {cell_size}, {k_size} into a formatted file")
                    for i in range(cell_size):
                        for k in range(k_size):
                            f.write("{0:7d} {1:7d}".format(i, k))
                            f.write(
                                " {0:.20e}\n".format(
                                    data[i, k]
                                )
                            )
                elif no_dim == 1:
                    cell_size = data.shape[0]
                    f.write("{0:7d}\n".format(cell_size))
                    log.info(f"Writing {title} with sizes of {cell_size} into a formatted file")
                    for i in range(cell_size):
                        f.write("{0:7d} {1:.20e}\n".format(i, data[i]))

        def printing_grid(data_lat, data_lon, vertical_grid):
            with open(self.run_config.output_path + "jw_grid.dat", "w") as f:
                cell_size = data_lat.shape[0]
                if cell_size != data_lon.shape[0]:
                    log.warning(f"Sizes of lat and lon are not equal, {cell_size}, {data_lon.shape[0]}, please check")
                k_size = vertical_grid.shape[0]
                log.info(f"Writing grid data with sizes of {cell_size}, {k_size} into a formatted file")
                f.write("{0:7d} {1:7d}\n".format(cell_size, k_size))
                for i in range(cell_size):
                    f.write("{0:7d}".format(i))
                    f.write(
                        " {0:.20e} {1:.20e}".format(
                            data_lat[i], data_lon[i]
                        )
                    )
                    f.write("\n")
                for i in range(k_size):
                    f.write("{0:7d}".format(i))
                    f.write(
                        " {0:.20e}".format(
                            vertical_grid[i]
                        )
                    )
                    f.write("\n")

        full_height = np.zeros(self.grid.num_levels, dtype=float)
        half_height = diagnostic_metric_state.vct_a.asnumpy()
        log.info(
            f"Writing grid file. vct_a size is {half_height.shape}"
        )
        for k in range(self.grid.num_levels):
            full_height[k] = 0.5 * (half_height[k] + half_height[k+1])
        nf4_levels[:] = full_height
        nf4_halflevels[:] = half_height
        printing_grid(
            diagnostic_metric_state.cell_center_lat.asnumpy(),
            diagnostic_metric_state.cell_center_lon.asnumpy(),
            full_height,
        )

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

        printing_data(diagnostic_state.temperature.asnumpy(),"temperature_init",first_write=True)
        printing_data(prognostic_state_list[self._now].vn.asnumpy(), "vn_init", first_write=True)
        printing_data(prognostic_state_list[self._now].rho.asnumpy(), "rho_init", first_write=True)
        printing_data(diagnostic_state.u.asnumpy(), "u_init", first_write=True)
        printing_data(diagnostic_state.v.asnumpy(), "v_init", first_write=True)
        printing_data(diagnostic_state.pressure_sfc.asnumpy(), "sfc_pres_init", first_write=True)
        printing_data(diagnostic_state.u.asnumpy()[:, 0], "sfc_u_init", first_write=True)
        printing_data(diagnostic_state.v.asnumpy()[:, 0], "sfc_v_init", first_write=True)

        nf4_u[0, :, :] = diagnostic_state.u.asnumpy()
        nf4_v[0, :, :] = diagnostic_state.v.asnumpy()
        nf4_w[0, :, :] = prognostic_state_list[self._now].w.asnumpy()
        nf4_temperature[0, :, :] = diagnostic_state.temperature.asnumpy()
        nf4_pressure[0, :, :] = diagnostic_state.pressure.asnumpy()
        nf4_pressure_sfc[0, :] = diagnostic_state.pressure_sfc.asnumpy()
        nf4_rho[0, :, :] = prognostic_state_list[self._now].rho.asnumpy()
        nf4_exner[0, :, :] = prognostic_state_list[self._now].exner.asnumpy()
        nf4_theta_v[0, :, :] = prognostic_state_list[self._now].theta_v.asnumpy()

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

        printing_data(solve_nonhydro_diagnostic_state.exner_pr.asnumpy(), "exner_pr_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.exner_pr.asnumpy()[:, 0], "sfc_exner_pr_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.exner_pr.asnumpy()[:, 20], "sfc20_exner_pr_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_exner_phy.asnumpy(), "ddt_exner_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_vn_phy.asnumpy(), "ddt_vn_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_w_adv_ntl1.asnumpy(), "ddt_w1_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_w_adv_ntl2.asnumpy(), "ddt_w2_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl1.asnumpy(), "ddt_vn1_init", first_write=True)
        printing_data(solve_nonhydro_diagnostic_state.ddt_vn_apc_ntl2.asnumpy(), "ddt_vn2_init", first_write=True)
        printing_data(prognostic_state_list[self._now].w.asnumpy(), "w_init", first_write=True)
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

            nf4_u[time_step + 1, :, :] = diagnostic_state.u.asnumpy()
            nf4_v[time_step + 1, :, :] = diagnostic_state.v.asnumpy()
            nf4_w[time_step + 1, :, :] = prognostic_state_list[self._now].w.asnumpy()
            nf4_temperature[time_step + 1, :, :] = diagnostic_state.temperature.asnumpy()
            nf4_pressure[time_step + 1, :, :] = diagnostic_state.pressure.asnumpy()
            nf4_pressure_sfc[time_step + 1, :] = diagnostic_state.pressure_sfc.asnumpy()
            nf4_rho[time_step + 1, :, :] = prognostic_state_list[self._now].rho.asnumpy()
            nf4_exner[time_step + 1, :, :] = prognostic_state_list[self._now].exner.asnumpy()
            nf4_theta_v[time_step + 1, :, :] = prognostic_state_list[self._now].theta_v.asnumpy()

        timer.summary(True)

        # printing_data(diagnostic_state.temperature.asnumpy(), "temperature")
        # printing_data(diagnostic_state.u.asnumpy(), "u")
        # printing_data(diagnostic_state.v.asnumpy(), "v")
        # printing_data(diagnostic_state.pressure_sfc.asnumpy(), "sfc_pres")

        printing_data(diagnostic_state.temperature.asnumpy(), "temperature_final", first_write=True)
        printing_data(prognostic_state_list[self._now].vn.asnumpy(), "vn_final", first_write=True)
        printing_data(prognostic_state_list[self._now].rho.asnumpy(), "rho_final", first_write=True)
        printing_data(diagnostic_state.u.asnumpy(), "u_final", first_write=True)
        printing_data(diagnostic_state.v.asnumpy(), "v_final", first_write=True)
        printing_data(diagnostic_state.pressure_sfc.asnumpy(), "sfc_pres_final", first_write=True)
        printing_data(diagnostic_state.u.asnumpy()[:, 0], "sfc_u_final", first_write=True)
        printing_data(diagnostic_state.v.asnumpy()[:, 0], "sfc_v_final", first_write=True)
        printing_data(z_fields.z_q.asnumpy(), "z_q_final", first_write=True)
        printing_data(prognostic_state_list[self._now].w.asnumpy(), "w_final", first_write=True)
        printing_data(z_fields.z_alpha.asnumpy(), "z_alpha_final", first_write=True)
        printing_data(z_fields.z_beta.asnumpy(), "z_beta_final", first_write=True)
        printing_data(z_fields.z_exner_expl.asnumpy(), "z_exner_expl_final", first_write=True)
        printing_data(z_fields.z_w_expl.asnumpy(), "z_w_expl_final", first_write=True)

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

def initialize(experiment_name: str, fname_prefix: str, ser_type: SerializationType, init_type: InitializationType, file_path: Path, props: ProcessProperties):
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
        z_fields,
        prep_adv,
        inital_divdamp_fac_o2,
    ) = initialize(experiment_name, fname_prefix, ser_type, init_type, Path(input_path), parallel_props)

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
    )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
