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
from typing import Callable

import click
import pytz
from devtools import Timer
from gt4py.next import Field, program
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn

from icon4py.model.atmosphere.dycore.nh_solve.solve_nonydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion
from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.atmosphere.diffusion.diffusion_utils import _identity_c_k, _identity_e_k

from icon4py.model.common.grid.horizontal import EdgeParams
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.states.prognostic_state import PrognosticState, copy_prognostics
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config
from icon4py.model.driver.io_utils import (
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)

from icon4py.model.common.test_utils.helpers import zero_field

log = logging.getLogger(__name__)


# TODO (magdalena) to be removed once there is a proper time stepping
@program
def _copy_diagnostic_and_prognostics(
    hdef_ic_new: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    div_ic_new: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    dwdx_new: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy_new: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
):
    _identity_c_k(hdef_ic_new, out=hdef_ic)
    _identity_c_k(div_ic_new, out=div_ic)
    _identity_c_k(dwdx_new, out=dwdx)
    _identity_c_k(dwdy_new, out=dwdy)
    _identity_e_k(vn_new, out=vn)
    _identity_c_k(w_new, out=w)
    _identity_c_k(exner_new, out=exner)
    _identity_c_k(theta_v_new, out=theta_v)



class TimeLoop:

    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: IconRunConfig,
        diffusion: Diffusion,
        non_hydro_solver: SolveNonhydro
    ):
        self.run_config: IconRunConfig = run_config
        self.diffusion = diffusion
        self.non_hydro_solver = non_hydro_solver

        self._n_time_steps: int = int((self.run_config.end_date - self.run_config.start_date) / timedelta(seconds=self.run_config.dtime))
        self._substep_timestep: float = float(self.run_config.dtime/self.run_config.ndyn_substeps)

        # check validity of configurations
        self._validate()

        # current simulation date
        # self.simulation_date = datetime.fromisoformat(SIMULATION_START_DATE)
        self._simulation_date: datetime = self.run_config.start_date

        self._l_init: bool = True

    def re_init(self):
        self._simulation_date: datetime = self.run_config.start_date
        self._l_init: bool = True

    def _validate(self):
        if (self._n_time_steps < 0):
            raise ValueError("end_date should be larger than start_date. Please check.")
        if (not self.diffusion.initialized):
            raise Exception("diffusion is not initialized before time loop")
        if (not self.non_hydro_solver.initialized):
            raise Exception("nonhydro solver is not initialized before time loop")


    def _not_first_step(self):
        self._l_init = False

    def _next_simulation_date(self):
        self._simulation_date += timedelta(seconds=self.run_config.dtime)

    @property
    def simulation_date(self):
        return self._simulation_date

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @property
    def substep_timestep(self):
        return self._substep_timestep

    def _full_name(self, func: Callable):
        return ":".join((self.__class__.__name__, func.__name__))

    def time_integration(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        # below is a long list of arguments for dycore time_step (suggestion: many can be moved to initialization of SolveNonhydro)
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams, # precomputed constants
        z_fields: ZFields, # local constants in solve_nh
        nh_constants: NHConstants, # user defined constants
        cfl_w_limit: float, # need to check
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float], # precomputed constants
        c_owner_mask: Field[[CellDim], bool], # precomputed constants
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float], # in EdgeParams, precomputed constants
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool
    ):
        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        if (self.diffusion.config.apply_to_horizontal_wind and self._l_init and not self.run_config.is_testcase):
            log.info("running initial step to diffuse fields before timeloop starts")
            self.diffusion.initial_run(
                diffusion_diagnostic_state,
                prognostic_state,
                self.run_config.dtime,
            )
        log.info(
            f"starting real time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for i_time_step in range(self._n_time_steps):
            log.info(f"run timestep : {i_time_step}")

            print(self._simulation_date, " large time step ", i_time_step, self._l_init)
            self._next_simulation_date()

            # update boundary condition

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state,
                prep_adv,
                nonhydro_config,
                nonhydro_params,
                edge_geometry,
                z_fields,
                nh_constants,
                cfl_w_limit,
                scalfac_exdiff,
                cell_areas,
                c_owner_mask,
                f_e,
                area_edge,
                bdy_divdamp,
                lprep_adv
            )
            timer.capture()

            # TODO IO


        timer.summary(True)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        c_owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool
    ):
        self._do_dyn_substepping(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prognostic_state,
            prep_adv,
            nonhydro_config,
            nonhydro_params,
            edge_geometry,
            z_fields,
            nh_constants,
            cfl_w_limit,
            scalfac_exdiff,
            cell_areas,
            c_owner_mask,
            f_e,
            area_edge,
            bdy_divdamp,
            lprep_adv
        )

        if (self.diffusion.config.apply_to_horizontal_wind and self.run_config.apply_horizontal_diff_at_large_dt):
            self.diffusion.run(
                diffusion_diagnostic_state,
                prognostic_state,
                self.run_config.dtime
            )

        # TODO tracer advection

    def _do_dyn_substepping(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        prep_adv: PrepAdvection,
        nonhydro_config: NonHydrostaticConfig,
        nonhydro_params: NonHydrostaticParams,
        edge_geometry: EdgeParams,
        z_fields: ZFields,
        nh_constants: NHConstants,
        cfl_w_limit: float,
        scalfac_exdiff: float,
        cell_areas: Field[[CellDim], float],
        c_owner_mask: Field[[CellDim], bool],
        f_e: Field[[EdgeDim], float],
        area_edge: Field[[EdgeDim], float],
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool
    ):

        # TODO compute airmass for prognostic_state

        # TODO remove this copying process, add new and now buffers in prognostic_state class
        rho_new = zero_field(self.non_hydro_solver.grid,CellDim,KDim)
        w_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        vn_new = zero_field(self.non_hydro_solver.grid, EdgeDim, KDim)
        exner_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        theta_v_new = zero_field(self.non_hydro_solver.grid, CellDim, KDim)
        copy_prognostics.with_backend(run_gtfn)(
            prognostic_state.rho,
            rho_new,
            prognostic_state.w,
            w_new,
            prognostic_state.vn,
            vn_new,
            prognostic_state.exner,
            exner_new,
            prognostic_state.theta_v,
            theta_v_new,
            offset_provider={},
        )
        prognostic_state_new = PrognosticState(
            rho=rho_new,
            w=w_new,
            vn=vn_new,
            exner=exner_new,
            theta_v=theta_v_new
        )
        prognostic_state_list = [prognostic_state, prognostic_state_new]
        time_n = 1
        time_n_plus_1 = 0
        l_recompute = True
        lclean_mflx = True
        for idyn_timestep in range(self.run_config.ndyn_substeps):
            print(self._simulation_date, " small time step ", idyn_timestep, self._l_init)
            time_n_swap = time_n_plus_1
            time_n_plus_1 = time_n
            time_n = time_n_swap
            self.non_hydro_solver.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                config=nonhydro_config,
                params=nonhydro_params,
                edge_geometry=edge_geometry,
                z_fields=z_fields,
                nh_constants=nh_constants,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                cell_areas=cell_areas,
                c_owner_mask=c_owner_mask,
                f_e=f_e,
                area_edge=area_edge,
                bdy_divdamp=bdy_divdamp,
                dtime=self._substep_timestep,
                idyn_timestep=idyn_timestep,
                l_recompute=l_recompute,
                l_init=self._l_init,
                nnew=time_n_plus_1,
                nnow=time_n,
                lclean_mflx=lclean_mflx,
                lprep_adv=lprep_adv
            )
            if (self.diffusion.config.apply_to_horizontal_wind and not self.run_config.apply_horizontal_diff_at_large_dt):
                self.diffusion.run(
                    diffusion_diagnostic_state,
                    prognostic_state_list[time_n_plus_1],
                    self._substep_timestep
                )

            l_recompute = False
            lclean_mflx = False

            self._not_first_step()

        if (time_n_plus_1 == 1):
            # TODO remove this copying process when the prognostic_state contains two buffers
            copy_prognostics.with_backend(run_gtfn)(
                prognostic_state_new.rho,
                prognostic_state.rho,
                prognostic_state_new.w,
                prognostic_state.w,
                prognostic_state_new.vn,
                prognostic_state.vn,
                prognostic_state_new.exner,
                prognostic_state.exner,
                prognostic_state_new.theta_v,
                prognostic_state.theta_v,
                offset_provider={},
            )

        # TODO compute airmass for prognostic_state



# TODO initialization of prognostic variables and topography of Jablonowski Williamson test
def model_initialization():

    # create two prognostic states, nnow and nnew?
    # at least two prognostic states are global because they are needed in the dycore, AND possibly nesting and restart processes in the future
    # one is enough for the JW test
    prognostic_state_1 = PrognosticState(
        w=None,
        vn=None,
        theta_v=None,
        rho=None,
        exner=None,
    )
    prognostic_state_2 = PrognosticState(
        w=None,
        vn=None,
        theta_v=None,
        rho=None,
        exner=None,
    )
    return (prognostic_state_1, prognostic_state_2)

'''
def initialize(file_path: Path, props: ProcessProperties):
    log.info("initialize parallel runtime")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name, n_time_steps=n_time_steps)

    decomp_info = read_decomp_info(file_path, props)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(file_path, rank=props.rank)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        file_path, rank=props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(file_path)

    log.info("initializing diffusion")
    diffusion_params = DiffusionParams(config.diffusion_config)
    exchange = create_exchange(props, decomp_info)
    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    data_provider, diagnostic_state, prognostic_state = read_initial_state(
        file_path, rank=props.rank
    )

    atmo_non_hydro = DummyAtmoNonHydro(data_provider)
    atmo_non_hydro.init(config=config.dycore_config)

    tl = Timeloop(
        config=config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
    )
    return tl, diagnostic_state, prognostic_state

def initialize_original(n_time_steps, file_path: Path, props: ProcessProperties):
    """
    Inititalize the driver run.

    "reads" in
        - configuration

        - grid information

        - (serialized) input fields, initial

     Returns:
         tl: configured timeloop,
         prognostic_state: initial state fro prognostic and diagnostic variables
         diagnostic_state:
    """
    log.info("initialize parallel runtime")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name, n_time_steps=n_time_steps)

    decomp_info = read_decomp_info(file_path, props)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(file_path, rank=props.rank)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        file_path, rank=props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(file_path)

    log.info("initializing diffusion")
    diffusion_params = DiffusionParams(config.diffusion_config)
    exchange = create_exchange(props, decomp_info)
    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
        edge_geometry,
        cell_geometry,
    )

    data_provider, diagnostic_state, prognostic_state = read_initial_state(
        file_path, rank=props.rank
    )

    atmo_non_hydro = DummyAtmoNonHydro(data_provider)
    atmo_non_hydro.init(config=config.dycore_config)

    tl = Timeloop(
        config=config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
    )
    return tl, diagnostic_state, prognostic_state

@click.command()
@click.argument("input_path")
@click.option("--run_path", default="", help="folder for output")
@click.option("--n_steps", default=5, help="number of time steps to run, max 5 is supported")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
def main(input_path, run_path, n_steps, mpi):
    """
    Run the driver.

    usage: python driver/dycore_driver.py ../../tests/ser_icondata/mch_ch_r04b09_dsl/ser_data

    steps:
    1. initialize model:

        a) load config

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) setup the time loop

    2. run time loop

    """


    timeloop = TimeLoop()
    timeloop.init()
    #start_time = datetime.now().astimezone(pytz.UTC)
    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    configure_logging(run_path, start_time, parallel_props)
    log.info(f"Starting ICON dycore run: {datetime.isoformat(start_time)}")
    log.info(f"input args: input_path={input_path}, n_time_steps={n_steps}")
    timeloop, diagnostic_state, prognostic_state = initialize(
        n_steps, Path(input_path), parallel_props
    )
    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    timeloop(diagnostic_state, prognostic_state)

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
'''
