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
from devtools import Timer

from gt4py.next import Field, as_field
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams, HorizontalMarkerIndex

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
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config
from icon4py.model.driver.io_utils import (
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)
import icon4py.model.common.constants as constants
import numpy as np
from icon4py.model.common.test_utils.helpers import dallclose

log = logging.getLogger(__name__)


# numpy version
def mo_solve_nonhydro_stencil_24_numpy(
    vn_nnow: np.array,
    ddt_vn_apc_ntl1: np.array,
    ddt_vn_phy: np.array,
    z_theta_v_e: np.array,
    z_gradh_exner: np.array,
    vn_nnew: np.array,
    dtime: float,
    cpd: float,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
):
    vn_nnew_update = np.array(vn_nnew)
    vn_nnew_update[horizontal_start:horizontal_end, vertical_start:vertical_end] = (
        vn_nnow[horizontal_start:horizontal_end, vertical_start:vertical_end] + dtime * (
            ddt_vn_apc_ntl1[horizontal_start:horizontal_end, vertical_start:vertical_end]
            - cpd * z_theta_v_e[horizontal_start:horizontal_end, vertical_start:vertical_end] *
            z_gradh_exner[horizontal_start:horizontal_end, vertical_start:vertical_end]
            + ddt_vn_phy[horizontal_start:horizontal_end, vertical_start:vertical_end]
        )
    )
    return vn_nnew_update

# stencil 24 of solve_nonhydro
def speed_test_step_numpy(
    prognostic_state_ls: list[PrognosticState],
    ddt_vn_apc_pc: np.array,
    ddt_vn_phy: np.array,
    z_theta_v_e: np.array,
    z_gradh_exner: np.array,
    dtime: float,
    nnow: int,
    nnew: int,
    start_edge_nudging_plus1: int,
    end_edge_local: int,
    num_levels: int
):

    # testing the performance of stencil 24
    vn_new = mo_solve_nonhydro_stencil_24_numpy(
        vn_nnow=prognostic_state_ls[nnow].vn.asnumpy(),
        ddt_vn_apc_ntl1=ddt_vn_apc_pc,
        ddt_vn_phy=ddt_vn_phy,
        z_theta_v_e=z_theta_v_e,
        z_gradh_exner=z_gradh_exner,
        vn_nnew=prognostic_state_ls[nnew].vn.asnumpy(),
        dtime=dtime,
        cpd=constants.CPD,
        horizontal_start=start_edge_nudging_plus1,
        horizontal_end=end_edge_local,
        vertical_start=0,
        vertical_end=num_levels
    )

    prognostic_state_ls[nnew].vn = as_field((EdgeDim, KDim), vn_new)

class TimeLoop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        run_config: IconRunConfig,
        diffusion: Diffusion,
        solve_nonhydro: SolveNonhydro,
    ):
        self.run_config: IconRunConfig = run_config
        self.diffusion = diffusion
        self.solve_nonhydro = solve_nonhydro

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
        if self._n_time_steps <= 0:
            raise ValueError("end_date should be larger than start_date. Please check.")
        #if not self.diffusion.initialized:
        #    raise Exception("diffusion is not initialized before time loop")
        #if not self.solve_nonhydro.initialized:
        #    raise Exception("nonhydro solver is not initialized before time loop")

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

    def speed_test_time_integration(
        self,
        prognostic_state_list: list[PrognosticState],
        start_edge_nudging_plus1,
        end_edge_local,
        num_edges,
        num_levels
    ):
        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )

        #start_edge_nudging_plus1 = self.solve_nonhydro.grid.get_start_index(
        #    EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        #)
        #end_edge_local = self.solve_nonhydro.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))
        #num_levels = self.solve_nonhydro.grid.num_levels
        #num_edges = self.solve_nonhydro.grid.num_edges

        log.info(
            f"num_edges={num_edges} num_levels={num_levels}"
        )
        ddt_vn_apc_pc_np = np.random.rand(num_edges,num_levels)
        ddt_vn_phy_np = np.random.rand(num_edges,num_levels)
        z_theta_v_e_np = np.random.rand(num_edges,num_levels)
        z_gradh_exner_np = np.random.rand(num_edges,num_levels)

        ddt_vn_apc_pc = as_field((EdgeDim,KDim), ddt_vn_apc_pc_np)
        ddt_vn_phy = as_field((EdgeDim,KDim), ddt_vn_phy_np)
        z_theta_v_e = as_field((EdgeDim,KDim), z_theta_v_e_np)
        z_gradh_exner = as_field((EdgeDim,KDim), z_gradh_exner_np)

        vn_now_np = prognostic_state_list[self._now].vn.asnumpy()

        ref_prognostic_now = PrognosticState(
            w=prognostic_state_list[self._now].w,
            vn=prognostic_state_list[self._now].vn,
            theta_v=prognostic_state_list[self._now].theta_v,
            rho=prognostic_state_list[self._now].rho,
            exner=prognostic_state_list[self._now].exner,
        )
        ref_prognostic_new = PrognosticState(
            w=prognostic_state_list[self._next].w,
            vn=prognostic_state_list[self._next].vn,
            theta_v=prognostic_state_list[self._next].theta_v,
            rho=prognostic_state_list[self._next].rho,
            exner=prognostic_state_list[self._next].exner,
        )
        ref_prognostic_state_list = [ref_prognostic_now, ref_prognostic_new]

        log.info(
            f"Checking numerics {np.abs(ddt_vn_apc_pc_np).max()} {np.abs(ddt_vn_phy_np).max()} {np.abs(z_theta_v_e_np).max()} {np.abs(z_gradh_exner_np).max()} {np.abs(vn_now_np).max()}"
        )
        log.info(
            f"Checking initial values {np.abs(prognostic_state_list[self._next].vn.asnumpy()).max()} {np.abs(prognostic_state_list[self._next].vn.asnumpy()).min()}"
        )
        log.info(
            f"Checking initial values {np.abs(ref_prognostic_state_list[self._next].vn.asnumpy()).max()} {np.abs(ref_prognostic_state_list[self._next].vn.asnumpy()).min()}"
        )
        log.info(
            f"starting speed test time loop in gt4py version for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(
                f"simulation date : {self._simulation_date} run timestep : {time_step} initial_stabilization : {self._do_initial_stabilization}"
            )

            self._next_simulation_date()

            # update boundary condition

            timer.start()
            self.solve_nonhydro.speed_test_step(
                prognostic_state_list,
                ddt_vn_apc_pc,
                ddt_vn_phy,
                z_theta_v_e,
                z_gradh_exner,
                self.run_config.dtime,
                self._now,
                self._next,
                start_edge_nudging_plus1,
                end_edge_local,
                num_levels
            )
            timer.capture()

        timer.summary(True)

        self.re_init()

        log.info(
            f"starting speed test time loop in numpy version for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        timer = Timer(self._full_name(self._integrate_one_time_step))
        for time_step in range(self._n_time_steps):
            log.info(
                f"simulation date : {self._simulation_date} run timestep : {time_step} initial_stabilization : {self._do_initial_stabilization}"
            )

            self._next_simulation_date()

            # update boundary condition

            timer.start()
            speed_test_step_numpy(
                ref_prognostic_state_list,
                ddt_vn_apc_pc_np,
                ddt_vn_phy_np,
                z_theta_v_e_np,
                z_gradh_exner_np,
                self.run_config.dtime,
                self._now,
                self._next,
                start_edge_nudging_plus1,
                end_edge_local,
                num_levels
            )
            timer.capture()

        timer.summary(True)

        assert dallclose(
            ref_prognostic_state_list[self._next].vn.asnumpy(),
            prognostic_state_list[self._next].vn.asnumpy(),
        )

    def time_integration(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        # TODO (Chia Rui): expand the PrognosticState to include indices of now and next, now it is always assumed that now = 0, next = 1 at the beginning
        prognostic_state_list: list[PrognosticState],
        # below is a long list of arguments for dycore time_step that many can be moved to initialization of SolveNonhydro)
        prep_adv: PrepAdvection,
        inital_divdamp_fac_o2: float,
        do_prep_adv: bool,
    ):
        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} initial_stabilization={self._do_initial_stabilization} dtime={self.run_config.dtime} substep_timestep={self._substep_timestep}"
        )

        # TODO (Chia Rui): Initialize vn tendencies that are used in solve_nh and advection to zero (init_ddt_vn_diagnostics subroutine)

        # TODO (Chia Rui): Compute diagnostic variables: P, T, zonal and meridonial winds, necessary for JW test output (diag_for_output_dyn subroutine)

        # TODO (Chia Rui): Initialize exner_pr used in solve_nh (compute_exner_pert subroutine)

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
                f"simulation date : {self._simulation_date} sub timestep : {dyn_substep}, initial_stabilization : {self._do_initial_stabilization}, nnow: {self._now}, nnew : {self._next}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
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


def initialize(props: ProcessProperties):
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
    log.info("initialize")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name)

    log.info("initializing diffusion")
    diffusion = Diffusion()

    solve_nonhydro = SolveNonhydro()

    timeloop = TimeLoop(
        run_config=config.run_config,
        diffusion=diffusion,
        solve_nonhydro=solve_nonhydro,
    )

    start_edge_nudging_plus1 = 100
    end_edge_local = 49900
    num_edges = 55000
    num_levels = 65
    vn = np.random.rand(num_edges, num_levels)
    w = np.random.rand(num_edges, num_levels)
    theta_v = np.random.rand(num_edges, num_levels)
    rho = np.random.rand(num_edges, num_levels)
    exner = np.random.rand(num_edges, num_levels)
    prognostic_state_now = PrognosticState(
        w=as_field((EdgeDim,KDim), w),
        vn=as_field((EdgeDim, KDim), vn),
        theta_v=as_field((EdgeDim, KDim), theta_v),
        rho=as_field((EdgeDim, KDim), rho),
        exner=as_field((EdgeDim, KDim), exner),
    )
    prognostic_state_next = PrognosticState(
        w=as_field((EdgeDim, KDim), w),
        vn=as_field((EdgeDim, KDim), vn),
        theta_v=as_field((EdgeDim, KDim), theta_v),
        rho=as_field((EdgeDim, KDim), rho),
        exner=as_field((EdgeDim, KDim), exner),
    )
    prognostic_state_list = [prognostic_state_now, prognostic_state_next]

    return (
        timeloop,
        prognostic_state_list,
        start_edge_nudging_plus1,
        end_edge_local,
        num_edges,
        num_levels,
    )


@click.command()
#@click.argument("input_path")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
def main(run_path, mpi):
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
    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    (
        timeloop,
        prognostic_state_list,
        start_edge_nudging_plus1,
        end_edge_local,
        num_edges,
        num_levels,
    ) = initialize(parallel_props)
    configure_logging(run_path, timeloop.simulation_date, parallel_props)
    log.info(
        f"num_edges={num_edges} num_levels={num_levels}"
    )
    log.info(f"Starting ICON dycore run: {timeloop.simulation_date.isoformat()}")
    log.info(
        f"input args: n_time_steps={timeloop.n_time_steps}, ending date={timeloop.run_config.end_date}"
    )

    log.info(f"input args: n_time_steps={timeloop.n_time_steps}")

    log.info("dycore configuring: DONE")
    log.info("timeloop: START")

    timeloop.speed_test_time_integration(
        prognostic_state_list,
        start_edge_nudging_plus1,
        end_edge_local,
        num_edges,
        num_levels
    )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
