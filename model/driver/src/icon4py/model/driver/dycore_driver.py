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
from gt4py.next import Field, program

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.atmosphere.diffusion.diffusion_utils import _identity_c_k, _identity_e_k
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
    SolveNonhydro,
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.nh_constants import NHConstants
from icon4py.model.atmosphere.dycore.state_utils.prep_adv_state import PrepAdvection
from icon4py.model.atmosphere.dycore.state_utils.z_fields import ZFields
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config
from icon4py.model.driver.io_utils import (
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)


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
        solve_nonhydro: SolveNonhydro,
        apply_initial_stabilization: bool = True,
    ):
        self.run_config: IconRunConfig = run_config
        self.diffusion = diffusion
        self.solve_nonhydro = solve_nonhydro

        self._n_time_steps: int = int(
            (self.run_config.end_date - self.run_config.start_date)
            / timedelta(seconds=self.run_config.dtime)
        )
        self._substep_timestep: float = float(self.run_config.dtime / self.run_config.ndyn_substeps)

        # check validity of configurations
        self._validate()

        # current simulation date
        self._simulation_date: datetime = self.run_config.start_date

        self._l_init: bool = apply_initial_stabilization

        self._now: int = 0  # TODO: move to PrognosticState
        self._next: int = 1  # TODO: move to PrognosticState

    def re_init(self):
        self._simulation_date = self.run_config.start_date
        self._l_init = True
        self._now: int = 0  # TODO: move to PrognosticState
        self._next: int = 1  # TODO: move to PrognosticState

    def _validate(self):
        if self._n_time_steps < 0:
            raise ValueError("end_date should be larger than start_date. Please check.")
        if not self.diffusion.initialized:
            raise Exception("diffusion is not initialized before time loop")
        if not self.solve_nonhydro.initialized:
            raise Exception("nonhydro solver is not initialized before time loop")

    def _not_first_step(self):
        self._l_init = False

    def _next_simulation_date(self):
        self._simulation_date += timedelta(seconds=self.run_config.dtime)

    @property
    def linit(self):
        return self._l_init

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
        # TODO: expand the PrognosticState to include indices of now and next, now it is always assumed that now = 0, next = 1 at the beginning
        prognostic_state_list: list[PrognosticState],
        # below is a long list of arguments for dycore time_step that many can be moved to initialization of SolveNonhydro)
        prep_adv: PrepAdvection,
        z_fields: ZFields,  # local constants in solve_nh
        nh_constants: NHConstants,
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool,
    ):

        log.info(
            f"starting time loop for dtime={self.run_config.dtime} n_timesteps={self._n_time_steps}"
        )
        log.info(
            f"apply_to_horizontal_wind={self.diffusion.config.apply_to_horizontal_wind} linit={self._l_init} apply_horizontal_diff_at_large_dt={self.run_config.apply_horizontal_diff_at_large_dt} dtime={self.run_config.dtime} substep_timestep={self._substep_timestep}"
        )
        if self.diffusion.config.apply_to_horizontal_wind and self._l_init:
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
        for i_time_step in range(self._n_time_steps):
            log.info(
                f"simulation date : {self._simulation_date} run timestep : {i_time_step} linit : {self._l_init}"
            )

            self._next_simulation_date()

            # update boundary condition

            timer.start()
            self._integrate_one_time_step(
                diffusion_diagnostic_state,
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv,
                z_fields,
                nh_constants,
                bdy_divdamp,
                lprep_adv,
            )
            timer.capture()

            # TODO IO

        timer.summary(True)

    def _integrate_one_time_step(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
        z_fields: ZFields,
        nh_constants: NHConstants,
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool,
    ):

        self._do_dyn_substepping(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prognostic_state_list,
            prep_adv,
            z_fields,
            nh_constants,
            bdy_divdamp,
            lprep_adv,
        )

        if (
            self.diffusion.config.apply_to_horizontal_wind
            and self.run_config.apply_horizontal_diff_at_large_dt
        ):
            self.diffusion.run(
                diffusion_diagnostic_state, prognostic_state_list[self._next], self.run_config.dtime
            )

        self._swap()

        # TODO tracer advection

    def _do_dyn_substepping(
        self,
        diffusion_diagnostic_state: DiffusionDiagnosticState,
        solve_nonhydro_diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state_list: list[PrognosticState],
        prep_adv: PrepAdvection,
        z_fields: ZFields,
        nh_constants: NHConstants,
        bdy_divdamp: Field[[KDim], float],
        lprep_adv: bool,
    ):

        # TODO compute airmass for prognostic_state

        l_recompute = True
        lclean_mflx = True
        for idyn_timestep in range(self.run_config.ndyn_substeps):
            log.info(
                f"simulation date : {self._simulation_date} sub timestep : {idyn_timestep}, linit : {self._l_init}, nnow: {self._now}, nnew : {self._next}"
            )
            self.solve_nonhydro.time_step(
                solve_nonhydro_diagnostic_state,
                prognostic_state_list,
                prep_adv=prep_adv,
                z_fields=z_fields,
                nh_constants=nh_constants,
                bdy_divdamp=bdy_divdamp,
                dtime=self._substep_timestep,
                idyn_timestep=idyn_timestep,
                l_recompute=l_recompute,
                l_init=self._l_init,
                nnew=self._next,
                nnow=self._now,
                lclean_mflx=lclean_mflx,
                lprep_adv=lprep_adv,
            )
            if (
                self.diffusion.config.apply_to_horizontal_wind
                and not self.run_config.apply_horizontal_diff_at_large_dt
            ):
                self.diffusion.run(
                    diffusion_diagnostic_state,
                    prognostic_state_list[self._next],
                    self._substep_timestep,
                )

            l_recompute = False
            lclean_mflx = False

            if idyn_timestep != self.run_config.ndyn_substeps - 1:
                self._swap()

            self._not_first_step()

        # TODO compute airmass for prognostic_state


def initialize(file_path: Path, props: ProcessProperties):
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
         other temporary fields:
    """
    log.info("initialize parallel runtime")
    experiment_name = "mch_ch_r04b09_dsl"
    log.info(f"reading configuration: experiment {experiment_name}")
    config = read_config(experiment_name)

    decomp_info = read_decomp_info(file_path, props)

    log.info(f"initializing the grid from '{file_path}'")
    icon_grid = read_icon_grid(file_path, rank=props.rank)
    log.info(f"reading input fields from '{file_path}'")
    (edge_geometry, cell_geometry, vertical_geometry, c_owner_mask) = read_geometry_fields(
        file_path, rank=props.rank
    )
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
    ) = read_static_fields(file_path)

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

    (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        z_fields,
        nh_constants,
        prep_adv,
        bdy_divdamp,
        prognostic_state_now,
        prognostic_state_next,
    ) = read_initial_state(file_path, rank=props.rank)
    prognostic_state_list = [prognostic_state_now, prognostic_state_next]

    nonhydro_config = NonHydrostaticConfig()
    nonhydro_params = NonHydrostaticParams(nonhydro_config)

    grid = SimpleGrid()
    enh_smag_fac = zero_field(grid, KDim)
    a_vec = random_field(grid, KDim, low=1.0, high=10.0, extend={KDim: 1})
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)

    solve_nonhydro = SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=nonhydro_config,
        params=nonhydro_params,
        metric_state_nonhydro=solve_nonhydro_metric_state,
        interpolation_state=solve_nonhydro_interpolation_state,
        vertical_params=vertical_geometry,
        edge_geometry=edge_geometry,
        cell_areas=cell_geometry.area,
        owner_mask=c_owner_mask,
        a_vec=a_vec,
        enh_smag_fac=enh_smag_fac,
        fac=fac,
        z=z,
    )

    timeloop = TimeLoop(
        run_config=config.run_config,
        diffusion=diffusion,
        solve_nonhydro=solve_nonhydro,
        apply_initial_stabilization=True,
    )
    return (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prognostic_state_list,
        z_fields,
        nh_constants,
        prep_adv,
        bdy_divdamp,
    )


@click.command()
@click.argument("input_path")
@click.option("--run_path", default="./", help="folder for output")
@click.option("--mpi", default=False, help="whether or not you are running with mpi")
def main(input_path, run_path, mpi):
    """
    Run the driver.

    usage: python driver/dycore_driver.py ../testdata/ser_icondata/mpitask1/mch_ch_r04b09_dsl/ser_data

    steps:
    1. initialize model:

        a) load config

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) setup the time loop

    2. run time loop
    """
    parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
    (
        timeloop,
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prognostic_state_list,
        z_fields,
        nh_constants,
        prep_adv,
        bdy_divdamp,
    ) = initialize(Path(input_path), parallel_props)
    configure_logging(run_path, timeloop.simulation_date, parallel_props)
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
        prognostic_state_list,
        prep_adv,
        z_fields,
        nh_constants,
        bdy_divdamp,
        lprep_adv=False,
    )

    log.info("timeloop:  DONE")


if __name__ == "__main__":
    main()
