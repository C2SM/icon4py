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
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
from icon4py.model.atmosphere.diffusion.diffusion_utils import _identity_c_k, _identity_e_k
from icon4py.model.common.decomposition.definitions import (
    ProcessProperties,
    create_exchange,
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.test_utils import serialbox_utils as sb
from icon4py.model.driver.icon_configuration import IconRunConfig, read_config
from icon4py.model.driver.io_utils import (
    SIMULATION_START_DATE,
    configure_logging,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)
from icon4py.model.driver.serialbox_helpers import construct_diagnostics_for_diffusion


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


class DummyAtmoNonHydro:
    def __init__(self, data_provider: sb.IconSerialDataProvider):
        self.config = None
        self.data_provider = data_provider
        self.simulation_date = datetime.fromisoformat(SIMULATION_START_DATE)

    def init(self, config):
        self.config = config

    def _next_physics_date(self, dtime: float):
        dynamics_dtime = dtime / self.config.n_substeps
        self.simulation_date += timedelta(seconds=dynamics_dtime)

    def _dynamics_timestep(self, dtime):
        """Show structure with this dummy fucntion called inside substepping loop."""
        self._next_physics_date(dtime)

    def do_dynamics_substepping(
        self,
        dtime,
        diagnostic_state: DiffusionDiagnosticState,
        prognostic_state: PrognosticState,
    ):
        for _ in range(self.config.n_substeps):
            self._dynamics_timestep(dtime)
        sp = self.data_provider.from_savepoint_diffusion_init(
            linit=False, date=self.simulation_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        )
        new_p = sp.construct_prognostics()
        new_d = construct_diagnostics_for_diffusion(sp)
        _copy_diagnostic_and_prognostics.with_backend(run_gtfn)(
            new_d.hdef_ic,
            diagnostic_state.hdef_ic,
            new_d.div_ic,
            diagnostic_state.div_ic,
            new_d.dwdx,
            diagnostic_state.dwdx,
            new_d.dwdy,
            diagnostic_state.dwdy,
            new_p.vn,
            prognostic_state.vn,
            new_p.w,
            prognostic_state.w,
            new_p.exner,
            prognostic_state.exner,
            new_p.theta_v,
            prognostic_state.theta_v,
            offset_provider={},
        )


class Timeloop:
    @classmethod
    def name(cls):
        return cls.__name__

    def __init__(
        self,
        config: IconRunConfig,
        diffusion: Diffusion,
        atmo_non_hydro: DummyAtmoNonHydro,
    ):
        self.config = config
        self.diffusion = diffusion
        self.atmo_non_hydro = atmo_non_hydro

    def _full_name(self, func: Callable):
        return ":".join((self.__class__.__name__, func.__name__))

    def _timestep(
        self,
        diagnostic_state: DiffusionDiagnosticState,
        prognostic_state: PrognosticState,
    ):
        self.atmo_non_hydro.do_dynamics_substepping(
            self.config.dtime, diagnostic_state, prognostic_state
        )
        self.diffusion.run(
            diagnostic_state,
            prognostic_state,
            self.config.dtime,
        )

    def __call__(
        self,
        diagnostic_state: DiffusionDiagnosticState,
        prognostic_state: PrognosticState,
    ):
        log.info(
            f"starting time loop for dtime={self.config.dtime} n_timesteps={self.config.n_time_steps}"
        )
        log.info("running initial step to diffuse fields before timeloop starts")
        self.diffusion.initial_run(
            diagnostic_state,
            prognostic_state,
            self.config.dtime,
        )
        log.info(
            f"starting real time loop for dtime={self.config.dtime} n_timesteps={self.config.n_time_steps}"
        )
        timer = Timer(self._full_name(self._timestep))
        for t in range(self.config.n_time_steps):
            log.info(f"run timestep : {t}")
            timer.start()
            self._timestep(diagnostic_state, prognostic_state)
            timer.capture()
        timer.summary(True)


def initialize(n_time_steps, file_path: Path, props: ProcessProperties):
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
    start_time = datetime.now().astimezone(pytz.UTC)
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
