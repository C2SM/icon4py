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

import click

from icon4py.diffusion.diagnostic_state import DiagnosticState
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.prognostic_state import PrognosticState
from icon4py.diffusion.utils import copy_diagnostic_and_prognostics
from icon4py.icon_configuration import IconRunConfig, read_config
from icon4py.io_utils import (
    SIMULATION_START_DATE,
    read_geometry_fields,
    read_icon_grid,
    read_initial_state,
    read_static_fields,
)
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


class DummyAtmoNonHydro:
    def __init__(self, data_provider: IconSerialDataProvider):
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
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):
        for _ in range(self.config.n_substeps):
            self._dynamics_timestep(dtime)
        sp = self.data_provider.from_savepoint_diffusion_init(
            linit=False, date=self.simulation_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        )
        new_p = sp.construct_prognostics()
        new_d = sp.construct_diagnostics()
        copy_diagnostic_and_prognostics(
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
            new_p.exner_pressure,
            prognostic_state.exner_pressure,
            new_p.theta_v,
            prognostic_state.theta_v,
            offset_provider={},
        )


def initialize(n_time_steps, file_path: str = "."):
    config = read_config("mch_ch_r04b09", n_time_steps=n_time_steps)
    icon_grid = read_icon_grid(file_path)

    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(file_path)

    (metric_state, interpolation_state) = read_static_fields(file_path)

    diffusion_params = DiffusionParams(config.diffusion_config)
    diffusion = Diffusion()
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
    )

    data_provider, diagnostic_state, prognostic_state = read_initial_state(file_path)

    atmo_non_hydro = DummyAtmoNonHydro(data_provider)
    atmo_non_hydro.init(config=config.dycore_config)

    tl = Timeloop(
        config=config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
    )
    return tl, diagnostic_state, prognostic_state


class Timeloop:
    def __init__(
        self,
        config: IconRunConfig,
        diffusion: Diffusion,
        atmo_non_hydro: DummyAtmoNonHydro,
        edge_geometry: EdgeParams,
        cell_geometry: CellParams,
    ):
        self.config = config
        self.diffusion = diffusion
        self.atmo_non_hydro = atmo_non_hydro
        self.edges = edge_geometry
        self.cells = cell_geometry
        self.log = logging.getLogger(__name__)

    def _timestep(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):

        self.atmo_non_hydro.do_dynamics_substepping(
            self.config.dtime, diagnostic_state, prognostic_state
        )
        self.diffusion.time_step(
            diagnostic_state,
            prognostic_state,
            self.config.dtime,
            self.edges.tangent_orientation,
            self.edges.inverse_primal_edge_lengths,
            self.edges.inverse_dual_edge_lengths,
            self.edges.inverse_vertex_vertex_lengths,
            self.edges.primal_normal_vert,
            self.edges.dual_normal_vert,
            self.edges.edge_areas,
            self.cells.area,
        )

    def __call__(
        self,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):
        self.log.info(
            "starting time loop for dtime={dtime} n_timesteps={self.config.n_time_steps}"
        )

        self.diffusion.initial_step(
            diagnostic_state,
            prognostic_state,
            self.config.dtime,
            self.edges.tangent_orientation,
            self.edges.inverse_primal_edge_lengths,
            self.edges.inverse_dual_edge_lengths,
            self.edges.inverse_vertex_vertex_lengths,
            self.edges.primal_normal_vert,
            self.edges.dual_normal_vert,
            self.edges.edge_areas,
            self.cells.area,
        )
        for _ in range(self.config.n_time_steps):
            self._timestep(diagnostic_state, prognostic_state)


@click.command()
@click.argument("input_path")
@click.option("--n_steps", default=5, help="number of time steps to run")
def run(input_path, n_steps):
    """
    Run the driver.

    steps:
    1. initialize model:

        a) read config

        b) initialize grid

        c) initialize/configure components ie "granules"

        d) setup the time loop

    2. run time loop
    3. collect output (not implemented)
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime) - %(name) - %(level) :---- %(message)"
    )
    log = logging.getLogger(__name__)
    log.info("Starting ICON dycore run")
    log.info(f"input args: input_path={input_path}, n_time_steps={n_steps}")
    timeloop, diagnostic_state, prognostic_state = initialize(n_steps, input_path)
    log.info("dycore configured")

    timeloop(diagnostic_state, prognostic_state)
    log.info("ICON dycore:  DONE")


if __name__ == "__main__":
    run()
