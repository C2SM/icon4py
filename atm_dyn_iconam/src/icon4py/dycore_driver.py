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
from datetime import datetime
from pathlib import Path

from icon4py.diffusion.diagnostic_state import DiagnosticState
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.prognostic_state import PrognosticState
from icon4py.diffusion.utils import copy_diagnostic_and_prognostics
from icon4py.icon_configuration import IconRunConfig, read_config
from icon4py.io_utils import (
    read_geometry_fields,
    read_icon_grid,
    read_static_fields,
)
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")


class DummyAtmoNonHydro:
    def __init__(self, data_provider: IconSerialDataProvider):
        self.config = None
        self.data_provider = data_provider
        self.simulation_date = datetime.fromisoformat("2021-06-20T12:00:10.00")

    def init(self, config):
        self.config = config

    def _next_physics_date(self, dtime: float):
        dynamics_dtime = dtime / self.config.n_substeps
        self.simulation_date += datetime.timedelta(seconds=dynamics_dtime)

    def _dynamics_timestep(self, dtime):
        """Show structure with this dummy fucntion called inside substepping loop."""
        self._next_physics_date(self, dtime)

    def do_dynamics_substepping(
        self,
        dtime,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):
        for _ in range(self.config.n_substeps):
            self._dynamics_timestep(dtime)
        sp = self.data_provider.from_savepoint_diffusion_init(
            linit=False, date=self.simulation_date.isoformat(sep="T")[:-3]
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


def configure_model(gridfile_path: str):
    config = read_config("mch_ch_r04b09")
    icon_grid = read_icon_grid(gridfile_path)

    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        gridfile_path
    )

    (metric_state, interpolation_state) = read_static_fields(gridfile_path)

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

    atmo_non_hydro = DummyAtmoNonHydro()
    atmo_non_hydro.init(config=config.dycore_config)
    tl = Timeloop(
        config.run_config,
        diffusion=diffusion,
        atmo_non_hydro=atmo_non_hydro,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
    )
    return tl


def run():
    """
    Run the driver.

    steps:
    1. initialize model
        a) read config
        b) initialize grid
        c) initialize/configure components ie "granules"
        d) setup the timestep
    2. run time loop loop
        run timeloop
    3. collect output
    """
    timeloop = configure_model(str(extracted_path))
    timeloop()


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

    def timestep(
        self,
        dtime: float,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):

        self.atmo_non_hydro.do_dynamics_substepping(diagnostic_state, prognostic_state)
        self.diffusion.time_step(
            diagnostic_state,
            prognostic_state,
            dtime,
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
        dtime,
        diagnostic_state: DiagnosticState,
        prognostic_state: PrognosticState,
    ):
        self.diffusion.initial_step(
            diagnostic_state,
            prognostic_state,
            dtime,
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
            self.timestep(dtime, diagnostic_state, prognostic_state)
