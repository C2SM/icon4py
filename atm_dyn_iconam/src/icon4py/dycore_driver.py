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

from pathlib import Path

from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.diffusion.icon_grid import VerticalModelParams
from icon4py.diffusion.interpolation_state import InterpolationState
from icon4py.diffusion.metric_state import MetricState
from icon4py.icon_configuration import IconRunConfig, read_config
from icon4py.io_utils import read_icon_grid, read_geometry_fields, read_static_fields

data_path = Path(__file__).parent.joinpath("ser_icondata")
extracted_path = data_path.joinpath("mch_ch_r04b09_dsl/ser_data")


class AtmoNonHydro:
    def __init__(self):
        self.config = None

    def init(self, config):
        self.config = config

    def _dynamics_timestep(self):
        pass

    def do_dynamics_substepping(self):
        for i in range(self.config.n_substeps):
            self._dynamics_timestep()


diffusion: Diffusion()
atmo_non_hydro: AtmoNonHydro()


def timestep(dtime: float):

    diffusion.initial_step(
        diagnostic_state,
        prognostic_state,
        dtime,
        tangent_orientation,
        inverse_primal_edge_lengths,
        inverse_dual_edge_length,
        inverse_vert_vert_lengths,
        primal_normal_vert,
        dual_normal_vert,
        edge_areas,
        cell_areas,
    )
    atmo_non_hydro.do_dynamics_substepping()
    diffusion.time_step()


def timeloop(run_config: IconRunConfig):
    """Runs the loop."""
    for t in range(run_config.n_time_steps):
        timestep(run_config.dtime)


def initialize_model(gridfile_path: str):
    config = read_config("mch_ch_r04b09")
    icon_grid = read_icon_grid(gridfile_path)

    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(gridfile_path)

    (metric_state, interpolation_state) = read_static_fields(gridfile_path)

    diffusion_params = DiffusionParams(config.diffusion_config)
    diffusion.init(
        icon_grid,
        config.diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
    )
    atmo_non_hydro.init(config=config.dycore_config)

    return config


def run():
    """
    "Runs the driver."
    steps:
    1. initialize model
        a) read config
        b) initialize grid
        c) initialize/configure components ie "granules"
    2. run time loop loop
        run timeloop
    3. collect output
    """
    config = initialize_model(str(extracted_path))
    timeloop(config.run_config)
