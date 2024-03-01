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

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.atmosphere.diffusion.test_utils.utils import (
    construct_config,
    construct_diagnostics,
    construct_interpolation_state,
    construct_metric_state,
)
from icon4py.model.common.decomposition.definitions import SingleNodeProcessProperties
from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.test_utils.datatest_utils import create_icon_serial_data_provider


def diffusion_test_case():
    # serialized data
    datapath = Path(
        "/home/sk/Dev/icon4py/testdata/ser_icondata/mpitask1/exclaim_ape_R02B04/ser_data"
    )
    processor_props = SingleNodeProcessProperties()
    data_provider = create_icon_serial_data_provider(datapath, processor_props)

    grid_savepoint = data_provider.from_savepoint_grid()
    interpolation_savepoint = data_provider.from_interpolation_savepoint()
    metrics_savepoint = data_provider.from_metrics_savepoint()
    diffusion_savepoint_init = data_provider.from_savepoint_diffusion_init(
        linit=False, date="2000-01-01T00:00:02.000"
    )

    icon_grid = grid_savepoint.construct_icon_grid(on_gpu=False)

    # constants
    dtime = 2.0
    damping_height = 50000
    experiment = "exclaim_ape_R02B04"
    ndyn_substeps = 2

    # input data
    edge_geometry: EdgeParams = grid_savepoint.construct_edge_geometry()
    cell_geometry: CellParams = grid_savepoint.construct_cell_geometry()
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    metric_state = construct_metric_state(metrics_savepoint)
    diagnostic_state = construct_diagnostics(diffusion_savepoint_init, grid_savepoint)
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    vertical_params = VerticalModelParams(
        vct_a=grid_savepoint.vct_a(),
        rayleigh_damping_height=damping_height,
        nflatlev=grid_savepoint.nflatlev(),
        nflat_gradp=grid_savepoint.nflat_gradp(),
    )
    config = construct_config(experiment, ndyn_substeps)
    additional_parameters = DiffusionParams(config)

    print("Initialising diffusion...")
    diffusion = Diffusion()
    diffusion.init(
        grid=icon_grid,
        config=config,
        params=additional_parameters,
        vertical_params=vertical_params,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print("Running diffusion...")
    diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )
    print("Successfully ran diffusion.")
    print("passed")


if __name__ == "__main__":
    diffusion_test_case()
