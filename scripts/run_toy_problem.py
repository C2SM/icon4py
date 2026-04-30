# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
from typing import Annotated

import numpy as np
import typer

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    saturation_adjustment,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.states import prognostic_state
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_utils, standalone_driver as driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import definitions as test_defs, grid_utils
from model.atmosphere.subgrid_scale_physics.muphys.tests.muphys.stencil_tests.test_saturation_adjustment import (
    saturation_adjustment_numpy,
)


log = logging.getLogger(__file__)


def toy_problem( icon4py_backend: Annotated[
        str,
        typer.Option(
            help=f"Backend for running the driver. Possible options are: {' / '.join([*model_backends.BACKENDS.keys()])}",
        ),
    ],
) -> None:
    # Select grid file and download if necessary
    grid_definition = test_defs.Grids.R02B04_GLOBAL
    grid_file_path = grid_utils._download_grid_file(grid_definition)

    # Initialize driver
    icon4py_driver = driver.initialize_driver(
        grid_file_path=grid_file_path,
        log_level="warning",
        backend_like=driver_utils.get_backend_from_name(icon4py_backend),
        output_path=pathlib.Path("./output"),
    )

    # Setup initial condition
    states = initial_condition.toy_problem(
        grid=icon4py_driver.grid,
        geometry_field_source=icon4py_driver.static_field_factories.geometry_field_source,
        backend=icon4py_driver.backend,
    )


    cell_domain = h_grid.domain(dims.CellDim)
    local_cell_end = icon4py_driver.grid.end_index(cell_domain(h_grid.Zone.END))

    # Numpy
    np_temperature, np_qv, np_qc = saturation_adjustment_numpy(
        te=states.diagnostic.temperature,
        rho=states.prognostics.current.rho,
        q_in=Q(*states.prognostics.current.tracer),
    )

    # GT4Py
    gt4py_temperature = data_alloc.as_field(states.diagnostic.temperature)
    gt4py_qv = data_alloc.as_field(states.prognostics.current.tracer[prognostic_state.QV])
    gt4py_qc = data_alloc.as_field(states.prognostics.current.tracer[prognostic_state.QC])
    #
    saturation_adjustment.with_backend(icon4py_driver.backend)(
        te=states.diagnostic.temperature,
        rho=states.prognostics.current.rho,
        q_in=Q(*states.prognostics.current.tracer),
        te_out=gt4py_temperature,
        qve_out=gt4py_qv,
        qce_out=gt4py_qc,
        horizontal_start=0,
        horizontal_end=local_cell_end,
        vertical_start=0,
        vertical_end=icon4py_driver.grid.num_levels,
    )



if __name__ == "__main__":
    typer.run(toy_problem)
