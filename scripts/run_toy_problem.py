# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
from typing import Annotated, Any

import numpy as np
import typer

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    saturation_adjustment,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment_numpy import (
    saturation_adjustment_numpy,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.states import prognostic_state
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_utils, standalone_driver as driver
from icon4py.model.standalone_driver.testcases import initial_condition
from icon4py.model.testing import config, definitions as test_defs, grid_utils


config.DALLCLOSE_PRINT_INSTEAD_OF_FAIL = True
config.DRIVER_LOGGING_LEVEL = "warning"

log = logging.getLogger(__file__)


def toy_problem(
    icon4py_backend: Annotated[
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

    fields = [
        {
            "name": "qc",
            "field_ref": np_qc,
            "field_cmp": data_alloc.as_numpy(gt4py_qc),
        },
        {
            "name": "qv",
            "field_ref": np_qv,
            "field_cmp": data_alloc.as_numpy(gt4py_qv),
        },
        {
            "name": "temperature",
            "field_ref": np_temperature,
            "field_cmp": data_alloc.as_numpy(gt4py_temperature),
        },
    ]
    for field in fields:
        verify_field(
            field=field,
            process_props=icon4py_driver.process_props,
        )


def verify_field(
    field: dict[str, Any],
    process_props: decomp_defs.ProcessProperties,
) -> None:
    field_name = field["name"]
    field_ref = field["field_ref"]
    field_cmp = field["field_cmp"]

    max_diff = np.max(np.abs(field_ref - field_cmp))

    color = "\033[1;31m" if max_diff > 0 else "\033[32m"
    log.info(f"\nverifying {field_name}")
    log.info(f"\tfield_ref.shape {field_ref.shape}, field_cmp.shape {field_cmp.shape}")

    if process_props.comm_size > 1:
        # barrier to ensure that all ranks have printed
        process_props.comm.barrier()
    print(  # print instead of log so all ranks print to stdout
        f"{color}\trank {process_props.rank}/{process_props.comm_size}: max diff {max_diff}\033[0m"
    )
    if process_props.comm_size > 1:
        # barrier to ensure that all ranks have printed
        process_props.comm.barrier()


if __name__ == "__main__":
    typer.run(toy_problem)
