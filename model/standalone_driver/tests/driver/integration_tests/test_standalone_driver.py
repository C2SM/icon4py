# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import typer

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import (
    driver_configuration as driver_configure,
    driver_utils as driver_init,
    standalone_driver as standalone_driver,
)
from icon4py.model.testing import datatest_utils as dt_utils, definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures.datatest import backend


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", definitions.Experiments.JW)
def test_standalone_driver(
    experiment,
    backend,
):
    """
    Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """

    backend_name = None
    for key, value in model_backends.BACKENDS.items():
        if value == backend:
            backend_name = key

    assert backend_name is not None

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    standalone_driver.run_icon4py_driver(
        configuration_file_path="./",
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path="./",
    )
