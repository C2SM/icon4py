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
    initialization_utils as driver_init,
    standalone_driver as driver,
)
from icon4py.model.testing import datatest_utils as dt_utils, definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures.datatest import backend


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_standalone_driver(
    experiment,
    experiment_type,
    *,
    data_provider,
    ranked_data_path,
    backend,
):
    """
    Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """
    data_path = dt_utils.get_datapath_for_experiment(
        ranked_base_path=ranked_data_path,
        experiment=experiment,
    )
    gm = grid_utils.get_grid_manager_from_experiment(
        experiment=experiment,
        keep_skip_values=True,
        backend=backend,
    )

    backend_name = None
    for key, value in model_backends.BACKENDS.items():
        if value == backend:
            backend_name = key

    assert backend_name is not None

    driver.run_icon4py_driver(
        [
            str(data_path),
            "--experiment_type",
            experiment_type,
            "--grid_file",
            str(gm._file_name),
            "--icon4py_driver_backend",
            backend_name,
        ],
        standalone_mode=False,
    )
