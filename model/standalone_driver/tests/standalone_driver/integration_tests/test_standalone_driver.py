# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import shutil

import pytest

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import run_driver
from icon4py.model.testing import definitions, grid_utils

from ..fixtures import *  # noqa: F403


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [definitions.Experiments.JW])
def test_standalone_driver(
    experiment: definitions.Experiments,
    backend_like,
):
    """
    Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """

    backend_name = None
    for k, v in model_backends.USER_BACKEND.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(experiment.grid)

    output_path = f"./ci_driver_output_for_backend_{backend_name}"
    run_driver.run_icon4py_driver(
        configuration_file_path="./",
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=output_path,
    )

    shutil.rmtree(pathlib.Path(output_path))
