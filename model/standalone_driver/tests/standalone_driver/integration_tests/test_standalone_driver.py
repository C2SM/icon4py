# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import pytest

from icon4py.model.common import model_backends
from icon4py.model.standalone_driver import main
from icon4py.model.testing import definitions, grid_utils
from icon4py.model.testing.fixtures.datatest import backend_like


@pytest.mark.embedded_remap_error
def test_standalone_driver(
    backend_like,
    tmp_path: pathlib.Path,
):
    """
    Currently, this is a only test to check if the driver runs from a grid file without verifying the end result.
    TODO(anyone): Modify this test for scientific validation after IO is ready.
    """

    backend_name = "embedded"
    for k, v in model_backends.BACKENDS.items():
        if backend_like == v:
            backend_name = k

    grid_file_path = grid_utils._download_grid_file(definitions.Grids.R02B04_GLOBAL)

    output_path = tmp_path / f"ci_driver_output_for_backend_{backend_name}"
    main.main(
        configuration_file_path=pathlib.Path("./"),
        grid_file_path=grid_file_path,
        icon4py_backend=backend_name,
        output_path=output_path,
    )
