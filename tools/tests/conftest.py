# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from importlib import reload

import icon4py.model.common.type_alias as type_alias
import pytest
from click.testing import CliRunner
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa F401
    damping_height,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    jstep_exit,
    jstep_init,
    linit,
    lowest_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_diffusion_exit,
    savepoint_diffusion_init,
    savepoint_nonhydro_init,
    step_date_exit,
    step_date_init,
    stretch_factor,
)


@pytest.fixture
def cli():
    yield CliRunner()
    os.environ["FLOAT_PRECISION"] = type_alias.DEFAULT_PRECISION
    reload(type_alias)


@pytest.fixture
def test_temp_dir():
    return os.getenv("TEST_TEMP_DIR", None)
