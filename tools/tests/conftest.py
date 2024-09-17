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
    istep_init,
    istep_exit,
    jstep_init,
    jstep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    data_provider,
    download_ser_data,
    processor_props,
    ranked_data_path,
    icon_grid,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    linit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
)


@pytest.fixture
def cli():
    yield CliRunner()
    os.environ["FLOAT_PRECISION"] = type_alias.DEFAULT_PRECISION
    reload(type_alias)


@pytest.fixture
def test_temp_dir():
    return os.getenv('CI_PROJECT_PATH', None)
