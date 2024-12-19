# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

import click.testing as click_testing
import pytest

import icon4py.model.common.type_alias as type_alias
from icon4py.model.testing.datatest_fixtures import (  # F401
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
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_exit,
    step_date_exit,
    step_date_init,
    stretch_factor,
)


# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_config import *  # noqa: F403


__all__ = [
    # local:
    "cli",
    "test_temp_dir",
    # imported fixtures:
    "damping_height",
    "data_provider",
    "download_ser_data",
    "experiment",
    "grid_savepoint",
    "icon_grid",
    "interpolation_savepoint",
    "istep_exit",
    "istep_init",
    "jstep_exit",
    "jstep_init",
    "linit",
    "lowest_layer_thickness",
    "metrics_savepoint",
    "model_top_height",
    "ndyn_substeps",
    "processor_props",
    "ranked_data_path",
    "savepoint_diffusion_exit",
    "savepoint_diffusion_init",
    "savepoint_nonhydro_exit",
    "savepoint_nonhydro_init",
    "savepoint_nonhydro_step_exit",
    "step_date_exit",
    "step_date_init",
    "stretch_factor",
]


@pytest.fixture
def cli():
    yield click_testing.CliRunner()
    os.environ["FLOAT_PRECISION"] = type_alias.DEFAULT_PRECISION
    type_alias.set_precision(type_alias.DEFAULT_PRECISION)


@pytest.fixture
def test_temp_dir():
    return os.getenv("TEST_TEMP_DIR", None)
