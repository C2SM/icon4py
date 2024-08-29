# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "fortran_samples"

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
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
    data_provider,
    download_ser_data,
    processor_props,
    ranked_data_path,
    icon_grid
)
