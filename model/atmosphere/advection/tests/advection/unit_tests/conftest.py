# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing.fixtures.datatest import (  # noqa: F401  # import fixtures from test_utils package
    damping_height,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    linit,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    step_date_exit,
    step_date_init,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
)


@pytest.fixture
def advection_init_savepoint(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    date,
):
    """
    Load data from advection init ICON savepoint.

    Date of the timestamp to be selected can be set seperately by overriding the 'date'
    fixture, passing 'date=<iso_string>'.
    """
    return data_provider.from_advection_init_savepoint(size=data_provider.grid_size, date=date)


@pytest.fixture
def advection_exit_savepoint(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    date,
):
    """
    Load data from advection exit ICON savepoint.

    Date of the timestamp to be selected can be set seperately by overriding the 'date'
    fixture, passing 'date=<iso_string>'.
    """
    return data_provider.from_advection_exit_savepoint(size=data_provider.grid_size, date=date)
