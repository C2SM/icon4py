# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from icon4py.model.testing.fixtures import (
    damping_height,
    data_provider,
    download_ser_data,
    experiment,
    flat_height,
    grid_savepoint,
    htop_moist_proc,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
)


@pytest.fixture
def timeloop_diffusion_savepoint_exit(
    data_provider,  # imported fixtures data_provider`
    step_date_exit,  # imported fixtures step_date_exit`
    timeloop_diffusion_linit_exit,
):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'timeloop_date'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(
        linit=timeloop_diffusion_linit_exit, date=step_date_exit
    )
    return sp
