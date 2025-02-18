# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime, timedelta

import pytest


# Make sure custom icon4py pytest hooks are loaded
try:
    import sys

    _ = sys.modules["icon4py.model.testing.pytest_config"]
except KeyError:
    from icon4py.model.testing.pytest_config import *  # noqa: F403 [undefined-local-with-import-star]


from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.driver import icon4py_configuration as driver_config
from icon4py.model.testing.datatest_fixtures import (
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
    jstep_exit,
    jstep_init,
    lowest_layer_thickness,
    maximal_layer_thickness,
    metrics_savepoint,
    model_top_height,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_exit,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
    stretch_factor,
    top_height_limit_for_maximal_layer_thickness,
    vn_only,
)


__all__ = [
    # local:
    "r04b09_diffusion_config",
    "r04b09_iconrun_config",
    "timeloop_diffusion_savepoint_init",
    "timeloop_diffusion_savepoint_exit",
    "timeloop_date_init",
    "timeloop_date_exit",
    # imported fixtures:
    "damping_height",
    "data_provider",
    "download_ser_data",
    "experiment",
    "flat_height",
    "grid_savepoint",
    "htop_moist_proc",
    "icon_grid",
    "interpolation_savepoint",
    "istep_exit",
    "istep_init",
    "jstep_exit",
    "jstep_init",
    "lowest_layer_thickness",
    "maximal_layer_thickness",
    "metrics_savepoint",
    "model_top_height",
    "ndyn_substeps",
    "processor_props",
    "ranked_data_path",
    "savepoint_nonhydro_exit",
    "savepoint_nonhydro_init",
    "savepoint_nonhydro_step_exit",
    "savepoint_velocity_init",
    "step_date_exit",
    "step_date_init",
    "stretch_factor",
    "top_height_limit_for_maximal_layer_thickness",
    "vn_only",
]


# TODO (Chia Rui): Reuse those pytest fixtures for diffusion test instead of creating here
@pytest.fixture
def r04b09_diffusion_config(ndyn_substeps) -> diffusion.DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        hdiff_w=True,
        hdiff_vn=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        zdiffu_t=True,
        velocity_boundary_diffusion_denom=150.0,
        max_nudging_coeff=0.075,
        n_substeps=ndyn_substeps,
    )


@pytest.fixture
def r04b09_iconrun_config(
    ndyn_substeps,
    timeloop_date_init,
    timeloop_date_exit,
    timeloop_diffusion_linit_init,
) -> driver_config.Icon4pyRunConfig:
    """
    Create Icon4pyRunConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return driver_config.Icon4pyRunConfig(
        dtime=timedelta(seconds=10.0),
        start_date=datetime.fromisoformat(timeloop_date_init),
        end_date=datetime.fromisoformat(timeloop_date_exit),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=timeloop_diffusion_linit_init,
    )


@pytest.fixture
def timeloop_diffusion_savepoint_init(
    data_provider,  # imported fixtures data_provider
    step_date_init,  # imported fixtures data_provider
    timeloop_diffusion_linit_init,
):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'timeloop_date_'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag is set to true
    """
    return data_provider.from_savepoint_diffusion_init(
        linit=timeloop_diffusion_linit_init, date=step_date_init
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


@pytest.fixture
def timeloop_date_init():
    return "2021-06-20T12:00:00.000"


@pytest.fixture
def timeloop_date_exit():
    return "2021-06-20T12:00:10.000"
