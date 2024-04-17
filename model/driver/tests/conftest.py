# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import random
from datetime import datetime
from pathlib import Path

import pytest

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401
    damping_height,
    data_provider,
    datapath,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    istep_exit,
    istep_init,
    jstep_exit,
    jstep_init,
    metrics_savepoint,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_exit,
    savepoint_velocity_init,
    step_date_exit,
    step_date_init,
    vn_only,
)
from icon4py.model.driver.icon_configuration import IconRunConfig


# TODO (Chia Rui): Reuse those pytest fixtures for diffusion test instead of creating here


@pytest.fixture
def r04b09_diffusion_config(
    ndyn_substeps,  # noqa: F811 # imported `ndyn_substeps` fixture
) -> DiffusionConfig:
    """
    Create DiffusionConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return DiffusionConfig(
        diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
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
    ndyn_substeps,  # noqa: F811 # imported `ndyn_substeps` fixture
    timeloop_date_init,
    timeloop_date_exit,
    timeloop_diffusion_linit_init,
) -> IconRunConfig:
    """
    Create IconRunConfig matching MCH_CH_r04b09_dsl.

    Set values to the ones used in the  MCH_CH_r04b09_dsl experiment where they differ
    from the default.
    """
    return IconRunConfig(
        dtime=10.0,
        start_date=datetime(
            int(timeloop_date_init[0:4]),
            int(timeloop_date_init[5:7]),
            int(timeloop_date_init[8:10]),
            int(timeloop_date_init[11:13]),
            int(timeloop_date_init[14:16]),
            int(timeloop_date_init[17:19]),
        ),
        end_date=datetime(
            int(timeloop_date_exit[0:4]),
            int(timeloop_date_exit[5:7]),
            int(timeloop_date_exit[8:10]),
            int(timeloop_date_exit[11:13]),
            int(timeloop_date_exit[14:16]),
            int(timeloop_date_exit[17:19]),
        ),
        n_substeps=ndyn_substeps,
        apply_initial_stabilization=timeloop_diffusion_linit_init,
    )


@pytest.fixture
def timeloop_diffusion_savepoint_init(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    step_date_init,  # noqa: F811 # imported fixtures data_provider
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
    data_provider,  # noqa: F811 # imported fixtures data_provider`
    step_date_exit,  # noqa: F811 # imported fixtures step_date_exit`
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


@pytest.fixture
def random_name():
    return "test" + str(random.randint(0, 100000))


def delete_recursive(p: Path):
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
        else:
            delete_recursive(child)
    p.rmdir()


@pytest.fixture
def test_path(tmp_path):
    base_path = tmp_path.joinpath("io_tests")
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)
    yield base_path
    delete_recursive(base_path)
