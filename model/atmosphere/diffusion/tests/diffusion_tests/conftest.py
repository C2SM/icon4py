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

import pytest

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    damping_height,
    data_provider,
    datapath,
    decomposition_info,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    linit,
    metrics_savepoint,
    ndyn_substeps,
    processor_props,
    ranked_data_path,
    step_date_exit,
    step_date_init,
)


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
def diffusion_savepoint_init(
    data_provider,  # noqa: F811 # imported fixtures data_provider
    linit,  # noqa: F811 # imported fixtures linit
    step_date_init,  # noqa: F811 # imported fixtures data_provider
):
    """
    Load data from ICON savepoint at start of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_date_init'
    fixture, passing 'step_date_init=<iso_string>'

    linit flag can be set by overriding the 'linit' fixture
    """
    return data_provider.from_savepoint_diffusion_init(linit=linit, date=step_date_init)


@pytest.fixture
def diffusion_savepoint_exit(
    data_provider,  # noqa: F811 # imported fixtures data_provider`
    linit,  # noqa: F811 # imported fixtures linit`
    step_date_exit,  # noqa: F811 # imported fixtures step_date_exit`
):
    """
    Load data from ICON savepoint at exist of diffusion module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    sp = data_provider.from_savepoint_diffusion_exit(linit=linit, date=step_date_exit)
    return sp
