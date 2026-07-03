# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import fortran_config
from icon4py.model.testing import datatest_utils as dt_utils, definitions
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
    step_date_exit,
    step_date_init,
)


@pytest.fixture
def tmx_config(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    download_ser_data: None,  # downloads data as side-effect
) -> tmx.TmxConfig:
    """TmxConfig read from the experiment's converted input namelists."""
    experiment_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    with (experiment_path / fortran_config.INPUT_DICT_FNAME).open() as f:
        input_dict = json.load(f)
    return tmx.TmxConfig.from_fortran_dict(input_dict)
