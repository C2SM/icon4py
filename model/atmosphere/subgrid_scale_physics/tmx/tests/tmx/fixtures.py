# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import typing

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import TmxConfig
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import fortran_config
from icon4py.model.testing import datatest_utils as dt_utils, definitions
from icon4py.model.testing.fixtures.datatest import (
    download_ser_data,
    experiment_description,
    process_props,
)


def load_fortran_dict(
    *,
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    fname: str,
) -> dict[str, typing.Any]:
    """Load one of the converted namelist dicts of an experiment."""
    experiment_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    with (experiment_path / fname).open() as f:
        return json.load(f)


@pytest.fixture
def tmx_config(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    download_ser_data: None,  # downloads data as side-effect
) -> TmxConfig:
    """TmxConfig read from the experiment's converted (echoed) namelists."""
    atm_dict = load_fortran_dict(
        experiment_description=experiment_description,
        process_props=process_props,
        fname=fortran_config.ATM_DICT_FNAME,
    )
    return TmxConfig.from_fortran_dict(atm_dict=atm_dict)
