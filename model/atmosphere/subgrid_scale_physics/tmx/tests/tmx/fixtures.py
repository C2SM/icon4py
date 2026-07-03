# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import json
import re
import typing

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


def load_fortran_dict(
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
) -> tmx.TmxConfig:
    """TmxConfig read from the experiment's converted input namelists."""
    input_dict = load_fortran_dict(
        experiment_description, process_props, fortran_config.INPUT_DICT_FNAME
    )
    return tmx.TmxConfig.from_fortran_dict(input_dict)


#: ISO 8601 duration, fixed-length components only (duplicated from
#: standalone_driver.config, which the tmx tests may not depend on)
_ISO8601_DURATION = re.compile(
    r"P(?:(?P<weeks>\d+)W)?(?:(?P<days>\d+)D)?"
    r"(?:T(?=\d)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
)

_SECONDS_PER = {
    "weeks": 604800.0,
    "days": 86400.0,
    "hours": 3600.0,
    "minutes": 60.0,
    "seconds": 1.0,
}


@pytest.fixture
def tmx_dtime(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
    download_ser_data: None,  # downloads data as side-effect
) -> float:
    """The tmx timestep [s], derived from the input namelists like in ICON.

    The Fortran passes ``dt_vdf`` of ``aes_phy_config`` as the tmx timestep;
    it must be explicitly set (as an ISO 8601 duration) for any run with
    active turbulent mixing, so it is always present in the input namelists.
    """
    input_dict = load_fortran_dict(
        experiment_description, process_props, fortran_config.INPUT_DICT_FNAME
    )
    dt_vdf = input_dict["aes_phy_nml"]["aes_phy_config"][0]["dt_vdf"]
    match = _ISO8601_DURATION.fullmatch(dt_vdf)
    if match is None or not any(match.groups()):
        raise ValueError(f"Invalid ISO 8601 duration: '{dt_vdf}'.")
    return sum(
        _SECONDS_PER[name] * float(value) for name, value in match.groupdict().items() if value
    )
