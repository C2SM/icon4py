# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from icon4py.model.common.config import config_io
from icon4py.model.driver import config as driver_config
from icon4py.model.testing import config as test_config


@pytest.fixture
def experiment_config(
    experiment_case: str, tmp_path: pathlib.Path
) -> driver_config.ExperimentConfig:
    config_path = test_config.EXPERIMENT_CONFIG_PATH / f"{experiment_case}.yaml"
    return config_io.read_yaml_str(
        config_path.read_text(), driver_config.ExperimentConfig
    ).with_overrides(driver={"output_path": tmp_path / "ci_driver_output"})
