# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from icon4py.model.driver.config import configuration_manager


def test_configuration_manager_happy_path():
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager.read_config()
    assert len(manager.get_configured_modules()) == 2
    assert "diffusion" in manager.get_configured_modules()
    assert manager.config.diffusion.temperature_boundary_diffusion_denom == 45.0
