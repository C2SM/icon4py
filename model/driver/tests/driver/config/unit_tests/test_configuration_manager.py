# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.config import configuration_manager


@pytest.fixture(scope="module")
def config() -> configuration_manager.ConfigurationManager:
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager.read_config()
    return manager


def test_configuration_manager_model_config(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.config.model.ndyn_substeps == 4
    assert config.config.model.vertical.SLEVE_minimum_layer_thickness_2 == 500.0
    assert config.config.model.vertical.htop_moist_proc == 3400.0
    assert config.config.model.vertical.num_levels == 80


def test_configuration_manager_run_config(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.config.run.dtime == 10
    assert config.config.run.start_date == "2021-06-20T12:00:10.000"
    assert config.config.run.output_path == pathlib.Path("model_out")


def test_configuration_manager_access_component_configs(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert len(config.get_configured_modules()) == 3
    assert "diffusion" in config.get_configured_modules()
    assert config.config.diffusion.temperature_boundary_diffusion_denominator == 45.0
    assert config.config.diffusion.type_vn_diffu == 1
    assert config.config.dycore.ndyn_substeps == 4
    assert config.config.dycore.divdamp_type == dycore_states.DivergenceDampingType.COMBINED


@pytest.mark.xfail
def test_configuration_manager_access_components_default(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.default.dycore.ndyn_substeps == 5
    assert (
        config.default.dycore.divdamp_type == dycore_states.DivergenceDampingType.THREE_DIMENSIONAL
    )


def test_configuration_manager_resolve_interpolations(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.config.model.ndyn_substeps == 4
    assert config.config.dycore.ndyn_substeps == 4
    assert config.config.diffusion.ndyn_substeps == 4
