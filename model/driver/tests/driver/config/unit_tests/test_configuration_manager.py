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
from icon4py.model.common.config import config as common_config, configuration_manager


@pytest.fixture(scope="module")
def config() -> configuration_manager.ConfigurationManager:
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager.read_config()
    return manager


def test_configuration_manager_model_config(
    config: configuration_manager.ConfigurationManager,
) -> None:
    model_config = config.get().model
    assert model_config.ndyn_substeps == 4
    assert model_config.vertical.SLEVE_minimum_layer_thickness_2 == 500.0
    assert model_config.vertical.htop_moist_proc == 3400.0
    assert model_config.vertical.num_levels == 80


def test_configuration_manager_run_config(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.get().run.dtime == 10
    assert config.get().run.start_date == "2021-06-20T12:00:10.000"
    assert config.get().run.output_path == pathlib.Path("model_out")


def test_configuration_manager_access_component_configs(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert len(config.get_configured_modules()) == 3
    assert "diffusion" in config.get_configured_modules()
    assert config.get().diffusion.temperature_boundary_diffusion_denominator == 45.0
    assert config.get().diffusion.type_vn_diffu == 1
    assert config.get().dycore.ndyn_substeps == 4
    assert config.get().dycore.divdamp_type == dycore_states.DivergenceDampingType.COMBINED


def test_configuration_manager_access_components_default(
    config: configuration_manager.ConfigurationManager,
) -> None:
    default_config = config.get(is_default=True)
    assert default_config.dycore.ndyn_substeps == 5
    assert (
        default_config.dycore.divdamp_type == dycore_states.DivergenceDampingType.THREE_DIMENSIONAL
    )


def test_configuration_manager_resolve_interpolations(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.get().model.ndyn_substeps == 4
    assert config.get().dycore.ndyn_substeps == 4
    assert config.get().diffusion.ndyn_substeps == 4
