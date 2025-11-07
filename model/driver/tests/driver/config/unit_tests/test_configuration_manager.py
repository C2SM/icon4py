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
from icon4py.model.testing import test_utils


@pytest.fixture(scope="module")
def config_manager() -> configuration_manager.ConfigurationManager:
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager()
    return manager


def test_configuration_manager_model_config(
    config_manager: configuration_manager.ConfigurationManager,
) -> None:
    model_config = config_manager.get().model
    assert model_config.ndyn_substeps == 4
    assert model_config.vertical.SLEVE_minimum_layer_thickness_2 == 500.0
    assert model_config.vertical.htop_moist_proc == 3455.0
    assert model_config.vertical.num_levels == 42


def test_configuration_manager_run_config(
    config_manager: configuration_manager.ConfigurationManager,
) -> None:
    run_config = config_manager.get().run
    assert run_config.dtime == 10
    assert run_config.start_date == "2021-06-20T12:00:10.000"
    assert run_config.output_path == pathlib.Path("model_out")


def test_configuration_manager_access_component_configs(
    config_manager: configuration_manager.ConfigurationManager,
) -> None:
    assert len(config_manager.get_configured_modules()) == 3
    assert "diffusion" in config_manager.get_configured_modules()
    assert config_manager.get().diffusion.temperature_boundary_diffusion_denominator == 45.0
    assert config_manager.get().diffusion.type_vn_diffu == 1
    assert config_manager.get().dycore.ndyn_substeps == 4
    assert config_manager.get().dycore.divdamp_type == dycore_states.DivergenceDampingType.COMBINED


def test_configuration_manager_access_components_default(
    config_manager: configuration_manager.ConfigurationManager,
) -> None:
    default_config = config_manager.get(is_default=True)
    assert default_config.dycore.ndyn_substeps == 5
    assert (
        default_config.dycore.divdamp_type == dycore_states.DivergenceDampingType.THREE_DIMENSIONAL
    )


def test_configuration_manager_resolve_interpolations(
    config_manager: configuration_manager.ConfigurationManager,
) -> None:
    assert config_manager.get().model.ndyn_substeps == 4
    assert config_manager.get().dycore.ndyn_substeps == 4
    assert config_manager.get().diffusion.ndyn_substeps == 4


def test_configuration_manager_to_yaml(
    config_manager: configuration_manager.ConfigurationManager, tmp_path: pathlib.Path
):
    file = tmp_path.joinpath("model_dump.yaml")
    config_manager.to_yaml(file)
    reference_file = pathlib.Path(__file__).parent.joinpath("references/full_model_config.yaml")
    assert test_utils.diff(reference_file, file)
