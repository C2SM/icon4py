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
from icon4py.model.driver.config import configuration_manager


@pytest.fixture(scope="module")
def config() -> configuration_manager.ConfigurationManager:
    path = pathlib.Path(__file__).parent.joinpath("model.yaml")
    manager = configuration_manager.ConfigurationManager(path)
    manager.read_config()
    return manager


def test_configuration_manager_access_component_configs(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert len(config.get_configured_modules()) == 2
    assert "diffusion" in config.get_configured_modules()
    assert config.config.diffusion.temperature_boundary_diffusion_denom == 45.0
    assert config.config.diffusion.type_vn_diffu == 1
    assert config.config.dycore.ndyn_substep == 4
    assert config.config.dycore.divdamp_type == dycore_states.DivergenceDampingType.COMBINED


@pytest.mark.skip
# TODO (halungge): not sure that should even be possible
def test_configuration_manager_access_components_default(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.default.dycore.ndyn_substep == 5
    assert (
        config.default.dycore.divdamp_type == dycore_states.DivergenceDampingType.THREE_DIMENSIONAL
    )


def test_configuration_manager_resolve_interpolations(
    config: configuration_manager.ConfigurationManager,
) -> None:
    assert config.config.model.nsubsteps == 4
    assert config.config.dycore.ndyn_substep == 4
    assert config.config.diffusion.n_substeps == 4
