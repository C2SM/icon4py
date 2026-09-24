# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
from icon4py.model.common.config import config_io


@pytest.mark.parametrize("turb_prandtl", [0.0, -1.0])
def test_config_rejects_non_positive_turb_prandtl(turb_prandtl: float) -> None:
    with pytest.raises(ValueError, match="turb_prandtl"):
        tmx_config.TmxConfig(turb_prandtl=turb_prandtl)


def test_config_rejects_negative_km_min() -> None:
    with pytest.raises(ValueError, match="km_min"):
        tmx_config.TmxConfig(km_min=-1.0)


def test_config_coerces_enums_from_ints() -> None:
    config = tmx_config.TmxConfig(solver_type=1, energy_type=1)
    assert config.solver_type is tmx_config.SolverType.EXPLICIT
    assert config.energy_type is tmx_config.EnergyType.DRY_STATIC


def test_config_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValueError):
        tmx_config.TmxConfig(solver_type=3)


def test_config_round_trips_through_config_io() -> None:
    """Every enum option is registered, so the config survives (un)structuring."""
    config = tmx_config.TmxConfig()
    unstructured = config_io.CONV.unstructure(config)

    assert unstructured["solver_type"] == "implicit"
    assert unstructured["energy_type"] == "internal"
    assert config_io.CONV.structure(unstructured, tmx_config.TmxConfig) == config
