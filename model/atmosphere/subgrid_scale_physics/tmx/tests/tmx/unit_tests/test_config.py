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
    config = tmx_config.TmxConfig(solver_type=2, energy_type=2)
    assert config.solver_type is tmx_config.SolverType.IMPLICIT
    assert config.energy_type is tmx_config.EnergyType.INTERNAL


@pytest.mark.parametrize("solver_type", [1, 3])
def test_config_rejects_unimplemented_solver_types(solver_type: int) -> None:
    with pytest.raises(ValueError, match="solver_type"):
        tmx_config.TmxConfig(solver_type=solver_type)


@pytest.mark.parametrize("energy_type", [1, 3])
def test_config_rejects_unimplemented_energy_types(energy_type: int) -> None:
    with pytest.raises(ValueError, match="energy_type"):
        tmx_config.TmxConfig(energy_type=energy_type)


def test_config_round_trips_through_config_io() -> None:
    """Every enum option is registered, so the config survives (un)structuring."""
    config = tmx_config.TmxConfig()
    unstructured = config_io.CONV.unstructure(config)

    assert unstructured["solver_type"] == "implicit"
    assert unstructured["energy_type"] == "internal"
    assert config_io.CONV.structure(unstructured, tmx_config.TmxConfig) == config


def _surface_config(**overrides: object) -> tmx_config.TmxSurfaceConfig:
    kwargs: dict = {"surface_flux_type": 1, "sea_surface_temperature": 303.15, **overrides}
    return tmx_config.TmxSurfaceConfig(**kwargs)


def test_surface_config_defaults_are_those_of_the_testcase_namelist() -> None:
    config = _surface_config()
    assert config.surface_flux_type is tmx_config.SurfaceFluxType.FIXED_HEAT_FLUXES
    # mo_nh_testcases_nml.f90: shflx = 0.1, lhflx = 0.0
    assert config.kinematic_sensible_heat_flux == 0.1
    assert config.kinematic_latent_heat_flux == 0.0


@pytest.mark.parametrize("surface_flux_type", [0, 2])
def test_surface_config_rejects_unimplemented_surface_flux_types(surface_flux_type: int) -> None:
    with pytest.raises(ValueError, match="surface_flux_type"):
        _surface_config(surface_flux_type=surface_flux_type)


def test_surface_config_round_trips_through_config_io() -> None:
    config = _surface_config()
    unstructured = config_io.CONV.unstructure(config)

    assert unstructured["surface_flux_type"] == "fixed_heat_fluxes"
    assert config_io.CONV.structure(unstructured, tmx_config.TmxSurfaceConfig) == config
