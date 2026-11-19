# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.surface import config as surface_config
from icon4py.model.common.config import config_io


def _ape_config(**overrides: object) -> surface_config.SurfaceConfig:
    """The aquaplanet setup of the reference data: prescribed fluxes over a 30 degC sea."""
    kwargs: dict[str, object] = dict(surface_flux_type=1, sea_surface_temperature=303.15)
    kwargs.update(overrides)
    return surface_config.SurfaceConfig(**kwargs)  # type: ignore[arg-type]


def test_config_defaults_are_the_fortran_defaults() -> None:
    """The defaults equal ICON's."""
    config = _ape_config()
    assert config.kinematic_sensible_heat_flux == 0.1
    assert config.kinematic_latent_heat_flux == 0.0


def test_config_coerces_enum_from_int() -> None:
    assert _ape_config().surface_flux_type is surface_config.SurfaceFluxType.FIXED_HEAT_FLUXES


@pytest.mark.parametrize("surface_flux_type", [0, 2])
def test_config_rejects_unimplemented_surface_flux_types(surface_flux_type: int) -> None:
    with pytest.raises(ValueError, match="surface_flux_type"):
        _ape_config(surface_flux_type=surface_flux_type)


def test_config_round_trips_through_config_io() -> None:
    """The enum option is registered, so the config survives (un)structuring."""
    config = _ape_config(kinematic_latent_heat_flux=2.0e-5)
    unstructured = config_io.CONV.unstructure(config)

    assert unstructured["surface_flux_type"] == "fixed_heat_fluxes"
    assert config_io.CONV.structure(unstructured, surface_config.SurfaceConfig) == config
