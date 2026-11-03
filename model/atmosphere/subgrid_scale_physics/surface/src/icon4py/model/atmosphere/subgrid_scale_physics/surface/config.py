# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the surface scheme."""

from __future__ import annotations

import dataclasses
import enum
import typing

from icon4py.model.common.config import config_io, options as common_conf_opt


@config_io.register_enum
class SurfaceFluxType(int, enum.Enum):
    """
    How the surface fluxes are obtained; ICON's interactive ocean, sea-ice and land tiles (0)
    are not ported.
    """

    FIXED_HEAT_FLUXES = 1
    """Fixed kinematic heat fluxes over a fixed surface temperature."""


@dataclasses.dataclass(kw_only=True)
class SurfaceConfig:
    """The surface boundary of the atmosphere: the surface fluxes tmx mixes into the air."""

    surface_flux_type: typing.Annotated[
        SurfaceFluxType,
        common_conf_opt.ConfigOption(
            description="How the surface fluxes are obtained (only the fixed kinematic heat "
            "fluxes are implemented).",
        ),
    ]

    kinematic_sensible_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic sensible heat flux at the surface [K m/s].",
        ),
    ] = 0.1

    kinematic_latent_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic latent heat flux at the surface [m/s].",
        ),
    ] = 0.0

    sea_surface_temperature: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Sea surface temperature, constant in space and time [K].",
        ),
    ]

    def __post_init__(self) -> None:
        try:
            self.surface_flux_type = SurfaceFluxType(self.surface_flux_type)
        except ValueError:
            raise ValueError(
                f"Invalid argument 'surface_flux_type': only fixed kinematic heat fluxes "
                f"({SurfaceFluxType.FIXED_HEAT_FLUXES.value}) are implemented, "
                f"got {self.surface_flux_type}."
            ) from None
