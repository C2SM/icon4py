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
    """How the surface fluxes are obtained (``isrfc_type`` in ``nh_testcase_nml``)."""

    INTERACTIVE = 0
    """The ocean, sea-ice and land tiles of ``mo_vdf_sfc.f90``; not ported."""
    FIXED_HEAT_FLUXES = 1
    """Fixed kinematic heat fluxes over a fixed surface temperature."""


@dataclasses.dataclass(kw_only=True)
class SurfaceConfig:
    """
    The surface boundary of the atmosphere, which ICON reads from ``nh_testcase_nml``.

    ICON's interactive surface models (``isrfc_type = 0``: ocean, sea ice, land) are not
    ported. Default values are those of ``mo_nh_testcases_nml.f90``.
    """

    surface_flux_type: typing.Annotated[
        SurfaceFluxType,
        common_conf_opt.ConfigOption(
            description="How the surface fluxes are obtained (`isrfc_type`); only the fixed "
            "kinematic heat fluxes are implemented.",
        ),
    ]

    kinematic_sensible_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic sensible heat flux at the surface (`shflx` in `nh_testcase_nml`) "
            "[K m/s]; the default is the one of mo_nh_testcases_nml.f90.",
        ),
    ] = 0.1

    kinematic_latent_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic latent heat flux at the surface (`lhflx` in `nh_testcase_nml`) "
            "[m/s]; the default is the one of mo_nh_testcases_nml.f90.",
        ),
    ] = 0.0

    sea_surface_temperature: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Sea surface temperature, constant in space and time "
            "(`ape_sst_val` of the `sst_const` case) [K].",
        ),
    ]

    def __post_init__(self) -> None:
        try:
            self.surface_flux_type = SurfaceFluxType(self.surface_flux_type)
        except ValueError:
            raise ValueError(
                f"Invalid argument 'surface_flux_type': got {self.surface_flux_type}, expected "
                f"one of {[member.value for member in SurfaceFluxType]}."
            ) from None
        if self.surface_flux_type is not SurfaceFluxType.FIXED_HEAT_FLUXES:
            raise ValueError(
                f"Invalid argument 'surface_flux_type': only fixed kinematic heat fluxes "
                f"({SurfaceFluxType.FIXED_HEAT_FLUXES.value}) are implemented, "
                f"got {self.surface_flux_type.value}."
            )
