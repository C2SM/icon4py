# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the tmx turbulent mixing component.

Kept apart from the component itself so that reading or building a configuration does
not import the stencils.
"""

from __future__ import annotations

import dataclasses
import enum
import typing

from icon4py.model.common.config import config_io, options as common_conf_opt


@config_io.register_enum
class SolverType(int, enum.Enum):
    """Type of the vertical diffusion solver; ICON's explicit solver (1) is not ported."""

    IMPLICIT = 2  # implicit time stepping


@config_io.register_enum
class EnergyType(int, enum.Enum):
    """Type of energy diffused by the heat diffusion; ICON's dry static energy (1) is not ported."""

    INTERNAL = 2  # internal energy cv*T


@config_io.register_enum
class SurfaceFluxType(int, enum.Enum):
    """How tmx gets its surface fluxes; ICON's surface models (0) are not ported."""

    FIXED_HEAT_FLUXES = 1  # fixed kinematic heat fluxes over a fixed surface temperature


@dataclasses.dataclass(kw_only=True)
class TmxConfig:
    """
    Default values are taken from ``vdiff_config_init`` in the corresponding ICON
    Fortran module ``mo_turb_vdiff_config.f90`` (namelist ``aes_vdf_nml``).
    """

    solver_type: typing.Annotated[
        SolverType,
        common_conf_opt.ConfigOption(
            description="Type of the vertical diffusion solver (only the implicit solver is implemented).",
        ),
    ] = SolverType.IMPLICIT

    energy_type: typing.Annotated[
        EnergyType,
        common_conf_opt.ConfigOption(
            description="Type of energy diffused by the heat diffusion (only internal energy is implemented).",
        ),
    ] = EnergyType.INTERNAL

    dissipation_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the kinetic energy dissipation heating.",
        ),
    ] = 1.0

    use_louis: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use the Louis (1979) stability correction function "
            "instead of the classic (Lilly 1962) one.",
        ),
    ] = True

    use_louis_land: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% land fraction "
            "from the Louis stability correction.",
        ),
    ] = True

    use_louis_ice: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% sea-ice fraction "
            "from the Louis stability correction.",
        ),
    ] = True

    louis_constant_b: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Louis constant b of the Louis stability correction function.",
        ),
    ] = 4.2

    use_km_const: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use a constant exchange coefficient instead of the "
            "Smagorinsky model.",
        ),
    ] = False

    km_const: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Constant exchange coefficient used if 'use_km_const' is True [m^2/s].",
        ),
    ] = 1.0

    use_scale_turb_energy_flux: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, scale the turbulent energy flux by 'scale_turb_energy_flux'.",
        ),
    ] = False

    scale_turb_energy_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the turbulent energy flux used if "
            "'use_scale_turb_energy_flux' is True.",
        ),
    ] = 1.0

    smag_constant: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Smagorinsky constant Cs of the Smagorinsky-Lilly eddy viscosity model.",
        ),
    ] = 0.23

    turb_prandtl: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Turbulent Prandtl number.",
        ),
    ] = 0.33333333333  # exact literal from mo_turb_vdiff_config.f90 (not 1/3)

    km_min: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum mass-weighted turbulent viscosity [kg/(m s)].",
        ),
    ] = 0.001

    max_turb_scale: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum turbulence length scale [m].",
        ),
    ] = 300.0

    def __post_init__(self) -> None:
        try:
            self.solver_type = SolverType(self.solver_type)
        except ValueError:
            raise ValueError(
                f"Invalid argument 'solver_type': only the implicit solver "
                f"({SolverType.IMPLICIT.value}) is implemented, got {self.solver_type}."
            ) from None
        try:
            self.energy_type = EnergyType(self.energy_type)
        except ValueError:
            raise ValueError(
                f"Invalid argument 'energy_type': only internal energy "
                f"({EnergyType.INTERNAL.value}) is implemented, got {self.energy_type}."
            ) from None

        if self.turb_prandtl <= 0.0:
            raise ValueError(
                f"Invalid argument 'turb_prandtl': should be positive, got {self.turb_prandtl}."
            )
        if self.km_min < 0.0:
            raise ValueError(
                f"Invalid argument 'km_min': should be non-negative, got {self.km_min}."
            )


@dataclasses.dataclass(kw_only=True)
class TmxSurfaceConfig:
    """
    The surface boundary of tmx, which ICON reads from ``nh_testcase_nml``.

    Default values are taken from ``mo_nh_testcases_nml.f90`` where ICON's default is ported.
    """

    surface_flux_type: typing.Annotated[
        SurfaceFluxType,
        common_conf_opt.ConfigOption(
            description="How tmx gets its surface fluxes (`isrfc_type`; only the fixed "
            "kinematic heat fluxes are implemented).",
        ),
    ]

    kinematic_sensible_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic sensible heat flux at the surface (`shflx`) [K m/s].",
        ),
    ] = 0.1

    kinematic_latent_heat_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Kinematic latent heat flux at the surface (`lhflx`) [m/s].",
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
