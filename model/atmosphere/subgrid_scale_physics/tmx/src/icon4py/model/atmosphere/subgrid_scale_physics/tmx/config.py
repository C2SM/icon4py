# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the tmx turbulent mixing granule.

Kept apart from the granule itself so that reading or building a configuration does
not import the stencils.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import typing
from typing import Any, NamedTuple

from icon4py.model.common.config import config_io, options as common_conf_opt


if typing.TYPE_CHECKING:
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


log = logging.getLogger(__name__)


@config_io.register_enum
class TurbulenceSolverType(int, enum.Enum):
    """
    Type of the vertical diffusion solver.

    Note: Called ``solver_type`` in ``mo_turb_vdiff_config.f90``.
    """

    EXPLICIT = 1  # explicit time stepping
    IMPLICIT = 2  # implicit time stepping


@config_io.register_enum
class EnergyType(int, enum.Enum):
    """
    Type of energy diffused by the temperature (heat) diffusion.

    Note: Called ``energy_type`` in ``mo_turb_vdiff_config.f90``.
    """

    DRY_STATIC = 1  # dry static energy cp*T + g*z
    INTERNAL = 2  # internal energy cv*T


@config_io.register_enum
class SurfaceType(int, enum.Enum):
    """
    Treatment of the surface fluxes.

    Note: called ``isrfc_type`` in ``mo_nh_testcases_nml.f90``.
    """

    INTERACTIVE = 0  # fluxes from the surface scheme
    FIXED_HEAT_FLUXES = 1  # fixed kinematic surface heat fluxes


class _DiffusedTracer(NamedTuple):
    """One hydrometeor of the scalar diffusion loop, with its per-step buffers."""

    name: str
    state: fa.CellKField[ta.wpfloat]
    tendency: fa.CellKField[ta.wpfloat]
    new_state: fa.CellKField[ta.wpfloat]
    surface_flux: fa.CellField[ta.wpfloat]


@dataclasses.dataclass(kw_only=True)
class TmxConfig:
    """
    Default values are taken from ``vdiff_config_init`` in the corresponding ICON
    Fortran module ``mo_turb_vdiff_config.f90`` (namelist ``aes_vdf_nml``).
    """

    solver_type: typing.Annotated[
        TurbulenceSolverType,
        common_conf_opt.ConfigOption(
            description="Type of the vertical diffusion solver (explicit or implicit).",
            icon_equivalent=common_conf_opt.IconOption(
                "solver_type", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=23
            ),
        ),
    ] = TurbulenceSolverType.IMPLICIT

    energy_type: typing.Annotated[
        EnergyType,
        common_conf_opt.ConfigOption(
            description="Type of energy diffused by the heat diffusion (dry static or internal).",
            icon_equivalent=common_conf_opt.IconOption(
                "energy_type", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=24
            ),
        ),
    ] = EnergyType.INTERNAL

    dissipation_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the kinetic energy dissipation heating.",
            icon_equivalent=common_conf_opt.IconOption(
                "dissipation_factor", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=25
            ),
        ),
    ] = 1.0

    use_louis: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use the Louis (1979) stability correction function "
            "instead of the classic (Lilly 1962) one.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=26
            ),
        ),
    ] = True

    use_louis_land: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% land fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis_land", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=27
            ),
        ),
    ] = True

    use_louis_ice: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If False, exclude cells with more than 50% sea-ice fraction "
            "from the Louis stability correction.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_louis_ice", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=28
            ),
        ),
    ] = True

    louis_constant_b: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Louis constant b of the Louis stability correction function.",
            icon_equivalent=common_conf_opt.IconOption(
                "louis_constant_b", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=29
            ),
        ),
    ] = 4.2

    use_km_const: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, use a constant exchange coefficient instead of the "
            "Smagorinsky model.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_km_const", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=30
            ),
        ),
    ] = False

    km_const: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Constant exchange coefficient used if 'use_km_const' is True [m^2/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "km_const", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=31
            ),
        ),
    ] = 1.0

    use_scale_turb_energy_flux: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, scale the turbulent energy flux by 'scale_turb_energy_flux'.",
            icon_equivalent=common_conf_opt.IconOption(
                "use_scale_turb_energy_flux", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=32
            ),
        ),
    ] = False

    scale_turb_energy_flux: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for the turbulent energy flux used if "
            "'use_scale_turb_energy_flux' is True.",
            icon_equivalent=common_conf_opt.IconOption(
                "scale_turb_energy_flux", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=33
            ),
        ),
    ] = 1.0

    smag_constant: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Smagorinsky constant Cs of the Smagorinsky-Lilly eddy viscosity model.",
            icon_equivalent=common_conf_opt.IconOption(
                "smag_constant", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=34
            ),
        ),
    ] = 0.23

    turb_prandtl: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Turbulent Prandtl number.",
            icon_equivalent=common_conf_opt.IconOption(
                "turb_prandtl", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=35
            ),
        ),
    ] = 0.33333333333  # exact literal from mo_turb_vdiff_config.f90 (not 1/3)

    km_min: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Minimum mass-weighted turbulent viscosity [kg/(m s)].",
            icon_equivalent=common_conf_opt.IconOption(
                "km_min", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=37
            ),
        ),
    ] = 0.001

    max_turb_scale: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum turbulence length scale [m].",
            icon_equivalent=common_conf_opt.IconOption(
                "max_turb_scale", ("aes_vdf_nml", "aes_vdf_config"), unnamed_index=38
            ),
        ),
    ] = 300.0

    surface_type: typing.Annotated[
        SurfaceType,
        common_conf_opt.ConfigOption(
            description="Treatment of the surface fluxes (interactive or fixed heat fluxes).",
            icon_equivalent=common_conf_opt.IconOption(
                "isrfc_type", ("nh_testcase_nml",), read_from_icon=False
            ),
        ),
    ] = SurfaceType.INTERACTIVE

    shflx: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Fixed kinematic sensible heat flux at the surface [K m/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "shflx", ("nh_testcase_nml",), read_from_icon=False
            ),
        ),
    ] = 0.1

    lhflx: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Fixed kinematic latent heat flux at the surface [m/s].",
            icon_equivalent=common_conf_opt.IconOption(
                "lhflx", ("nh_testcase_nml",), read_from_icon=False
            ),
        ),
    ] = 0.0

    def __post_init__(self) -> None:
        self.solver_type = TurbulenceSolverType(self.solver_type)
        self.energy_type = EnergyType(self.energy_type)
        self.surface_type = SurfaceType(self.surface_type)

        if self.turb_prandtl <= 0.0:
            raise ValueError(
                f"Invalid argument 'turb_prandtl': should be positive, got {self.turb_prandtl}."
            )
        if self.km_min < 0.0:
            raise ValueError(
                f"Invalid argument 'km_min': should be non-negative, got {self.km_min}."
            )

    @classmethod
    def from_fortran_dict(
        cls, *, atm_dict: dict[str, Any], input_dict: dict[str, Any], **overrides: Any
    ) -> TmxConfig:
        """
        Construct the configuration from the echoed ICON namelists.

        ``aes_vdf_nml`` is a derived-type namelist (``t_vdiff_config``), which
        ICON echoes as an anonymous positional array of the member values in
        declaration order, so the options are located by ``unnamed_index``
        (pinned to mo_turb_vdiff_config.f90) instead of by name. Only the
        first domain is read. The guards below make a change of the Fortran
        type fail loudly instead of silently mis-assigning values.

        The surface-flux options come from the *input* namelist dict instead,
        which holds only the members the experiment sets explicitly, so absent
        ones are left out and keep the class default rather than being indexed
        strictly.
        """
        # number of members of the Fortran t_vdiff_config derived type
        # (mo_turb_vdiff_config.f90); the echoed aes_vdf_nml namelist holds this
        # many values per domain, in declaration order. Must be kept in sync with
        # the 'unnamed_index' positions of the options above.
        num_members = 42
        # position of 'use_tmx' in t_vdiff_config, used as an order canary
        use_tmx_index = 22

        flat = atm_dict["aes_vdf_nml"]["aes_vdf_config"]
        if len(flat) % num_members != 0:
            raise ValueError(
                f"'aes_vdf_config' has {len(flat)} values, not a multiple of the "
                f"{num_members} members of t_vdiff_config: the Fortran type changed "
                "and the pinned 'unnamed_index' positions must be revised."
            )
        use_tmx = flat[use_tmx_index]
        if use_tmx is not True:
            raise ValueError(
                f"expected 'use_tmx' (True) at position {use_tmx_index} of "
                f"'aes_vdf_config', found {use_tmx!r}: either the run does not use tmx or "
                "the t_vdiff_config member order changed."
            )
        testcase = input_dict.get("nh_testcase_nml", {})
        # 'nh_testcase_nml' member -> (TmxConfig field, converter); members
        # absent from the namelist keep the TmxConfig default
        surface_options = {
            "isrfc_type": ("surface_type", SurfaceType),
            "shflx": ("shflx", float),
            "lhflx": ("lhflx", float),
        }
        surface_fluxes = {
            field: convert(testcase[name])
            for name, (field, convert) in surface_options.items()
            if name in testcase
        }
        return common_conf_opt.construct_config_from_icon(
            cls, atm_dict, **(surface_fluxes | overrides)
        )
