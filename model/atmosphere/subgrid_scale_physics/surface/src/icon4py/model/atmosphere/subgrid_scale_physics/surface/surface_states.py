# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""States of the surface scheme."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@dataclasses.dataclass(frozen=True)
class SurfaceInputState:
    """Atmospheric input of the surface scheme: the lowest level of the atmosphere."""

    surface_pressure: fa.CellField[ta.wpfloat]
    """Pressure at the surface, the lowest half level (``psfc``) [Pa]."""
    temperature: fa.CellField[ta.wpfloat]
    """Air temperature at the lowest full level (``ta(nlev)``) [K]."""
    virtual_temperature: fa.CellField[ta.wpfloat]
    """Virtual temperature at the lowest full level (``tv``) [K]."""
    pressure: fa.CellField[ta.wpfloat]
    """Air pressure at the lowest full level (``pa``) [Pa]."""
    specific_humidity: fa.CellField[ta.wpfloat]
    """Specific humidity at the lowest full level (``qa``) [kg/kg]."""
    u: fa.CellField[ta.wpfloat]
    """Zonal wind at the lowest full level (``ua``) [m/s]."""
    v: fa.CellField[ta.wpfloat]
    """Meridional wind at the lowest full level (``va``) [m/s]."""


@dataclasses.dataclass(frozen=True)
class SurfaceMetricState:
    """Static geometry of the surface layer."""

    z_mc: fa.CellField[ta.wpfloat]
    """Height of the lowest full level (``zf`` in the surface scheme) [m]."""
    ddqz_z_half: fa.CellField[ta.wpfloat]
    """
    Thickness of the lowest half-level layer, at the surface (``dz`` in the surface scheme) [m];
    half of it is the reference height of the surface layer in the exchange coefficients.
    """


@dataclasses.dataclass(frozen=True)
class SurfaceDiagnosticState:
    """
    Diagnostics of the surface scheme (``t_vdf_sfc_diags`` in mo_vdf_sfc_memory.f90), for one
    surface tile: ICON's ``*_tile`` fields and their aggregates coincide.
    """

    wind_speed: fa.CellField[ta.wpfloat]
    """Wind speed at the lowest level, floored at the configured minimum (``wind``) [m/s]."""
    potential_temperature: fa.CellField[ta.wpfloat]
    """Potential temperature of the air at the lowest level (``theta_atm``) [K]."""
    virtual_potential_temperature: fa.CellField[ta.wpfloat]
    """Virtual potential temperature of the air at the lowest level (``thetav_atm``) [K]."""
    saturation_specific_humidity: fa.CellField[ta.wpfloat]
    """Saturation specific humidity at the surface temperature (``qsat_tile``) [kg/kg]."""
    density: fa.CellField[ta.wpfloat]
    """Air density at the surface (``rho_tile``) [kg/m^3]."""
    surface_potential_temperature: fa.CellField[ta.wpfloat]
    """Potential temperature of the surface (``theta_tile``) [K]."""
    surface_virtual_potential_temperature: fa.CellField[ta.wpfloat]
    """Virtual potential temperature of the surface (``thetav_tile``) [K]."""
    richardson_number: fa.CellField[ta.wpfloat]
    """Moist bulk Richardson number of the surface layer (``moist_rich_tile``) [1]."""
    roughness_length_momentum: fa.CellField[ta.wpfloat]
    """Roughness length for momentum (``rough_m``) [m], carried from step to step."""
    roughness_length_heat: fa.CellField[ta.wpfloat]
    """Roughness length for heat (``rough_h``) [m], equal to the momentum one over the sea."""
    exchange_coefficient_momentum: fa.CellField[ta.wpfloat]
    """Surface exchange coefficient for momentum (``km_sfc``), the lowest level of tmx's ``km``."""
    exchange_coefficient_heat: fa.CellField[ta.wpfloat]
    """Surface exchange coefficient for heat (``kh_sfc``), the lowest level of tmx's ``kh``."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> SurfaceDiagnosticState:
        """Allocate a diagnostic state with all fields initialized to zero."""
        zero_field = functools.partial(
            data_alloc.zero_field, grid, dtype=ta.wpfloat, allocator=allocator
        )
        return cls(**{field.name: zero_field(dims.CellDim) for field in dataclasses.fields(cls)})
