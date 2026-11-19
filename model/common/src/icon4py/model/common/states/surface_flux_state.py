# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The surface fluxes handed from the surface scheme to the atmosphere."""

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
class SurfaceFluxState:
    """
    Surface fluxes of the surface scheme: the bottom boundary of tmx's vertical diffusion
    and, later, inputs to radiation.
    """

    evapotranspiration: fa.CellField[ta.wpfloat]
    """Surface evapotranspiration flux (``evspsbl``) [kg/(m^2 s)]."""
    sensible_heat_flux: fa.CellField[ta.wpfloat]
    """Surface sensible heat flux (``hfss``) [W/m^2]."""
    latent_heat_flux: fa.CellField[ta.wpfloat]
    """Surface latent heat flux (``hfls``) [W/m^2]."""
    u_stress: fa.CellField[ta.wpfloat]
    """Zonal surface wind stress (``tauu``) [N/m^2]."""
    v_stress: fa.CellField[ta.wpfloat]
    """Meridional surface wind stress (``tauv``) [N/m^2]."""
    q_snocpymlt: fa.CellField[ta.wpfloat]
    """Heat used to melt snow on the canopy (``q_snocpymlt``), zero off land [W/m^2]."""
    thermal_energy_flux: fa.CellField[ta.wpfloat]
    """Surface flux of internal energy from thermal exchange (``ufts``) [W/m^2]."""
    vapor_energy_flux: fa.CellField[ta.wpfloat]
    """Surface flux of internal energy from vapor exchange (``ufvs``) [W/m^2]."""

    @classmethod
    def allocate(
        cls, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None = None
    ) -> SurfaceFluxState:
        """Allocate a flux state with all fields initialized to zero."""
        zero_field = functools.partial(
            data_alloc.zero_field, grid, dtype=ta.wpfloat, allocator=allocator
        )
        return cls(
            evapotranspiration=zero_field(dims.CellDim),
            sensible_heat_flux=zero_field(dims.CellDim),
            latent_heat_flux=zero_field(dims.CellDim),
            u_stress=zero_field(dims.CellDim),
            v_stress=zero_field(dims.CellDim),
            q_snocpymlt=zero_field(dims.CellDim),
            thermal_energy_flux=zero_field(dims.CellDim),
            vapor_energy_flux=zero_field(dims.CellDim),
        )
