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

from icon4py.model.common import field, field_type_aliases as fa, state, type_alias as ta
from icon4py.model.common.framework import quantities as qty


@dataclasses.dataclass(frozen=True)
class SurfaceInputState:
    """Atmospheric input of the surface scheme: the lowest level of the atmosphere."""

    surface_pressure: fa.CellField[ta.wpfloat]
    """Pressure at the surface, the lowest half level (``psfc``) [Pa]."""
    temperature: fa.CellField[ta.wpfloat]
    """Air temperature at the lowest full level (``ta(nlev)``) [K]."""


class SurfaceFluxState(state.State):
    """The surface fluxes of the surface scheme: the bottom boundary of tmx's vertical diffusion."""

    evapotranspiration: field.Field[qty.EvapotranspirationOnCell]
    """Surface evapotranspiration flux (``evspsbl``) [kg/(m^2 s)]."""
    sensible_heat_flux: field.Field[qty.SensibleHeatFluxOnCell]
    """Surface sensible heat flux (``hfss``) [W/m^2]."""
    latent_heat_flux: field.Field[qty.LatentHeatFluxOnCell]
    """Surface latent heat flux (``hfls``) [W/m^2]."""
    u_stress: field.Field[qty.UStressOnCell]
    """Zonal surface wind stress (``tauu``) [N/m^2]."""
    v_stress: field.Field[qty.VStressOnCell]
    """Meridional surface wind stress (``tauv``) [N/m^2]."""
    q_snocpymlt: field.Field[qty.SnowCanopyMeltHeatOnCell]
    """Heat used to melt snow on the canopy (``q_snocpymlt``), zero off land [W/m^2]."""
    thermal_energy_flux: field.Field[qty.ThermalEnergyFluxOnCell]
    """Surface flux of internal energy from thermal exchange (``ufts``) [W/m^2]."""
    vapor_energy_flux: field.Field[qty.VaporEnergyFluxOnCell]
    """Surface flux of internal energy from vapor exchange (``ufvs``) [W/m^2]."""
