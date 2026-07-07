# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_energy_flux(
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    use_internal_energy: bool,
) -> fa.CellField[wpfloat]:
    """
    Compute the grid-mean surface flux of the energy diffused by the tmx heat
    diffusion.

    Port of 'compute_flux_x' (mo_vdf_atmo.f90) with the ``ufts`` / ``ufvs``
    energy fluxes inlined from 'compute_energy_fluxes' (mo_tmx_surface.f90,
    called on the grid-mean fluxes at the end of the surface Compute in
    mo_vdf_sfc.f90):

    - dry static energy (``energy_type = 1``, ``use_internal_energy = False``):

          flux_x = sensible_heat_flux * cpd / cvd

    - internal energy (``energy_type = 2``, ``use_internal_energy = True``):

          ufts   = sensible_heat_flux
          ufvs   = temperature_sfc * evapotranspiration * (cvv - cvd)
          flux_x = ufts + ufvs

      ``ufts`` is the surface energy flux from thermal exchange and ``ufvs``
      the one from vapor exchange.

    ``use_internal_energy`` is a scalar configuration flag; it can be passed as
    a static (compile-time) argument so that only the selected variant is
    compiled.

    Domain (Fortran): the tmx ``t_domain`` cell range (``grf_bdywidth_c + 1``
    to ``min_rlcell_int``), which maps to the horizontal domain
    ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``; 2D surface fields only.

    Args:
        sensible_heat_flux: grid-mean surface sensible heat flux (``shfl``) [W/m^2]
        evapotranspiration: grid-mean surface evapotranspiration flux [kg/(m^2 s)]
        temperature_sfc: air temperature at the lowest full level (``ta(nlev)``) [K]
        use_internal_energy: True for internal energy, False for dry static energy

    Returns:
        grid-mean surface energy flux (``flux_x``) [W/m^2]
    """
    if use_internal_energy:
        ufts = sensible_heat_flux
        ufvs = temperature_sfc * evapotranspiration * (PhysicsConstants.cvv - PhysicsConstants.cvd)
        flux_x = ufts + ufvs
    else:
        flux_x = sensible_heat_flux * PhysicsConstants.cpd / PhysicsConstants.cvd
    return flux_x


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_energy_flux(
    sensible_heat_flux: fa.CellField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    temperature_sfc: fa.CellField[wpfloat],
    flux_x: fa.CellField[wpfloat],
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_energy_flux(
        sensible_heat_flux=sensible_heat_flux,
        evapotranspiration=evapotranspiration,
        temperature_sfc=temperature_sfc,
        use_internal_energy=use_internal_energy,
        out=flux_x,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
