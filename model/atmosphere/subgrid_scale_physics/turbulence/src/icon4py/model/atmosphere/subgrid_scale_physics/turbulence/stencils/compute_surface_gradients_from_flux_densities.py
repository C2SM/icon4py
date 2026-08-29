# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Effective surface-layer gradients implied by the prescribed surface flux densities.

Translated from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE 'vertdiff'
(:614-634 at icon commit 26d6b98cce), the block "Load effective surface layer gradients due to
given flux values", which runs for a variable whose 'lsfli(n)' is true. The scientific
commentary in that file is by Matthias Raschendorfer (DWD).

    wert = rhon(i,ke1)*vtyp_tkv(i,ke1)
    IF (n.EQ.tem) zvari(i,ke1,m) = dvar_sv(i)/(wert*cp_d*eprs(i,ke1))
    ELSE          zvari(i,ke1,m) = dvar_sv(i)/wert

WHEN THIS RUNS. 'lsfli' is set by 'lsfluse' -- "use explicit heat flux densities at the
surface" -- which the NWP interface passes as '.TRUE.', and it makes the surface value of
temperature and of water vapour a FLUX rather than a concentration: 'dvar(tem)%sv' is
repointed from 't_g' to 'shfl_s' and 'dvar(vap)%sv' from 'qv_s' to 'qvfl_s' (:399-400). It is
never true for the two wind components or for cloud water, so the two outputs here are the
complete set.

Inverting the surface-layer flux law 'F = -rho*K*dc/dz' for the gradient is all this is; the
sensible heat flux carries an extra 'c_pd*eprs' because it is an enthalpy flux and the
diffused variable is the potential temperature. The Fortran's warning is worth repeating: the
division needs 'tkv(ke1) > 0', so 'tkvh(ke1) = 0' must never be forced when 'lsfli' is true.

These gradients land in 'zvari(:,ke1,m)', which the solve later overwrites with the explicit
surface flux; they exist only to give
'compute_surface_profile_value_from_flux_gradient' its lower boundary value.
"""

import enum

import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.type_alias import wpfloat


class SurfaceFluxConstants(ta.wpfloat, enum.Enum):
    """The one physical constant this stencil reads.

    An enum because GT4Py can only fold a closure variable that is an attribute of an enum class
    or of an 'eve.FrozenNamespace'; see the docstring of
    'thermodynamic_functions.ThermoConstants', which holds every other constant the turbulence
    stencils use. 'c_pd' is not among them, since nothing before 'vertdiff' needed it.
    """

    #: 'cp_d', the specific heat of dry air at constant pressure [J/K/kg].
    CP_D = constants.CPD


@gtx.field_operator
def _compute_surface_temperature_gradient(
    sensible_heat_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_exner_factor: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'shfl_s/(rhon*tkvh*c_pd*eprs)' [K/m], with the denominator associated to the left."""
    transfer_momentum = air_density * diffusion_coefficient
    return broadcast(sensible_heat_flux, (dims.CellDim, dims.KDim)) / (
        transfer_momentum * SurfaceFluxConstants.CP_D * surface_exner_factor
    )


@gtx.field_operator
def _compute_surface_vapour_gradient(
    water_vapour_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'qvfl_s/(rhon*tkvh)' [1/m]."""
    transfer_momentum = air_density * diffusion_coefficient
    return broadcast(water_vapour_flux, (dims.CellDim, dims.KDim)) / transfer_momentum


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_gradients_from_flux_densities(
    sensible_heat_flux: fa.CellField[wpfloat],
    water_vapour_flux: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    diffusion_coefficient: fa.CellKField[wpfloat],
    surface_exner_factor: fa.CellKField[wpfloat],
    surface_temperature_gradient: fa.CellKField[wpfloat],
    surface_vapour_gradient: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Write the surface row of the two effective gradients implied by the surface fluxes.

    One program for both because the Fortran is one loop with one 'IF' in it, because both rows
    are one deep, and because 'lsfluse' makes them either both live or both dead.

    Args:
        sensible_heat_flux: 'shfl_s' [W/m2], positive downward.
        water_vapour_flux: 'qvfl_s' [kg/m2/s], positive downward.
        air_density: 'rhon' [kg/m3]; only the surface row is read.
        diffusion_coefficient: 'tkvh' [m2/s], the SCALAR type's; only the surface row is read.
        surface_exner_factor: 'eprs' [-], from
            'compute_surface_air_density_and_exner_factor'.
        surface_temperature_gradient: In-out, 'zvari(:,:,tem)' [K/m]; surface row only.
        surface_vapour_gradient: In-out, 'zvari(:,:,vap)' [1/m]; surface row only.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The surface half level, 'nlev' (Fortran 'ke1').
        vertical_end: 'nlev + 1'.
    """
    _compute_surface_temperature_gradient(
        sensible_heat_flux=sensible_heat_flux,
        air_density=air_density,
        diffusion_coefficient=diffusion_coefficient,
        surface_exner_factor=surface_exner_factor,
        out=surface_temperature_gradient,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_surface_vapour_gradient(
        water_vapour_flux=water_vapour_flux,
        air_density=air_density,
        diffusion_coefficient=diffusion_coefficient,
        out=surface_vapour_gradient,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
