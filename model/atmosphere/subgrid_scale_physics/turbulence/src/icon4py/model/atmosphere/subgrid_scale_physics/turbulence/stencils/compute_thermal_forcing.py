# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Thermal forcing of the TKE equation: the buoyancy production term.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads "1b) Calculation of the basic single-column forcing functions for
TKE" (:1209-1240 at icon commit 26d6b98cce, which is the commit that produced the reference
capture; the section is :1196-1229 in the uninstrumented upstream file). The scientific
commentary in that file is by Matthias Raschendorfer (DWD).

The Fortran is 'frh(i,k) = zaux(i,k,4)*zvari(i,k,tet_l) + zaux(i,k,5)*zvari(i,k,h2o_g)', i.e.
the vertical gradients of the two quasi-conserved thermodynamic variables weighted by their
buoyancy factors. 'zaux(:,:,4)' and 'zaux(:,:,5)' are the 'g_tet' and 'g_h2o' outputs of
'adjust_satur_equil' (turb_utilities.f90), produced by section 0); the two gradients are
produced by section 1a). All four are half-level fields.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_thermal_forcing(
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    vertical_gradient_tet_l: fa.CellKField[wpfloat],
    vertical_gradient_h2o_g: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Buoyancy production of turbulent kinetic energy from the two conserved-variable gradients."""
    return (
        buoyancy_factor_tet_l * vertical_gradient_tet_l
        + buoyancy_factor_h2o_g * vertical_gradient_h2o_g
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_thermal_forcing(
    buoyancy_factor_tet_l: fa.CellKField[wpfloat],
    buoyancy_factor_h2o_g: fa.CellKField[wpfloat],
    vertical_gradient_tet_l: fa.CellKField[wpfloat],
    vertical_gradient_h2o_g: fa.CellKField[wpfloat],
    thermal_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the thermal (buoyancy) forcing 'frh' of the TKE equation on half levels.

    'tet_l' is the liquid-water potential temperature and 'h2o_g' the total water content --
    Raschendorfer's names for the two variables that are conserved under condensation and
    evaporation, and therefore the pair the buoyancy flux is expressed in. Their buoyancy
    factors carry the moisture and cloud-cover dependence, so the sum below is the full moist
    buoyancy production and not a dry-air approximation.

    The vertical range runs to the surface half level 'ke1', one half level below where the
    mechanical forcing stops. The note the Fortran leaves at that point
    (turb_diffusion.f90:1225) is the reason:

        "'frh' at "0"-level (k=ke1) is used for calculating the acceleration of non-turbulent
        near-surface circulations."

    and the local declaration (:783) agrees that the array is not always a forcing: "thermal
    forcing (1/s2) or thermal acceleration (m/s2)". Which later section consumes the 'ke1' row
    is not settled here -- section 6) overwrites the whole array with the circulation-kinetic-
    energy flux density -- but the asymmetry against 'frm' is deliberate, not an oversight, and
    has to survive translation.

    Args:
        buoyancy_factor_tet_l: Buoyancy factor of the liquid-water potential temperature,
            'zaux(:,:,4)' = 'g_tet' [m/s2 per K], half levels.
        buoyancy_factor_h2o_g: Buoyancy factor of the total water content, 'zaux(:,:,5)' =
            'g_h2o' [m/s2 per (kg/kg)], half levels.
        vertical_gradient_tet_l: Vertical gradient of the liquid-water potential temperature,
            'zvari(:,:,tet_l)' [K/m], half levels.
        vertical_gradient_h2o_g: Vertical gradient of the total water content,
            'zvari(:,:,h2o_g)' [(kg/kg)/m], half levels.
        thermal_forcing: Output, 'frh' [1/s2], half levels; an acceleration [m/s2] at 'ke1'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. Level 0 is never written.
        vertical_end: End of the half levels; 'ke1', mirroring Fortran 'k=...,ke1'.
    """
    _compute_thermal_forcing(
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        vertical_gradient_tet_l=vertical_gradient_tet_l,
        vertical_gradient_h2o_g=vertical_gradient_h2o_g,
        out=thermal_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
