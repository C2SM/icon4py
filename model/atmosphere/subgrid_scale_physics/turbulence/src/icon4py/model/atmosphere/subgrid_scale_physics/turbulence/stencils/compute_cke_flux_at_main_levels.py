# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The CKE flux density interpolated from half levels onto the flux (main) levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
6) (:2118-2310 at icon commit 26d6b98cce), from the loop Raschendorfer heads

    "Interpolation der skalierten CKE-Flussdichte auf Hauptflaechen:"
    -- "Interpolation of the scaled CKE flux density onto main levels"

at :2239-2249. The scientific commentary in that file is by Matthias Raschendorfer (DWD).
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_cke_flux_at_main_levels(
    cke_flux_density: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Linear interpolation of the scaled flux onto the main level between two half levels.

    The Fortran is

        frm(i,k) = (frh(i,k) + frh(i,k-1)) / (len_scale(i,k) + len_scale(i,k-1))

    which is the mean of the two scaled fluxes divided by the mean of the two length scales --
    the two halves cancel, which is why neither appears. The result is the UNSCALED flux
    density [kg/s3] at the flux level, so this operator both interpolates and undoes the
    scaling that 'compute_cke_flux_density' applied.

    THIS IS NOT A RECURRENCE. 'frh' is an input, written by a preceding program and never by
    this one; the Fortran writes into a different array, 'frm'. The Fortran loop is a plain
    '!$ACC LOOP GANG VECTOR COLLAPSE(2)'.
    """
    return (cke_flux_density + cke_flux_density(Koff[-1])) / (
        mixing_length + mixing_length(Koff[-1])
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cke_flux_at_main_levels(
    cke_flux_density: fa.CellKField[wpfloat],
    mixing_length: fa.CellKField[wpfloat],
    cke_flux_at_main_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Interpolate 'frh' onto the flux levels as 'frm' (turb_diffusion.f90:2241-2249).

    Section 8) divides this by 'expl_mom' to obtain the contribution of the circulation flux to
    the virtual TKE profile, so it carries the same staggering: index k is the flux level ABOVE
    half level k, i.e. main level k-1. See 'compute_explicit_tke_diffusion_momentum' for
    Raschendorfer's statement of that convention.

    THE STORAGE IS REUSED. 'frm' held the mechanical (shear) forcing of the TKE equation from
    section 1b) to section 5); this section overwrites it. The two are not related, and the
    savepoint reader gives them separate accessors ('mech_forcing()' and
    'cke_flux_at_main_levels()') for that reason.

    Args:
        cke_flux_density: 'frh' [kg m/s3], half levels, from 'compute_cke_flux_density'.
        mixing_length: 'len_scale', the turbulent master length scale [m], half levels -- the
            same field that scaled 'frh', divided out here.
        cke_flux_at_main_levels: Output, 'frm' [kg/s3], at the flux levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First flux level; 2, mirroring Fortran 'k=3'. Rows 0 and 1 are not
            written -- there is no flux level above the topmost half level that carries a
            defined 'frh' pair.
        vertical_end: End of the flux levels; 'ke1', mirroring Fortran 'k=...,ke1'.
    """
    _compute_cke_flux_at_main_levels(
        cke_flux_density=cke_flux_density,
        mixing_length=mixing_length,
        out=cke_flux_at_main_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
