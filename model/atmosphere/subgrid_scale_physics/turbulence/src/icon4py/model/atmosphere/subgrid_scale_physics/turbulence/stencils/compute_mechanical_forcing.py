# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanical forcing of the TKE equation: the single-column shear production term.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads "1b) Calculation of the basic single-column forcing functions for
TKE" (:1209-1240 at icon commit 26d6b98cce, which is the commit that produced the reference
capture; the section is :1196-1229 in the uninstrumented upstream file). The scientific
commentary in that file is by Matthias Raschendorfer (DWD).

The Fortran is 'frm(i,k) = MAX( zvari(i,k,u_m)**2 + zvari(i,k,v_m)**2, fc_min(i) )', the
squared vertical shear of the horizontal wind at the mass centre, floored by 'fc_min'. The two
gradients are produced by section 1a); 'fc_min' comes from 'turb_setup'.

This is only the *pure single-column* part, as the Fortran's own sub-heading says. Section 2a)
adds the three-dimensional shear complements, the separated horizontal shear mode, the SSO wake
production and the convective circulation on top of the same storage.
"""

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_mechanical_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Shear production of turbulent kinetic energy, floored by the minimal forcing."""
    return maximum(vertical_gradient_u**2 + vertical_gradient_v**2, min_forcing)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_mechanical_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
    mechanical_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the mechanical (shear) forcing 'frm' of the TKE equation on half levels.

    The floor 'fc_min = (vel_min / MAX(l_hori, tur_len))**2' is set once per column in
    'turb_setup' (turb_utilities.f90:338) and is the shear that a velocity scale of 'vel_min'
    over the effective horizontal length scale would produce. It is a physical floor on the
    forcing, not a division guard: nothing here divides by 'frm'. Raschendorfer records having
    tested its removal, at turb_utilities.f90:337 immediately above the assignment, alongside
    the commented-out alternative 'fc_min(i)=z0':

        "test: frm ohne fc_min-Beschraenkung: Bewirkt Unterschiede!"
        -- "test: 'frm' without the 'fc_min' restriction: causes differences!"

    which is why the floor is kept rather than dropped as a numerical nicety.

    The vertical range stops one half level above the thermal forcing, at 'kem = ke'
    (turb_diffusion.f90:843, "lowest model-layer, SUB 'turbdiff' is applied to"). No statement
    anywhere in 'turbdiff' writes 'frm(:,ke1)', so that row keeps whatever the routine-local
    array was allocated with. It is in particular NOT the surface shear that 'turbtran'
    computes with the same expression at turb_transfer.f90:954: that is a different local array
    of the same name in a different routine.

    Args:
        vertical_gradient_u: Vertical gradient of the zonal wind at the mass centre,
            'zvari(:,:,u_m)' [1/s], half levels.
        vertical_gradient_v: Vertical gradient of the meridional wind at the mass centre,
            'zvari(:,:,v_m)' [1/s], half levels.
        min_forcing: Lower limit of the TKE forcing, 'fc_min' [1/s2], one value per column.
        mechanical_forcing: Output, 'frm' [1/s2], half levels.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'. Level 0 is never written.
        vertical_end: End of the half levels; 'ke', mirroring Fortran 'k=...,kem' with
            'kem = ke'. The surface half level 'ke1' is not written by this section.
    """
    _compute_mechanical_forcing(
        vertical_gradient_u=vertical_gradient_u,
        vertical_gradient_v=vertical_gradient_v,
        min_forcing=min_forcing,
        out=mechanical_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
