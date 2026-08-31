# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Vertical smoothing of a TKE forcing term, ICON's SUBROUTINE 'vert_smooth'.

Translated from 'icon/src/atm_phy_schemes/turb_utilities.f90:3098-3227' at icon commit
26d6b98cce, as called by 'turbdiff' section 2c) at turb_diffusion.f90:1720-1738 on the
mechanical forcing 'frm' and then on the thermal forcing 'frh'.

NOT VALIDATED AGAINST ICON DATA. Every other stencil of this package is bit-exact against a
serialized ICON run; this one is not, and cannot be made so from the capture in use. The call
is guarded by

    IF (tdc%frcsmot > z0) THEN
      luse_mask = (tdc%imode_frcsmot == 2 .AND. .NOT.lini)
      IF (luse_mask) lcond = ANY(trop_mask(ivstart:ivend) > z0)

and 'trop_mask' is identically zero at all 8276 computed columns of the Swiss LAM domain of
'exp.mch_icon-ch2_small'. So no capture from that experiment exercises 'vert_smooth' at ANY
'frcsmot': the guard is false, and even if it were forced true the smoothing weight
'versmot = frcsmot*trop_mask' would be zero and the routine the identity. What stands behind
this file is a numpy transcription of the Fortran and a set of structural properties
('integration_tests/test_turbdiff_section_2c.py', the 'vert_smooth' section), not ICON output.
It is ported anyway because 28 top-level 'exp.*' configurations set 'frcsmot = 0.2' -- 13 MCH
operational setups and 7 DWD NWP ones among them -- so refusing it locks out configurations ICON
operates. (0.2 is NOT the Fortran default: 'mo_turbdiff_config.f90:143' defaults 'frcsmot' to
0.0, which is what the capture ran.) A first tropical or global capture should be used to
validate it before it is trusted.

IT IS NOT A RECURRENCE, despite the '!$ACC LOOP SEQ' over 'k'. The Fortran rotates two saved
columns, 'sav_tend(:,j1)' and 'sav_tend(:,j2)', and writes 'cur_tend' in place:

    k = k_tp+1:  sav_tend(i,j1) = cur_tend(i,k)              ! j1=1, j2=2
                 cur_tend(i,k)  = remfact*cur_tend(i,k) + versmot*cur_tend(i,k+1)*dm(k+1)/dm(k)
    k = k_tp+2 .. k_sf-2:
                 j0=j1; j1=j2; j2=j0                          ! swap, so j2 is the PREVIOUS j1
                 sav_tend(i,j1) = cur_tend(i,k)               ! the value BEFORE it is smoothed
                 cur_tend(i,k)  = remfact*cur_tend(i,k)
                                + versmot*(sav_tend(i,j2)*dm(k-1) + cur_tend(i,k+1)*dm(k+1))/dm(k)
    k = k_sf-1:  j2=j1
                 cur_tend(i,k)  = remfact*cur_tend(i,k) + versmot*sav_tend(i,j2)*dm(k-1)/dm(k)

Tracing the rotation by hand: at level 'k' the slot 'j2' holds what 'cur_tend(i,k-1)' was
before level 'k-1' was overwritten. The saved column exists for exactly that reason -- to undo
the in-place write -- so every output row is a function of INPUT rows only. This is a plain
three-point stencil over 'Koff[-1]' and 'Koff[1]', a 'concat_where' and not a 'scan_operator'
(port spec 3.2, and the package README's "Vertical recurrences").

THE THREE ROWS. With the call's 'k_tp = 1' and 'k_sf = ke1', on zero-based half levels
'0..nlev' and writing 's' for 'versmot' and 'dm' for 'disc_mom':

    row 0            untouched -- the Fortran writes 'k_tp+1' upwards
    row 1            (1-s) *f(1)     + s*f(2)*dm(2)/dm(1)
    rows 2..nlev-2   (1-2s)*f(k)     + s*(f(k-1)*dm(k-1) + f(k+1)*dm(k+1))/dm(k)
    row nlev-1       (1-s) *f(nlev-1)+ s*f(nlev-2)*dm(nlev-2)/dm(nlev-1)
    row nlev         untouched -- the surface half level

The two ends carry '(1-s)' rather than '(1-2s)' and one neighbour rather than two, which is
what makes the smoothing conservative: 'sum_k out(k)*dm(k)' equals 'sum_k in(k)*dm(k)', because
the weight the missing neighbour would have taken is the one added back to the row itself.

THAT IS A STATEMENT ABOUT THE COLUMNS OF THE WEIGHT MATRIX, NOT ITS ROWS, and the difference is
not pedantic. Each neighbour weight carries the mass ratio 'dm(k')/dm(k)', so what is
redistributed is 'f*dm' and not 'f'. The weights of one ROW sum to
'1 - 2s + s*(dm(k-1) + dm(k+1))/dm(k)', which is one only where 'dm' is uniform -- so this
operator does NOT preserve a constant profile on a stretched grid, and on an idealized column it
moves one by up to 7 per cent in a single pass at 'frcsmot = 0.2'. It is a mass-conservative
redistribution, not an average. Measured and asserted level by level in
'analytic_tests/test_vertical_smoothing_weights.py'.

WHY THE UNTOUCHED ROWS ARE COPIED HERE. The Fortran smooths in place, so a row it does not
write keeps its value by construction. A GT4Py program computes its whole domain into a
separate output field, and this one cannot be run in place because it reads 'Koff[+-1]' of
what it writes. Rows 0 and 'nlev' are therefore copied through explicitly, so that the output
field is the complete profile 'turbdiff' would have had in 'frm'/'frh' -- section 3)'s
circulation acceleration reads row 'nlev' of the thermal forcing.

THE SMOOTHING WEIGHT IS PER COLUMN. 'versmot(i) = vertsmot*smotfac(i)' when 'smotfac' is
present and 'luse_mask' holds, which on the ported path it always does: 'imode_frcsmot' is
frozen at 2, the call passes 'smotfac = trop_mask', and 'lini' is false by construction (the
granule replaces 'mo_nwp_turbdiff_interface.f90:576', which passes 'iini = 0' as a literal).
The unmasked branch, 'versmot(i) = vertsmot', is what 'imode_frcsmot = 1' would take; it is
not ported, and 'TurbulenceConfig' freezes 'imode_frcsmot' at 2 for that reason.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _smooth_tke_forcing_vertically(
    tke_forcing: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    smoothing_mask: fa.CellField[wpfloat],
    smoothing_weight: wpfloat,
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """One smoothing pass over a forcing profile, both ends and the two untouched rows.

    The three expressions are written with the Fortran's own parenthesisation. That is not
    pedantry: 'a + b' and the order in which 'versmot', the neighbouring value and the two
    momenta are multiplied decide the rounding, and this is the only statement of what the
    Fortran computed that the port has -- there is no reference capture to fall back on.
    """
    smoothing = smoothing_weight * smoothing_mask
    remaining_at_an_end = wpfloat("1.0") - smoothing
    remaining_inside = wpfloat("1.0") - wpfloat("2.0") * smoothing

    at_the_top = (
        remaining_at_an_end * tke_forcing
        + smoothing
        * tke_forcing(Koff[1])
        * discretisation_momentum(Koff[1])
        / discretisation_momentum
    )
    inside = (
        remaining_inside * tke_forcing
        + smoothing
        * (
            tke_forcing(Koff[-1]) * discretisation_momentum(Koff[-1])
            + tke_forcing(Koff[1]) * discretisation_momentum(Koff[1])
        )
        / discretisation_momentum
    )
    at_the_bottom = (
        remaining_at_an_end * tke_forcing
        + smoothing
        * tke_forcing(Koff[-1])
        * discretisation_momentum(Koff[-1])
        / discretisation_momentum
    )

    # NESTED HALF-SPACES, NOT ONE EQUALITY PER SPECIAL ROW. Each 'concat_where' below splits
    # the interval its parent left it, so every branch is inferred over an INTERVAL and none of
    # them is evaluated on a row whose 'Koff[+-1]' neighbour lies outside the field. The rows
    # and the arithmetic are exactly those of the table in the module docstring: 0 and 'nlev'
    # are copied through, 1 takes the one-sided top form, 'nlev'-1 the one-sided bottom form,
    # and 2..'nlev'-2 the interior form.
    #
    # WRITTEN AS FOUR EQUALITIES THIS PROGRAM READ ONE ROW OFF EITHER END OF ITS INPUTS. The
    # complement of a point is not an interval -- the rows other than 1 are '[0,1)' together
    # with '(1,nlev]' -- so the domain inferred for the fallback branch could only be the whole
    # column, and the fused kernel of 'inside' then evaluated at row 0 and at row 'nlev',
    # loading 'tke_forcing' and 'discretisation_momentum' at rows -1 and 'nlev'+1. The
    # selection discarded those values, so the answer was right and every backend agreed with
    # the transcription -- but the loads happened. 'compute-sanitizer --tool memcheck' on
    # 'dace_gpu' reports them as "Invalid __global__ read of size 8 bytes ... is out of bounds",
    # thousands per launch. Whether an out-of-bounds load faults depends on what CuPy's memory
    # pool has mapped next to the array, which is why it surfaced as an INTERMITTENT
    # 'cudaErrorIllegalAddress' that killed the CUDA context and failed every later test in the
    # process. 'gtfn' happened not to fault on the same reads; that is luck, not safety.
    smoothed = concat_where(dims.KDim < nlev, at_the_bottom, tke_forcing)
    smoothed = concat_where(dims.KDim < nlev - 1, inside, smoothed)
    smoothed = concat_where(dims.KDim < 2, at_the_top, smoothed)
    return concat_where(dims.KDim < 1, tke_forcing, smoothed)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def smooth_tke_forcing_vertically(
    tke_forcing: fa.CellKField[wpfloat],
    discretisation_momentum: fa.CellKField[wpfloat],
    smoothing_mask: fa.CellField[wpfloat],
    smoothing_weight: wpfloat,
    nlev: gtx.int32,
    smoothed_tke_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Smooth one TKE forcing profile vertically, on half levels; ICON's 'vert_smooth'.

    UNVALIDATED AGAINST ICON DATA -- see the module docstring for why no capture from
    'exp.mch_icon-ch2_small' can exercise it, and what stands in place of a reference.

    Run it over the WHOLE column, 'vertical_start = 0' and 'vertical_end = nlev + 1': the two
    rows the Fortran leaves alone are copied through here, because the port cannot smooth in
    place and the output field has to be the complete profile.

    'smoothed_tke_forcing' must not be 'tke_forcing'. The program reads both neighbours of
    every interior row, so in place is a different computation -- which is exactly what the
    Fortran's 'sav_tend' exists to prevent.

    Args:
        tke_forcing: 'cur_tend' on entry, 'frm' or 'frh' [1/s2], half levels.
        discretisation_momentum: 'disc_mom', the 'dicke' storage as section 1a) leaves it,
            'rho_n*dz/dt' [kg/m2/s] on half levels. Defined on rows 1 to 'nlev' - 1, which are
            the only rows this program reads it on.
        smoothing_mask: 'smotfac', the tropics mask 'trop_mask' [-], one value per column.
        smoothing_weight: 'vertsmot', the namelist parameter 'frcsmot' [-].
        nlev: 'ke', the zero-based row of the surface half level; the row that is copied
            through, and the row whose predecessor takes the one-sided form.
        smoothed_tke_forcing: Output, the smoothed profile over the whole column.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: 0, the model top.
        vertical_end: 'nlev + 1'.
    """
    _smooth_tke_forcing_vertically(
        tke_forcing=tke_forcing,
        discretisation_momentum=discretisation_momentum,
        smoothing_mask=smoothing_mask,
        smoothing_weight=smoothing_weight,
        nlev=nlev,
        out=smoothed_tke_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
