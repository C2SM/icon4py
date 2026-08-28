# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Interpolation of the super-saturation standard deviation back to main levels.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', from the
section whose banner reads

    "11) Interpolationen auf Hauptflaechen fuer die Standardabweichnung des
     Saettigungsdefizites"
    -- "11) Interpolations to main levels for the standard deviation of the saturation deficit"

(:2559-2589 at icon commit 26d6b98cce, the commit that produced the reference capture; the
'#ifdef __INTEL_COMPILER' FORALL at :2572-2574 is the same statement written twice, not a
second case). The scientific commentary in that file is by Matthias Raschendorfer (DWD).
"Standardabweichnung" is a typo for "Standardabweichung" in the original.

This is the last thing 'turbdiff' does, and it is a staggering correction rather than physics.
The declaration of the dummy argument (:602-607) states the contract the caller sees:

    rcld  ! standard deviation of local super-saturation (SDSS)
          !  at MAIN levels including the lower boundary  (---)
          ! AUX: cloud-cover at main levels (as output of SUB 'adjust_satur_equil'
          !        and later at half levels (as output of SUB 'bound_level_interp'
          !                                  and input of SUB 'solve_turb_budgets')

so 'rcld' enters and leaves 'turbdiff' on main levels, and is on half levels only for the
scheme's own duration. Section 3) put SDSS there ('solve_turb_budgets' computes it on the half
levels where the turbulent budgets live); this section puts it back. What consumes it is the
statistical cloud scheme, which the routine's own preamble (:356-359) introduces as

    "Angeschlossen ist auch ein optionales statistisches Wolkenschema (nach Sommeria und
     Deardorff), SUB 'turb_cloud', welches auch subskalige Bewoelkung mit Hilfe der ueber das
     Feld 'rcld' ausgegebenen Standardabweichung des Saettigungsdefizites (SDSS) berechnet."
    -- "An optional statistical cloud scheme (after Sommeria and Deardorff), SUB 'turb_cloud',
        is also attached, which computes subgrid cloudiness too, using the standard deviation of
        the saturation deficit (SDSS) exported in the field 'rcld'."
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_supersaturation_deviation_to_main_levels(
    supersaturation_deviation_on_half_levels: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """Average each pair of neighbouring half levels onto the main level between them.

    Main level 'k' lies between half levels 'k' and 'k+1', so the interpolation is the plain
    two-point mean

        rcld(i,k) = (rcld(i,k) + rcld(i,k+1)) * z1d2        ! k = 2 .. kem-1

    with 'z1d2 = z1/z2', a PARAMETER equal to one half.

    THE MODEL TOP IS A COPY, NOT A MEAN, and the reason is that the half level above it holds no
    SDSS at all. 'solve_turb_budgets' is called for 'k_st = 2', so the topmost half level keeps
    what section 0) left there -- measured over all four serialized timesteps, exactly zero --
    and averaging into it would halve the topmost main-level value instead of extrapolating it.
    Raschendorfer therefore writes the top row as

        rcld(i,1) = rcld(i,2)

    in a loop of its own, ahead of the averaging loop so that it still sees the un-averaged
    'rcld(i,2)'. Here the two are one program and the row selects its coefficients: the same two
    half levels are read on every row, weighted (0, 1) at the top and (1/2, 1/2) below it. See
    the package README, "Boundary rows", for when that merge is the right one. The topmost half
    level is consequently never read, which is what
    'test_the_model_top_is_a_copy_and_never_reads_the_half_level_above_it' measures.

    THE SEQUENTIAL FORTRAN LOOP IS NOT A RECURRENCE. The averaging loop is marked '!$ACC LOOP
    SEQ' and writes 'rcld(i,k)' while reading 'rcld(i,k+1)', but it sweeps upward -- k ascending,
    writing at 'k' -- so 'rcld(i,k+1)' is always still the input when row 'k' is written. Every
    output row is a function of input rows only. The aliasing is a Fortran storage economy and
    the sequential marker guards nothing; both vanish here, where input and output are separate
    fields (port spec 3.2, which lists this loop among the apparent k-recurrences that are
    ordinary 'Koff' stencils). Verified against the capture: an out-of-place evaluation
    reproduces 'turbdiff-exit' bit for bit.

    BIT-EXACTNESS IS FREE HERE. Multiplication by one half is exact in binary floating point, so
    the only rounding is the single addition, and there is no operand order left to get wrong:
    addition is commutative under IEEE 754. Nothing in this expression is sensitive to
    multiply-add contraction, to re-association, or to reciprocal substitution.
    """
    return concat_where(
        dims.KDim == 0,
        supersaturation_deviation_on_half_levels(Koff[1]),
        (
            supersaturation_deviation_on_half_levels
            + supersaturation_deviation_on_half_levels(Koff[1])
        )
        * wpfloat("0.5"),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_supersaturation_deviation_to_main_levels(
    supersaturation_deviation_on_half_levels: fa.CellKField[wpfloat],
    supersaturation_deviation_on_main_levels: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Interpolate 'rcld', the SDSS, from the half levels of the scheme back to main levels.

    THE TWO LOWEST ROWS ARE NOT WRITTEN, and the vertical domain is what says so. The Fortran
    loop stops at 'kem-1', with 'kem = ke' in 'turbdiff' (:843), and leaves a note where it
    stops:

        "Fuer die unterste Hauptflaeche (k=ke) wird bei kem=ke der Wert auf der entspr.
         Nebenflaeche beibehalten."
        -- "For the lowest main level (k=ke), when kem=ke, the value on the corresponding half
            level is retained."

    so the last main level keeps the half-level value it already holds rather than being averaged
    with the surface, and the surface row 'ke1' -- the lower boundary that the dummy argument's
    "including the lower boundary" refers to -- is not a main level and is left alone as well.
    Neither row is a second case to select, so neither appears in the field operator: run this
    with 'vertical_end = nlev - 1' and the domain states it (package README, "Boundary rows").

    Args:
        supersaturation_deviation_on_half_levels: 'rcld' as section 3) left it, the standard
            deviation of the local super-saturation on half levels [-]. Rows 1 to 'nlev' - 1
            are read; the model top is not, and neither is the surface.
        supersaturation_deviation_on_main_levels: Output, 'rcld' on main levels [-]. The Fortran
            writes this back into the input storage; here it is a separate field, which is what
            makes the claim that the sweep is not a recurrence testable.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First main level; 0, mirroring Fortran 'k = 1'.
        vertical_end: End of the written main levels; 'nlev' - 1, mirroring the Fortran loop's
            'k = 2, kem-1' with 'kem = ke'.
    """
    _interpolate_supersaturation_deviation_to_main_levels(
        supersaturation_deviation_on_half_levels=supersaturation_deviation_on_half_levels,
        out=supersaturation_deviation_on_main_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
