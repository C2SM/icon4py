# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Lower limits of the vertical diffusion coefficients, the 'effective' coefficients.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
4), :1918-1995 at icon commit 26d6b98cce. The scientific commentary in that file is by Matthias
Raschendorfer (DWD), the tuning blocks by Guenther Zaengl (DWD), who signs them '!GZ:'.

The section is the whole of Raschendorfer's 4). What is left out of this stencil is the three
blocks the operational configuration does not reach, each guarded by a flag this port freezes:

    :1943  IF (imode_tkemini == 3)   adapt 'tke' by SQRT of the amplification of 'tkvh'
    :1982  IF (ltkeadapt)            adapt 'tke' by the full amplification of 'tkvh'
    :1999  IF (lsrfshear)            turn 'tfm'/'tfh' into LLDC drag and shear factors
    :2015  IF (l3dturb)              load the horizontal diffusion coefficients

'ltkeadapt' is '(imode_tkemini == 2)' (:967) and 'lsrfshear' is
'(rsur_sher > 0 .OR. (imode_trancnf < 4 .AND. imode_suradap >= 1))' (:968), so 'TurbulenceConfig'
freezing 'imode_tkemini = 1', 'imode_suradap = 0' and 'l3dturb = False' and defaulting
'rsur_sher = 0' switches all four off. That is why section 4) writes 'tkvm' and 'tkvh' and
nothing else -- in particular NOT 'tke', which the two 'tke'-adaptation branches above would
otherwise modify. 'test_section_4_writes_only_the_two_diffusion_coefficients' measures it.

WHAT AN 'LLDC' IS. Raschendorfer's abbreviation, used throughout the section: a Lower Limit of a
Diffusion Coefficient. The scheme's own closure can produce arbitrarily small coefficients in
stable stratification, which decouples adjacent model levels; the limits imposed here stand in
for transport processes the 1D scheme does not describe -- unresolved gravity-wave momentum
transport in the stratosphere, and near-surface circulations lower down. They are tuning, and
Raschendorfer says so at :1970-1977: "this treatment degenerates the turbulence model in its
principal conception".

THE THREE FLOORS, in the order they are applied:

  * a constant floor, 'MAX(con_m, tkmmin)' for momentum and 'MAX(con_h, tkhmin)' for scalars.
    The molecular diffusivities only bind if the namelist sets a minimum below them, which no
    operational setup does. Both are scalars and are computed by the caller.
  * a Richardson-number dependent modification of that floor, proportional to '1/Ri**(2/3)'
    through 'xri' and to the height above ground, and reduced near the surface by the two
    'tkred_sfc' factors ICON's tuning supplies. This is the 'imode_tkvmini == 2' block.
  * a stratospheric enhancement above 12.5 km, widened in the tropics, which takes over from the
    other two by a 'MAX'.

THE 'fakt2' TRAP. ':1939' is two statements on one line:

    fakt1=tkred_sfc(i)*fakt1; fakt2=tkred_sfc_h(i)*fakt1

The second reads the 'fakt1' the first has just written, so the scalar reduction factor is
'tkred_sfc_h * tkred_sfc * fakt1', with BOTH reduction factors in it, and the 'fakt2' computed
at :1937 is dead once the 'MERGE' at :1938 has consumed it. Translating the line as two
independent products -- the obvious reading -- gives a scalar floor that is too large wherever
'tkred_sfc < 1', which is near the surface, which is where the floor binds.
"""

import gt4py.next as gtx
from gt4py.next import broadcast, maximum, minimum, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _minimum_diffusion_coefficients(
    height_of_half_levels: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number: fa.CellKField[wpfloat],
    roughness_length_times_gravity: fa.CellField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    surface_reduction_for_momentum: fa.CellField[wpfloat],
    surface_reduction_for_scalars: fa.CellField[wpfloat],
    tropics_mask: fa.CellField[wpfloat],
    inner_tropics_mask: fa.CellField[wpfloat],
    minimum_coefficient_for_momentum: wpfloat,
    minimum_coefficient_for_scalars: wpfloat,
    stratospheric_minimum_for_momentum: wpfloat,
    stratospheric_minimum_for_scalars: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """The lower limits 'val1' and 'val2' of the two diffusion coefficients [m2/s].

    Mirrors :1928-1962 statement by statement, with the two 'tke'-adaptation branches left out
    (see the module docstring). The Fortran reuses the names 'fakt1', 'fakt2' and 'fakt' for
    unrelated quantities; the names here follow the quantity, and the Fortran name is given in
    the comment of each group.

    Args:
        height_of_half_levels: 'hhl(:,k)' [m].
        surface_height: 'hhl(:,ke1)', the surface half level, pre-sliced by the caller because
            GT4Py offsets are relative and this is an absolute row [m].
        inverse_richardson_number: 'xri', '1/Ri**(2/3)' as section 2a) computed it [-].
        roughness_length_times_gravity: 'gz0' [m2/s2].
        pattern_length_scale: 'l_pat', the near-surface thermal pattern scale [m].
        surface_reduction_for_momentum: 'tkred_sfc' [-].
        surface_reduction_for_scalars: 'tkred_sfc_h' [-].
        tropics_mask: 'trop_mask', 1 in the tropics and 0 outside [-].
        inner_tropics_mask: 'innertrop_mask' [-].
        minimum_coefficient_for_momentum: 'vel1 = MAX(con_m, tkmmin)' [m2/s].
        minimum_coefficient_for_scalars: 'vel2 = MAX(con_h, tkhmin)' [m2/s].
        stratospheric_minimum_for_momentum: 'tkmmin_strat' [m2/s].
        stratospheric_minimum_for_scalars: 'tkhmin_strat' [m2/s].

    Returns:
        'val1' for momentum and 'val2' for scalars [m2/s].
    """
    # ':1936-:1938'. 'fakt1' over glaciers -- bare ice is smooth, so a small roughness length
    # together with a resolved surface pattern is what identifies one -- and 'fakt2' elsewhere,
    # both growing with the height above ground. The two branches of the Fortran 'MERGE'.
    height_above_ground = height_of_half_levels - surface_height
    glacier = (roughness_length_times_gravity < wpfloat("0.01")) & (
        pattern_length_scale > wpfloat("0.0")
    )
    height_factor = where(
        broadcast(glacier, (dims.CellDim, dims.KDim)),
        wpfloat("4.0e-3") * height_above_ground,
        wpfloat("0.25") + wpfloat("7.5e-3") * height_above_ground,
    )

    # ':1939'. Both reductions accumulate into the scalar factor; see "THE 'fakt2' TRAP" above.
    momentum_factor = surface_reduction_for_momentum * height_factor
    scalar_factor = surface_reduction_for_scalars * momentum_factor

    # ':1940-:1941'. The floors follow '1/Ri**(2/3)', clipped to [0.01, 2.5] of themselves.
    limit_for_momentum = minimum_coefficient_for_momentum * minimum(
        wpfloat("2.5"),
        maximum(
            wpfloat("0.01"),
            minimum(wpfloat("1.0"), momentum_factor) * inverse_richardson_number,
        ),
    )
    limit_for_scalars = minimum_coefficient_for_scalars * minimum(
        wpfloat("2.5"),
        maximum(
            wpfloat("0.01"),
            minimum(wpfloat("1.0"), scalar_factor) * inverse_richardson_number,
        ),
    )

    # ':1956'. 'fakt', ramping in linearly between 12.5 and 17.5 km.
    stratospheric_ramp = minimum(
        wpfloat("1.0"),
        wpfloat("2.0e-4") * maximum(wpfloat("0.0"), height_of_half_levels - wpfloat("12500.0")),
    )
    # ':1958-:1959'. 'x4' and 'x4i', which widen the transition zone towards the tropopause and
    # weaken the enhancement there; 'z1d3' and 'z2d3' are 'z1/z3' and 'z2/z3' (:269-:270), so
    # they are the correctly rounded thirds and not a product with a reciprocal.
    tropical_weakening = wpfloat("1.0") - (
        wpfloat("1.0") / wpfloat("3.0")
    ) * tropics_mask * minimum(
        wpfloat("1.0"),
        wpfloat("2.0e-4") * maximum(wpfloat("0.0"), wpfloat("22500.0") - height_of_half_levels),
    )
    inner_tropical_weakening = wpfloat("1.0") - (
        wpfloat("2.0") / wpfloat("3.0")
    ) * inner_tropics_mask * minimum(
        wpfloat("1.0"),
        wpfloat("2.0e-4") * maximum(wpfloat("0.0"), wpfloat("27500.0") - height_of_half_levels),
    )
    # ':1960-:1961'. The stability dependence enters as SQRT(xri), clipped from below at 0.25 and
    # from above at '1.5*x4'.
    stratospheric_ramp = stratospheric_ramp * minimum(
        tropical_weakening * wpfloat("1.5"),
        maximum(wpfloat("0.25"), sqrt(inverse_richardson_number)),
    )

    # ':1962'. The stratospheric enhancement takes over from the Richardson-number floor.
    limit_for_momentum = maximum(
        limit_for_momentum,
        stratospheric_minimum_for_momentum
        * minimum(tropical_weakening, inner_tropical_weakening)
        * stratospheric_ramp,
    )
    limit_for_scalars = maximum(
        limit_for_scalars,
        stratospheric_minimum_for_scalars * tropical_weakening * stratospheric_ramp,
    )
    return limit_for_momentum, limit_for_scalars


@gtx.field_operator
def _compute_effective_diffusion_coefficients(
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    height_of_half_levels: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number: fa.CellKField[wpfloat],
    roughness_length_times_gravity: fa.CellField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    surface_reduction_for_momentum: fa.CellField[wpfloat],
    surface_reduction_for_scalars: fa.CellField[wpfloat],
    tropics_mask: fa.CellField[wpfloat],
    inner_tropics_mask: fa.CellField[wpfloat],
    minimum_coefficient_for_momentum: wpfloat,
    minimum_coefficient_for_scalars: wpfloat,
    stratospheric_minimum_for_momentum: wpfloat,
    stratospheric_minimum_for_scalars: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """'tkvm' and 'tkvh' raised to their lower limits, ':1990-:1991'."""
    limit_for_momentum, limit_for_scalars = _minimum_diffusion_coefficients(
        height_of_half_levels=height_of_half_levels,
        surface_height=surface_height,
        inverse_richardson_number=inverse_richardson_number,
        roughness_length_times_gravity=roughness_length_times_gravity,
        pattern_length_scale=pattern_length_scale,
        surface_reduction_for_momentum=surface_reduction_for_momentum,
        surface_reduction_for_scalars=surface_reduction_for_scalars,
        tropics_mask=tropics_mask,
        inner_tropics_mask=inner_tropics_mask,
        minimum_coefficient_for_momentum=minimum_coefficient_for_momentum,
        minimum_coefficient_for_scalars=minimum_coefficient_for_scalars,
        stratospheric_minimum_for_momentum=stratospheric_minimum_for_momentum,
        stratospheric_minimum_for_scalars=stratospheric_minimum_for_scalars,
    )
    return (
        maximum(limit_for_momentum, diffusion_coefficient_for_momentum),
        maximum(limit_for_scalars, diffusion_coefficient_for_scalars),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_effective_diffusion_coefficients(
    diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    height_of_half_levels: fa.CellKField[wpfloat],
    surface_height: fa.CellField[wpfloat],
    inverse_richardson_number: fa.CellKField[wpfloat],
    roughness_length_times_gravity: fa.CellField[wpfloat],
    pattern_length_scale: fa.CellField[wpfloat],
    surface_reduction_for_momentum: fa.CellField[wpfloat],
    surface_reduction_for_scalars: fa.CellField[wpfloat],
    tropics_mask: fa.CellField[wpfloat],
    inner_tropics_mask: fa.CellField[wpfloat],
    minimum_coefficient_for_momentum: wpfloat,
    minimum_coefficient_for_scalars: wpfloat,
    stratospheric_minimum_for_momentum: wpfloat,
    stratospheric_minimum_for_scalars: wpfloat,
    effective_diffusion_coefficient_for_momentum: fa.CellKField[wpfloat],
    effective_diffusion_coefficient_for_scalars: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Apply the lower limits to both diffusion coefficients on half levels 2..ke.

    The vertical domain is the Fortran 'DO k=2, ke': half levels 1 to 'ke' - 1 zero-based. Half
    level 'ke1' carries the surface coefficients 'turbtran' produced and is not touched, and row
    0 is not written by 'turbdiff' anywhere.

    That domain is also what keeps 'inverse_richardson_number' in bounds. 'xri' is declared
    '(nvec,ke)' (:802), one row shorter than every other half-level array here, and section 2a)
    fills exactly rows 1..ke-1 of it (:1403-:1409); its row 0 is undefined memory.

    Args:
        diffusion_coefficient_for_momentum: 'tkvm' as section 3) left it [m2/s].
        diffusion_coefficient_for_scalars: 'tkvh' as section 3) left it [m2/s].
        height_of_half_levels: 'hhl' [m].
        surface_height: 'hhl(:,ke1)' [m].
        inverse_richardson_number: 'xri' [-].
        roughness_length_times_gravity: 'gz0' [m2/s2].
        pattern_length_scale: 'l_pat' [m].
        surface_reduction_for_momentum: 'tkred_sfc' [-].
        surface_reduction_for_scalars: 'tkred_sfc_h' [-].
        tropics_mask: 'trop_mask' [-].
        inner_tropics_mask: 'innertrop_mask' [-].
        minimum_coefficient_for_momentum: 'vel1' [m2/s].
        minimum_coefficient_for_scalars: 'vel2' [m2/s].
        stratospheric_minimum_for_momentum: 'tkmmin_strat' [m2/s].
        stratospheric_minimum_for_scalars: 'tkhmin_strat' [m2/s].
        effective_diffusion_coefficient_for_momentum: Output, 'tkvm' [m2/s].
        effective_diffusion_coefficient_for_scalars: Output, 'tkvh' [m2/s].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k = 2'.
        vertical_end: End of the half levels; 'ke'.
    """
    _compute_effective_diffusion_coefficients(
        diffusion_coefficient_for_momentum=diffusion_coefficient_for_momentum,
        diffusion_coefficient_for_scalars=diffusion_coefficient_for_scalars,
        height_of_half_levels=height_of_half_levels,
        surface_height=surface_height,
        inverse_richardson_number=inverse_richardson_number,
        roughness_length_times_gravity=roughness_length_times_gravity,
        pattern_length_scale=pattern_length_scale,
        surface_reduction_for_momentum=surface_reduction_for_momentum,
        surface_reduction_for_scalars=surface_reduction_for_scalars,
        tropics_mask=tropics_mask,
        inner_tropics_mask=inner_tropics_mask,
        minimum_coefficient_for_momentum=minimum_coefficient_for_momentum,
        minimum_coefficient_for_scalars=minimum_coefficient_for_scalars,
        stratospheric_minimum_for_momentum=stratospheric_minimum_for_momentum,
        stratospheric_minimum_for_scalars=stratospheric_minimum_for_scalars,
        out=(
            effective_diffusion_coefficient_for_momentum,
            effective_diffusion_coefficient_for_scalars,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
