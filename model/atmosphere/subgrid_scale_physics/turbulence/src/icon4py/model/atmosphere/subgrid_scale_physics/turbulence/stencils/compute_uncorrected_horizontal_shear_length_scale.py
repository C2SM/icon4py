# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The uncorrected length scale of the separated horizontal shear mode, one value per column.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a), :1421-1428 at icon commit 26d6b98cce, the commit that produced the reference capture. The
scientific commentary in that file is by Matthias Raschendorfer (DWD).

    wert = tdc%a_hshr*tdc%akt*z1d2
    layr(i) = wert*l_hori(i)          !uncorrected effective horizontal length scale

'l_hori' is the horizontal mesh size, so this is the mesh size scaled by the von Karman constant
and the tuning factor 'a_hshr': the size of the largest eddy the separated horizontal shear mode
can hold. 'compute_effective_horizontal_shear_length_scale' then corrects it by the local
stability and by the height above ground.

Nothing downstream of section 2a) reads it -- 'layr' is a Fortran scratch vector reused all over
'turbdiff' -- so it is here as a stencil of its own only because that is what the Fortran is,
and because the capture serializes it, which makes 'a_hshr' recoverable from the data. The
reference run used 'a_hshr = 2.0' and not the compiled-in 1.0;
'test_the_capture_used_a_horizontal_shear_factor_of_two' recovers it.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_uncorrected_horizontal_shear_length_scale(
    horizontal_mesh_size: fa.CellField[wpfloat],
    horizontal_shear_length_factor: wpfloat,
    karman_constant: wpfloat,
) -> fa.CellField[wpfloat]:
    """'a_hshr * akt / 2 * l_hori'.

    The parenthesisation follows the Fortran, which forms the scalar factor 'wert' once outside
    the loop and multiplies the field by it. Grouping the three scalars first is what keeps the
    result independent of the mesh size's own rounding.
    """
    return (
        horizontal_shear_length_factor * karman_constant * wpfloat("0.5")
    ) * horizontal_mesh_size


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_uncorrected_horizontal_shear_length_scale(
    horizontal_mesh_size: fa.CellField[wpfloat],
    horizontal_shear_length_factor: wpfloat,
    karman_constant: wpfloat,
    uncorrected_horizontal_shear_length_scale: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    """Compute 'layr' [m], one value per column.

    Args:
        horizontal_mesh_size: 'l_hori' [m]. ICON fills every entry of it with the single scalar
            'phy_params%mean_charlen', so it is constant over a domain, but it is a field.
        horizontal_shear_length_factor: 'a_hshr', the length-scale factor of the separated
            horizontal shear circulations [-]. Zero switches the whole mode off, which the
            caller expresses by not running any of it.
        karman_constant: 'akt' [-].
        uncorrected_horizontal_shear_length_scale: Output, 'layr' [m].
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
    """
    _compute_uncorrected_horizontal_shear_length_scale(
        horizontal_mesh_size=horizontal_mesh_size,
        horizontal_shear_length_factor=horizontal_shear_length_factor,
        karman_constant=karman_constant,
        out=uncorrected_horizontal_shear_length_scale,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
