# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Mechanical forcing of the TKE equation by the full three-dimensional mean-flow shear.

Translated from 'icon/src/atm_phy_schemes/turb_diffusion.f90', SUBROUTINE 'turbdiff', section
2a) ("Adding 3D-complements of mechanical shear-forcing by the mean flow and all shear-forcing
of the non-turbulent sub-grid flow", :1323-1623 at icon commit 26d6b98cce, the commit that
produced the reference capture). The scientific commentary in that file is by Matthias
Raschendorfer (DWD).

Two consecutive loops, both writing 'frm':

    :1338  frm(i,k) = MAX( (zvari(i,k,u_m)+dwdx(i,k))**2 + (zvari(i,k,v_m)+dwdy(i,k))**2
                          + z3*hdiv(i,k)**2, fc_min(i) )        ! itype_sher == 2
    :1368  frm(i,k) = frm(i,k) + hdef2(i,k)                     ! itype_sher >= 1

They are two loops because they are guarded separately, not because anything separates them
numerically, and at 'itype_sher = 2' -- which is what the reference capture ran, established
from the data by 'test_the_capture_runs_the_full_three_dimensional_shear' -- both apply.

WHAT THIS OUTPUT IS, AND WHY IT IS NOT 'frm'. Section 2a) adds three further contributions to
the same storage before its exit savepoint, so the value written here never reaches a
savepoint of its own. It is not a nameless intermediate, though: it is exactly the quantity
Raschendorfer saves into 'ftm' at :1382, "save traditional (pure mean) shear", in the
configurations where the scale-interaction terms have to be kept separable
('lssintact .OR. loutbms'). Neither holds in this capture, so 'ftm' is untouched here and this
port carries the mean shear in a field of its own rather than accumulating into 'frm' in
place. Its oracle is indirect but tight: 'xri' is a strictly monotone function of it and is
compared bit-for-bit.

THE FOUR 'vp' FIELDS. 'hdef2', 'hdiv', 'dwdx' and 'dwdy' are declared 'REAL(KIND=vp)'
(turb_diffusion.f90:612-618) and come from the dycore's diffusion. This section is the only
consumer of them in the whole of 'turbdiff'. They are typed 'wpfloat' here, as the savepoint
reader returns them: the default double build makes 'vp' equal to 'wp'. A mixed-precision
build would need an 'astype' at the call site, which is the granule driver's business and not
this stencil's.
"""

import gt4py.next as gtx
from gt4py.next import maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_three_dimensional_shear_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    dwdx: fa.CellKField[wpfloat],
    dwdy: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """The squared shear of the mean flow, floored by 'fc_min' and extended by the deformation.

    Section 1b) formed the single-column part of this, 'MAX(du/dz**2 + dv/dz**2, fc_min)'. Two
    corrections turn it into the three-dimensional one, and the Fortran's own comment block at
    :1345-1355 says what they are:

      * the vertical wind contributes to the horizontal-momentum shear, so its horizontal
        derivatives are added to the vertical derivatives of the horizontal wind, component by
        component. Incompressibility ('vel_div = hdiv + dw/dz = 0', :1351) turns the remaining
        diagonal term into '3 * hdiv**2';
      * 'hdef2', the squared horizontal deformation
        '(d1v2+d2v1)**2 + (d1v1-d2v2)**2' (:1345), is the purely horizontal shear.

    The floor is applied to the first group only, before 'hdef2' is added -- which is what the
    two separate Fortran loops encode and is the reason they are reproduced in this order and
    not fused into a single 'MAX'.

    THE SQUARES ARE WRITTEN AS PRODUCTS. Fortran's 'x**2' with an integer literal exponent is a
    multiplication, while GT4Py lowers Python's 'x**2' to 'math.pow'; CUDA's 'pow' carries up to
    2 ulp, so on a GPU backend the two are different numbers. See the docstring of
    '_compute_mechanical_forcing', where this was measured.
    """
    shear_u = vertical_gradient_u + dwdx
    shear_v = vertical_gradient_v + dwdy
    single_column_and_vertical_wind_shear = maximum(
        shear_u * shear_u
        + shear_v * shear_v
        + wpfloat("3.0") * (horizontal_divergence * horizontal_divergence),
        min_forcing,
    )
    return single_column_and_vertical_wind_shear + horizontal_deformation_square


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_three_dimensional_shear_forcing(
    vertical_gradient_u: fa.CellKField[wpfloat],
    vertical_gradient_v: fa.CellKField[wpfloat],
    dwdx: fa.CellKField[wpfloat],
    dwdy: fa.CellKField[wpfloat],
    horizontal_divergence: fa.CellKField[wpfloat],
    horizontal_deformation_square: fa.CellKField[wpfloat],
    min_forcing: fa.CellField[wpfloat],
    mean_shear_forcing: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Compute the mechanical forcing by the three-dimensional mean shear, on half levels.

    ONLY 'itype_sher = 2' IS IMPLEMENTED HERE. The Fortran selects among three formulations
    (:1353-1355): 0 is the single-column vertical shear alone, which section 1b) already
    computed and which this program would then not run at all; 1 adds 'hdef2'; 2 adds the
    vertical-wind terms as well. The reference capture ran 2 -- measured, not read off a
    namelist -- so 0 and 1 have no oracle in this data and are not written out as separate
    branches. A run that needs them has to add them together with the reference data that
    validates them.

    Args:
        vertical_gradient_u: Vertical gradient of the zonal wind at the mass centre,
            'zvari(:,:,u_m)' [1/s], half levels.
        vertical_gradient_v: Vertical gradient of the meridional wind, 'zvari(:,:,v_m)' [1/s].
        dwdx: Zonal derivative of the vertical wind, 'dwdx' [1/s], half levels.
        dwdy: Meridional derivative of the vertical wind, 'dwdy' [1/s], half levels.
        horizontal_divergence: Horizontal wind divergence, 'hdiv' [1/s], half levels.
        horizontal_deformation_square: Squared horizontal deformation, 'hdef2' [1/s2], half
            levels.
        min_forcing: Lower limit of the TKE forcing, 'fc_min' [1/s2], one value per column.
        mean_shear_forcing: Output [1/s2], half levels; the Fortran's 'frm' before the
            non-turbulent contributions are added, i.e. its 'ftm'.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: First half level; 1, mirroring Fortran 'k=2'.
        vertical_end: End of the half levels; 'ke', mirroring Fortran 'k=...,kem' with
            'kem = ke'. The surface half level 'ke1' is not written by this section.
    """
    _compute_three_dimensional_shear_forcing(
        vertical_gradient_u=vertical_gradient_u,
        vertical_gradient_v=vertical_gradient_v,
        dwdx=dwdx,
        dwdy=dwdy,
        horizontal_divergence=horizontal_divergence,
        horizontal_deformation_square=horizontal_deformation_square,
        min_forcing=min_forcing,
        out=mean_shear_forcing,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
