# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencil selection of the hybrid quadratic scheme (ihadv_tracer=132).

Port of mo_advection_hflux.f90 3547-3574 (upwind_hflux_miura_weno_hyb): per cell and
level, the residual of the full 9-point quadratic fit

    lsqe = sum_is (lsq_error(1:5, is) . c(2:6) - z_b(is))**2

is compared with ``5e-5 * (p_cc + 1e-10)**2`` (paper eq. 6 with max(q) = the cell's own
value); at or below the threshold the plain quadratic reconstruction is used, above it
the 27-candidate WENO blend. ``lsq_error`` is ICON's name for the transposed, distance-
weighted design matrix of the fit, stored in single precision; the residual is
accumulated in single precision too, with the coefficients and z_b rounded to single
precision first (``REAL(sp) :: zlc, lsqe``; ``real(z_b(is))``). Only the comparison is
in double precision: the single-precision literals are promoted, so the threshold and
epsilon passed in must be the double values of the single-precision constants.

Rows: the 9 stencil positions are split over the C2E2C and C2E2C2E2C offsets as the
pseudoinverse is (weno_least_squares.scatter_to_offsets); ``lsq_butterfly_active`` (int32)
is 1 on the six claimed butterfly slots and 0 on the three that hold the centre cell or a
duplicated direct neighbour, whose z_b must not enter the residual. The summation runs
over the direct slots then the butterfly slots, not in the Fortran's stencil order.
"""

import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C, C2E2C2E2C, C2E2C2E2CDim, C2E2CDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_weno_hybrid_stencil_selection(
    p_cc: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    lsq_error_direct_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_butterfly_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_butterfly_active: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.int32],
    selection_threshold: ta.wpfloat,
    selection_eps: ta.wpfloat,
) -> fa.CellKField[bool]:
    # f90 3564: zlc(1:5) = z_lsq_coeff(2:6), single precision
    zlc_1 = astype(p_coeff_2, gtx.float32)
    zlc_2 = astype(p_coeff_3, gtx.float32)
    zlc_3 = astype(p_coeff_4, gtx.float32)
    zlc_4 = astype(p_coeff_5, gtx.float32)
    zlc_5 = astype(p_coeff_6, gtx.float32)

    # f90 3547-3549: z_b = p_cc(stencil cell) - p_cc(center), rounded as real(z_b(is));
    # the padding slots of the butterfly offset are masked out
    zb_direct = astype(p_cc(C2E2C) - p_cc, gtx.float32)
    zb_butterfly = astype(p_cc(C2E2C2E2C) - p_cc, gtx.float32) * astype(
        lsq_butterfly_active, gtx.float32
    )

    # f90 3565-3568: DOT_PRODUCT(lsq_error(1:5, is), zlc(1:5)) - real(z_b(is)), squared and
    # summed over the stencil rows
    residual_direct = (
        lsq_error_direct_1 * zlc_1
        + lsq_error_direct_2 * zlc_2
        + lsq_error_direct_3 * zlc_3
        + lsq_error_direct_4 * zlc_4
        + lsq_error_direct_5 * zlc_5
    ) - zb_direct
    residual_butterfly = (
        lsq_error_butterfly_1 * zlc_1
        + lsq_error_butterfly_2 * zlc_2
        + lsq_error_butterfly_3 * zlc_3
        + lsq_error_butterfly_4 * zlc_4
        + lsq_error_butterfly_5 * zlc_5
    ) - zb_butterfly
    lsqe = neighbor_sum(residual_direct * residual_direct, axis=C2E2CDim) + neighbor_sum(
        residual_butterfly * residual_butterfly, axis=C2E2C2E2CDim
    )

    # f90 3574: if (lsqe .le. 5e-5 * (p_cc + 1e-10)**2) -> plain quadratic, else WENO
    threshold = selection_threshold * ((p_cc + selection_eps) * (p_cc + selection_eps))
    use_weno = astype(lsqe, wpfloat) > threshold
    return use_weno


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_weno_hybrid_stencil_selection(
    p_cc: fa.CellKField[ta.wpfloat],
    p_coeff_2: fa.CellKField[ta.wpfloat],
    p_coeff_3: fa.CellKField[ta.wpfloat],
    p_coeff_4: fa.CellKField[ta.wpfloat],
    p_coeff_5: fa.CellKField[ta.wpfloat],
    p_coeff_6: fa.CellKField[ta.wpfloat],
    lsq_error_direct_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_direct_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.float32],
    lsq_error_butterfly_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_error_butterfly_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.float32],
    lsq_butterfly_active: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], gtx.int32],
    use_weno: fa.CellKField[bool],
    selection_threshold: ta.wpfloat,
    selection_eps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_weno_hybrid_stencil_selection(
        p_cc=p_cc,
        p_coeff_2=p_coeff_2,
        p_coeff_3=p_coeff_3,
        p_coeff_4=p_coeff_4,
        p_coeff_5=p_coeff_5,
        p_coeff_6=p_coeff_6,
        lsq_error_direct_1=lsq_error_direct_1,
        lsq_error_direct_2=lsq_error_direct_2,
        lsq_error_direct_3=lsq_error_direct_3,
        lsq_error_direct_4=lsq_error_direct_4,
        lsq_error_direct_5=lsq_error_direct_5,
        lsq_error_butterfly_1=lsq_error_butterfly_1,
        lsq_error_butterfly_2=lsq_error_butterfly_2,
        lsq_error_butterfly_3=lsq_error_butterfly_3,
        lsq_error_butterfly_4=lsq_error_butterfly_4,
        lsq_error_butterfly_5=lsq_error_butterfly_5,
        lsq_butterfly_active=lsq_butterfly_active,
        selection_threshold=selection_threshold,
        selection_eps=selection_eps,
        out=use_weno,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
