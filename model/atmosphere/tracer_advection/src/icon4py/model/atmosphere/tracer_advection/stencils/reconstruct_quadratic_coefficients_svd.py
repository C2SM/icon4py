# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C, C2E2C2E2C, C2E2C2E2CDim, C2E2CDim


# Conservative quadratic least-squares reconstruction over the 9-cell stencil
# (recon_lsq_cell_q_svd; mo_advection_hflux.f90 2447-2495 for one WENO candidate, :4632 for
# miura3). z_b is the increment of the neighbour cell averages relative to the center cell.
# The pseudoinverse (5 unknowns [x, y, x^2, y^2, xy], f90 index 1..5 -> coeff 2..6) is split
# into a direct part on C2E2C rows and a butterfly part on C2E2C2E2C rows; unused slots hold 0
# so the two neighbour sums together reproduce the Fortran 9-point dot products. The WENO
# scheme passes one candidate's pseudoinverse per call, miura3 passes the full-stencil one.


@gtx.field_operator
def _reconstruct_quadratic_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_direct_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_moments_1: fa.CellField[ta.wpfloat],
    lsq_moments_2: fa.CellField[ta.wpfloat],
    lsq_moments_3: fa.CellField[ta.wpfloat],
    lsq_moments_4: fa.CellField[ta.wpfloat],
    lsq_moments_5: fa.CellField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # f90 2448: z_b = p_cc(stencil cell) - p_cc(center), on the direct and butterfly rows
    zb_direct = p_cc(C2E2C) - p_cc
    zb_butterfly = p_cc(C2E2C2E2C) - p_cc

    # f90 2463-2472: coeff 2..6 = pseudoinv(1..5) . z_b, reassembled from the two row sets
    p_coeff_2_dsl = neighbor_sum(lsq_pseudoinv_direct_1 * zb_direct, axis=C2E2CDim) + neighbor_sum(
        lsq_pseudoinv_butterfly_1 * zb_butterfly, axis=C2E2C2E2CDim
    )
    p_coeff_3_dsl = neighbor_sum(lsq_pseudoinv_direct_2 * zb_direct, axis=C2E2CDim) + neighbor_sum(
        lsq_pseudoinv_butterfly_2 * zb_butterfly, axis=C2E2C2E2CDim
    )
    p_coeff_4_dsl = neighbor_sum(lsq_pseudoinv_direct_3 * zb_direct, axis=C2E2CDim) + neighbor_sum(
        lsq_pseudoinv_butterfly_3 * zb_butterfly, axis=C2E2C2E2CDim
    )
    p_coeff_5_dsl = neighbor_sum(lsq_pseudoinv_direct_4 * zb_direct, axis=C2E2CDim) + neighbor_sum(
        lsq_pseudoinv_butterfly_4 * zb_butterfly, axis=C2E2C2E2CDim
    )
    p_coeff_6_dsl = neighbor_sum(lsq_pseudoinv_direct_5 * zb_direct, axis=C2E2CDim) + neighbor_sum(
        lsq_pseudoinv_butterfly_5 * zb_butterfly, axis=C2E2C2E2CDim
    )

    # f90 2494-2495: c0 from the linear constraint c0 = p_cc - coeff(2:6) . moments(1:5)
    p_coeff_1_dsl = p_cc - (
        p_coeff_2_dsl * lsq_moments_1
        + p_coeff_3_dsl * lsq_moments_2
        + p_coeff_4_dsl * lsq_moments_3
        + p_coeff_5_dsl * lsq_moments_4
        + p_coeff_6_dsl * lsq_moments_5
    )

    return (
        p_coeff_1_dsl,
        p_coeff_2_dsl,
        p_coeff_3_dsl,
        p_coeff_4_dsl,
        p_coeff_5_dsl,
        p_coeff_6_dsl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_quadratic_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_direct_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_direct_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_butterfly_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_moments_1: fa.CellField[ta.wpfloat],
    lsq_moments_2: fa.CellField[ta.wpfloat],
    lsq_moments_3: fa.CellField[ta.wpfloat],
    lsq_moments_4: fa.CellField[ta.wpfloat],
    lsq_moments_5: fa.CellField[ta.wpfloat],
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_4_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_5_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_6_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_quadratic_coefficients_svd(
        p_cc=p_cc,
        lsq_pseudoinv_direct_1=lsq_pseudoinv_direct_1,
        lsq_pseudoinv_direct_2=lsq_pseudoinv_direct_2,
        lsq_pseudoinv_direct_3=lsq_pseudoinv_direct_3,
        lsq_pseudoinv_direct_4=lsq_pseudoinv_direct_4,
        lsq_pseudoinv_direct_5=lsq_pseudoinv_direct_5,
        lsq_pseudoinv_butterfly_1=lsq_pseudoinv_butterfly_1,
        lsq_pseudoinv_butterfly_2=lsq_pseudoinv_butterfly_2,
        lsq_pseudoinv_butterfly_3=lsq_pseudoinv_butterfly_3,
        lsq_pseudoinv_butterfly_4=lsq_pseudoinv_butterfly_4,
        lsq_pseudoinv_butterfly_5=lsq_pseudoinv_butterfly_5,
        lsq_moments_1=lsq_moments_1,
        lsq_moments_2=lsq_moments_2,
        lsq_moments_3=lsq_moments_3,
        lsq_moments_4=lsq_moments_4,
        lsq_moments_5=lsq_moments_5,
        out=(
            p_coeff_1_dsl,
            p_coeff_2_dsl,
            p_coeff_3_dsl,
            p_coeff_4_dsl,
            p_coeff_5_dsl,
            p_coeff_6_dsl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
