# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys  # Increase recusion depth, otherwise it doesn't compile

from gt4py.next.common import Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.common.dimension import (
    C2CECEC,
    C2E2C2E2C,
    CECECDim,
    CellDim,
    KDim,
)


sys.setrecursionlimit(6000)


@field_operator
def _recon_lsq_cell_c_stencil(
    p_cc: Field[[CellDim, KDim], float],
    lsq_qtmat_c_1: Field[[CECECDim], float],
    lsq_qtmat_c_2: Field[[CECECDim], float],
    lsq_qtmat_c_3: Field[[CECECDim], float],
    lsq_qtmat_c_4: Field[[CECECDim], float],
    lsq_qtmat_c_5: Field[[CECECDim], float],
    lsq_qtmat_c_6: Field[[CECECDim], float],
    lsq_qtmat_c_7: Field[[CECECDim], float],
    lsq_qtmat_c_8: Field[[CECECDim], float],
    lsq_qtmat_c_9: Field[[CECECDim], float],
    lsq_rmat_rdiag_c_1: Field[[CellDim], float],
    lsq_rmat_rdiag_c_2: Field[[CellDim], float],
    lsq_rmat_rdiag_c_3: Field[[CellDim], float],
    lsq_rmat_rdiag_c_4: Field[[CellDim], float],
    lsq_rmat_rdiag_c_5: Field[[CellDim], float],
    lsq_rmat_rdiag_c_6: Field[[CellDim], float],
    lsq_rmat_rdiag_c_7: Field[[CellDim], float],
    lsq_rmat_rdiag_c_8: Field[[CellDim], float],
    lsq_rmat_rdiag_c_9: Field[[CellDim], float],
    lsq_rmat_utri_c_1: Field[[CellDim], float],
    lsq_rmat_utri_c_2: Field[[CellDim], float],
    lsq_rmat_utri_c_3: Field[[CellDim], float],
    lsq_rmat_utri_c_4: Field[[CellDim], float],
    lsq_rmat_utri_c_5: Field[[CellDim], float],
    lsq_rmat_utri_c_6: Field[[CellDim], float],
    lsq_rmat_utri_c_7: Field[[CellDim], float],
    lsq_rmat_utri_c_8: Field[[CellDim], float],
    lsq_rmat_utri_c_9: Field[[CellDim], float],
    lsq_rmat_utri_c_10: Field[[CellDim], float],
    lsq_rmat_utri_c_11: Field[[CellDim], float],
    lsq_rmat_utri_c_12: Field[[CellDim], float],
    lsq_rmat_utri_c_13: Field[[CellDim], float],
    lsq_rmat_utri_c_14: Field[[CellDim], float],
    lsq_rmat_utri_c_15: Field[[CellDim], float],
    lsq_rmat_utri_c_16: Field[[CellDim], float],
    lsq_rmat_utri_c_17: Field[[CellDim], float],
    lsq_rmat_utri_c_18: Field[[CellDim], float],
    lsq_rmat_utri_c_19: Field[[CellDim], float],
    lsq_rmat_utri_c_20: Field[[CellDim], float],
    lsq_rmat_utri_c_21: Field[[CellDim], float],
    lsq_rmat_utri_c_22: Field[[CellDim], float],
    lsq_rmat_utri_c_23: Field[[CellDim], float],
    lsq_rmat_utri_c_24: Field[[CellDim], float],
    lsq_rmat_utri_c_25: Field[[CellDim], float],
    lsq_rmat_utri_c_26: Field[[CellDim], float],
    lsq_rmat_utri_c_27: Field[[CellDim], float],
    lsq_rmat_utri_c_28: Field[[CellDim], float],
    lsq_rmat_utri_c_29: Field[[CellDim], float],
    lsq_rmat_utri_c_30: Field[[CellDim], float],
    lsq_rmat_utri_c_31: Field[[CellDim], float],
    lsq_rmat_utri_c_32: Field[[CellDim], float],
    lsq_rmat_utri_c_33: Field[[CellDim], float],
    lsq_rmat_utri_c_34: Field[[CellDim], float],
    lsq_rmat_utri_c_35: Field[[CellDim], float],
    lsq_rmat_utri_c_36: Field[[CellDim], float],
    lsq_moments_1: Field[[CellDim], float],
    lsq_moments_2: Field[[CellDim], float],
    lsq_moments_3: Field[[CellDim], float],
    lsq_moments_4: Field[[CellDim], float],
    lsq_moments_5: Field[[CellDim], float],
    lsq_moments_6: Field[[CellDim], float],
    lsq_moments_7: Field[[CellDim], float],
    lsq_moments_8: Field[[CellDim], float],
    lsq_moments_9: Field[[CellDim], float],
) -> tuple[
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
    Field[[CellDim, KDim], float],
]:
    lsq_rmat_rdiag_c_1 = broadcast(lsq_rmat_rdiag_c_1, (CellDim, KDim))
    lsq_rmat_rdiag_c_2 = broadcast(lsq_rmat_rdiag_c_2, (CellDim, KDim))
    lsq_rmat_rdiag_c_3 = broadcast(lsq_rmat_rdiag_c_3, (CellDim, KDim))
    lsq_rmat_rdiag_c_4 = broadcast(lsq_rmat_rdiag_c_4, (CellDim, KDim))
    lsq_rmat_rdiag_c_5 = broadcast(lsq_rmat_rdiag_c_5, (CellDim, KDim))
    lsq_rmat_rdiag_c_6 = broadcast(lsq_rmat_rdiag_c_6, (CellDim, KDim))
    lsq_rmat_rdiag_c_7 = broadcast(lsq_rmat_rdiag_c_7, (CellDim, KDim))
    lsq_rmat_rdiag_c_8 = broadcast(lsq_rmat_rdiag_c_8, (CellDim, KDim))
    lsq_rmat_rdiag_c_9 = broadcast(lsq_rmat_rdiag_c_9, (CellDim, KDim))
    lsq_qtmat_c_1 = broadcast(lsq_qtmat_c_1, (CECECDim, KDim))
    lsq_qtmat_c_2 = broadcast(lsq_qtmat_c_2, (CECECDim, KDim))
    lsq_qtmat_c_3 = broadcast(lsq_qtmat_c_3, (CECECDim, KDim))
    lsq_qtmat_c_4 = broadcast(lsq_qtmat_c_4, (CECECDim, KDim))
    lsq_qtmat_c_5 = broadcast(lsq_qtmat_c_5, (CECECDim, KDim))
    lsq_qtmat_c_6 = broadcast(lsq_qtmat_c_6, (CECECDim, KDim))
    lsq_qtmat_c_7 = broadcast(lsq_qtmat_c_7, (CECECDim, KDim))
    lsq_qtmat_c_8 = broadcast(lsq_qtmat_c_8, (CECECDim, KDim))
    lsq_qtmat_c_9 = broadcast(lsq_qtmat_c_9, (CECECDim, KDim))

    p_coeff_10 = lsq_rmat_rdiag_c_9 * (
        lsq_qtmat_c_9(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_9(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
    )

    p_coeff_9 = lsq_rmat_rdiag_c_8 * (
        lsq_qtmat_c_8(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_8(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - lsq_rmat_utri_c_1 * p_coeff_10
    )

    p_coeff_8 = lsq_rmat_rdiag_c_7 * (
        lsq_qtmat_c_7(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_7(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (lsq_rmat_utri_c_2 * p_coeff_9 + lsq_rmat_utri_c_3 * p_coeff_10)
    )

    p_coeff_7 = lsq_rmat_rdiag_c_6 * (
        lsq_qtmat_c_6(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_6(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_4 * p_coeff_8
            + lsq_rmat_utri_c_5 * p_coeff_9
            + lsq_rmat_utri_c_6 * p_coeff_10
        )
    )

    p_coeff_6 = lsq_rmat_rdiag_c_5 * (
        lsq_qtmat_c_5(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_5(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_7 * p_coeff_7
            + lsq_rmat_utri_c_8 * p_coeff_8
            + lsq_rmat_utri_c_9 * p_coeff_9
            + lsq_rmat_utri_c_10 * p_coeff_10
        )
    )

    p_coeff_5 = lsq_rmat_rdiag_c_4 * (
        lsq_qtmat_c_4(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_4(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_11 * p_coeff_6
            + lsq_rmat_utri_c_12 * p_coeff_7
            + lsq_rmat_utri_c_13 * p_coeff_8
            + lsq_rmat_utri_c_14 * p_coeff_9
            + lsq_rmat_utri_c_15 * p_coeff_10
        )
    )

    p_coeff_4 = lsq_rmat_rdiag_c_3 * (
        lsq_qtmat_c_3(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_3(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_16 * p_coeff_5
            + lsq_rmat_utri_c_17 * p_coeff_6
            + lsq_rmat_utri_c_18 * p_coeff_7
            + lsq_rmat_utri_c_19 * p_coeff_8
            + lsq_rmat_utri_c_20 * p_coeff_9
            + lsq_rmat_utri_c_21 * p_coeff_10
        )
    )

    p_coeff_3 = lsq_rmat_rdiag_c_2 * (
        lsq_qtmat_c_2(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_2(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_22 * p_coeff_4
            + lsq_rmat_utri_c_23 * p_coeff_5
            + lsq_rmat_utri_c_24 * p_coeff_6
            + lsq_rmat_utri_c_25 * p_coeff_7
            + lsq_rmat_utri_c_26 * p_coeff_8
            + lsq_rmat_utri_c_27 * p_coeff_9
            + lsq_rmat_utri_c_28 * p_coeff_10
        )
    )

    p_coeff_2 = lsq_rmat_rdiag_c_1 * (
        lsq_qtmat_c_1(C2CECEC[0]) * (p_cc(C2E2C2E2C[0]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[1]) * (p_cc(C2E2C2E2C[1]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[2]) * (p_cc(C2E2C2E2C[2]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[3]) * (p_cc(C2E2C2E2C[3]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[4]) * (p_cc(C2E2C2E2C[4]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[5]) * (p_cc(C2E2C2E2C[5]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[6]) * (p_cc(C2E2C2E2C[6]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[7]) * (p_cc(C2E2C2E2C[7]) - p_cc)
        + lsq_qtmat_c_1(C2CECEC[8]) * (p_cc(C2E2C2E2C[8]) - p_cc)
        - (
            lsq_rmat_utri_c_29 * p_coeff_3
            + lsq_rmat_utri_c_30 * p_coeff_4
            + lsq_rmat_utri_c_31 * p_coeff_5
            + lsq_rmat_utri_c_32 * p_coeff_6
            + lsq_rmat_utri_c_33 * p_coeff_7
            + lsq_rmat_utri_c_34 * p_coeff_8
            + lsq_rmat_utri_c_35 * p_coeff_9
            + lsq_rmat_utri_c_36 * p_coeff_10
        )
    )

    p_coeff_1 = p_cc - (
        p_coeff_2 * lsq_moments_1
        + p_coeff_3 * lsq_moments_2
        + p_coeff_4 * lsq_moments_3
        + p_coeff_5 * lsq_moments_4
        + p_coeff_6 * lsq_moments_5
        + p_coeff_7 * lsq_moments_6
        + p_coeff_8 * lsq_moments_7
        + p_coeff_9 * lsq_moments_8
        + p_coeff_10 * lsq_moments_9
    )
    return (
        p_coeff_1,
        p_coeff_2,
        p_coeff_3,
        p_coeff_4,
        p_coeff_5,
        p_coeff_6,
        p_coeff_7,
        p_coeff_8,
        p_coeff_9,
        p_coeff_10,
    )


@program
def recon_lsq_cell_c_stencil(
    p_cc: Field[[CellDim, KDim], float],
    lsq_qtmat_c_1: Field[[CECECDim], float],
    lsq_qtmat_c_2: Field[[CECECDim], float],
    lsq_qtmat_c_3: Field[[CECECDim], float],
    lsq_qtmat_c_4: Field[[CECECDim], float],
    lsq_qtmat_c_5: Field[[CECECDim], float],
    lsq_qtmat_c_6: Field[[CECECDim], float],
    lsq_qtmat_c_7: Field[[CECECDim], float],
    lsq_qtmat_c_8: Field[[CECECDim], float],
    lsq_qtmat_c_9: Field[[CECECDim], float],
    lsq_rmat_rdiag_c_1: Field[[CellDim], float],
    lsq_rmat_rdiag_c_2: Field[[CellDim], float],
    lsq_rmat_rdiag_c_3: Field[[CellDim], float],
    lsq_rmat_rdiag_c_4: Field[[CellDim], float],
    lsq_rmat_rdiag_c_5: Field[[CellDim], float],
    lsq_rmat_rdiag_c_6: Field[[CellDim], float],
    lsq_rmat_rdiag_c_7: Field[[CellDim], float],
    lsq_rmat_rdiag_c_8: Field[[CellDim], float],
    lsq_rmat_rdiag_c_9: Field[[CellDim], float],
    lsq_rmat_utri_c_1: Field[[CellDim], float],
    lsq_rmat_utri_c_2: Field[[CellDim], float],
    lsq_rmat_utri_c_3: Field[[CellDim], float],
    lsq_rmat_utri_c_4: Field[[CellDim], float],
    lsq_rmat_utri_c_5: Field[[CellDim], float],
    lsq_rmat_utri_c_6: Field[[CellDim], float],
    lsq_rmat_utri_c_7: Field[[CellDim], float],
    lsq_rmat_utri_c_8: Field[[CellDim], float],
    lsq_rmat_utri_c_9: Field[[CellDim], float],
    lsq_rmat_utri_c_10: Field[[CellDim], float],
    lsq_rmat_utri_c_11: Field[[CellDim], float],
    lsq_rmat_utri_c_12: Field[[CellDim], float],
    lsq_rmat_utri_c_13: Field[[CellDim], float],
    lsq_rmat_utri_c_14: Field[[CellDim], float],
    lsq_rmat_utri_c_15: Field[[CellDim], float],
    lsq_rmat_utri_c_16: Field[[CellDim], float],
    lsq_rmat_utri_c_17: Field[[CellDim], float],
    lsq_rmat_utri_c_18: Field[[CellDim], float],
    lsq_rmat_utri_c_19: Field[[CellDim], float],
    lsq_rmat_utri_c_20: Field[[CellDim], float],
    lsq_rmat_utri_c_21: Field[[CellDim], float],
    lsq_rmat_utri_c_22: Field[[CellDim], float],
    lsq_rmat_utri_c_23: Field[[CellDim], float],
    lsq_rmat_utri_c_24: Field[[CellDim], float],
    lsq_rmat_utri_c_25: Field[[CellDim], float],
    lsq_rmat_utri_c_26: Field[[CellDim], float],
    lsq_rmat_utri_c_27: Field[[CellDim], float],
    lsq_rmat_utri_c_28: Field[[CellDim], float],
    lsq_rmat_utri_c_29: Field[[CellDim], float],
    lsq_rmat_utri_c_30: Field[[CellDim], float],
    lsq_rmat_utri_c_31: Field[[CellDim], float],
    lsq_rmat_utri_c_32: Field[[CellDim], float],
    lsq_rmat_utri_c_33: Field[[CellDim], float],
    lsq_rmat_utri_c_34: Field[[CellDim], float],
    lsq_rmat_utri_c_35: Field[[CellDim], float],
    lsq_rmat_utri_c_36: Field[[CellDim], float],
    lsq_moments_1: Field[[CellDim], float],
    lsq_moments_2: Field[[CellDim], float],
    lsq_moments_3: Field[[CellDim], float],
    lsq_moments_4: Field[[CellDim], float],
    lsq_moments_5: Field[[CellDim], float],
    lsq_moments_6: Field[[CellDim], float],
    lsq_moments_7: Field[[CellDim], float],
    lsq_moments_8: Field[[CellDim], float],
    lsq_moments_9: Field[[CellDim], float],
    p_coeff_1: Field[[CellDim, KDim], float],
    p_coeff_2: Field[[CellDim, KDim], float],
    p_coeff_3: Field[[CellDim, KDim], float],
    p_coeff_4: Field[[CellDim, KDim], float],
    p_coeff_5: Field[[CellDim, KDim], float],
    p_coeff_6: Field[[CellDim, KDim], float],
    p_coeff_7: Field[[CellDim, KDim], float],
    p_coeff_8: Field[[CellDim, KDim], float],
    p_coeff_9: Field[[CellDim, KDim], float],
    p_coeff_10: Field[[CellDim, KDim], float],
):
    _recon_lsq_cell_c_stencil(
        p_cc,
        lsq_qtmat_c_1,
        lsq_qtmat_c_2,
        lsq_qtmat_c_3,
        lsq_qtmat_c_4,
        lsq_qtmat_c_5,
        lsq_qtmat_c_6,
        lsq_qtmat_c_7,
        lsq_qtmat_c_8,
        lsq_qtmat_c_9,
        lsq_rmat_rdiag_c_1,
        lsq_rmat_rdiag_c_2,
        lsq_rmat_rdiag_c_3,
        lsq_rmat_rdiag_c_4,
        lsq_rmat_rdiag_c_5,
        lsq_rmat_rdiag_c_6,
        lsq_rmat_rdiag_c_7,
        lsq_rmat_rdiag_c_8,
        lsq_rmat_rdiag_c_9,
        lsq_rmat_utri_c_1,
        lsq_rmat_utri_c_2,
        lsq_rmat_utri_c_3,
        lsq_rmat_utri_c_4,
        lsq_rmat_utri_c_5,
        lsq_rmat_utri_c_6,
        lsq_rmat_utri_c_7,
        lsq_rmat_utri_c_8,
        lsq_rmat_utri_c_9,
        lsq_rmat_utri_c_10,
        lsq_rmat_utri_c_11,
        lsq_rmat_utri_c_12,
        lsq_rmat_utri_c_13,
        lsq_rmat_utri_c_14,
        lsq_rmat_utri_c_15,
        lsq_rmat_utri_c_16,
        lsq_rmat_utri_c_17,
        lsq_rmat_utri_c_18,
        lsq_rmat_utri_c_19,
        lsq_rmat_utri_c_20,
        lsq_rmat_utri_c_21,
        lsq_rmat_utri_c_22,
        lsq_rmat_utri_c_23,
        lsq_rmat_utri_c_24,
        lsq_rmat_utri_c_25,
        lsq_rmat_utri_c_26,
        lsq_rmat_utri_c_27,
        lsq_rmat_utri_c_28,
        lsq_rmat_utri_c_29,
        lsq_rmat_utri_c_30,
        lsq_rmat_utri_c_31,
        lsq_rmat_utri_c_32,
        lsq_rmat_utri_c_33,
        lsq_rmat_utri_c_34,
        lsq_rmat_utri_c_35,
        lsq_rmat_utri_c_36,
        lsq_moments_1,
        lsq_moments_2,
        lsq_moments_3,
        lsq_moments_4,
        lsq_moments_5,
        lsq_moments_6,
        lsq_moments_7,
        lsq_moments_8,
        lsq_moments_9,
        out=(
            p_coeff_1,
            p_coeff_2,
            p_coeff_3,
            p_coeff_4,
            p_coeff_5,
            p_coeff_6,
            p_coeff_7,
            p_coeff_8,
            p_coeff_9,
            p_coeff_10,
        ),
    )
