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

import numpy as np
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider

from icon4py.advection.recon_lsq_cell_c_stencil import (
    recon_lsq_cell_c_stencil,
)
from icon4py.common.dimension import C2E2C2E2CDim, CECECDim, CellDim, KDim

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def recon_lsq_cell_c_stencil_numpy(
    c2e2c2e2c: np.ndarray,
    p_cc:  np.ndarray,
    lsq_qtmat_c_1: np.ndarray,
    lsq_qtmat_c_2: np.ndarray,
    lsq_qtmat_c_3: np.ndarray,
    lsq_qtmat_c_4: np.ndarray,
    lsq_qtmat_c_5: np.ndarray,
    lsq_qtmat_c_6: np.ndarray,
    lsq_qtmat_c_7: np.ndarray,
    lsq_qtmat_c_8: np.ndarray,
    lsq_qtmat_c_9: np.ndarray,
    lsq_rmat_rdiag_c_1: np.ndarray,
    lsq_rmat_rdiag_c_2: np.ndarray,
    lsq_rmat_rdiag_c_3: np.ndarray,
    lsq_rmat_rdiag_c_4: np.ndarray,
    lsq_rmat_rdiag_c_5: np.ndarray,
    lsq_rmat_rdiag_c_6: np.ndarray,
    lsq_rmat_rdiag_c_7: np.ndarray,
    lsq_rmat_rdiag_c_8: np.ndarray,
    lsq_rmat_rdiag_c_9: np.ndarray,
    lsq_rmat_utri_c_1: np.ndarray,
    lsq_rmat_utri_c_2: np.ndarray,
    lsq_rmat_utri_c_3: np.ndarray,
    lsq_rmat_utri_c_4: np.ndarray,
    lsq_rmat_utri_c_5: np.ndarray,
    lsq_rmat_utri_c_6: np.ndarray,
    lsq_rmat_utri_c_7: np.ndarray,
    lsq_rmat_utri_c_8: np.ndarray,
    lsq_rmat_utri_c_9: np.ndarray,
    lsq_rmat_utri_c_10: np.ndarray,
    lsq_rmat_utri_c_11: np.ndarray,
    lsq_rmat_utri_c_12: np.ndarray,
    lsq_rmat_utri_c_13: np.ndarray,
    lsq_rmat_utri_c_14: np.ndarray,
    lsq_rmat_utri_c_15: np.ndarray,
    lsq_rmat_utri_c_16: np.ndarray,
    lsq_rmat_utri_c_17: np.ndarray,
    lsq_rmat_utri_c_18: np.ndarray,
    lsq_rmat_utri_c_19: np.ndarray,
    lsq_rmat_utri_c_20: np.ndarray,
    lsq_rmat_utri_c_21: np.ndarray,
    lsq_rmat_utri_c_22: np.ndarray,
    lsq_rmat_utri_c_23: np.ndarray,
    lsq_rmat_utri_c_24: np.ndarray,
    lsq_rmat_utri_c_25: np.ndarray,
    lsq_rmat_utri_c_26: np.ndarray,
    lsq_rmat_utri_c_27: np.ndarray,
    lsq_rmat_utri_c_28: np.ndarray,
    lsq_rmat_utri_c_29: np.ndarray,
    lsq_rmat_utri_c_30: np.ndarray,
    lsq_rmat_utri_c_31: np.ndarray,
    lsq_rmat_utri_c_32: np.ndarray,
    lsq_rmat_utri_c_33: np.ndarray,
    lsq_rmat_utri_c_34: np.ndarray,
    lsq_rmat_utri_c_35: np.ndarray,
    lsq_rmat_utri_c_36: np.ndarray,
    lsq_moments_1: np.ndarray,
    lsq_moments_2: np.ndarray,
    lsq_moments_3: np.ndarray,
    lsq_moments_4: np.ndarray,
    lsq_moments_5: np.ndarray,
    lsq_moments_6: np.ndarray,
    lsq_moments_7: np.ndarray,
    lsq_moments_8: np.ndarray,
    lsq_moments_9: np.ndarray
) -> tuple[np.ndarray]:
    p_cc_e = np.expand_dims(p_cc, axis=-1)
#    n_diff = p_cc[c2e2c] - p_cc_e

    lsq_rmat_rdiag_c_1 = np.expand_dims(lsq_rmat_rdiag_c_1, axis=-1)
    lsq_rmat_rdiag_c_2 = np.expand_dims(lsq_rmat_rdiag_c_2, axis=-1)
    lsq_rmat_rdiag_c_3 = np.expand_dims(lsq_rmat_rdiag_c_3, axis=-1)
    lsq_rmat_rdiag_c_4 = np.expand_dims(lsq_rmat_rdiag_c_4, axis=-1)
    lsq_rmat_rdiag_c_5 = np.expand_dims(lsq_rmat_rdiag_c_5, axis=-1)
    lsq_rmat_rdiag_c_6 = np.expand_dims(lsq_rmat_rdiag_c_6, axis=-1)
    lsq_rmat_rdiag_c_7 = np.expand_dims(lsq_rmat_rdiag_c_7, axis=-1)
    lsq_rmat_rdiag_c_8 = np.expand_dims(lsq_rmat_rdiag_c_8, axis=-1)
    lsq_rmat_rdiag_c_9 = np.expand_dims(lsq_rmat_rdiag_c_9, axis=-1)
#    lsq_rmat_rdiag_c_1 = np.broadcast_to(lsq_rmat_rdiag_c_1, p_cc.shape)
#    lsq_rmat_rdiag_c_2 = np.broadcast_to(lsq_rmat_rdiag_c_2, p_cc.shape)
#    lsq_rmat_rdiag_c_3 = np.broadcast_to(lsq_rmat_rdiag_c_3, p_cc.shape)
#    lsq_rmat_rdiag_c_4 = np.broadcast_to(lsq_rmat_rdiag_c_4, p_cc.shape)
#    lsq_rmat_rdiag_c_5 = np.broadcast_to(lsq_rmat_rdiag_c_5, p_cc.shape)
#    lsq_rmat_rdiag_c_6 = np.broadcast_to(lsq_rmat_rdiag_c_6, p_cc.shape)
#    lsq_rmat_rdiag_c_7 = np.broadcast_to(lsq_rmat_rdiag_c_7, p_cc.shape)
#    lsq_rmat_rdiag_c_8 = np.broadcast_to(lsq_rmat_rdiag_c_8, p_cc.shape)
#    lsq_rmat_rdiag_c_9 = np.broadcast_to(lsq_rmat_rdiag_c_9, p_cc.shape)
    lsq_moments_1 = np.expand_dims(lsq_moments_1, axis=-1)
    lsq_moments_2 = np.expand_dims(lsq_moments_2, axis=-1)
    lsq_moments_3 = np.expand_dims(lsq_moments_3, axis=-1)
    lsq_moments_4 = np.expand_dims(lsq_moments_4, axis=-1)
    lsq_moments_5 = np.expand_dims(lsq_moments_5, axis=-1)
    lsq_moments_6 = np.expand_dims(lsq_moments_6, axis=-1)
    lsq_moments_7 = np.expand_dims(lsq_moments_7, axis=-1)
    lsq_moments_8 = np.expand_dims(lsq_moments_8, axis=-1)
    lsq_moments_9 = np.expand_dims(lsq_moments_9, axis=-1)
    lsq_moments_1 = np.broadcast_to(lsq_moments_1, p_cc.shape)
    lsq_moments_2 = np.broadcast_to(lsq_moments_2, p_cc.shape)
    lsq_moments_3 = np.broadcast_to(lsq_moments_3, p_cc.shape)
    lsq_moments_4 = np.broadcast_to(lsq_moments_4, p_cc.shape)
    lsq_moments_5 = np.broadcast_to(lsq_moments_5, p_cc.shape)
    lsq_moments_6 = np.broadcast_to(lsq_moments_6, p_cc.shape)
    lsq_moments_7 = np.broadcast_to(lsq_moments_7, p_cc.shape)
    lsq_moments_8 = np.broadcast_to(lsq_moments_8, p_cc.shape)
    lsq_moments_9 = np.broadcast_to(lsq_moments_9, p_cc.shape)
    lsq_rmat_utri_c_1 = np.expand_dims(lsq_rmat_utri_c_1, axis=-1)
    lsq_rmat_utri_c_2 = np.expand_dims(lsq_rmat_utri_c_2, axis=-1)
    lsq_rmat_utri_c_3 = np.expand_dims(lsq_rmat_utri_c_3, axis=-1)
    lsq_rmat_utri_c_4 = np.expand_dims(lsq_rmat_utri_c_4, axis=-1)
    lsq_rmat_utri_c_5 = np.expand_dims(lsq_rmat_utri_c_5, axis=-1)
    lsq_rmat_utri_c_6 = np.expand_dims(lsq_rmat_utri_c_6, axis=-1)
    lsq_rmat_utri_c_7 = np.expand_dims(lsq_rmat_utri_c_7, axis=-1)
    lsq_rmat_utri_c_8 = np.expand_dims(lsq_rmat_utri_c_8, axis=-1)
    lsq_rmat_utri_c_9 = np.expand_dims(lsq_rmat_utri_c_9, axis=-1)
    lsq_rmat_utri_c_10 = np.expand_dims(lsq_rmat_utri_c_10, axis=-1)
    lsq_rmat_utri_c_11 = np.expand_dims(lsq_rmat_utri_c_11, axis=-1)
    lsq_rmat_utri_c_12 = np.expand_dims(lsq_rmat_utri_c_12, axis=-1)
    lsq_rmat_utri_c_13 = np.expand_dims(lsq_rmat_utri_c_13, axis=-1)
    lsq_rmat_utri_c_14 = np.expand_dims(lsq_rmat_utri_c_14, axis=-1)
    lsq_rmat_utri_c_15 = np.expand_dims(lsq_rmat_utri_c_15, axis=-1)
    lsq_rmat_utri_c_16 = np.expand_dims(lsq_rmat_utri_c_16, axis=-1)
    lsq_rmat_utri_c_17 = np.expand_dims(lsq_rmat_utri_c_17, axis=-1)
    lsq_rmat_utri_c_18 = np.expand_dims(lsq_rmat_utri_c_18, axis=-1)
    lsq_rmat_utri_c_19 = np.expand_dims(lsq_rmat_utri_c_19, axis=-1)
    lsq_rmat_utri_c_20 = np.expand_dims(lsq_rmat_utri_c_20, axis=-1)
    lsq_rmat_utri_c_21 = np.expand_dims(lsq_rmat_utri_c_21, axis=-1)
    lsq_rmat_utri_c_22 = np.expand_dims(lsq_rmat_utri_c_22, axis=-1)
    lsq_rmat_utri_c_23 = np.expand_dims(lsq_rmat_utri_c_23, axis=-1)
    lsq_rmat_utri_c_24 = np.expand_dims(lsq_rmat_utri_c_24, axis=-1)
    lsq_rmat_utri_c_25 = np.expand_dims(lsq_rmat_utri_c_25, axis=-1)
    lsq_rmat_utri_c_26 = np.expand_dims(lsq_rmat_utri_c_26, axis=-1)
    lsq_rmat_utri_c_27 = np.expand_dims(lsq_rmat_utri_c_27, axis=-1)
    lsq_rmat_utri_c_28 = np.expand_dims(lsq_rmat_utri_c_28, axis=-1)
    lsq_rmat_utri_c_29 = np.expand_dims(lsq_rmat_utri_c_29, axis=-1)
    lsq_rmat_utri_c_30 = np.expand_dims(lsq_rmat_utri_c_30, axis=-1)
    lsq_rmat_utri_c_31 = np.expand_dims(lsq_rmat_utri_c_31, axis=-1)
    lsq_rmat_utri_c_32 = np.expand_dims(lsq_rmat_utri_c_32, axis=-1)
    lsq_rmat_utri_c_33 = np.expand_dims(lsq_rmat_utri_c_33, axis=-1)
    lsq_rmat_utri_c_34 = np.expand_dims(lsq_rmat_utri_c_34, axis=-1)
    lsq_rmat_utri_c_35 = np.expand_dims(lsq_rmat_utri_c_35, axis=-1)
    lsq_rmat_utri_c_36 = np.expand_dims(lsq_rmat_utri_c_36, axis=-1)
    lsq_rmat_utri_c_1 = np.broadcast_to(lsq_rmat_utri_c_1, p_cc.shape)
    lsq_rmat_utri_c_2 = np.broadcast_to(lsq_rmat_utri_c_2, p_cc.shape)
    lsq_rmat_utri_c_3 = np.broadcast_to(lsq_rmat_utri_c_3, p_cc.shape)
    lsq_rmat_utri_c_4 = np.broadcast_to(lsq_rmat_utri_c_4, p_cc.shape)
    lsq_rmat_utri_c_5 = np.broadcast_to(lsq_rmat_utri_c_5, p_cc.shape)
    lsq_rmat_utri_c_6 = np.broadcast_to(lsq_rmat_utri_c_6, p_cc.shape)
    lsq_rmat_utri_c_7 = np.broadcast_to(lsq_rmat_utri_c_7, p_cc.shape)
    lsq_rmat_utri_c_8 = np.broadcast_to(lsq_rmat_utri_c_8, p_cc.shape)
    lsq_rmat_utri_c_9 = np.broadcast_to(lsq_rmat_utri_c_9, p_cc.shape)
    lsq_rmat_utri_c_10 = np.broadcast_to(lsq_rmat_utri_c_10, p_cc.shape)
    lsq_rmat_utri_c_11 = np.broadcast_to(lsq_rmat_utri_c_11, p_cc.shape)
    lsq_rmat_utri_c_12 = np.broadcast_to(lsq_rmat_utri_c_12, p_cc.shape)
    lsq_rmat_utri_c_13 = np.broadcast_to(lsq_rmat_utri_c_13, p_cc.shape)
    lsq_rmat_utri_c_14 = np.broadcast_to(lsq_rmat_utri_c_14, p_cc.shape)
    lsq_rmat_utri_c_15 = np.broadcast_to(lsq_rmat_utri_c_15, p_cc.shape)
    lsq_rmat_utri_c_16 = np.broadcast_to(lsq_rmat_utri_c_16, p_cc.shape)
    lsq_rmat_utri_c_17 = np.broadcast_to(lsq_rmat_utri_c_17, p_cc.shape)
    lsq_rmat_utri_c_18 = np.broadcast_to(lsq_rmat_utri_c_18, p_cc.shape)
    lsq_rmat_utri_c_19 = np.broadcast_to(lsq_rmat_utri_c_19, p_cc.shape)
    lsq_rmat_utri_c_20 = np.broadcast_to(lsq_rmat_utri_c_20, p_cc.shape)
    lsq_rmat_utri_c_21 = np.broadcast_to(lsq_rmat_utri_c_21, p_cc.shape)
    lsq_rmat_utri_c_22 = np.broadcast_to(lsq_rmat_utri_c_22, p_cc.shape)
    lsq_rmat_utri_c_23 = np.broadcast_to(lsq_rmat_utri_c_23, p_cc.shape)
    lsq_rmat_utri_c_24 = np.broadcast_to(lsq_rmat_utri_c_24, p_cc.shape)
    lsq_rmat_utri_c_25 = np.broadcast_to(lsq_rmat_utri_c_25, p_cc.shape)
    lsq_rmat_utri_c_26 = np.broadcast_to(lsq_rmat_utri_c_26, p_cc.shape)
    lsq_rmat_utri_c_27 = np.broadcast_to(lsq_rmat_utri_c_27, p_cc.shape)
    lsq_rmat_utri_c_28 = np.broadcast_to(lsq_rmat_utri_c_28, p_cc.shape)
    lsq_rmat_utri_c_29 = np.broadcast_to(lsq_rmat_utri_c_29, p_cc.shape)
    lsq_rmat_utri_c_30 = np.broadcast_to(lsq_rmat_utri_c_30, p_cc.shape)
    lsq_rmat_utri_c_31 = np.broadcast_to(lsq_rmat_utri_c_31, p_cc.shape)
    lsq_rmat_utri_c_32 = np.broadcast_to(lsq_rmat_utri_c_32, p_cc.shape)
    lsq_rmat_utri_c_33 = np.broadcast_to(lsq_rmat_utri_c_33, p_cc.shape)
    lsq_rmat_utri_c_34 = np.broadcast_to(lsq_rmat_utri_c_34, p_cc.shape)
    lsq_rmat_utri_c_35 = np.broadcast_to(lsq_rmat_utri_c_35, p_cc.shape)
    lsq_rmat_utri_c_36 = np.broadcast_to(lsq_rmat_utri_c_36, p_cc.shape)
#   lsq_rmat_utri_c_9 lsq_qtmat_c_1 = nplsq_rmat_utri_c_9.broadcast_to(lsq_qtmat_c_1, (CECECDim, KDim))
#    lsq_qtmat_c_2 = np.broadcast_to(lsq_qtmat_c_2, (CECECDim, KDim))
#    lsq_qtmat_c_3 = np.broadcast_to(lsq_qtmat_c_3, (CECECDim, KDim))
#    lsq_qtmat_c_4 = np.broadcast_to(lsq_qtmat_c_4, (CECECDim, KDim))
#    lsq_qtmat_c_5 = np.broadcast_to(lsq_qtmat_c_5, (CECECDim, KDim))
#    lsq_qtmat_c_6 = np.broadcast_to(lsq_qtmat_c_6, (CECECDim, KDim))
#    lsq_qtmat_c_7 = np.broadcast_to(lsq_qtmat_c_7, (CECECDim, KDim))
#    lsq_qtmat_c_8 = np.broadcast_to(lsq_qtmat_c_8, (CECECDim, KDim))
#    lsq_qtmat_c_9 = np.broadcast_to(lsq_qtmat_c_9, (CECECDim, KDim))
#    lsq_qtmat_c_1 = np.broadcast_to(lsq_qtmat_c_1, p_cc.shape)
#    lsq_qtmat_c_2 = np.broadcast_to(lsq_qtmat_c_2, p_cc.shape)
#    lsq_qtmat_c_3 = np.broadcast_to(lsq_qtmat_c_3, p_cc.shape)
#    lsq_qtmat_c_4 = np.broadcast_to(lsq_qtmat_c_4, p_cc.shape)
#    lsq_qtmat_c_5 = np.broadcast_to(lsq_qtmat_c_5, p_cc.shape)
#    lsq_qtmat_c_6 = np.broadcast_to(lsq_qtmat_c_6, p_cc.shape)
#    lsq_qtmat_c_7 = np.broadcast_to(lsq_qtmat_c_7, p_cc.shape)
#    lsq_qtmat_c_8 = np.broadcast_to(lsq_qtmat_c_8, p_cc.shape)
#    lsq_qtmat_c_9 = np.broadcast_to(lsq_qtmat_c_9, p_cc.shape)
#    lsq_qtmat_c_1 = np.repeat(lsq_qtmat_c_1[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_2 = np.repeat(lsq_qtmat_c_2[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_3 = np.repeat(lsq_qtmat_c_3[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_4 = np.repeat(lsq_qtmat_c_4[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_5 = np.repeat(lsq_qtmat_c_5[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_6 = np.repeat(lsq_qtmat_c_6[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_7 = np.repeat(lsq_qtmat_c_7[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_8 = np.repeat(lsq_qtmat_c_8[:, np.newaxis], p_cc.shape[1], axis=1)
#    lsq_qtmat_c_9 = np.repeat(lsq_qtmat_c_9[:, np.newaxis], p_cc.shape[1], axis=1)
    lsq_qtmat_c_9 = np.expand_dims(lsq_qtmat_c_9, axis=-1)
    lsq_qtmat_c_8 = np.expand_dims(lsq_qtmat_c_8, axis=-1)
    lsq_qtmat_c_7 = np.expand_dims(lsq_qtmat_c_7, axis=-1)
    lsq_qtmat_c_6 = np.expand_dims(lsq_qtmat_c_6, axis=-1)
    lsq_qtmat_c_5 = np.expand_dims(lsq_qtmat_c_5, axis=-1)
    lsq_qtmat_c_4 = np.expand_dims(lsq_qtmat_c_4, axis=-1)
    lsq_qtmat_c_3 = np.expand_dims(lsq_qtmat_c_3, axis=-1)
    lsq_qtmat_c_2 = np.expand_dims(lsq_qtmat_c_2, axis=-1)
    lsq_qtmat_c_1 = np.expand_dims(lsq_qtmat_c_1, axis=-1)

    p_coeff_10 = lsq_rmat_rdiag_c_9 * (
          lsq_qtmat_c_9[:, 0]  * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_9[:, 1]  * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_9[:, 2]  * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_9[:, 3]  * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_9[:, 4]  * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_9[:, 5]  * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_9[:, 6]  * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_9[:, 7]  * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_9[:, 8]  * (p_cc_e[:, 8] - p_cc)
    )


    p_coeff_9 = lsq_rmat_rdiag_c_8 * (
          lsq_qtmat_c_8[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_8[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_8[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_8[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_8[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_8[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_8[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_8[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_8[:, 8] * (p_cc_e[:, 8] - p_cc)
                - lsq_rmat_utri_c_1 * p_coeff_10
                )

    p_coeff_8 = lsq_rmat_rdiag_c_8 * (
          lsq_qtmat_c_7[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_7[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_7[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_7[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_7[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_7[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_7[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_7[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_7[:, 8] * (p_cc_e[:, 8] - p_cc)
                - (lsq_rmat_utri_c_2 * p_coeff_9 + lsq_rmat_utri_c_3 * p_coeff_10)
                )

    p_coeff_7 = lsq_rmat_rdiag_c_6 * (
          lsq_qtmat_c_6[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_6[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_6[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_6[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_6[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_6[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_6[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_6[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_6[:, 8] * (p_cc_e[:, 8] - p_cc)
        - (
            lsq_rmat_utri_c_4 * p_coeff_8
            + lsq_rmat_utri_c_5 * p_coeff_9
            + lsq_rmat_utri_c_6 * p_coeff_10
        )
    )

    p_coeff_6 = lsq_rmat_rdiag_c_5 * (
          lsq_qtmat_c_5[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_5[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_5[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_5[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_5[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_5[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_5[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_5[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_5[:, 8] * (p_cc_e[:, 8] - p_cc)
        - (
            lsq_rmat_utri_c_7 * p_coeff_7
            + lsq_rmat_utri_c_8 * p_coeff_8
            + lsq_rmat_utri_c_9 * p_coeff_9
            + lsq_rmat_utri_c_10 * p_coeff_10
        )
    )

    p_coeff_5 = lsq_rmat_rdiag_c_4 * (
          lsq_qtmat_c_4[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_4[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_4[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_4[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_4[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_4[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_4[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_4[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_4[:, 8] * (p_cc_e[:, 8] - p_cc)
       - (
            lsq_rmat_utri_c_11 * p_coeff_6
            + lsq_rmat_utri_c_12 * p_coeff_7
            + lsq_rmat_utri_c_13 * p_coeff_8
            + lsq_rmat_utri_c_14 * p_coeff_9
            + lsq_rmat_utri_c_15 * p_coeff_10
        )
    )

    p_coeff_4 = lsq_rmat_rdiag_c_3 * (
          lsq_qtmat_c_3[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_3[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_3[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_3[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_3[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_3[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_3[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_3[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_3[:, 8] * (p_cc_e[:, 8] - p_cc)
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
          lsq_qtmat_c_2[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_2[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_2[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_2[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_2[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_2[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_2[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_2[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_2[:, 8] * (p_cc_e[:, 8] - p_cc)
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
          lsq_qtmat_c_1[:, 0] * (p_cc_e[:, 0] - p_cc)
        + lsq_qtmat_c_1[:, 1] * (p_cc_e[:, 1] - p_cc)
        + lsq_qtmat_c_1[:, 2] * (p_cc_e[:, 2] - p_cc)
        + lsq_qtmat_c_1[:, 3] * (p_cc_e[:, 3] - p_cc)
        + lsq_qtmat_c_1[:, 4] * (p_cc_e[:, 4] - p_cc)
        + lsq_qtmat_c_1[:, 5] * (p_cc_e[:, 5] - p_cc)
        + lsq_qtmat_c_1[:, 6] * (p_cc_e[:, 6] - p_cc)
        + lsq_qtmat_c_1[:, 7] * (p_cc_e[:, 7] - p_cc)
        + lsq_qtmat_c_1[:, 8] * (p_cc_e[:, 8] - p_cc)
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

def test_recon_lsq_cell_c_stencil():
    mesh = SimpleMesh()
    p_cc = random_field(mesh, CellDim, KDim)
    lsq_qtmat_c_1 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_2 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_3 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_4 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_5 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_6 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_7 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_8 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_9 = random_field(mesh, CellDim, C2E2C2E2CDim)
    lsq_qtmat_c_1_field = as_1D_sparse_field(lsq_qtmat_c_1, CECECDim)
    lsq_qtmat_c_2_field = as_1D_sparse_field(lsq_qtmat_c_2, CECECDim)
    lsq_qtmat_c_3_field = as_1D_sparse_field(lsq_qtmat_c_3, CECECDim)
    lsq_qtmat_c_4_field = as_1D_sparse_field(lsq_qtmat_c_4, CECECDim)
    lsq_qtmat_c_5_field = as_1D_sparse_field(lsq_qtmat_c_5, CECECDim)
    lsq_qtmat_c_6_field = as_1D_sparse_field(lsq_qtmat_c_6, CECECDim)
    lsq_qtmat_c_7_field = as_1D_sparse_field(lsq_qtmat_c_7, CECECDim)
    lsq_qtmat_c_8_field = as_1D_sparse_field(lsq_qtmat_c_8, CECECDim)
    lsq_qtmat_c_9_field = as_1D_sparse_field(lsq_qtmat_c_9, CECECDim)
    lsq_rmat_rdiag_c_1 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_2 = random_field(mesh, CellDim) 
    lsq_rmat_rdiag_c_3 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_4 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_5 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_6 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_7 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_8 = random_field(mesh, CellDim)
    lsq_rmat_rdiag_c_9 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_1 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_2 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_3 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_4 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_5 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_6 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_7 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_8 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_9 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_10 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_11 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_12 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_13 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_14 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_15 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_16 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_17 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_18 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_19 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_20 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_21 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_22 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_23 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_24 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_25 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_26 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_27 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_28 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_29 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_30 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_31 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_32 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_33 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_34 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_35 = random_field(mesh, CellDim)
    lsq_rmat_utri_c_36 = random_field(mesh, CellDim)
    lsq_moments_1 = random_field(mesh, CellDim)
    lsq_moments_2 = random_field(mesh, CellDim)
    lsq_moments_3 = random_field(mesh, CellDim)
    lsq_moments_4 = random_field(mesh, CellDim)
    lsq_moments_5 = random_field(mesh, CellDim)
    lsq_moments_6 = random_field(mesh, CellDim)
    lsq_moments_7 = random_field(mesh, CellDim)
    lsq_moments_8 = random_field(mesh, CellDim)
    lsq_moments_9 = random_field(mesh, CellDim)
    p_coeff_1 = zero_field(mesh, CellDim, KDim)
    p_coeff_2 = zero_field(mesh, CellDim, KDim)
    p_coeff_3 = zero_field(mesh, CellDim, KDim)
    p_coeff_4 = zero_field(mesh, CellDim, KDim)
    p_coeff_5 = zero_field(mesh, CellDim, KDim)
    p_coeff_6 = zero_field(mesh, CellDim, KDim)
    p_coeff_7 = zero_field(mesh, CellDim, KDim)
    p_coeff_8 = zero_field(mesh, CellDim, KDim)
    p_coeff_9 = zero_field(mesh, CellDim, KDim)
    p_coeff_10 = zero_field(mesh, CellDim, KDim)

    (ref_1, 
     ref_2,
     ref_3,
     ref_4,
     ref_5,
     ref_6,
     ref_7,
     ref_8,
     ref_9,
     ref_10) = recon_lsq_cell_c_stencil_numpy(
        mesh.c2e2c2e2c,
        np.asarray(p_cc),
        np.asarray(lsq_qtmat_c_1),
        np.asarray(lsq_qtmat_c_2),
        np.asarray(lsq_qtmat_c_3),
        np.asarray(lsq_qtmat_c_4),
        np.asarray(lsq_qtmat_c_5),
        np.asarray(lsq_qtmat_c_6),
        np.asarray(lsq_qtmat_c_7),
        np.asarray(lsq_qtmat_c_8),
        np.asarray(lsq_qtmat_c_9),
        np.asarray(lsq_rmat_rdiag_c_1),
        np.asarray(lsq_rmat_rdiag_c_2),
        np.asarray(lsq_rmat_rdiag_c_3),
        np.asarray(lsq_rmat_rdiag_c_4),
        np.asarray(lsq_rmat_rdiag_c_5),
        np.asarray(lsq_rmat_rdiag_c_6),
        np.asarray(lsq_rmat_rdiag_c_7),
        np.asarray(lsq_rmat_rdiag_c_8),
        np.asarray(lsq_rmat_rdiag_c_9),
        np.asarray(lsq_rmat_utri_c_1),
        np.asarray(lsq_rmat_utri_c_2),
        np.asarray(lsq_rmat_utri_c_3),
        np.asarray(lsq_rmat_utri_c_4),
        np.asarray(lsq_rmat_utri_c_5),
        np.asarray(lsq_rmat_utri_c_6),
        np.asarray(lsq_rmat_utri_c_7),
        np.asarray(lsq_rmat_utri_c_8),
        np.asarray(lsq_rmat_utri_c_9),
        np.asarray(lsq_rmat_utri_c_10),
        np.asarray(lsq_rmat_utri_c_11),
        np.asarray(lsq_rmat_utri_c_12),
        np.asarray(lsq_rmat_utri_c_13),
        np.asarray(lsq_rmat_utri_c_14),
        np.asarray(lsq_rmat_utri_c_15),
        np.asarray(lsq_rmat_utri_c_16),
        np.asarray(lsq_rmat_utri_c_17),
        np.asarray(lsq_rmat_utri_c_18),
        np.asarray(lsq_rmat_utri_c_19),
        np.asarray(lsq_rmat_utri_c_20),
        np.asarray(lsq_rmat_utri_c_21),
        np.asarray(lsq_rmat_utri_c_22),
        np.asarray(lsq_rmat_utri_c_23),
        np.asarray(lsq_rmat_utri_c_24),
        np.asarray(lsq_rmat_utri_c_25),
        np.asarray(lsq_rmat_utri_c_26),
        np.asarray(lsq_rmat_utri_c_27),
        np.asarray(lsq_rmat_utri_c_28),
        np.asarray(lsq_rmat_utri_c_29),
        np.asarray(lsq_rmat_utri_c_30),
        np.asarray(lsq_rmat_utri_c_31),
        np.asarray(lsq_rmat_utri_c_32),
        np.asarray(lsq_rmat_utri_c_33),
        np.asarray(lsq_rmat_utri_c_34),
        np.asarray(lsq_rmat_utri_c_35),
        np.asarray(lsq_rmat_utri_c_36),
        np.asarray(lsq_moments_1),
        np.asarray(lsq_moments_2),
        np.asarray(lsq_moments_3),
        np.asarray(lsq_moments_4),
        np.asarray(lsq_moments_5),
        np.asarray(lsq_moments_6),
        np.asarray(lsq_moments_7),
        np.asarray(lsq_moments_8),
        np.asarray(lsq_moments_9)
    )

    recon_lsq_cell_c_stencil(
        p_cc,
        lsq_qtmat_c_1_field,
        lsq_qtmat_c_2_field,
        lsq_qtmat_c_3_field,
        lsq_qtmat_c_4_field,
        lsq_qtmat_c_5_field,
        lsq_qtmat_c_6_field,
        lsq_qtmat_c_7_field,
        lsq_qtmat_c_8_field,
        lsq_qtmat_c_9_field,
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
        offset_provider={
            "C2E2C2E2C": mesh.get_c2e2c_offset_provider(),
            "C2CECEC": StridedNeighborOffsetProvider(CellDim, CECECDim, mesh.n_c2e2c2e2c),
        },
    )
    co1 = np.asarray(p_coeff_1)
    co2 = np.asarray(p_coeff_2)
    co3 = np.asarray(p_coeff_3)
    co4 = np.asarray(p_coeff_4)
    co5 = np.asarray(p_coeff_5)
    co6 = np.asarray(p_coeff_6)
    co7 = np.asarray(p_coeff_7)
    co8 = np.asarray(p_coeff_8)
    co9 = np.asarray(p_coeff_9)
    co10 = np.asarray(p_coeff_10)
    assert np.allclose(ref_1, co1)
    assert np.allclose(ref_2, co2)
    assert np.allclose(ref_3, co3)
    assert np.allclose(ref_4, co4)
    assert np.allclose(ref_5, co5)
    assert np.allclose(ref_6, co6)
    assert np.allclose(ref_7, co7)
    assert np.allclose(ref_8, co8)
    assert np.allclose(ref_9, co9)
    assert np.allclose(ref_10, co10)
