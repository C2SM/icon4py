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

from dataclasses import dataclass

from functional.common import Field

from icon4py.common.dimension import C2E2CDim, CellDim, EdgeDim, KDim


@dataclass
class MetricState:
    theta_ref_mc: Field[[CellDim, KDim], float]
    enhfac_diffu: Field[
        [KDim], float
    ]  # Enhancement factor for nabla4 background diffusion TODO check dimension
    wgtfac_e: Field[
        [EdgeDim, KDim], float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_e)
    wgtfac_c: Field[
        [CellDim, KDim], float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    wgtfacq1_e: Field[
        [
            EdgeDim,
        ],
        float,
    ]  # weighting factor for quadratic interpolation to model top (nproma,3,nblks_e)
    wgtfacq_e: Field[
        [
            EdgeDim,
        ],
        float,
    ]  # weighting factor for quadratic interpolation to surface (nproma,3,nblks_e)
    ddqz_z_full_e: Field[
        [EdgeDim, KDim], float
    ]  # functional determinant of the metrics [sqrt(gamma)] (nproma,nlev,nblks_e)
    mask_hdiff: Field[[CellDim, KDim], int]
    zd_vertidx_dsl: Field[[CellDim, C2E2CDim, KDim], int]
    zd_diffcoef: Field[[CellDim, KDim], float]
    zd_intcoef: Field[[CellDim, KDim], float]
