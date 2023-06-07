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

from gt4py.next.common import Field
from numpy import int32

from icon4py.common.dimension import CECDim, CellDim, KDim


@dataclass
class MetricState:
    theta_ref_mc: Field[[CellDim, KDim], float]
    wgtfac_c: Field[
        [CellDim, KDim], float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    mask_hdiff: Field[[CellDim, KDim], bool]
    zd_vertoffset: Field[[CECDim, KDim], int32]
    zd_diffcoef: Field[[CellDim, KDim], float]
    zd_intcoef: Field[[CECDim, KDim], float]
