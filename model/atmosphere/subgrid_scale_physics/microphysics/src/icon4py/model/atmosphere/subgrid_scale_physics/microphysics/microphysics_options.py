# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Literal

import gt4py.next as gtx
from gt4py.eve import utils as eve_utils


class LiquidAutoConversionType(eve_utils.FrozenNamespace[gtx.int32]):
    """
    Options for computing liquid auto conversion rate
    """

    #: Kessler (1969) liquid auto conversion mode
    KESSLER = 0
    #: Seifert & Beheng (2006) liquid auto conversion mode
    SEIFERT_BEHENG = 1


class SnowInterceptParametererization(eve_utils.FrozenNamespace[gtx.int32]):
    """
    Options for deriving snow intercept parameter
    """

    #: Estimated intercept parameter for the snow size distribution from the best-fit line in figure 10(a) of Field et al. (2005)
    FIELD_BEST_FIT_ESTIMATION = 1
    #: Estimated intercept parameter for the snow size distribution from the general moment equation in table 2 of Field et al. (2005)
    FIELD_GENERAL_MOMENT_ESTIMATION = 2


ValidLiquidAutoConversionType = Literal[
    LiquidAutoConversionType.KESSLER, LiquidAutoConversionType.SEIFERT_BEHENG
]
ValidSnowInterceptParametererization = Literal[
    SnowInterceptParametererization.FIELD_BEST_FIT_ESTIMATION,
    SnowInterceptParametererization.FIELD_GENERAL_MOMENT_ESTIMATION,
]
