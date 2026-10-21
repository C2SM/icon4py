# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Constants of the surface scheme that are not shared physical constants."""

import enum
import math

from icon4py.model.common import type_alias as ta


# a wpfloat-based Enum: what gtfn needs for constants referenced inside field operators
class LouisConstants(ta.wpfloat, enum.Enum):
    """Parameters of the Louis surface-layer scheme (mo_vdf_diag_smag.f90)."""

    #: Businger-Dyer coefficient of the stable momentum profile, bsm in ICON
    businger_stable_momentum = 5.0
    #: Businger-Dyer coefficient of the unstable momentum profile, bum in ICON
    businger_unstable_momentum = 16.0
    #: Businger-Dyer coefficient of the stable heat profile, bsh in ICON
    businger_stable_heat = 5.0
    #: Businger-Dyer coefficient of the unstable heat profile, buh in ICON
    businger_unstable_heat = 16.0
    #: lower bound of the square root of the neutral transfer coefficients, zepsec in ICON
    minimum_sqrt_neutral_coefficient = 0.028
    #: pi / 2 of the unstable momentum profile (mo_math_constants.f90)
    half_pi = math.pi / 2.0
    #: ln 2 of the unstable profiles (mo_math_constants.f90)
    ln_2 = math.log(2.0)
