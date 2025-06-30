# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

class graupel_ct:
    qmin = 1.0e-15  # threshold for computation

class idx:
    prefactor_r = 14.58
    exponent_r = 0.111
    offset_r = 1.0e-12
    prefactor_i = 1.25
    exponent_i = 0.160
    offset_i = 1.0e-12
    prefactor_s = 57.80
    exponent_s = 0.5
    offset_s = 1.0e-12
    prefactor_g = 12.24
    exponent_g = 0.217
    offset_g = 1.0e-08
