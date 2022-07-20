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

"""
Provide description.

TODO:
- Change documentation such that description appears when hovering over symbol  in IDE's
- Finish port
"""

from icon4py.shared.mo_physical_constants import (
    als,
    alv,
    cpd,
    rd,
    rv,
    tmelt,
)


# Constants used for the computation of lookup tables of the saturation
# mixing ratio over liquid water (*c_les*) or ice(*c_ies*)
c1es = 610.78
c2es = c1es * rd / rv
c3les = 17.269
c3ies = 21.875
c4les = 35.86
c4ies = 7.66
c5les = c3les * (tmelt - c4les)
c5ies = c3ies * (tmelt - c4ies)
c5alvcp = c5les * alv / cpd
c5alscp = c5ies * als / cpd
alvdcp = alv / cpd
alsdcp = als / cpd
