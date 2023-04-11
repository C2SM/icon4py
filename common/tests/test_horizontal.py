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

from icon4py.common.dimension import CellDim
from icon4py.grid.horizontal import Nudging, nudging


def test_nudging_class_for_cells():
    nudging = Nudging(CellDim)
    nudging_plus_one = Nudging(CellDim) + 1
    assert nudging() == 13
    assert nudging_plus_one == 14


def test_nudging_for_cells():
    nudge = nudging(CellDim)
    nudge_p1 = nudging(CellDim, 1)
    assert nudge == 13
    assert nudge_p1 == 14


def get_start_index(i):
    pass


def test_start_index():
    get_start_index(nudging(CellDim, 1))
