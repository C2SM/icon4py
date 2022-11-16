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


class GridConfig:
    def __init__(self):
        self._n_lev = 65
        # see mo_model_domimp_patches.f90 l. 405
        self.n_shift_total = 0

    def get_k_size(self):
        return self._n_lev

    def get_n_shift(self):
        return self.n_shift_total
