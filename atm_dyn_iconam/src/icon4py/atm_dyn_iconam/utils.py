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
from functional.common import Dimension
from functional.iterator.embedded import np_as_located_field


# TODO fix duplication: duplicated from test testutils/utils.py
def zero_field(mesh, *dims: Dimension, dtype=float):
    shapex = tuple(map(lambda x: mesh.size[x], dims))
    return np_as_located_field(*dims)(np.zeros(shapex, dtype=dtype))
