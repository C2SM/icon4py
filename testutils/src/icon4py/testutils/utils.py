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
from functional.iterator.embedded import np_as_located_field


def random_field(mesh, *dims):
    return np_as_located_field(*dims)(
        np.random.randn(*map(lambda x: mesh.size[x], dims))
    )


def zero_field(mesh, *dims):
    return np_as_located_field(*dims)(
        np.zeros(shape=tuple(map(lambda x: mesh.size[x], dims)))
    )


def get_cell_to_k_table(k_arr, k):
    """Create cell to k table based on an input array and k value.

    Args:
        k_arr: 1D input array holding k values.
        k: k value to use within k table.

    Returns:
        2D array filled with k values
    """
    # creating cell to k table
    c2k = np.expand_dims(k_arr, axis=-1)
    return np.repeat(c2k[:], k, axis=-1)


def get_stencil_module_path(module, stencil_name) -> str:
    return f"icon4py.{module}.{stencil_name}:{stencil_name}"
