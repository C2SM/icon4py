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
import numpy.typing as npt
from functional import common as gt_common
from functional.iterator import embedded as it_embedded

from . import simple_mesh


def random_mask(
    mesh: simple_mesh.SimpleMesh, *dims: gt_common.Dimension, numeric: bool = False
) -> it_embedded.MutableLocatedField:
    shape = tuple(map(lambda x: mesh.size[x], dims))
    arr = np.full(shape, False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if numeric:
        arr = arr.astype("int")
    return it_embedded.np_as_located_field(*dims)(arr)


def random_field(
    mesh: simple_mesh.SimpleMesh, *dims: gt_common.Dimension
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.random.randn(*map(lambda x: mesh.size[x], dims))
    )


def broadcasted_field(
    num: float, mesh: simple_mesh.SimpleMesh, *dims: gt_common.Dimension
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.full(tuple(map(lambda x: mesh.size[x], dims)), num)
    )


def zero_field(
    mesh: simple_mesh.SimpleMesh, *dims: gt_common.Dimension
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.zeros(shape=tuple(map(lambda x: mesh.size[x], dims)))
    )


def get_cell_to_k_table(k_arr: npt.NDArray, k: int) -> npt.NDArray:
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


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"icon4py.{stencil_module}.{stencil_name}:{stencil_name}"
