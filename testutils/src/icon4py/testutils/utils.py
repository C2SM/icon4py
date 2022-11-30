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

from typing import Optional

import numpy as np
import numpy.typing as npt
from functional import common as gt_common
from functional.iterator import embedded as it_embedded

from . import simple_mesh


def random_mask(
    mesh: simple_mesh.SimpleMesh,
    *dims: gt_common.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
) -> it_embedded.MutableLocatedField:
    shape = tuple(map(lambda x: mesh.size[x], dims))
    arr = np.full(shape, False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if dtype:
        arr = arr.astype(dtype)
    return it_embedded.np_as_located_field(*dims)(arr)


def random_field(
    mesh, *dims, low: float = -1.0, high: float = 1.0
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.random.default_rng().uniform(
            low=low, high=high, size=tuple(map(lambda x: mesh.size[x], dims))
        )
    )


def zero_field(
    mesh: simple_mesh.SimpleMesh, *dims: gt_common.Dimension, dtype=float
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.zeros(shape=tuple(map(lambda x: mesh.size[x], dims)), dtype=dtype)
    )


def constant_field(
    mesh: simple_mesh.SimpleMesh, value: float, *dims: gt_common.Dimension, dtype=float
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        value * np.ones(shape=tuple(map(lambda x: mesh.size[x], dims)), dtype=dtype)
    )


def as_1D_sparse_field(
    field: it_embedded.MutableLocatedField, dim: gt_common.Dimension
) -> it_embedded.MutableLocatedField:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    old_shape = np.asarray(field).shape
    assert len(old_shape) == 2
    new_shape = (old_shape[0] * old_shape[1],)
    return it_embedded.np_as_located_field(dim)(np.asarray(field).reshape(new_shape))


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"icon4py.{stencil_module}.{stencil_name}:{stencil_name}"
