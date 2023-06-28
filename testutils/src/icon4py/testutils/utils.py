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


from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from gt4py.next import common as gt_common
from gt4py.next.iterator import embedded as it_embedded
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import arrays as hypothesis_array

from . import simple_mesh


def objShape(
    obj: Union[tuple, np.ndarray, simple_mesh.SimpleMesh], *dims: gt_common.Dimension
):

    if isinstance(obj, simple_mesh.SimpleMesh):
        return tuple(map(lambda x: obj.size[x], dims))
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.shape


def _shape(
    mesh,
    *dims: gt_common.Dimension,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
):
    if extend is None:
        extend = {}
    for d in dims:
        if d not in extend.keys():
            extend[d] = 0
    return tuple(mesh.size[dim] + extend[dim] for dim in dims)


def random_mask(
    mesh: simple_mesh.SimpleMesh,
    *dims: gt_common.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
) -> it_embedded.MutableLocatedField:
    shape = _shape(mesh, *dims, extend=extend)
    arr = np.full(shape, False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if dtype:
        arr = arr.astype(dtype)
    return it_embedded.np_as_located_field(*dims)(arr)


def to_icon4py_field(
    field: Union[tuple, np.ndarray, simple_mesh.SimpleMesh],
    *dims: gt_common.Dimension,
    dtype=float,
) -> it_embedded.MutableLocatedField:
    """Copy a numpy field into an field with named dimensions."""
    return it_embedded.np_as_located_field(*dims)(field.copy())


def random_field(
    mesh,
    *dims,
    low: float = -1.0,
    high: float = 1.0,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.random.default_rng().uniform(
            low=low, high=high, size=_shape(mesh, *dims, extend=extend)
        )
    )


def zero_field(
    mesh: simple_mesh.SimpleMesh,
    *dims: gt_common.Dimension,
    dtype=float,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.zeros(shape=_shape(mesh, *dims, extend=extend), dtype=dtype)
    )


def constant_field(
    mesh: simple_mesh.SimpleMesh, value: float, *dims: gt_common.Dimension, dtype=float
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        value * np.ones(shape=tuple(map(lambda x: mesh.size[x], dims)), dtype=dtype)
    )


def random_field_strategy(
    mesh: Union[tuple, np.ndarray, simple_mesh.SimpleMesh],
    *dims,
    min_value=None,
    max_value=None,
) -> st.SearchStrategy[float]:
    """Return a hypothesis strategy of a random field."""
    return hypothesis_array(
        dtype=np.float64,
        shape=objShape(mesh, *dims),
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            exclude_min=min_value is not None,
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(it_embedded.np_as_located_field(*dims))


def as_1D_sparse_field(
    field: it_embedded.MutableLocatedField, dim: gt_common.Dimension
) -> it_embedded.MutableLocatedField:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    old_shape = np.asarray(field).shape
    assert len(old_shape) == 2
    new_shape = (old_shape[0] * old_shape[1],)
    return it_embedded.np_as_located_field(dim)(np.asarray(field).reshape(new_shape))


def flatten_first_two_dims(
    *dims: gt_common.Dimension, field: it_embedded.MutableLocatedField
) -> it_embedded.MutableLocatedField:
    """Convert a n-D sparse field to a (n-1)-D flattened (Felix-style) sparse field."""
    old_shape = np.asarray(field).shape
    assert len(old_shape) >= 2
    flattened_size = old_shape[0] * old_shape[1]
    flattened_shape = (flattened_size,)
    new_shape = flattened_shape + old_shape[2:]
    newarray = np.asarray(field).reshape(new_shape)
    return it_embedded.np_as_located_field(*dims)(newarray)


def get_stencil_module_path(stencil_module: str, stencil_name: str) -> str:
    return f"icon4py.{stencil_module}.{stencil_name}:{stencil_name}"
