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
from functional import common as gt_common
from functional.iterator import embedded as it_embedded
from hypothesis import strategies as st
from hypothesis import target
from hypothesis.extra.numpy import arrays as hypothesis_array

from . import simple_mesh


def shape(
    obj: Union[tuple, np.ndarray, simple_mesh.SimpleMesh], *dims: gt_common.Dimension
):
    if isinstance(obj, simple_mesh.SimpleMesh):
        return tuple(map(lambda x: obj.size[x], dims))
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.shape
    raise NotImplementedError(f"Cannot get shape of {type(obj)}")


def random_mask(
    mesh: simple_mesh.SimpleMesh,
    *dims: gt_common.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
) -> it_embedded.MutableLocatedField:
    arr = np.full(shape, False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=shape(mesh, dims))
    if dtype:
        arr = arr.astype(dtype)
    return it_embedded.np_as_located_field(*dims)(arr)


def to_icon4py_field(
    field, *dims: gt_common.Dimension, dtype=float
) -> it_embedded.MutableLocatedField:
    """Copy a numpy field into an field with named dimensions."""
    return it_embedded.np_as_located_field(*dims)(field)


def random_field(
    mesh, *dims, low: float = -1.0, high: float = 1.0
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.random.default_rng().uniform(low=low, high=high, size=shape(mesh, dims))
    )


def zero_field(
    mesh: Union[simple_mesh.SimpleMesh, np.ndarray],
    *dims: gt_common.Dimension,
    dtype=float,
) -> it_embedded.MutableLocatedField:
    return it_embedded.np_as_located_field(*dims)(
        np.zeros(shape=shape(mesh, dims), dtype=dtype)
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


def random_field_strategy(
    mesh, *dims, min_value=None, max_value=None
) -> st.SearchStrategy[float]:
    """Return a hypothesis strategy of a random field."""
    return hypothesis_array(
        dtype=np.float64,
        shape=tuple(map(lambda x: mesh.size[x], dims)),
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            exclude_min=min_value is not None,
            allow_nan=False,
            allow_infinity=False,
        ),
    ).map(it_embedded.np_as_located_field(*dims))


def maximizeTendency(fld, refFld, varname):
    """Make hypothesis maximize mean and std of tendency."""
    tendency = np.asarray(fld) - refFld
    target(np.mean(np.abs(tendency)), label=f"{varname} mean tendency")
    target(np.std(tendency), label=f"{varname} stdev. tendency")
