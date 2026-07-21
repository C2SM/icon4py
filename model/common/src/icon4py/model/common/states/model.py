# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict, runtime_checkable

import gt4py.next as gtx
import gt4py.next.common as gt_common
import numpy.typing as np_t
from gt4py._core import definitions as gt_coredefs

import icon4py.model.common.type_alias as ta


"""Contains type definitions used for the model`s state representation."""
type DimensionNames = Literal["cell", "edge", "vertex"]
type BufferT = np_t.ArrayLike | gtx.Field
type DTypeT = (
    type[ta.wpfloat]
    | type[ta.vpfloat]
    | type[gtx.int32]
    | type[gtx.int64]
    | type[gtx.float32]
    | type[gtx.float64]
    | type[bool]
    | type[int]
)


class OptionalMetaData(TypedDict, total=False):
    #: is optional in CF conventions for downwards compatibility with COARDS
    long_name: str
    #: we might not have this one for all fields. But it is useful to have it for tractability with ICON
    icon_var_name: str
    #: list index for variables stored in fortran lists (e.g. tracers)
    icon_var_list_index: int
    # TODO(halungge): dims should probably be required?
    dims: Sequence[gtx.Dimension]
    dtype: (
        type[ta.wpfloat]
        | type[ta.vpfloat]
        | type[gtx.int32]
        | type[gtx.int64]
        | type[gtx.float32]
        | type[gtx.float64]
        | type[bool]
        | type[int]
    )
    #: whether the vertical dimension of the field lives on interface (half) levels
    #: rather than full levels
    is_on_half_levels: bool
    #: CF convention: direction of increase for vertical coordinate
    positive: str


class RequiredMetaData(TypedDict, total=True):
    #: CF conventions
    standard_name: str
    #: CF conventions
    units: str


class FieldMetaData(RequiredMetaData, OptionalMetaData):
    pass


@runtime_checkable
class DataField(Protocol):
    """Protocol that should be implemented by icon4py model fields and xarray.DataArray"""

    data: BufferT
    attrs: dict


@dataclasses.dataclass
class ModelField(DataField):
    data: gtx.Field[gtx.Dims[gt_common.DimsT], gt_coredefs.Scalar]
    attrs: FieldMetaData  # type: ignore[assignment]  # FieldMetaData is a TypedDict, not a subtype of dict in mypy

    @functools.cached_property
    def metadata(self) -> FieldMetaData:
        return self.attrs
