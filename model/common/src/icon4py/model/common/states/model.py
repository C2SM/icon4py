# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import functools
from typing import Literal, Protocol, TypedDict, Union, runtime_checkable

import gt4py._core.definitions as gt_coredefs
import gt4py.next as gtx
import gt4py.next.common as gt_common
import numpy.typing as np_t

import icon4py.model.common.type_alias as ta


"""Contains type definitions used for the model`s state representation."""
DimensionNames = Literal["cell", "edge", "vertex"]
DimensionT = Union[gtx.Dimension, DimensionNames]  # TODO use Literal instead of str
BufferT = Union[np_t.ArrayLike, gtx.Field]
DTypeT = Union[ta.wpfloat, ta.vpfloat, gtx.int32, gtx.int64, gtx.float32, gtx.float64]


class OptionalMetaData(TypedDict, total=False):
    #: is optional in CF conventions for downwards compatibility with COARDS
    long_name: str
    #: we might not have this one for all fields. But it is useful to have it for tractability with ICON
    icon_var_name: str
    # TODO (@halungge) dims should probably be required?
    dims: tuple[DimensionT, ...]
    dtype: Union[ta.wpfloat, ta.vpfloat, gtx.int32, gtx.int64, gtx.float32, gtx.float64]


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
    data: gtx.Field[gtx.Dims[gt_common.DimsT], gt_coredefs.ScalarT]
    attrs: FieldMetaData

    @functools.cached_property
    def metadata(self) -> FieldMetaData:
        return self.attrs
