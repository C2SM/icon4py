# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import enum
import functools
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import gt4py._core.definitions as gt_coredefs
import gt4py.next as gtx
import gt4py.next.common as gt_common
import numpy.typing as np_t

import icon4py.model.common.type_alias as ta


"""Contains type definitions used for the model`s state representation."""
type DimensionNames = Literal["cell", "edge", "vertex"]
type BufferT = np_t.ArrayLike | gtx.Field
type DTypeT = ta.wpfloat | ta.vpfloat | gtx.int32 | gtx.int64 | gtx.float32 | gtx.float64


class FieldKind(enum.StrEnum):
    """A component output that its consumer must handle specially.

    Only kinds that change what a consumer does with an output belong here; an
    output with no kind is stored as it comes. There is deliberately no member
    for that default: it would mean the same thing as declaring nothing, and a
    later kind (prognostics, say) would make "everything else" ambiguous.
    """

    #: a rate: applied to the target field as ``field += value * dt``
    TENDENCY = "tendency"


@dataclasses.dataclass(frozen=True, kw_only=True)
class FieldMetaData:
    """CF-style metadata describing one model field.

    Attribute access is the interface (``meta.units``, ``meta.kind``); an unset
    optional entry reads as ``None``. ``as_dict`` renders the set entries as the
    plain attribute mapping the IO layer writes to netCDF.
    """

    #: CF conventions
    standard_name: str
    #: CF conventions
    units: str
    #: is optional in CF conventions for downwards compatibility with COARDS
    long_name: str | None = None
    #: CF direction of a vertical coordinate ("up" / "down")
    positive: Literal["up", "down"] | None = None
    #: we might not have this one for all fields. But it is useful to have it for tractability with ICON
    icon_var_name: str | None = None
    #: list index for variables stored in fortran lists (e.g. tracers)
    icon_var_list_index: int | None = None
    # TODO(halungge): dims should probably be required?
    dims: Sequence[gtx.Dimension] | None = None
    dtype: ta.wpfloat | ta.vpfloat | gtx.int32 | gtx.int64 | gtx.float32 | gtx.float64 | None = None
    #: whether the vertical dimension of the field lives on interface (half) levels
    #: rather than full levels
    is_on_half_levels: bool | None = None
    #: set when a consumer must handle this output specially; see ``FieldKind``
    kind: FieldKind | None = None

    def as_dict(self) -> dict[str, Any]:
        """The set entries, as the attribute mapping used for IO.

        Shallow by design: ``dims`` stays a tuple of ``gtx.Dimension`` rather
        than being recursed into, and unset optional entries are omitted so the
        written attributes carry only what was actually declared.
        """
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }


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
