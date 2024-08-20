from typing import Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import xarray as xa

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.settings import xp


T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)
DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)
Scalar: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64]

FieldType: TypeAlias = Union[gtx.Field[Sequence[gtx.Dims[DimT]], T], xp.ndarray]

def to_data_array(field:FieldType, attrs:dict):
    data = field if isinstance(field, xp.ndarray) else field.ndarray
    return xa.DataArray(data, attrs=attrs)
