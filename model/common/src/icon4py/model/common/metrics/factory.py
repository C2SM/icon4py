from enum import IntEnum
from typing import Sequence

import gt4py.next as gtx
import xarray as xa

import icon4py.model.common.metrics.metric_fields as metrics
import icon4py.model.common.type_alias as ta
from icon4py.model.common.dimension import CellDim, KDim, KHalfDim
from icon4py.model.common.grid import icon
from icon4py.model.common.grid.base import BaseGrid


class RetrievalType(IntEnum):
    FIELD = 0,
    DATA_ARRAY = 1,
    METADATA = 2,

_attrs = {"functional_determinant_of_the_metrics_on_half_levels":dict(
            standard_name="functional_determinant_of_the_metrics_on_half_levels",
            long_name="functional determinant of the metrics [sqrt(gamma)] on half levels",
            units="",
            dims=(CellDim, KHalfDim),
            icon_var_name="ddqz_z_half",
            ), 
        "height": dict(standard_name="height", long_name="height", units="m", dims=(CellDim, KDim), icon_var_name="z_mc"), 
        "height_on_interface_levels": dict(standard_name="height_on_interface_levels", long_name="height_on_interface_levels", units="m", dims=(CellDim, KHalfDim), icon_var_name="z_ifc")
    }


class FieldProviderImpl:
    """
    In charge of computing a field and providing metadata about it.
    TODO: change for tuples of fields

    """

    # TODO that should be a sequence or a dict of fields, since func -> tuple[...]
    def __init__(self, grid: BaseGrid, deps: Sequence['FieldProvider'], attrs: dict):
        self.grid = grid
        self.dependencies = deps
        self._attrs = attrs
        self.func = metrics.compute_z_mc
        self.fields:Sequence[gtx.Field|None] = []

    # TODO (@halungge) handle DType
    def _allocate(self, fields:Sequence[gtx.Field], dimensions: Sequence[gtx.Dimension]):
        domain = {dim: (0, self.grid.size[dim]) for dim in dimensions}
        return [gtx.constructors.zeros(domain, dtype=ta.wpfloat) for _ in fields]

    def __call__(self):
        if not self.fields:
            self.field = self._allocate(self.fields, self._attrs["dims"])
            domain = (0, self.grid.num_cells, 0, self.grid.num_levels)
            args = [dep(RetrievalType.FIELD) for dep in self.dependencies]
            self.field = self.func(*args, self.field, *domain,
                                   offset_provider=self.grid.offset_providers)
        return self.field


class SimpleFieldProvider:
    def id(x: gtx.Field) -> gtx.Field:
        return x

    def __init__(self, grid: BaseGrid, field, attrs):
        super().__init__(grid, deps=(), attrs=attrs)
        self.func = self.id
        self.field = field


# class FieldProvider(Protocol):
#     
#     func = metrics.compute_ddqz_z_half
#     field: gtx.Field[gtx.Dims[CellDim, KDim], ta.wpfloat] = None
#     
#     def __init__(self, grid:BaseGrid, func,  deps: Sequence['FieldProvider''], attrs):
#         super().__init__(grid, deps=deps, attrs=attrs)
#         self.func = func

class MetricsFieldsFactory:
    """
    Factory for metric fields.
    """
    def __init__(self, grid:icon.IconGrid, z_ifc:gtx.Field):
        self.grid = grid
        self.z_ifc_provider = SimpleFieldProvider(self.grid, z_ifc, _attrs["height_on_interface_levels"])
        self._providers = {"height_on_interface_levels": self.z_ifc_provider}
    
        z_mc_provider = None
        z_ddqz_provider = None
    #   TODO (@halungge) use TypedDict
        self._providers["functional_determinant_of_the_metrics_on_half_levels"]= z_ddqz_provider
        self._providers["height"] = z_mc_provider
    
    
    def get(self, field_name: str, type_: RetrievalType):
        if field_name not in _attrs:
            raise ValueError(f"Field {field_name} not found in metric fields")
        if type_ == RetrievalType.METADATA:
            return _attrs[field_name]
        if type_ == RetrievalType.FIELD:
            return self._providers[field_name]()
        if type_ == RetrievalType.DATA_ARRAY:
            return to_data_array(self._providers[field_name](), _attrs[field_name])
        raise ValueError(f"Invalid retrieval type {type_}")


def to_data_array(field, attrs):
    return xa.DataArray(field, attrs=attrs)






    
    


