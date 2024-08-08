import functools
import operator
from enum import IntEnum
from typing import Optional, Protocol, Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import gt4py.next.ffront.decorator as gtx_decorator
import xarray as xa

import icon4py.model.common.type_alias as ta
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.grid import icon
from icon4py.model.common.settings import xp


T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)
DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)
Scalar: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64]

FieldType:TypeAlias = gtx.Field[Sequence[gtx.Dims[DimT]], T]
class RetrievalType(IntEnum):
    FIELD = 0,
    DATA_ARRAY = 1,
    METADATA = 2,

_attrs = {"functional_determinant_of_the_metrics_on_half_levels":dict(
            standard_name="functional_determinant_of_the_metrics_on_half_levels",
            long_name="functional determinant of the metrics [sqrt(gamma)] on half levels",
            units="",
            dims=(dims.CellDim, dims.KHalfDim),
            dtype=ta.wpfloat,
            icon_var_name="ddqz_z_half",
            ), 
        "height": dict(standard_name="height", 
                       long_name="height", 
                       units="m", 
                       dims=(dims.CellDim, dims.KDim), 
                       icon_var_name="z_mc", dtype = ta.wpfloat) ,
        "height_on_interface_levels": dict(standard_name="height_on_interface_levels", 
                                           long_name="height_on_interface_levels", 
                                           units="m", 
                                           dims=(dims.CellDim, dims.KHalfDim), 
                                           icon_var_name="z_ifc", 
                                           dtype = ta.wpfloat),
        "model_level_number": dict(standard_name="model_level_number", 
                                   long_name="model level number", 
                                   units="", dims=(dims.KHalfDim,), 
                                   icon_var_name="k_index", 
                                   dtype = gtx.int32),
    }

class FieldProvider(Protocol):
    """
    Protocol for field providers.
    
    A field provider is responsible for the computation and caching of a set of fields.
    The fields can be accessed by their field_name (str).
    
    A FieldProvider has to methods:
     - evaluate: computes the fields based on the instructions of concrete implementation
     - get: returns the field with the given field_name.
    
    """
    def evaluate(self) -> None:
        pass
    
    def get(self, field_name: str) -> FieldType:
        pass
    
    def fields(self) -> Sequence[str]:
        pass
        


class PrecomputedFieldsProvider:
    """Simple FieldProvider that does not do any computation but gets its fields at construction and returns it upon provider.get(field_name)."""
    
    def __init__(self, fields: dict[str, FieldType]):
        self._fields = fields
        
    def evaluate(self):
        pass
    def get(self, field_name: str) -> FieldType:
        return self._fields[field_name]

    def fields(self) -> Sequence[str]:
        return self._fields.keys()

class ProgramFieldProvider:
        """
        Computes a field defined by a GT4Py Program.

        """

        def __init__(self,
                     outer: 'MetricsFieldsFactory',  #
                     func: gtx_decorator.Program,
                     domain: dict[gtx.Dimension:tuple[int, int]],  # the compute domain 
                     fields: Sequence[str],
                     deps: Sequence[str] = [],  # the dependencies of func
                     params: Sequence[str] = [],  # the parameters of func
                     ):
            self._factory = outer
            self._compute_domain = domain
            self._dims = domain.keys()
            self._func = func
            self._dependencies = {k: self._factory._providers[k] for k in deps}
            self._params = {k: self._factory._params[k] for k in params}

            self._fields: dict[str, Optional[gtx.Field | Scalar]] = {name: None for name in fields}

        def _map_dim(self, dim: gtx.Dimension) -> gtx.Dimension:
            if dim == dims.KHalfDim:
                return dims.KDim
            return dim

        def _allocate(self):
            field_domain = {self._map_dim(dim): (0, self._factory._sizes[dim]) for dim in
                            self._dims}
            return {k: self._factory._allocator(field_domain, dtype=_attrs[k]["dtype"]) for k, v in
                    self._fields.items()}

        def _unallocated(self) -> bool:
            return not all(self._fields.values())

        def evaluate(self):
            self._fields = self._allocate()

            domain = functools.reduce(operator.add, self._compute_domain.values())
            # args = {k: provider.get(k) for k, provider in self._dependencies.items()}
            args = [self._dependencies[k].get(k) for k in self._dependencies.keys()]
            params = [p for p in self._params.values()]
            output = [f for f in self._fields.values()]
            self._func(*args, *output, *params, *domain,
                       offset_provider=self._factory._grid.offset_providers)

        def fields(self):
            return self._fields.keys()
        def get(self, field_name: str):
            if field_name not in self._fields.keys():
                raise ValueError(f"Field {field_name} not provided by f{self._func.__name__}")
            if self._unallocated():
                self.evaluate()
            return self._fields[field_name]


class MetricsFieldsFactory:
    """
    Factory for metric fields.
    """

    
    def __init__(self, grid:icon.IconGrid, z_ifc:gtx.Field, backend=settings.backend):
        self._grid = grid
        self._sizes = grid.size
        self._sizes[dims.KHalfDim] = self._sizes[dims.KDim] + 1
        self._providers: dict[str, 'FieldProvider'] = {}
        self._params = {"num_lev": grid.num_levels, }
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)

        k_index = gtx.as_field((dims.KDim,), xp.arange(grid.num_levels + 1, dtype=gtx.int32))

        pre_computed_fields = PrecomputedFieldsProvider(
            {"height_on_interface_levels": z_ifc, "model_level_number": k_index})
        self.register_provider(pre_computed_fields)
        
    def register_provider(self, provider:FieldProvider):
        for field in provider.fields():
            self._providers[field] = provider
        
    
    def get(self, field_name: str, type_: RetrievalType):
        if field_name not in _attrs:
            raise ValueError(f"Field {field_name} not found in metric fields")
        if type_ == RetrievalType.METADATA:
            return _attrs[field_name]
        if type_ == RetrievalType.FIELD:
            return self._providers[field_name].get(field_name)
        if type_ == RetrievalType.DATA_ARRAY:
            return to_data_array(self._providers[field_name].get(field_name), _attrs[field_name])
        raise ValueError(f"Invalid retrieval type {type_}")


def to_data_array(field, attrs):
    return xa.DataArray(field, attrs=attrs)






    
    


