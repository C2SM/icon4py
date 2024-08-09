import abc
import functools
import operator
from enum import IntEnum
from typing import Iterable, Optional, Protocol, Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import gt4py.next.ffront.decorator as gtx_decorator
import xarray as xa

import icon4py.model.common.type_alias as ta
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.io import cf_utils


T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)
DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)
Scalar: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64]

FieldType:TypeAlias = gtx.Field[Sequence[gtx.Dims[DimT]], T]
class RetrievalType(IntEnum):
    FIELD = 0,
    DATA_ARRAY = 1,
    METADATA = 2,

_attrs = {"functional_determinant_of_metrics_on_interface_levels":dict(
            standard_name="functional_determinant_of_metrics_on_interface_levels",
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
                                   units="", dims=(dims.KDim,), 
                                   icon_var_name="k_index", 
                                   dtype = gtx.int32),
        cf_utils.INTERFACE_LEVEL_STANDARD_NAME: dict(standard_name=cf_utils.INTERFACE_LEVEL_STANDARD_NAME,
                               long_name="model interface level number",
                               units="", dims=(dims.KHalfDim,),
                               icon_var_name="k_index",
                               dtype=gtx.int32),
    }





class FieldProvider(Protocol):
    """
    Protocol for field providers.
    
    A field provider is responsible for the computation and caching of a set of fields.
    The fields can be accessed by their field_name (str).
    
    A FieldProvider has three methods:
     - evaluate: computes the fields based on the instructions of concrete implementation
     - get: returns the field with the given field_name.
    - fields: returns the list of field names provided by the
    
    """
    @abc.abstractmethod
    def _evaluate(self, factory:'FieldsFactory') -> None:
        pass

    @abc.abstractmethod
    def __call__(self, field_name: str, factory:'FieldsFactory') -> FieldType:
        pass

    @abc.abstractmethod
    def dependencies(self) -> Iterable[str]:
        pass

    @abc.abstractmethod
    def fields(self) -> Iterable[str]:
        pass
        

class PrecomputedFieldsProvider(FieldProvider):
    """Simple FieldProvider that does not do any computation but gets its fields at construction and returns it upon provider.get(field_name)."""
    
    def __init__(self, fields: dict[str, FieldType]):
        self._fields = fields
        
    def _evaluate(self, factory: 'FieldsFactory') -> None:
        pass
    
    def dependencies(self) -> Sequence[str]:
        return []
    
    def __call__(self, field_name: str, factory:'FieldsFactory') -> FieldType:
        return self._fields[field_name]

    def fields(self) -> Iterable[str]:
        return self._fields.keys()
    

class ProgramFieldProvider:
    """
    Computes a field defined by a GT4Py Program.

    """

    def __init__(self,
                 func: gtx_decorator.Program,
                 domain: dict[gtx.Dimension:tuple[gtx.int32, gtx.int32]],  # the compute domain 
                 fields: Sequence[str],
                 deps: Sequence[str] = [],  # the dependencies of func
                 params: dict[str, Scalar] = {},  # the parameters of func
                 ):
        self._compute_domain = domain
        self._dims = domain.keys()
        self._func = func
        self._dependencies = deps
        self._params = params
        self._fields: dict[str, Optional[gtx.Field | Scalar]] = {name: None for name in fields}



    def _allocate(self, allocator, grid:base_grid.BaseGrid) -> dict[str, FieldType]:
        def _map_size(dim:gtx.Dimension, grid:base_grid.BaseGrid) -> int:
            if dim == dims.KHalfDim:
                return grid.num_levels + 1
            return grid.size[dim]

        def _map_dim(dim: gtx.Dimension) -> gtx.Dimension:
            if dim == dims.KHalfDim:
                return dims.KDim
            return dim

        field_domain = {_map_dim(dim): (0, _map_size(dim, grid)) for dim in
                        self._compute_domain.keys()}
        return {k: allocator(field_domain, dtype=_attrs[k]["dtype"]) for k in
                self._fields.keys()}

    def _unallocated(self) -> bool:
        return not all(self._fields.values())

    def _evaluate(self, factory: 'FieldsFactory'):
        self._fields = self._allocate(factory._allocator, factory.grid)
        domain = functools.reduce(operator.add, self._compute_domain.values())
        args = [factory.get(k) for k in self.dependencies()]
        params = [p for p in self._params.values()]
        output = [f for f in self._fields.values()]
        self._func(*args, *output, *params, *domain,
                   offset_provider=factory.grid.offset_providers)

    def fields(self)->Iterable[str]:
        return self._fields.keys()
    
    def dependencies(self)->Iterable[str]:
        return self._dependencies
    def __call__(self, field_name: str, factory:'FieldsFactory') -> FieldType:
        if field_name not in self._fields.keys():
            raise ValueError(f"Field {field_name} not provided by f{self._func.__name__}")
        if self._unallocated():
            self._evaluate(factory)
        return self._fields[field_name]


class FieldsFactory:
    """
    Factory for fields.
    
    Lazily compute fields and cache them.  
    """

    
    def __init__(self, grid:base_grid.BaseGrid, backend=settings.backend):
        self._grid = grid
        self._providers: dict[str, 'FieldProvider'] = {}
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)


    @property
    def grid(self):
        return self._grid
        
    def register_provider(self, provider:FieldProvider):
        
        for dependency in provider.dependencies():
            if dependency not in self._providers.keys():
                raise ValueError(f"Dependency '{dependency}' not found in registered providers")
    
        
        for field in provider.fields():
            self._providers[field] = provider
        
    
    def get(self, field_name: str, type_: RetrievalType = RetrievalType.FIELD) -> Union[FieldType, xa.DataArray, dict]:
        if field_name not in _attrs:
            raise ValueError(f"Field {field_name} not found in metric fields")
        if type_ == RetrievalType.METADATA:
            return _attrs[field_name]
        if type_ == RetrievalType.FIELD:
            return self._providers[field_name](field_name, self)
        if type_ == RetrievalType.DATA_ARRAY:
            return to_data_array(self._providers[field_name](field_name), _attrs[field_name])
        raise ValueError(f"Invalid retrieval type {type_}")


def to_data_array(field, attrs):
    return xa.DataArray(field, attrs=attrs)





    

    
    


