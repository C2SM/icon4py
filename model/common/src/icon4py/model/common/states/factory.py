import abc
import functools
import inspect
import operator
from enum import IntEnum
from typing import Callable, Iterable, Optional, Protocol, Sequence, TypeAlias, TypeVar, Union

import gt4py.next as gtx
import gt4py.next.ffront.decorator as gtx_decorator
import xarray as xa

import icon4py.model.common.states.metadata as metadata
from icon4py.model.common import dimension as dims, exceptions, settings, type_alias as ta
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.settings import xp
from icon4py.model.common.utils import builder


T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)
DimT = TypeVar("DimT", dims.KDim, dims.KHalfDim, dims.CellDim, dims.EdgeDim, dims.VertexDim)
Scalar: TypeAlias = Union[ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64]

FieldType:TypeAlias = gtx.Field[Sequence[gtx.Dims[DimT]], T]
class RetrievalType(IntEnum):
    FIELD = 0,
    DATA_ARRAY = 1,
    METADATA = 2,




def valid(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.validate():
            raise exceptions.IncompleteSetupError("Factory not fully instantiated, missing grid or allocator")
        return func(self, *args, **kwargs)
    return wrapper


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
                 func: Union[gtx_decorator.Program, Callable],
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

    def _unallocated(self) -> bool:
        return not all(self._fields.values())

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
        return {k: allocator(field_domain, dtype=metadata.attrs[k]["dtype"]) for k in
                self._fields.keys()}



    def _evaluate(self, factory: 'FieldsFactory'):
        self._fields = self._allocate(factory.allocator, factory.grid)
        domain = functools.reduce(operator.add, self._compute_domain.values())
        deps = [factory.get(k) for k in self.dependencies()]
        params = [p for p in self._params.values()]
        output = [f for f in self._fields.values()]
        # it might be safer to call the field_operator here? then we can use the keyword only args for out= and domain=
        self._func(*deps, *output, *params, *domain,
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


class NumpyFieldsProvider(FieldProvider):
    def __init__(self, func:Callable, 
                 domain:dict[gtx.Dimension:tuple[gtx.int32, gtx.int32]], 
                 fields:Sequence[str], 
                 deps:dict[str, str], 
                 params:dict[str, Scalar] = {}):
        self._compute_domain = domain
        self._func = func
        self._fields:dict[str, Optional[FieldType]] = {name: None for name in fields}
        self._dependencies = deps
        self._params = params
        
    def _evaluate(self, factory: 'FieldsFactory') -> None:
        domain = {dim: range(*self._compute_domain[dim]) for dim in self._compute_domain.keys()}
       
        # validate deps:
        self._validate_dependencies(factory)
        args = {k: factory.get(v).ndarray for k, v in self._dependencies.items()}
        args.update(self._params)
        results = self._func(**args)
        ## TODO: check order of return values
        results = (results,) if isinstance(results, xp.ndarray) else results
    
        self._fields = {k: gtx.as_field(tuple(self._compute_domain.keys()), results[i]) for i, k in enumerate(self._fields.keys())}

    def _validate_dependencies(self, factory):
        func_signature = inspect.signature(self._func)
        parameters = func_signature.parameters
        for dep_key in self._dependencies.keys():
            try:
                parameter_definition = parameters[dep_key]
                if parameter_definition.annotation != xp.ndarray: # also allow for gtx.Field ???
                    raise ValueError(f"Dependency {dep_key} in function {self._func.__name__} : {func_signature} is not of type xp.ndarray")
            except KeyError:
                raise ValueError(f"Argument {dep_key} does not exist in {self._func.__name__} : {func_signature}.")    
        
        
        for param_key, param_value in self._params.items():
            try:
                parameter_definition = parameters[param_key]
                if parameter_definition.annotation != type(param_value):
                    raise ValueError(f"parameter {parameter_definition} to function {self._func.__name__} has the wrong type")
            except KeyError:
                raise ValueError(f"Argument {param_key} does not exist in {self._func.__name__} : {func_signature}.")

    def dependencies(self) -> Iterable[str]:
        return self._dependencies.values()
    
    def fields(self) -> Iterable[str]:
        return self._fields.keys()
    
    def __call__(self, field_name: str, factory:'FieldsFactory') -> FieldType:
        if field_name not in self._fields.keys():
            raise ValueError(f"Field {field_name} not provided by f{self._func.__name__}")
        if any([f is None for f in self._fields.values()]):
            self._evaluate(factory)
        return self._fields[field_name]

def inspect_func(func:Callable):
    signa = inspect.signature(func)
    print(f"signature: {signa}")
    print(f"parameters: {signa.parameters}")
   
    print(f"return : {signa.return_annotation}")
    return signa

    
    
    

class FieldsFactory:
    """
    Factory for fields.
    
    Lazily compute fields and cache them.  
    """

    
    def __init__(self, grid:base_grid.BaseGrid = None, backend=settings.backend):
        self._grid = grid
        self._providers: dict[str, 'FieldProvider'] = {}
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)


    def validate(self):
        return self._grid is not None and self._allocator is not None
        
    @builder.builder
    def with_grid(self, grid:base_grid.BaseGrid):
        self._grid = grid
        
    @builder.builder
    def with_allocator(self, backend = settings.backend):
        self._allocator = backend
        
    
    
    @property
    def grid(self):
        return self._grid
    
    @property
    def allocator(self):
        return self._allocator
        
    def register_provider(self, provider:FieldProvider):
        
        for dependency in provider.dependencies():
            if dependency not in self._providers.keys():
                raise ValueError(f"Dependency '{dependency}' not found in registered providers")
    
        
        for field in provider.fields():
            self._providers[field] = provider
        
    @valid
    def get(self, field_name: str, type_: RetrievalType = RetrievalType.FIELD) -> Union[FieldType, xa.DataArray, dict]:
        if field_name not in metadata.attrs:
            raise ValueError(f"Field {field_name} not found in metric fields")
        if type_ == RetrievalType.METADATA:
            return metadata.attrs[field_name]
        if type_ == RetrievalType.FIELD:
            return self._providers[field_name](field_name, self)
        if type_ == RetrievalType.DATA_ARRAY:
            return to_data_array(self._providers[field_name](field_name), metadata.attrs[field_name])
        raise ValueError(f"Invalid retrieval type {type_}")





def to_data_array(field, attrs):
    return xa.DataArray(field, attrs=attrs)





    

    
    


