# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Provides Protocols and default implementations for Fields factories, which can be used to compute static
fields and manage their dependencies

- `FieldSource`: allows to query for a field, by a `.get(field_name, retrieval_type)` method:

Three `RetrievalMode` s are available:
_ `FIELD`: return the buffer containing the computed values as a GT4Py `Field`
- `METADATA`:  return metadata (`FieldMetaData`) such as units, CF standard_name or similar, dimensions...
- `DATA_ARRAY`: combination of the two above in the form of `xarray.dataarray`

The factory can be used to "store" already computed fields or register functions and call arguments
and only compute the fields lazily upon request. In order to do so the user registers the fields
computation with factory by setting up a `FieldProvider`

It should be possible to setup the factory and computations and the factory independent of concrete runtime parameters that define
the computation, passing those only once they are defined at runtime, for example
---
factory = Factory(metadata, ...)
foo_provider = FieldProvider("foo", func = f1, dependencies, fields)
bar_provider = FieldProvider("bar", func = f2, dependencies = ["foo"])

factory.register_provider(foo_provider)
factory.register_provider(bar_provider)
(...)

val = factory.get("foo", RetrievalType.DATA_ARRAY)


TODO: @halungge: allow to read configuration data

"""

import collections
import enum
import functools
import inspect
import logging
from types import ModuleType
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    get_args,
)

import gt4py.next as gtx
import gt4py.next.backend as gtx_backend
import gt4py.next.ffront.decorator as gtx_decorator
import xarray as xa

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import (
    base as base_grid,
    horizontal as h_grid,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.states import model, utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)
DomainType = TypeVar("DomainType", h_grid.Domain, v_grid.Domain)


class GridProvider(Protocol):
    @property
    def grid(self) -> Optional[icon_grid.IconGrid]: ...

    @property
    def vertical_grid(self) -> Optional[v_grid.VerticalGrid]: ...


class FieldProvider(Protocol):
    """
    Protocol for field providers.

    A field provider is responsible for the computation (and caching) of a set of fields.
    The fields can be accessed by their field_name (str).

    A FieldProvider is a callable and additionally has three properties (except for __call__):
     - func: the function used to compute the fields
     - fields: Mapping of a field_name to the data buffer holding the computed values
     - dependencies: returns a list of field_names that the fields provided by this provider depend on.

    """

    def __call__(
        self,
        field_name: str,
        field_src: Optional["FieldSource"],
        backend: Optional[gtx_backend.Backend],
        grid: Optional[GridProvider],
    ) -> state_utils.FieldType: ...

    @property
    def dependencies(self) -> Sequence[str]: ...

    @property
    def fields(self) -> Mapping[str, Any]: ...

    @property
    def func(self) -> Callable: ...


class RetrievalType(enum.Enum):
    FIELD = 0
    DATA_ARRAY = 1
    METADATA = 2


class FieldSource(GridProvider, Protocol):
    """
    Protocol for object that can be queried for fields and field metadata

    Provides a default implementation of the get method.
    """

    _providers: MutableMapping[str, FieldProvider] = {}  # noqa:  RUF012 instance variable

    @property
    def _sources(self) -> "FieldSource":
        return self

    @property
    def metadata(self) -> MutableMapping[str, model.FieldMetaData]:
        """Returns metadata for the fields that this field source provides."""
        ...

    # TODO @halungge: this is the target Backend: not necessarily the one that the field is computed and
    #      there are fields which need to be computed on a specific backend, which can be different from the
    #      general run backend
    @property
    def backend(self) -> gtx_backend.Backend: ...

    def _backend_name(self) -> str:
        return "embedded" if self.backend is None else self.backend.name

    def get(
        self, field_name: str, type_: RetrievalType = RetrievalType.FIELD
    ) -> Union[state_utils.FieldType, xa.DataArray, model.FieldMetaData]:
        """
        Get a field or its metadata from the factory.

        Fields are computed upon first call to `get`.
        Args:
            field_name:
            type_: RetrievalType, determines whether only the field (databuffer) or Metadata or both will be returned

        Returns:
            gt4py field containing allocated using this factories backend, a fields metadata or a
            dataarray containing both.

        """
        if field_name not in self._providers:
            raise ValueError(f"Field '{field_name}' not provided by the source '{self.__class__}'")
        match type_:
            case RetrievalType.METADATA:
                return self.metadata[field_name]
            case RetrievalType.FIELD | RetrievalType.DATA_ARRAY:
                provider = self._providers[field_name]
                if field_name not in provider.fields:
                    raise ValueError(
                        f"Field {field_name} not provided by f{provider.func.__name__}."
                    )

                buffer = provider(field_name, self._sources, self.backend, self)
                return (
                    buffer
                    if type_ == RetrievalType.FIELD
                    else state_utils.to_data_array(buffer, self.metadata[field_name])
                )
            case _:
                raise ValueError(f"Invalid retrieval type {type_}")

    def _provided_by_source(self, name):
        return name in self._sources._providers or name in self._sources.metadata.keys()

    def register_provider(self, provider: FieldProvider):
        # dependencies must be provider by this field source or registered in sources
        for dependency in provider.dependencies:
            if not (dependency in self._providers.keys() or self._provided_by_source(dependency)):
                raise ValueError(
                    f"Missing dependency: '{dependency}' in registered of sources {self.__class__}"
                )

        for field in provider.fields:
            self._providers[field] = provider


class CompositeSource(FieldSource):
    def __init__(self, me: FieldSource, others: tuple[FieldSource, ...]):
        self._backend = me.backend
        self._grid = me.grid
        self._vertical_grid = me.vertical_grid
        self._metadata = collections.ChainMap(me.metadata, *(s.metadata for s in others))
        self._providers = collections.ChainMap(me._providers, *(s._providers for s in others))

    @functools.cached_property
    def metadata(self) -> MutableMapping[str, model.FieldMetaData]:
        return self._metadata

    @property
    def backend(self) -> gtx_backend.Backend:
        return self._backend

    @property
    def vertical_grid(self) -> Optional[v_grid.VerticalGrid]:
        return self._vertical_grid

    @property
    def grid(self) -> Optional[icon_grid.IconGrid]:
        return self._grid


class PrecomputedFieldProvider(FieldProvider):
    """Simple FieldProvider that does not do any computation but gets its fields at construction
    and returns it upon provider.get(field_name)."""

    def __init__(self, fields: dict[str, state_utils.FieldType]):
        self._fields = fields

    @property
    def dependencies(self) -> Sequence[str]:
        return ()

    def __call__(
        self, field_name: str, field_src=None, backend=None, grid=None
    ) -> state_utils.FieldType:
        return self.fields[field_name]

    @property
    def fields(self) -> Mapping[str, state_utils.FieldType]:
        return self._fields

    @property
    def func(self) -> Callable:
        return lambda: self.fields


class EmbeddedFieldOperatorProvider(FieldProvider):
    """Provider that calls a GT4Py Fieldoperator.

    # TODO (@halungge) for now to be used only on FieldView Embedded GT4Py backend.
    - restrictions:
         - (if only called on FieldView-Embedded, this is not a necessary restriction)
            calls field operators without domain args, so it can only be used for full field computations
    - plus:
        - can write sparse/local fields
    """

    def __init__(
        self,
        func: gtx_decorator.FieldOperator,
        domain: tuple[gtx.Dimension, ...],
        fields: dict[str, str],  # keyword arg to (field_operator, field_name)
        deps: dict[str, str],  # keyword arg to (field_operator, field_name) need: src
        params: Optional[
            dict[str, state_utils.ScalarType]
        ] = None,  # keyword arg to (field_operator, field_name)
    ):
        self._func = func
        self._dims = domain
        self._dependencies = deps
        self._output = fields
        self._params = {} if params is None else params
        self._fields: dict[str, Optional[gtx.Field | state_utils.ScalarType]] = {
            name: None for name in fields.values()
        }

    @property
    def dependencies(self) -> Sequence[str]:
        return list(self._dependencies.values())

    @property
    def fields(self) -> Mapping[str, state_utils.FieldType]:
        return self._fields

    @property
    def func(self) -> Callable:
        return self._func

    def __call__(
        self,
        field_name: str,
        field_src: Optional[FieldSource],
        backend: Optional[gtx_backend.Backend],
        grid: GridProvider,
    ) -> state_utils.FieldType:
        if any([f is None for f in self.fields.values()]):
            self._compute(field_src, grid)
        return self.fields[field_name]

    def _compute(self, factory: FieldSource, grid_provider: GridProvider) -> None:
        # allocate output buffer
        compute_backend = self._func.backend
        log.info(
            f"computing {self._func.__name__}: compute backend is: "
            f"{data_alloc.backend_name(compute_backend)}, target backend is: "
            f"{data_alloc.backend_name(factory.backend)}"
        )
        xp = data_alloc.import_array_ns(factory.backend)
        metadata = {k: factory.get(k, RetrievalType.METADATA) for k in self.fields.keys()}
        self._fields = self._allocate_fields(compute_backend, grid_provider, xp, metadata)
        # call field operator
        log.debug(f"transferring dependencies to compute backend: {self._dependencies.keys()}")

        deps = {
            k: data_alloc.as_field(factory.get(v), backend=compute_backend)
            for k, v in self._dependencies.items()
        }

        providers = self._get_offset_providers(grid_provider.grid)
        self._func(**deps, out=self._unravel_output_fields(), offset_provider=providers)
        # transfer to target backend, the fields might have been computed on a compute backend
        for k, v in self._fields.items():
            log.debug(
                f"transferring result {k} to target backend: "
                f"{data_alloc.backend_name(factory.backend)}"
            )
            self._fields[k] = data_alloc.as_field(v, backend=factory.backend)

    def _unravel_output_fields(self):
        out_fields = tuple(self._fields.values())
        if len(out_fields) == 1:
            out_fields = out_fields[0]
        return out_fields

    # TODO: do we need that here?
    def _get_offset_providers(self, grid: icon_grid.IconGrid) -> dict[str, gtx.FieldOffset]:
        offset_providers = {}
        for dim in self._dims:
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                horizontal_offsets = {
                    k: v
                    for k, v in grid.connectivities.items()
                    if isinstance(v, gtx.Connectivity)
                    and v.domain.dims[0].kind == gtx.DimensionKind.HORIZONTAL
                }
                offset_providers.update(horizontal_offsets)
            if dim.kind == gtx.DimensionKind.VERTICAL:
                vertical_offsets = {
                    k: v
                    for k, v in grid.connectivities.items()
                    if isinstance(v, gtx.Dimension) and v.kind == gtx.DimensionKind.VERTICAL
                }
                offset_providers.update(vertical_offsets)
                # used for different compute backend in function call
        return offset_providers

    def _allocate_fields(
        self,
        backend: Optional[gtx_backend.Backend],
        grid_provider: GridProvider,
        xp: ModuleType,
        metadata: dict[str, model.FieldMetaData],
    ) -> dict[str, state_utils.FieldType]:
        def _map_size(dim: gtx.Dimension, grids: GridProvider) -> int:
            match dim:
                case dims.KHalfDim:
                    return grids.vertical_grid.num_levels + 1
                case dims.KDim:
                    return grids.vertical_grid.num_levels
                case _:
                    return grids.grid.size[dim]

        def _map_dim(dim: gtx.Dimension) -> gtx.Dimension:
            match dim:
                case dims.KHalfDim:
                    return dims.KDim
                case _:
                    return dim

        def _allocate(
            grid_provider: GridProvider,
            backend: gtx_backend.Backend,
            array_ns: ModuleType,
            dtype: state_utils.ScalarType = ta.wpfloat,
        ) -> gtx.Field:
            shape = tuple(_map_size(dim, grid_provider) for dim in self._dims)
            dims = tuple(_map_dim(dim) for dim in self._dims)
            buffer = array_ns.zeros(shape, dtype=dtype)
            return gtx.as_field(dims, data=buffer, allocator=backend, dtype=dtype)

        return {
            k: _allocate(grid_provider, backend, xp, dtype=dtype_or_default(k, metadata))
            for k in self._fields.keys()
        }


class ProgramFieldProvider(FieldProvider):
    """
    Computes a field defined by a GT4Py Program.

    TODO (halungge): need a way to specify where the dependencies and params can be retrieved.
       As not all parameters can be resolved at the definition time

    Args:
        func: GT4Py Program that computes the fields
        domain: the compute domain used for the stencil computation
        fields: dict[str, str], fields computed by this stencil:  the key is the variable name of
            the out arguments used in the program and the value the name the field is registered
            under and declared in the metadata.
        deps: dict[str, str], input fields used for computing this stencil:
            the key is the variable name used in the `gtx.program` and the value the name
            of the field it depends on.
        params: scalar parameters used in the program
    """

    def __init__(
        self,
        func: gtx_decorator.Program,
        domain: dict[gtx.Dimension, tuple[DomainType, DomainType]],
        fields: dict[str, str],
        deps: dict[str, str],
        params: Optional[dict[str, state_utils.ScalarType]] = None,
    ):
        self._func = func
        self._compute_domain = domain
        self._dims = domain.keys()
        self._dependencies = deps
        self._output = fields
        self._params = params if params is not None else {}
        self._fields: dict[str, Optional[gtx.Field | state_utils.ScalarType]] = {
            name: None for name in fields.values()
        }

    def _allocate(
        self,
        backend: Optional[gtx_backend.Backend],
        grid: base_grid.Grid,  # TODO @halungge: change to vertical grid
        dtype: dict[str, state_utils.ScalarType],
    ) -> dict[str, state_utils.FieldType]:
        def _map_size(dim: gtx.Dimension, grid: base_grid.Grid) -> int:
            if dim == dims.KHalfDim:
                return grid.num_levels + 1
            return grid.size[dim]

        def _map_dim(dim: gtx.Dimension) -> gtx.Dimension:
            if dim == dims.KHalfDim:
                return dims.KDim
            return dim

        allocate = gtx.constructors.zeros.partial(allocator=backend)
        field_domain = {_map_dim(dim): (0, _map_size(dim, grid)) for dim in self._dims}
        return {k: allocate(field_domain, dtype=dtype[k]) for k in self._fields.keys()}

    # TODO (@halungge) this can be simplified when completely disentangling vertical and horizontal grid.
    #   the IconGrid should then only contain horizontal connectivities and no longer any Koff which should be moved to the VerticalGrid
    def _get_offset_providers(self, grid: icon_grid.IconGrid) -> dict[str, gtx.FieldOffset]:
        offset_providers = {}
        for dim in self._compute_domain.keys():
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                horizontal_offsets = {
                    k: v
                    for k, v in grid.connectivities.items()
                    # TODO(halungge): review this workaround, as the fix should be available in the gt4py baseline
                    if isinstance(v, gtx.Connectivity)
                    and v.domain.dims[0].kind == gtx.DimensionKind.HORIZONTAL
                }
                offset_providers.update(horizontal_offsets)
            if dim.kind == gtx.DimensionKind.VERTICAL:
                vertical_offsets = {
                    k: v
                    for k, v in grid.connectivities.items()
                    if isinstance(v, gtx.Dimension) and v.kind == gtx.DimensionKind.VERTICAL
                }
                offset_providers.update(vertical_offsets)
        return offset_providers

    def _domain_args(
        self, grid: icon_grid.IconGrid, vertical_grid: v_grid.VerticalGrid
    ) -> dict[str : gtx.int32]:
        domain_args = {}

        for dim in self._compute_domain:
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                domain_args.update(
                    {
                        "horizontal_start": grid.start_index(self._compute_domain[dim][0]),
                        "horizontal_end": grid.end_index(self._compute_domain[dim][1]),
                    }
                )
            elif dim.kind == gtx.DimensionKind.VERTICAL:
                domain_args.update(
                    {
                        "vertical_start": vertical_grid.index(self._compute_domain[dim][0]),
                        "vertical_end": vertical_grid.index(self._compute_domain[dim][1]),
                    }
                )
            else:
                raise ValueError(f"DimensionKind '{dim.kind}' not supported in Program Domain")
        return domain_args

    def __call__(
        self,
        field_name: str,
        factory: FieldSource,
        backend: Optional[gtx_backend.Backend],
        grid_provider: GridProvider,
    ):
        if any([f is None for f in self.fields.values()]):
            self._compute(factory, backend, grid_provider)
        return self.fields[field_name]

    def _compute(
        self,
        factory: FieldSource,
        backend: Optional[gtx_backend.Backend],
        grid_provider: GridProvider,
    ) -> None:
        try:
            metadata = {v: factory.get(v, RetrievalType.METADATA) for k, v in self._output.items()}
            dtype = {v: metadata[v]["dtype"] for v in self._output.values()}
        except (ValueError, KeyError):
            dtype = {v: ta.wpfloat for v in self._output.values()}

        self._fields = self._allocate(backend, grid_provider.grid, dtype=dtype)
        deps = {k: factory.get(v) for k, v in self._dependencies.items()}
        deps.update(self._params)
        deps.update({k: self._fields[v] for k, v in self._output.items()})
        dims = self._domain_args(grid_provider.grid, grid_provider.vertical_grid)
        offset_providers = self._get_offset_providers(grid_provider.grid)
        deps.update(dims)
        self._func.with_backend(backend)(**deps, offset_provider=offset_providers)

    @property
    def fields(self) -> Mapping[str, state_utils.FieldType]:
        return self._fields

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def dependencies(self) -> Sequence[str]:
        return list(self._dependencies.values())


class NumpyFieldsProvider(FieldProvider):
    """
    Computes a field defined by a numpy function.

    Args:
        func: numpy function that computes the fields
        domain: the compute domain used for the stencil computation
        fields: Seq[str] names under which the results fo the function will be registered
        deps: dict[str, str] input fields used for computing this stencil: the key is the variable name
            used in the function and the value the name of the field it depends on.
        connectivities: dict[str, Dimension] dict where the key is the variable named used in the
            function and the value the sparse Dimension of the connectivity field
        params: scalar arguments for the function
    """

    def __init__(
        self,
        func: Callable,
        domain: Sequence[gtx.Dimension],
        fields: Sequence[str],
        deps: dict[str, str],
        connectivities: Optional[dict[str, gtx.Dimension]] = None,
        params: Optional[dict[str, state_utils.ScalarType]] = None,
    ):
        self._func = func
        self._dims = domain
        self._fields: dict[str, Optional[state_utils.FieldType]] = {name: None for name in fields}
        self._dependencies = deps
        self._connectivities = connectivities if connectivities is not None else {}
        self._params = params if params is not None else {}

    def __call__(
        self,
        field_name: str,
        factory: FieldSource,
        backend: Optional[gtx_backend.Backend],
        grid: GridProvider,
    ) -> state_utils.FieldType:
        if any([f is None for f in self.fields.values()]):
            self._compute(factory, backend, grid)
        return self.fields[field_name]

    def _compute(
        self,
        factory: FieldSource,
        backend: Optional[gtx_backend.Backend],
        grid_provider: GridProvider,
    ) -> None:
        self._validate_dependencies()
        args = {k: factory.get(v).ndarray for k, v in self._dependencies.items()}
        offsets = {
            k: grid_provider.grid.get_connectivity(v.value).ndarray
            for k, v in self._connectivities.items()
        }
        args.update(offsets)
        args.update(self._params)
        results = self._func(**args)
        ## TODO: can the order of return values be checked?
        results = (results,) if isinstance(results, data_alloc.NDArray) else results
        self._fields = {
            k: gtx.as_field(tuple(self._dims), results[i], allocator=backend)
            for i, k in enumerate(self.fields)
        }

    def _validate_dependencies(self):
        func_signature = inspect.signature(self._func)
        parameters = func_signature.parameters
        for dep_key in self._dependencies.keys():
            parameter_definition = parameters.get(dep_key)
            checked = _check_union(parameter_definition, union=data_alloc.NDArray)
            assert checked, (
                f"Dependency '{dep_key}' in function '{_func_name(self._func)}':  does not exist or has "
                f"wrong type ('expected ndarray') but was '{parameter_definition}'."
            )

        for param_key, param_value in self._params.items():
            parameter_definition = parameters.get(param_key)
            checked = _check_union_and_type(
                parameter_definition, param_value, union=state_utils.IntegerType
            ) or _check_union_and_type(
                parameter_definition, param_value, union=state_utils.FloatType
            )
            assert checked, (
                f"Parameter '{param_key}' in function '{_func_name(self._func)}' does not "
                f"exist or has the wrong type: '{type(param_value)}'."
            )

    @property
    def func(self) -> Callable:
        return self._func

    @property
    def dependencies(self) -> Sequence[str]:
        return list(self._dependencies.values())

    @property
    def fields(self) -> Mapping[str, state_utils.FieldType]:
        return self._fields


def _check_union_and_type(
    parameter_definition: inspect.Parameter,
    value: Union[state_utils.ScalarType, gtx.Field],
    union: Union,
) -> bool:
    _check_union(parameter_definition, union) and type(value) in get_args(union)
    members = get_args(union)
    return (
        parameter_definition is not None
        and parameter_definition.annotation in members
        and type(value) in members
    )


def _check_union(
    parameter_definition: inspect.Parameter,
    union: Union,
) -> bool:
    members = get_args(union)
    # fix for unions with only one member, which implicitly are not Union but fallback to the type
    if not members:
        members = (union,)
    annotation = parameter_definition.annotation
    return parameter_definition is not None and (annotation == union or annotation in members)


def _func_name(callable_: Callable[..., Any]) -> str:
    if isinstance(callable_, functools.partial):
        return callable_.func.__name__
    else:
        return callable_.__name__


def dtype_or_default(
    field_name: str, metadata: dict[str, model.FieldMetaData]
) -> state_utils.ScalarType:
    return metadata[field_name].get("dtype", ta.wpfloat)
