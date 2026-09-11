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

`FieldSource`: allows to query for a field, by the following methods:
- `.get(field_name)`:  return computed values as a GT4Py `Field` with dtype according to metadata
- `.get_full_precision(field_name)`:  return computed values as a GT4Py `Field` with the dtype the computation returned
- `.get_metadata(field_name)`:  return metadata such as units, CF standard_name or similar, dimensions...

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

val = factory.get("foo")


TODO: @halungge: allow to read configuration data

"""

from __future__ import annotations

import collections
import contextlib
import enum
import functools
import inspect
import logging
import types
import typing
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Literal, Protocol, TypeVar, cast, overload

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import model, utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)
DomainType = TypeVar("DomainType", h_grid.Domain, v_grid.Domain)


class GridProvider(Protocol):
    @property
    def grid(self) -> icon_grid.IconGrid: ...

    @property
    def vertical_grid(self) -> v_grid.VerticalGrid | None: ...


@contextlib.contextmanager
def as_exchangeable_field(field: state_utils.GTXFieldType) -> Iterator[state_utils.GTXFieldType]:
    """Create a 2d View of the field that can be passed to GHEX."""
    original_dims = field.domain.dims
    if len(original_dims) > 2:
        original_shape = field.ndarray.shape
        tail_size = original_shape[1] * original_shape[2]
        field = gtx_common._field(
            field.ndarray.reshape(original_shape[0], -1),
            domain={original_dims[0]: (0, original_shape[0]), original_dims[1]: (0, tail_size)},
        )
    yield field


class NeedsExchange(Protocol):
    def needs_exchange(self) -> bool: ...

    def exchange(
        self,
        fields: Mapping[str, state_utils.FieldType],
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        log.debug(f"provider for fields {fields.keys()} needs exchange {self.needs_exchange()}")
        if self.needs_exchange():
            # ghex assumes all fields to in one call to have the same `dtype`, this is not the case for all producer functions in icon4py,
            # hence as a simple workaround we loop over the fields
            for name, field in fields.items():
                log.debug(f"preparing exchange of {name} - {field}")
                first_dim = field.domain.dims[0]
                assert first_dim.kind == gtx.DimensionKind.HORIZONTAL, (
                    f"1st dimension {first_dim} needs to be one of {list(dims.horizontal_dims())} for exchange"
                )
                with as_exchangeable_field(field) as buffer:
                    exchange.exchange(first_dim, buffer, stream=decomposition.BLOCK)
                log.debug(f"exchanged buffer for {name}")


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
        *,
        field_name: str,
        field_src: FieldSource,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ) -> state_utils.GTXFieldType | state_utils.ScalarType: ...

    @property
    def dependencies(self) -> Sequence[str]: ...

    @property
    def fields(
        self,
    ) -> Mapping[str, state_utils.FieldType | state_utils.ScalarType]: ...

    @property
    def func(self) -> Callable: ...


class FieldSource(GridProvider, Protocol):
    """
    Protocol for object that can be queried for fields and field metadata

    Provides a default implementation of the get method.
    """

    _providers: MutableMapping[str, FieldProvider] = {}  # noqa:  RUF012 instance variable
    _exchange: decomposition.ExchangeRuntime

    @property
    def _sources(self) -> FieldSource:
        return self

    @property
    def metadata(self) -> MutableMapping[str, model.FieldMetaData]:
        """Returns metadata for the fields that this field source provides."""
        ...

    @property
    def backend(self) -> gtx_typing.Backend | None:
        """Target backend: this is the backend that the field should be produced for when requested from the source.
        The field computation might
        be done on a different backend, as there are FieldOperators that require a specific backend (mostly embedded)
        to be used."""
        ...

    def _backend_name(self) -> str:
        return "embedded" if self.backend is None else self.backend.name

    def check_field_in_provider(self, field_name: str) -> None:
        if field_name not in self._providers:
            raise ValueError(f"Field '{field_name}' not provided by the source '{self.__class__}'")

    def get_metadata(self, field_name: str) -> model.FieldMetaData:
        self.check_field_in_provider(field_name)
        return self.metadata[field_name]

    def get_full_precision(self, field_name: str) -> state_utils.GTXFieldType |  state_utils.ScalarType:
        log.info(f" retrieving field {field_name}")
        self.check_field_in_provider(field_name)
        provider = self._providers[field_name]
        if field_name not in provider.fields:
            raise ValueError(
                f"Field {field_name} not provided by f{provider.func.__name__}."
            )

        return provider(
            field_name=field_name,
            field_src=self._sources,
            backend=self.backend,
            grid=self,
            exchange=self._exchange,
        )

    def get_scalar(self, field_name: str) -> state_utils.ScalarType:
        scalar = self.get_full_precision(field_name)
        this_metadata = self.metadata[field_name]
        if "dims" in this_metadata:
            raise TypeError(f"This function is intended to return a Scalar. Field name {field_name!r} looks like a Field (contains 'dims' in metadata).")
        return scalar


    def output_dtype(self, field_name: str) -> state_utils.ScalarType:
        return self.get_metadata(field_name)["dtype"]

    def internal_dtype(self, field_name: str) -> state_utils.ScalarType:
        return allfloats_as_double(self.output_dtype(field_name))

    def dtypes_for_factory(self, field_names: Iterator[str]) -> dict[str, state_utils.ScalarType]:
        dtypes = {field_name: self.internal_dtype(field_name) for field_name in field_names}
        return dtypes

    def _provided_by_source(self, name) -> bool:
        return name in self._sources._providers or name in self._sources.metadata

    def get(self, field_name: str) -> state_utils.GTXFieldType:
        """Export a field from the factory in the dtype provided by the metadata."""
        field = self.get_full_precision(field_name)
        this_metadata = self.metadata[field_name]
        if "dims" not in this_metadata:
            raise TypeError(f"This function is intended to return a Field. Field name {field_name!r} looks like a Scalar ('dims' missing in metadata).")
        dtype_metadata = this_metadata.get("dtype", ta.wpfloat)
        # `astype` is a `BuiltInFunction`, whose overloads are erased by the decorator.
        return cast("state_utils.GTXFieldType", gtx.astype(field, dtype_metadata))

    def register_provider(self, provider: FieldProvider) -> None:
        # dependencies must be provider by this field source or registered in sources
        for dependency in provider.dependencies:
            if not (dependency in self._providers or self._provided_by_source(dependency)):
                raise ValueError(
                    f"Missing dependency: '{dependency}' in registered of sources {self.__class__}"
                )

        for field in provider.fields:
            self._providers[field] = provider


class CompositeSource(FieldSource):
    def __init__(self, *, me: FieldSource, others: tuple[FieldSource, ...]):
        self._backend = me.backend
        self._grid = me.grid
        self._vertical_grid = me.vertical_grid
        self._exchange = me._exchange
        self._metadata = collections.ChainMap(me.metadata, *(s.metadata for s in others))
        self._providers = collections.ChainMap(me._providers, *(s._providers for s in others))

    @functools.cached_property
    def metadata(self) -> MutableMapping[str, model.FieldMetaData]:
        return self._metadata

    @property
    def backend(self) -> gtx_typing.Backend | None:
        return self._backend

    @property
    def vertical_grid(self) -> v_grid.VerticalGrid | None:
        return self._vertical_grid

    @property
    def grid(self) -> icon_grid.IconGrid:
        return self._grid


class PrecomputedFieldProvider(FieldProvider):
    """Simple FieldProvider that does not do any computation but gets its fields at construction
    and returns it upon provider.get(field_name)."""

    def __init__(
        self, fields: Mapping[str, state_utils.GTXFieldType | state_utils.ScalarType]
    ) -> None:
        self._fields = fields

    @property
    def dependencies(self) -> Sequence[str]:
        return ()

    def __call__(
        self,
        *,
        field_name: str,
        field_src: FieldSource,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ) -> state_utils.GTXFieldType | state_utils.ScalarType:
        return self.fields[field_name]

    @property
    def fields(self) -> Mapping[str, state_utils.GTXFieldType | state_utils.ScalarType]:
        return self._fields

    @property
    def func(self) -> Callable:
        return lambda: self.fields


def _field_extent[DomainT: (h_grid.Domain, v_grid.Domain)](
    dim: gtx.Dimension, declared: tuple[DomainT, DomainT] | None, grid: GridProvider
) -> tuple[int, int]:
    """
    The range a provider allocates for `dim`.

    A declared vertical range is the field's extent: there is no vertical decomposition and no
    exchange, and a gt4py field keeps absolute level indices, so a sub-range is a field on those
    levels. Horizontal dimensions are always allocated at full local size, because the halo exchange
    fills entries outside the compute range and neighbor access indexes the field by absolute local
    index; so are local (sparse) dimensions and any dimension declared without a range.
    """
    if declared is not None and dim.kind == gtx.DimensionKind.VERTICAL:
        assert grid.vertical_grid is not None
        start, end = declared
        return grid.vertical_grid.index(start), grid.vertical_grid.index(end)
    return 0, grid.grid.size[dim]


class EmbeddedFieldOperatorProvider(FieldProvider, NeedsExchange):
    """Provider that calls a GT4Py Fieldoperator.

    # TODO(halungge): for now to be used only on FieldView Embedded GT4Py backend.
    The field operator is called without domain args, so it computes on the whole extent of its
    output fields: the declared vertical range and the full horizontal size, as `_field_extent`
    describes. A `domain` given as a tuple of dimensions allocates full size in every dimension,
    which is how sparse/local fields are written.
    """

    def __init__(
        self,
        *,
        func: gtx_typing.FieldOperator,
        domain: dict[gtx.Dimension, tuple[DomainType, DomainType]] | tuple[gtx.Dimension, ...],
        fields: dict[str, str],
        deps: dict[str, str],
        do_exchange: bool,
        params: dict[str, state_utils.ScalarType] | None = None,
    ):
        self._func = func
        self._domain = domain if isinstance(domain, dict) else dict.fromkeys(domain)
        self._dims = tuple(self._domain)
        self._dependencies = deps
        self._output = fields
        self._params = {} if params is None else params
        self._fields: dict[str, gtx.Field | state_utils.ScalarType | None] = {
            name: None for name in fields.values()
        }
        self._do_exchange = do_exchange

    def needs_exchange(self) -> bool:
        return self._do_exchange

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
        *,
        field_name: str,
        field_src: FieldSource | None,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ) -> state_utils.FieldType:
        if any([f is None for f in self.fields.values()]):
            log.debug(f"computing fields  {self.fields.keys()}")
            self._compute(field_src, grid)
            self.exchange(self.fields, exchange)
        return self.fields[field_name]

    def _compute(self, factory: FieldSource, grid_provider: GridProvider) -> None:
        # allocate output buffer
        compute_backend = self._func.backend
        log.info(
            f"computing {self._func.__name__}: compute backend is: "
            f"{data_alloc.backend_name(compute_backend)}, target backend is: "
            f"{data_alloc.backend_name(factory.backend)}"
        )
        dtypes = factory.dtypes_for_factory(self._fields)
        # the outputs live on the target backend's device: embedded computes in place on them
        self._fields = self._allocate_fields(factory.backend, grid_provider, dtypes)
        # call field operator
        log.debug(f"transferring dependencies to compute backend: {self._dependencies.keys()}")

        deps = {
            k: data_alloc.reallocate(factory.get_full_precision(v), allocator=compute_backend)
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
            self._fields[k] = data_alloc.reallocate(v, allocator=factory.backend)

    def _unravel_output_fields(self):
        out_fields = tuple(self._fields.values())
        if len(out_fields) == 1:
            out_fields = out_fields[0]
        return out_fields

    # TODO(): do we need that here?
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
        backend: gtx_typing.Backend | None,
        grid_provider: GridProvider,
        dtypes: dict[str, state_utils.ScalarType],
    ) -> dict[str, state_utils.FieldType]:
        allocate = gtx.constructors.zeros.partial(allocator=backend)
        field_domain = {
            dim: _field_extent(dim, declared, grid_provider)
            for dim, declared in self._domain.items()
        }
        return {k: allocate(field_domain, dtype=dtypes[k]) for k in self._fields}


class ProgramFieldProvider(FieldProvider, NeedsExchange):
    """
    Computes a field defined by a GT4Py Program.

    TODO(halungge): need a way to specify where the dependencies and params can be retrieved.
       As not all parameters can be resolved at the definition time

    Args:
        func: GT4Py Program that computes the fields
        domain: the domain of the computed fields and the compute domain of the program. It is
            the fields' extent only in the vertical, see `_field_extent`.
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
        *,
        func: gtx_typing.Program,
        domain: dict[gtx.Dimension, tuple[DomainType, DomainType]],
        fields: dict[str, str],
        deps: dict[str, str],
        do_exchange: bool,
        params: dict[str, state_utils.ScalarType] | None = None,
    ):
        self._func = func
        self._domain = domain
        self._dims = domain.keys()
        self._dependencies = deps
        self._output = fields
        self._params = params if params is not None else {}
        self.ready = False
        self._fields: dict[str, gtx.Field | state_utils.ScalarType | None] = {
            name: None for name in fields.values()
        }
        self._do_exchange = do_exchange

    def _allocate(
        self,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        dtypes: dict[str, state_utils.ScalarType],
    ) -> dict[str, state_utils.FieldType]:
        allocate = gtx.constructors.zeros.partial(allocator=backend)
        field_domain = {
            dim: _field_extent(dim, declared, grid) for dim, declared in self._domain.items()
        }
        return {k: allocate(field_domain, dtype=dtypes[k]) for k in self._fields}

    # TODO(halungge): this can be simplified when completely disentangling vertical and horizontal grid.
    #   the IconGrid should then only contain horizontal connectivities and no longer any Koff which should be moved to the VerticalGrid
    def _get_offset_providers(self, grid: icon_grid.IconGrid) -> dict[str, gtx.FieldOffset]:
        offset_providers = {}
        for dim in self._domain:
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

    def _domain_args(self, grid: GridProvider) -> dict[str, gtx.int32]:
        domain_args = {}

        for dim in self._domain:
            if dim.kind == gtx.DimensionKind.HORIZONTAL:
                domain_args.update(
                    {
                        "horizontal_start": grid.grid.start_index(self._domain[dim][0]),
                        "horizontal_end": grid.grid.end_index(self._domain[dim][1]),
                    }
                )
            elif dim.kind == gtx.DimensionKind.VERTICAL:
                vertical_start, vertical_end = _field_extent(dim, self._domain[dim], grid)
                domain_args.update({"vertical_start": vertical_start, "vertical_end": vertical_end})
            else:
                raise ValueError(f"DimensionKind '{dim.kind}' not supported in Program Domain")
        return domain_args

    def needs_exchange(self) -> bool:
        return self._do_exchange

    def __call__(
        self,
        *,
        field_name: str,
        field_src: FieldSource | None,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ):
        if any([f is None for f in self.fields.values()]):
            self._compute(field_src=field_src, grid=grid, backend=backend)
            self.exchange(self.fields, exchange=exchange)
        return self.fields[field_name]

    def _compute(
        self,
        *,
        field_src: FieldSource,
        grid: GridProvider,
        backend: gtx_typing.Backend | None,
    ) -> None:
        dtypes = field_src.dtypes_for_factory(self._output.values())
        self._fields = self._allocate(backend, grid.grid, dtypes=dtypes)
        log.debug(f" getting dependencies {self._dependencies.values()} from {field_src}")
        deps = {k: field_src.get_full_precision(v) for k, v in self._dependencies.items()}
        deps.update(self._params)
        deps.update({k: self._fields[v] for k, v in self._output.items()})
        dims = self._domain_args(grid)
        offset_providers = self._get_offset_providers(grid.grid)
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


class NumpyDataProvider(FieldProvider, NeedsExchange):
    """
    Computes a field defined by a numpy function.

    Args:
        func: numpy function that computes the fields
        domain: the domain of the computed fields, following `_field_extent` when given with
            ranges; as a bare tuple of dimensions the returned arrays' shapes are the extent, which
            is how a field on a dimension without a grid size (e.g. `LsqUnkDim`) is labelled.
            Empty for a scalar result.
        fields: Seq[str] names under which the results fo the function will be registered
        deps: dict[str, str] input fields used for computing this stencil: the key is the variable name
            used in the function and the value the name of the field it depends on.
        connectivities: dict[str, Dimension] dict where the key is the variable named used in the
            function and the value the sparse Dimension of the connectivity field
        params: scalar arguments for the function
        do_exchange: a flag that governs whether or not a halo exchange is needed after the field has been computed. Defaults to False
    """

    def __init__(
        self,
        *,
        func: Callable,
        domain: dict[gtx.Dimension, tuple[DomainType, DomainType]] | tuple[gtx.Dimension, ...],
        fields: Sequence[str],
        deps: dict[str, str],
        connectivities: dict[str, gtx.Dimension] | None = None,
        params: dict[str, state_utils.ScalarType] | None = None,
        do_exchange: bool = False,
    ):
        self._func = func
        self._domain = domain if isinstance(domain, dict) else None
        self._dims = tuple(domain)
        self._fields: dict[str, state_utils.ScalarType | state_utils.FieldType | None] = {
            name: None for name in fields
        }
        self._dependencies = deps
        self._connectivities = connectivities if connectivities is not None else {}
        self._params = params if params is not None else {}
        self._do_exchange = do_exchange

    def __call__(
        self,
        *,
        field_name: str,
        field_src: FieldSource,
        backend: gtx_typing.Backend | None,
        grid: GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ) -> state_utils.FieldType:
        if any([f is None for f in self.fields.values()]):
            log.info(f"computing field {field_name}")
            self._compute(field_src, backend, grid)
            exchangeable_fields = {
                name: field for name, field in self.fields.items() if isinstance(field, gtx.Field)
            }
            self.exchange(exchangeable_fields, exchange=exchange)
        return self.fields[field_name]

    def _compute(
        self,
        factory: FieldSource,
        backend: gtx_typing.Backend | None,
        grid_provider: GridProvider,
    ) -> None:
        self._validate_dependencies()
        args = {
            k: buffer.ndarray if hasattr(buffer := factory.get_full_precision(v), "ndarray") else buffer
            for k, v in self._dependencies.items()
        }
        offsets = {
            k: grid_provider.grid.get_connectivity(v.value).ndarray
            for k, v in self._connectivities.items()
        }
        args.update(offsets)
        args.update(self._params)
        results = self._func(**args)
        # convert to tuple
        results = (results,) if not isinstance(results, tuple) else results
        # force double for floating-precision
        dtypes = factory.dtypes_for_factory(self.fields.keys())
        self._fields = {
            k: self._as_field(backend, results[i], dtype=dtypes[k], grid=grid_provider) if self._dims else results[i]
            for i, k in enumerate(self.fields)
        }

    def _as_field(
        self, backend: gtx_typing.Backend | None, value: data_alloc.NDArray, dtype, grid: GridProvider
    ) -> state_utils.GTXFieldType:
        if self._domain is None:
            return gtx.as_field(self._dims, value, allocator=backend, dtype=dtype)
        field_domain = gtx.domain(
            {dim: _field_extent(dim, declared, grid) for dim, declared in self._domain.items()}
        )
        return gtx.as_field(field_domain, value, allocator=backend, dtype=dtype)

    def _validate_dependencies(self) -> None:
        # TODO(egparedes): dealing with type annotations at run-time is error prone
        #   and requires robust utility functions. This snippet should use a better
        #   solution in the future.
        obj = inspect.unwrap(self._func)
        while isinstance(obj, functools.partial):
            obj = inspect.unwrap(obj.func)
        annotations = typing.get_type_hints(obj)
        for dep_key in self._dependencies:
            parameter_annotation = annotations.get(dep_key, gtx.float64)
            checked = _is_compatible_union(
                parameter_annotation, expected=data_alloc.NDArray | np.float64
            )
            assert checked, (
                f"Dependency '{dep_key}' in function '{_func_name(self._func)}':  does not exist or has "
                f"wrong type ('expected ndarray or float64') but was '{parameter_annotation}'."
            )

        supported_scalars = state_utils.IntegerType | state_utils.FloatType
        for param_key, param_value in self._params.items():
            parameter_annotation = annotations.get(param_key, gtx.float64)
            checked = _is_compatible_union(
                parameter_annotation, expected=supported_scalars
            ) and _is_compatible_value(param_value, expected=supported_scalars)

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

    def needs_exchange(self) -> bool:
        return self._do_exchange


def _is_compatible_union(annotation: Any, expected: types.UnionType | typing._SpecialForm) -> bool:
    possible_types = (
        typing.get_args(annotation)
        if typing.get_origin(annotation) in {types.UnionType, typing.Union}
        else (annotation,)
    )
    expected_types = (
        typing.get_args(expected)
        if typing.get_origin(expected) in {types.UnionType, typing.Union}
        else (expected,)
    )
    return set(possible_types) <= set(expected_types) and None not in possible_types


def _is_compatible_value(
    value: state_utils.ScalarType | gtx.Field, expected: types.UnionType | typing._SpecialForm
) -> bool:
    return type(value) in set(
        typing.get_args(expected)
        if typing.get_origin(expected) in {types.UnionType, typing.Union}
        else (expected,)
    )


def _func_name(callable_: Callable[..., Any]) -> str:
    if isinstance(callable_, functools.partial):
        return callable_.func.__name__
    else:
        return callable_.__name__


def allfloats_as_double(dtype_metadata: state_utils.ScalarType) -> state_utils.ScalarType:
    if dtype_metadata in [gtx.int32, bool]:
        return dtype_metadata
    else:
        return gtx.float64
