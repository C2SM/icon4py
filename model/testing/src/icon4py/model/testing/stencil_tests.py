# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Base class and helpers for writing GT4Py stencil test suites.

A suite is a subclass of `StencilTest` declaring the program under test plus two members:

- `reference(grid, ...)`, decorated with `@static_reference`: a NumPy implementation
  computing the expected outputs.
- `input_data(data_alloc, ...)`, decorated with `@input_data_fixture`: a pytest fixture
  building the program arguments, allocating them through `data_alloc` so that they end up
  on the device the selected backend expects.

Neither takes `self`; both are turned into static methods by their decorator. `data_alloc`
and `grid` are ordinary fixtures, so a suite requests `grid` only when it needs it.

Both conventions are enforced in `StencilTest.__init_subclass__`, so a suite that does not
follow them fails when its module is imported rather than misbehaving at run time.
`__init_subclass__` also attaches the test function itself, named after the subclass so it
is identifiable in pytest output.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import functools
import inspect
import os
import textwrap
import types
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias, cast

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py.next import common as gtx_common, typing as gtx_typing

# TODO(havogt): import will disappear after FieldOperators support `.compile`
from gt4py.next.ffront.decorator import FieldOperator
from gt4py.next.instrumentation import hooks as gtx_hooks, metrics as gtx_metrics

from icon4py.model.common import exceptions, model_backends, model_options, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation, device_utils
from icon4py.model.testing import test_utils


if TYPE_CHECKING:
    import numpy.typing as npt

_STENCIL_REFERENCE_MARKER: Final = "__stencil_test_reference__"
_INPUT_DATA_FIXTURE_MARKER: Final = "__stencil_test_input_fixture__"
_METRICS_KEY_EXTRACTOR: Final = "metrics_id_extractor"

#: gt4py constructors that allocate a field, and so need an allocator.
_GT4PY_FIELD_CONSTRUCTORS: Final = frozenset({"as_field", "zeros", "ones", "full", "empty"})

#: Members every `StencilTest` subclass must define, as (name, marker, decorator name).
_REQUIRED_MEMBERS: Final = (
    ("reference", _STENCIL_REFERENCE_MARKER, "static_reference"),
    ("input_data", _INPUT_DATA_FIXTURE_MARKER, "input_data_fixture"),
)

# TODO(iomaganaris, havogt, nfarabullini): tolerance was increased from 1e-7 to 1e-6 to
# cover floating point discrepancies observed in CI tests. Failing CI can be found in
# https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/5125340235196978/2255149825504673/-/pipelines/2184694383
# from PR#861. Reason is probably derivatives of random data. Investigate and lower the
# tolerance back to 1e-7 if possible.
_RELATIVE_TOLERANCE: Final = 3e-6


def _validate_signature(
    func: types.FunctionType | staticmethod,
    *,
    name: str,
    leading_params: tuple[str, ...],
    allowed_types: tuple[type, ...],
    allowed_description: str,
) -> None:
    """Check that `func` is named `name` and its signature starts with `leading_params`."""
    if not isinstance(func, allowed_types):
        raise TypeError(f"The '{name}' method must be {allowed_description} but got {type(func)}.")
    if func.__name__ != name:
        raise ValueError(f"The '{name}' method must be named '{name}' but got '{func.__name__}'.")
    signature = inspect.signature(func)
    params = tuple(signature.parameters)
    if params[: len(leading_params)] != leading_params:
        raise ValueError(
            f"The '{name}' method signature must be '{name}({', '.join(leading_params)}, ...)'"
            f" but got '{name}({', '.join(params)})'."
        )
    # Both are always passed by name -- `reference` by `test_and_benchmark`, `input_data` by
    # pytest -- so a positional-only parameter would only fail once the test runs.
    positional_only = [
        param
        for param in leading_params
        if signature.parameters[param].kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        raise ValueError(
            f"The '{name}' method is called with keyword arguments, so"
            f" {', '.join(repr(p) for p in positional_only)} cannot be positional-only."
        )


def _reject_direct_data_allocation(func: types.FunctionType, tree: ast.AST) -> None:
    """
    Ensure `func` allocates through its `data_alloc` argument, not by calling `data_allocation`.

    Only the wrapper has the grid and the backend's allocator bound, so a direct call would
    silently allocate on the wrong device.

    The names `func` reads are collected from its AST, which covers nested functions and
    comprehensions, and which never mistakes an attribute access for a name because that is
    an `ast.Attribute`. What the AST cannot tell apart is a global from a local, hence the
    subtraction below. `inspect.getclosurevars(func).globals` is the reverse trade and no
    better: it sees only the top-level code object, and resolves every name in `co_names`,
    attribute names included.
    """
    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    # A parameter shadows any global of the same name, `data_alloc` above all: the fixture
    # receives the wrapper under exactly the name test modules bind the wrapped module to.
    definition = tree.body[0]
    if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = definition.args
        referenced -= {
            arg.arg
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
            if arg is not None
        }

    def is_data_allocation(value: Any) -> bool:
        # Not just the module: `from ...data_allocation import zero_field` binds the
        # constructor itself, which allocates on the wrong device just as readily.
        return value is data_allocation or (
            inspect.isfunction(value) and value.__module__ == data_allocation.__name__
        )

    scopes = (func.__globals__, inspect.getclosurevars(func).nonlocals)
    if any(is_data_allocation(scope.get(name)) for name in referenced for scope in scopes):
        raise TypeError(
            "The 'input_data_fixture' should not call 'data_allocation' functions directly. "
            "Use the 'data_alloc' fixture argument to access data allocation functions instead."
        )


def _reject_host_field_construction(tree: ast.AST) -> None:
    """
    Ensure a gt4py constructor called directly is given an allocator.

    Without one it allocates on the host, and a GPU backend then rejects the field. The
    `data_alloc` wrapper passes the allocator for you; when a domain it cannot express
    forces a direct call, pass `data_alloc.allocator` explicitly.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _GT4PY_FIELD_CONSTRUCTORS:
            continue
        root: ast.expr = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if not (isinstance(root, ast.Name) and root.id.startswith("gtx")):
            continue
        if not any(keyword.arg == "allocator" for keyword in node.keywords):
            raise TypeError(
                f"'{ast.unparse(node.func)}' is called without an 'allocator', so it"
                " allocates on the host and a GPU backend cannot use the result. Allocate"
                " through the 'data_alloc' fixture argument, or pass"
                " 'allocator=data_alloc.allocator' when it cannot express the domain."
            )


def _static_reference(func: types.FunctionType | staticmethod) -> staticmethod:
    """Runtime implementation of the public `static_reference` decorator."""
    _validate_signature(
        func,
        name="reference",
        leading_params=("grid",),
        allowed_types=(types.FunctionType, staticmethod),
        allowed_description="a regular function or a staticmethod",
    )
    marked = func if isinstance(func, staticmethod) else staticmethod(func)
    setattr(marked, _STENCIL_REFERENCE_MARKER, True)

    return marked


def _input_data_fixture(
    func: types.FunctionType | staticmethod | None = None, **kwargs: Any
) -> Any:
    """Runtime implementation of the public `input_data_fixture` decorator."""
    if func is None:  # called with parentheses: return the actual decorator
        return functools.partial(_input_data_fixture, **kwargs)

    _validate_signature(
        func,
        name="input_data",
        leading_params=("data_alloc",),
        allowed_types=(types.FunctionType, staticmethod),
        allowed_description="a regular function or a staticmethod",
    )
    plain = cast(types.FunctionType, func.__func__) if isinstance(func, staticmethod) else func
    try:
        source: str | None = inspect.getsource(plain)
    except OSError:  # no source available, e.g. a dynamically generated function
        source = None
    if source is not None:
        tree = ast.parse(textwrap.dedent(source))
        _reject_direct_data_allocation(plain, tree)
        _reject_host_field_construction(tree)

    kwargs.setdefault("scope", "class")
    # Wrapped in a `staticmethod` because pytest deprecated class-scoped fixtures defined
    # as instance methods: `self` is a different instance there than in the test.
    fixture = staticmethod(pytest.fixture(**kwargs)(plain))
    setattr(fixture, _INPUT_DATA_FIXTURE_MARKER, True)

    return fixture


if TYPE_CHECKING:
    # Type checkers see the decorators as what they effectively are, so that decorated
    # members keep their usual typing. `static_reference` is deliberately a `TypeAlias`
    # rather than a PEP 695 `type` statement: it has to *be* `staticmethod` for the
    # descriptor semantics to survive.
    static_reference: TypeAlias = staticmethod  # noqa: UP040 [non-pep695-type-alias]
    input_data_fixture: Final = pytest.fixture
else:
    static_reference = _static_reference
    input_data_fixture = _input_data_fixture


@dataclasses.dataclass(frozen=True)
class DataAllocationWrapper:
    """
    Field constructors with `grid` and `allocator` already bound.

    Mostly the `icon4py.model.common.utils.data_allocation` ones; `as_field` and
    `connectivity_field` go straight to gt4py, having no grid-shaped counterpart there.

    A `StencilTest` suite receives one as the `data_alloc` argument of its `input_data`
    fixture. See the wrapped module for the meaning of the remaining arguments.
    """

    grid: base.Grid
    allocator: gtx_typing.Allocator | None

    def connectivity_field(self, offset: str | gtx.FieldOffset) -> gtx.Field:
        """
        A connectivity table as a regular field, for stencils consuming it as data.

        `Grid.get_connectivity` returns a `NeighborTable`, which cannot be passed as a
        program argument; re-allocating it yields a plain field on the right device.
        """
        connectivity = self.grid.get_connectivity(offset)
        return gtx.as_field(
            domain=connectivity.domain, data=connectivity.ndarray, allocator=self.allocator
        )

    def as_field(
        self,
        data: npt.NDArray,
        *dims: type[gtx.Dimension],
        dtype: npt.DTypeLike | None = None,
    ) -> gtx.Field:
        """
        A field over `dims` holding `data`, for inputs built with NumPy first.

        Writing into an already allocated field instead only works while its buffer is
        host memory, which it is not under a GPU backend.
        """
        return gtx.as_field(dims, data, dtype=dtype, allocator=self.allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any]"; expected "NDArrayObject"

    def constant_field(
        self,
        value: float,
        *dims: type[gtx.Dimension],
        dtype: npt.DTypeLike = ta.wpfloat,
        extend: dict[type[gtx.Dimension], int] | None = None,
    ) -> gtx.Field:
        """A field filled with `value`."""
        return data_allocation.constant_field(
            self.grid, value, *dims, dtype=dtype, extend=extend, allocator=self.allocator
        )

    def index_field(
        self,
        dim: type[gtx.Dimension],
        *,
        extend: dict[type[gtx.Dimension], int] | None = None,
        dtype: npt.DTypeLike = gtx.int32,
    ) -> gtx.Field:
        """A field over `dim` holding each element's own index."""
        return data_allocation.index_field(
            grid=self.grid, dim=dim, extend=extend, dtype=dtype, allocator=self.allocator
        )

    def random_field(
        self,
        *dims: type[gtx.Dimension],
        low: float = -1.0,
        high: float = 1.0,
        dtype: npt.DTypeLike | None = None,
        extend: dict[type[gtx.Dimension], int] | None = None,
    ) -> gtx.Field:
        """A field of uniform random values in `[low, high)`."""
        return data_allocation.random_field(
            self.grid,
            *dims,
            low=low,
            high=high,
            dtype=dtype,
            allocator=self.allocator,
            extend=extend,
        )

    def random_mask(
        self,
        *dims: type[gtx.Dimension],
        dtype: npt.DTypeLike | None = None,
        extend: dict[type[gtx.Dimension], int] | None = None,
    ) -> gtx.Field:
        """A field of random booleans, or of `dtype` if given."""
        return data_allocation.random_mask(
            self.grid, *dims, dtype=dtype, allocator=self.allocator, extend=extend
        )

    def random_sign(
        self,
        *dims: type[gtx.Dimension],
        dtype: npt.DTypeLike | None = None,
        extend: dict[type[gtx.Dimension], int] | None = None,
    ) -> gtx.Field:
        """A field of random values in `{-1, 1}`."""
        return data_allocation.random_sign(
            self.grid, *dims, dtype=dtype, allocator=self.allocator, extend=extend
        )

    def zero_field(
        self,
        *dims: type[gtx.Dimension],
        dtype: npt.DTypeLike = ta.wpfloat,
        extend: dict[type[gtx.Dimension], int] | None = None,
    ) -> gtx.Field:
        """A field filled with zeros."""
        return data_allocation.zero_field(
            self.grid, *dims, dtype=dtype, allocator=self.allocator, extend=extend
        )


class _NumPyGridConnectivitiesView(Mapping[str | gtx.FieldOffset, np.ndarray]):
    """Read-only `Mapping` exposing a grid's neighbor tables as NumPy arrays."""

    def __init__(self, grid: base.Grid) -> None:
        self._grid = grid

    def __getitem__(self, key: str | gtx.FieldOffset) -> np.ndarray:
        # `KeyError` rather than what the grid raises: `Mapping` builds `get` and `in` on
        # top of this, and both have to see a missing key as missing rather than as an error.
        try:
            connectivity = self._grid.get_connectivity(key)
        except exceptions.MissingConnectivityError as error:
            raise KeyError(key) from error
        if not gtx_common.is_neighbor_table(connectivity):
            raise KeyError(f"Connectivity '{key}' is not a neighbor table.")
        return connectivity.asnumpy()

    def __iter__(self) -> Iterator[str | gtx.FieldOffset]:
        return (
            key
            for key, connectivity in self._grid.connectivities.items()
            if gtx_common.is_neighbor_table(connectivity)
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)


def connectivities_asnumpy(grid: base.Grid) -> Mapping[gtx.FieldOffset, np.ndarray]:
    """
    A read-only view of `grid`'s neighbor tables as NumPy arrays.

    Entries can be looked up by `FieldOffset`, as the return annotation advertises, or by
    name. The cast is needed because the underlying mapping is keyed by name and `Mapping`
    is invariant in its key type, so the honest `Mapping[str | FieldOffset, ...]` would not
    be accepted where reference helpers ask for `Mapping[FieldOffset, ...]`.
    """
    return cast(Mapping[gtx.FieldOffset, np.ndarray], _NumPyGridConnectivitiesView(grid))


@dataclasses.dataclass(frozen=True)
class Output:
    """
    An output to verify, optionally comparing only part of it.

    `refslice` selects the region of the reference array and `gtslice` the region of the
    computed field; both default to the whole field. Use it in `StencilTest.OUTPUTS` in
    place of a plain name whenever the two do not line up, e.g. because the program only
    writes an interior sub-domain.
    """

    name: str
    refslice: tuple[slice, ...] = (slice(None),)
    gtslice: tuple[slice, ...] = (slice(None),)


class StandardStaticVariants(eve.StrEnum):
    """Common `StencilTest.STATIC_PARAMS` categories for compile-time specialization."""

    NONE = "none"
    COMPILE_TIME_DOMAIN = "compile_time_domain"
    COMPILE_TIME_VERTICAL = "compile_time_vertical"


def _collect_compute_samples(
    configured_program: Callable[..., None],
    program_kwargs: dict[str, Any],
    *,
    iterations_to_skip: int,
) -> list[Any]:
    """
    Run the program once more to recover its metrics key, and return its `compute` samples.

    The leading samples are dropped because they do not measure the benchmarked steady
    state: they come from the verification run (if any), from pytest-benchmark's
    calibration round, from the warmup iterations, and from this extra run itself.
    """
    metrics_key: str | None = None

    @contextlib.contextmanager
    def _capture_metrics_key(
        program: gtx_typing.Program,
        args: tuple[Any, ...],
        offset_provider: gtx_common.OffsetProvider,
        enable_jit: bool,
        kwargs: dict[str, Any],
    ) -> Generator[None, None, None]:
        yield
        # Collect the key after running the program to make sure it is set
        nonlocal metrics_key
        metrics_key = gtx_metrics.get_current_source_key()

    gtx_hooks.program_call_context.register(_capture_metrics_key, name=_METRICS_KEY_EXTRACTOR)
    try:
        configured_program(**program_kwargs)
    finally:
        gtx_hooks.program_call_context.remove(_METRICS_KEY_EXTRACTOR)

    if metrics_key is None:
        raise RuntimeError("Metrics key could not be recovered during run.")
    if not metrics_key.startswith(configured_program.__name__):
        raise RuntimeError(
            f"Metrics key ({metrics_key}) does not start with the program name"
            f" ({configured_program.__name__})"
        )
    if len(configured_program._compiled_programs.compiled_programs) != 1:
        raise RuntimeError("Multiple compiled programs found, cannot extract metrics.")

    samples = gtx_metrics.sources[metrics_key].metrics["compute"].samples
    if len(samples) <= iterations_to_skip:
        raise RuntimeError("Not enough samples collected to compute metrics.")

    return samples[iterations_to_skip:]


def test_and_benchmark(
    self: StencilTest,
    *,
    benchmark: Any,  # should be `pytest_benchmark.fixture.BenchmarkFixture` but pytest_benchmark is not typed
    grid: base.Grid,
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    configured_program: Callable[..., None],
    request: pytest.FixtureRequest,
) -> None:
    """
    Verify the stencil program against its reference, then benchmark it.

    Note that it is defined as a standalone function and then attached to the `StencilTest`
    subclasses in order to use a meaningful name for the test in pytest output.
    """
    # skip verification if the `--skip-stenciltest-verification` CLI option is set
    skip_verification = request.config.getoption("skip_stenciltest_verification")
    skip_benchmark = benchmark is None or not benchmark.enabled
    program_kwargs: dict[str, Any] = {**input_data, "offset_provider": grid.connectivities}

    if not skip_verification:
        reference_outputs = self.reference(
            grid=grid,
            **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
        )
        configured_program(**program_kwargs)
        self.verify_data(input_data=input_data, reference_outputs=reference_outputs)

    if not skip_benchmark:
        warmup_rounds = int(os.getenv("ICON4PY_STENCIL_TEST_WARMUP_ROUNDS", "1"))
        iterations = int(os.getenv("ICON4PY_STENCIL_TEST_ITERATIONS", "10"))

        # Use of `pedantic` to explicitly control warmup rounds and iterations
        benchmark.pedantic(
            configured_program,
            args=(),
            kwargs=program_kwargs,
            rounds=int(
                os.getenv("ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS", "3")
            ),  # 30 iterations in total should be stable enough
            warmup_rounds=warmup_rounds,
            iterations=iterations,
        )

        if gtx_metrics.is_any_level_enabled():
            benchmark.extra_info["gtx_metrics"] = _collect_compute_samples(
                configured_program,
                program_kwargs,
                iterations_to_skip=warmup_rounds * iterations + (2 if skip_verification else 3),
            )


class StencilTest:
    """
    Base class to be used for testing stencils.

    Example (pseudo-code):

        >>> class TestMultiplyByTwo(StencilTest):  # doctest: +SKIP
        ...     PROGRAM = multiply_by_two  # noqa: F821
        ...     OUTPUTS = ("some_output",)
        ...     STATIC_PARAMS = {"category_a": ["flag0"], "category_b": ["flag0", "flag1"]}
        ...
        ...     @static_reference
        ...     def reference(grid, some_input, **kwargs):
        ...         return dict(some_output=np.asarray(some_input) * 2)
        ...
        ...     @input_data_fixture
        ...     def input_data(data_alloc):
        ...         return {
        ...             "some_input": data_alloc.random_field(dims.CellDim),  # noqa: F821
        ...             "some_output": data_alloc.zero_field(dims.CellDim),  # noqa: F821
        ...         }

    `reference` must be decorated with `@static_reference` and take `grid` first;
    `input_data` must be decorated with `@input_data_fixture` and take `data_alloc` first.
    Both are checked in `__init_subclass__`.
    """

    PROGRAM: ClassVar[gtx_typing.Program | gtx_typing.FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str]] | None] = None

    reference: ClassVar[Callable[..., Mapping[str, np.ndarray | tuple[np.ndarray, ...]]]]
    input_data: ClassVar[Callable[..., dict[str, Any]]]

    @pytest.fixture
    def configured_program(
        self,
        backend_like: model_backends.BackendLike,
        static_variant: Sequence[str],
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        grid: base.Grid,
    ) -> Callable[..., None]:
        """The program under test, compiled for the selected backend and static variant."""
        unused_static_params = set(static_variant) - set(input_data.keys())
        if unused_static_params:
            raise ValueError(
                f"Parameter defined in 'STATIC_PARAMS' not in 'input_data': {unused_static_params}"
            )
        static_args = {name: [input_data[name]] for name in static_variant}
        backend = model_options.customize_backend(self.PROGRAM, backend_like)
        program = self.PROGRAM.with_backend(backend)
        if backend is not None:
            if isinstance(program, FieldOperator):
                if len(static_args) > 0:
                    raise NotImplementedError(
                        "'FieldOperator's do not support static arguments yet."
                    )
            else:
                program.compile(
                    offset_provider=grid.connectivities,
                    **static_args,  # type: ignore[arg-type]
                )

        return device_utils.synchronized_function(program, allocator=backend)

    def verify_data(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        reference_outputs: Mapping[str, np.ndarray | tuple[np.ndarray, ...]],
    ) -> None:
        """Compare every entry of `OUTPUTS` against the reference, honouring its slices."""
        for entry in self.OUTPUTS:
            out = entry if isinstance(entry, Output) else Output(entry)
            computed = input_data[out.name]
            expected = reference_outputs[out.name]

            # Normalize the scalar and tuple cases so both are verified the same way.
            computed_fields = computed if isinstance(computed, tuple) else (computed,)
            expected_arrays = expected if isinstance(expected, tuple) else (expected,)

            for index, (field, reference) in enumerate(
                zip(computed_fields, expected_arrays, strict=True)
            ):
                label = f"{out.name}[{index}]" if isinstance(computed, tuple) else out.name
                test_utils.assert_dallclose(
                    field.asnumpy()[out.gtslice],
                    reference[out.refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{label}'",
                    rtol=_RELATIVE_TOLERANCE,
                )

    @staticmethod
    def static_variant(request: pytest.FixtureRequest) -> Sequence[str]:
        """
        Fixture for parametrization over the `STATIC_PARAMS` of the test class.

        Note: the actual `pytest.fixture()`  decoration happens inside `__init_subclass__`,
          when all information is available.
        """
        _, variant = request.param
        return () if variant is None else variant

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        """Enforce the suite conventions and attach the test function to the subclass."""
        super().__init_subclass__(*args, **kwargs)

        for member_name, marker, decorator_name in _REQUIRED_MEMBERS:
            # `getattr_static` returns the descriptor carrying the marker (a plain
            # `getattr` would unwrap the staticmethod) and still searches the MRO.
            member = inspect.getattr_static(cls, member_name, None)
            if member is None:
                raise TypeError(
                    f"'{cls.__name__}' StencilTest subclass does not implement"
                    f" the required '{member_name}' method."
                )
            if not getattr(member, marker, False):
                raise TypeError(
                    f"The '{member_name}' method of '{cls.__name__}' must be decorated"
                    f" with '@{decorator_name}'."
                )

        setattr(cls, f"test_{cls.__name__}", test_and_benchmark)

        # Decorate `static_variant` with parametrized fixtures, since the
        # parametrization is only available in the concrete subclass definition
        if cls.STATIC_PARAMS is None:
            # not parametrized, return an empty tuple
            cls.static_variant = staticmethod(pytest.fixture(lambda: ()))  # type: ignore[method-assign] # we override with a non-parametrized function
        else:
            cls.static_variant = staticmethod(  # type: ignore[method-assign]
                pytest.fixture(params=cls.STATIC_PARAMS.items(), scope="class", ids=lambda p: p[0])(
                    cls.static_variant
                )
            )
