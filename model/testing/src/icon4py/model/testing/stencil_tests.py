# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import os
import types
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol, TypeAlias, cast

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py.next import typing as gtx_typing

# TODO(havogt): import will disappear after FieldOperators support `.compile`
from gt4py.next.ffront.decorator import FieldOperator
from gt4py.next.instrumentation import hooks as gtx_hooks, metrics as gtx_metrics

from icon4py.model.common import model_backends, model_options, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation, device_utils
from icon4py.model.testing import test_utils


if TYPE_CHECKING:
    import numpy.typing as npt

_STENCIL_REFERENCE_MARKER: Final = "__stencil_test_reference__"
_INPUT_DATA_FIXTURE_MARKER: Final = "__stencil_test_input_fixture__"


def _static_reference(func: types.FunctionType | staticmethod) -> staticmethod:
    """Decorator to mark the `reference` method of a `StencilTest` suite."""
    if not isinstance(func, (types.FunctionType, staticmethod)):
        raise TypeError(
            f"The 'reference' function must be a regular function or staticmethod but got {type(func)}."
        )
    if func.__name__ != "reference":
        raise ValueError(
            f"The 'reference' method must be named 'reference' but got '{func.__name__}'."
        )
    if not isinstance(func, staticmethod):
        func = staticmethod(func)

    setattr(func, _STENCIL_REFERENCE_MARKER, True)

    return func


def _input_data_fixture(
    func: types.FunctionType | None = None, **kwargs: Any
) -> types.FunctionType | Callable[[types.FunctionType], types.FunctionType]:
    """
    Decorator to mark the `input_data` method of a `StencilTest` suite as a pytest fixture.

    Perform some checks on the decorated function and forward all the keyword
    arguments to `pytest.fixture` for parametrization and scoping (default: "class").
    """
    if func is None:
        return functools.partial(input_data_fixture, **kwargs)

    if not isinstance(func, types.FunctionType):
        raise TypeError(f"The 'input_data' method must be a regular function but got {type(func)}.")
    if func.__name__ != "input_data":
        raise ValueError(
            f"The 'input_data' method must be named 'input_data' but got '{func.__name__}'."
        )
    func_params = tuple(inspect.signature(func).parameters.keys())
    if func_params[:2] != ("self", "grid"):
        raise ValueError(
            f"The 'input_data' method signature must be 'input_data(self, grid, ...)' but got"
            f" '{func.__name__}{func_params}'."
        )

    # This allows us to check that the `input_data` fixture does not call any `data_allocation`
    # functions directly and thus it only uses the `self.data_alloc` wrapper, which ensures
    # that the backend and grid are properly bound. However, it might be a bit too strict,
    # since it means that the `data_allocation` module cannot be imported in the global scope
    # of the test module (it can be still imported in the local scope of other functions).
    # We can remove it in the future if it causes many issues.
    cv = inspect.getclosurevars(func)
    if any(ref is data_allocation for ref in [*cv.globals.values(), *cv.nonlocals.values()]):
        raise TypeError(
            "The 'input_data_fixture' should not call 'data_allocation' functions directly. "
            "Use self.data_alloc inside the fixture to access data allocation functions instead."
        )

    kwargs.setdefault("scope", "class")
    fixt = pytest.fixture(**kwargs)(func)
    setattr(fixt, _INPUT_DATA_FIXTURE_MARKER, True)

    return fixt


if TYPE_CHECKING:
    static_reference: TypeAlias = staticmethod
    input_data_fixture: Final = pytest.fixture
else:
    static_reference = _static_reference
    input_data_fixture = _input_data_fixture


@dataclasses.dataclass
class _ConnectivityConceptFixer:
    """
    This works around a misuse of dimensions as an identifier for connectivities.
    Since GT4Py might change the way the mesh is represented, we could
    keep this for a while, otherwise we need to touch all StencilTests.
    """

    _grid: base.Grid

    def __getitem__(self, dim: gtx.Dimension | str) -> np.ndarray:
        if isinstance(dim, gtx.Dimension):
            dim = dim.value
        return self._grid.get_connectivity(dim).asnumpy()


class DataAllocation(Protocol):
    """
    This protocol mimics the 'icon4py.model.common.utils.data_allocation',
    but with 'backend' bound in the respective functions.
    """

    def constant_field(
        self,
        value: float,
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = ta.wpfloat,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    def index_field(
        self,
        dim: gtx.Dimension,
        extend: dict[gtx.Dimension, int] | None = None,
        dtype: npt.DTypeLike = gtx.int32,
        allocator: gtx_typing.Allocator | None = None,
    ) -> gtx.Field: ...

    def random_field(
        self,
        *dims: gtx.Dimension,
        low: float = -1.0,
        high: float = 1.0,
        dtype: npt.DTypeLike | None = None,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    def random_mask(
        self,
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = None,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...

    def random_sign(
        self,
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = None,
        extend: dict[gtx.Dimension, int] | None = None,
        allocator: gtx_typing.Allocator | None = None,
    ) -> gtx.Field: ...

    def zero_field(
        self,
        *dims: gtx.Dimension,
        dtype: npt.DTypeLike | None = ta.wpfloat,
        extend: dict[gtx.Dimension, int] | None = None,
    ) -> gtx.Field: ...


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))


class StandardStaticVariants(eve.StrEnum):
    NONE = "none"
    COMPILE_TIME_DOMAIN = "compile_time_domain"
    COMPILE_TIME_VERTICAL = "compile_time_vertical"


def test_and_benchmark(
    self: StencilTest,
    benchmark: Any,  # should be `pytest_benchmark.fixture.BenchmarkFixture` but pytest_benchmark is not typed
    grid: base.Grid,
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    configured_program: Callable[..., None],
    request: pytest.FixtureRequest,
) -> None:
    """
    Test and benchmark the stencil program.

    Note that it is defined as a standalone function and then attached to the `StencilTest`
    subclasses in order to use a meaningful name for the test in pytest output.
    """
    skip_stenciltest_verification = request.config.getoption(
        "skip_stenciltest_verification"
    )  # skip verification if `--skip-stenciltest-verification` CLI option is set
    skip_stenciltest_benchmark = benchmark is None or not benchmark.enabled

    if not skip_stenciltest_verification:
        reference_outputs = self.reference(
            grid=_ConnectivityConceptFixer(grid),
            **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
        )

        configured_program(**input_data, offset_provider=grid.connectivities)
        self.verify_data(input_data=input_data, reference_outputs=reference_outputs)

    if not skip_stenciltest_benchmark:
        warmup_rounds = int(os.getenv("ICON4PY_STENCIL_TEST_WARMUP_ROUNDS", "1"))
        iterations = int(os.getenv("ICON4PY_STENCIL_TEST_ITERATIONS", "10"))

        # Use of `pedantic` to explicitly control warmup rounds and iterations
        benchmark.pedantic(
            configured_program,
            args=(),
            kwargs=dict(**input_data, offset_provider=grid.connectivities),
            rounds=int(
                os.getenv("ICON4PY_STENCIL_TEST_BENCHMARK_ROUNDS", "3")
            ),  # 30 iterations in total should be stable enough
            warmup_rounds=warmup_rounds,
            iterations=iterations,
        )

        # Collect GT4Py runtime metrics if enabled
        if gtx_metrics.is_any_level_enabled():
            metrics_key = None
            # Run the program one final time to get the metrics key
            METRICS_KEY_EXTRACTOR: Final = "metrics_id_extractor"

            @contextlib.contextmanager
            def _get_metrics_id_program_callback(
                program: gtx_typing.Program,
                args: tuple[Any, ...],
                offset_provider: gtx.common.OffsetProvider,
                enable_jit: bool,
                kwargs: dict[str, Any],
            ) -> Generator[None, None, None]:
                yield
                # Collect the key after running the program to make sure it is set
                nonlocal metrics_key
                metrics_key = gtx_metrics.get_current_source_key()

            gtx_hooks.program_call_context.register(
                _get_metrics_id_program_callback, name=METRICS_KEY_EXTRACTOR
            )
            configured_program(**input_data, offset_provider=grid.connectivities)
            gtx_hooks.program_call_context.remove(METRICS_KEY_EXTRACTOR)

            if metrics_key is None:
                raise RuntimeError("Metrics key could not be recovered during run.")
            if not metrics_key.startswith(configured_program.__name__):
                raise RuntimeError(
                    f"Metrics key ({metrics_key}) does not start with the program name ({configured_program.__name__})"
                )
            if len(configured_program._compiled_programs.compiled_programs) != 1:
                raise RuntimeError("Multiple compiled programs found, cannot extract metrics.")

            metrics_data = gtx_metrics.sources
            compute_samples = metrics_data[metrics_key].metrics["compute"].samples
            # exclude:
            #  - one for validation (if executed)
            #  - one extra warmup round for calibrating pytest-benchmark
            #  - warmup iterations
            #  - one last round to get the metrics key
            initial_program_iterations_to_skip = warmup_rounds * iterations + (
                2 if skip_stenciltest_verification else 3
            )

            if len(compute_samples) <= initial_program_iterations_to_skip:
                raise RuntimeError("Not enough samples collected to compute metrics.")

            benchmark.extra_info["gtx_metrics"] = compute_samples[
                initial_program_iterations_to_skip:
            ]


class StencilTest:
    """
    Base class to be used for testing stencils.

    Example (pseudo-code):

        >>> class TestMultiplyByTwo(StencilTest):  # doctest: +SKIP
        ...     PROGRAM = multiply_by_two  # noqa: F821
        ...     OUTPUTS = ("some_output",)
        ...     STATIC_PARAMS = {"category_a": ["flag0"], "category_b": ["flag0", "flag1"]}
        ...
        ...     @pytest.fixture
        ...     def input_data(self):
        ...         return {"some_input": ..., "some_output": ...}
        ...
        ...     @staticmethod
        ...     def reference(some_input, **kwargs):
        ...         return dict(some_output=np.asarray(some_input) * 2)
    """

    PROGRAM: ClassVar[gtx_typing.Program | gtx_typing.FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str]] | None] = None

    reference: ClassVar[Callable[..., Mapping[str, np.ndarray | tuple[np.ndarray, ...]]]]
    input_data: ClassVar[Callable[..., dict[str, Any]]]

    data_alloc: DataAllocation

    @pytest.fixture
    def configured_program(
        self,
        backend_like: model_backends.BackendLike,
        static_variant: Sequence[str],
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        grid: base.Grid,
    ) -> Callable[..., None]:
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

        test_func = device_utils.synchronized_function(program, allocator=backend)
        return test_func

    @pytest.fixture(autouse=True, scope="class")
    def _provision_data_alloc_fixture(
        self, backend_like: model_backends.BackendLike, grid: base.Grid
    ) -> Generator[None, None, None]:
        """
        Convenience fixture to provide data allocation functions with backend and grid already bound.
        """
        allocator = model_backends.get_allocator(backend_like)

        class DataAllocationWrapper:
            def __getattr__(self, name: str) -> Callable[..., gtx.Field]:
                if name in (
                    "constant_field",
                    "index_field",
                    "random_field",
                    "random_mask",
                    "random_sign",
                    "zero_field",
                ):
                    alloc_fun = getattr(data_allocation, name)
                    return functools.partial(alloc_fun, grid, allocator=allocator)
                else:
                    raise AttributeError(f"Invalid data allocation function '{name}'.")

        try:
            self.data_alloc = cast(DataAllocation, DataAllocationWrapper())
            yield

        finally:
            del self.data_alloc

    def verify_data(
        self,
        input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
        reference_outputs: Mapping[str, np.ndarray | tuple[np.ndarray, ...]],
    ) -> None:
        for out in self.OUTPUTS:
            name, refslice, gtslice = (
                (out.name, out.refslice, out.gtslice)
                if isinstance(out, Output)
                else (out, (slice(None),), (slice(None),))
            )

            input_data_name = input_data[name]  # for mypy
            # TODO(iomaganaris, havogt, nfarabullini): tolerance was increased from 1e-7 to 1e-6
            # to cover floating point descripancies observed in CI tests. Failing CI can be found in
            # https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/5125340235196978/2255149825504673/-/pipelines/2184694383
            # from PR#861. Reason is probably derivatives of random data. Investigate and lower tolerance back to 1e-7 if possible.
            relative_tolerance = 3e-6
            if isinstance(input_data_name, tuple):
                for i_out_field, out_field in enumerate(input_data_name):
                    test_utils.assert_dallclose(
                        out_field.asnumpy()[gtslice],
                        reference_outputs[name][i_out_field][refslice],
                        equal_nan=True,
                        err_msg=f"Verification failed for '{name}[{i_out_field}]'",
                        rtol=relative_tolerance,  # TODO(iomaganaris, havogt, nfarabullini): check above comment
                    )
            else:
                reference_outputs_name = reference_outputs[name]  # for mypy
                assert isinstance(reference_outputs_name, np.ndarray)
                test_utils.assert_dallclose(
                    input_data_name.asnumpy()[gtslice],
                    reference_outputs_name[refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{name}'",
                    rtol=relative_tolerance,  # TODO(iomaganaris, havogt, nfarabullini): check above comment
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
        super().__init_subclass__(*args, **kwargs)

        # Check the conventions for `reference` and `input_data` methods
        if not hasattr(cls, "reference"):
            raise TypeError(
                f"{cls.__name__} StencilTest subclass does not implement a 'reference' method."
            )
        if not getattr(cls.__dict__["reference"], _STENCIL_REFERENCE_MARKER, False):
            raise RuntimeError(
                f"The 'reference' method of {cls.__name__} must be decorated with '@static_reference'."
            )
        if not hasattr(cls, "input_data"):
            raise TypeError(
                f"{cls.__name__} StencilTest subclass does not implement an 'input_data' method."
            )
        if not getattr(cls.__dict__["input_data"], _INPUT_DATA_FIXTURE_MARKER, False):
            raise RuntimeError(
                f"The 'input_data' method of {cls.__name__} must be decorated with '@input_data_fixture'."
            )

        setattr(cls, f"test_{cls.__name__}", test_and_benchmark)

        # Decorate `static_variant` with parametrized fixtures, since the
        # parametrization is only available in the concrete subclass definition
        if cls.STATIC_PARAMS is None:
            # not parametrized, return an empty tuple
            cls.static_variant = staticmethod(pytest.fixture(lambda: ()))  # type: ignore[method-assign, assignment] # we override with a non-parametrized function
        else:
            cls.static_variant = staticmethod(  # type: ignore[method-assign]
                pytest.fixture(params=cls.STATIC_PARAMS.items(), scope="class", ids=lambda p: p[0])(
                    cls.static_variant
                )
            )
