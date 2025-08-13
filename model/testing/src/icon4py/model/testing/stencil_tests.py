# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import functools
import hashlib
import typing
from typing import Any, Callable, ClassVar, Optional, Sequence

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py import eve
from gt4py._core.definitions import is_scalar_type
from gt4py.next import backend as gtx_backend, constructors
from gt4py.next.ffront.decorator import FieldOperator, Program
from typing_extensions import Buffer

from icon4py.model.common.grid import base
from icon4py.model.common.utils import device_utils


def is_python(backend: gtx_backend.Backend | None) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_dace(backend: gtx_backend.Backend | None) -> bool:
    return backend.name.startswith("run_dace_") if backend else False


def is_embedded(backend: gtx_backend.Backend | None) -> bool:
    return backend is None


def is_gtfn_backend(backend: gtx_backend.Backend | None) -> bool:
    return "gtfn" in backend.name if backend else False


def is_roundtrip(backend: gtx_backend.Backend | None) -> bool:
    return backend.name == "roundtrip" if backend else False


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]  # type: ignore[arg-type]


def allocate_data(
    backend: Optional[gtx_backend.Backend],
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
) -> dict[str, gtx.Field | tuple[gtx.Field, ...]]:
    _allocate_field = constructors.as_field.partial(allocator=backend)  # type:ignore[attr-defined] # TODO: check why it does understand the fluid_partial
    input_data = {
        k: tuple(_allocate_field(domain=field.domain, data=field.ndarray) for field in v)
        if isinstance(v, tuple)
        else _allocate_field(domain=v.domain, data=v.ndarray)
        if not is_scalar_type(v) and k != "domain"
        else v
        for k, v in input_data.items()
    }
    return input_data


def apply_markers(
    markers: tuple[pytest.Mark | pytest.MarkDecorator, ...],
    grid: base.Grid,
    backend: gtx_backend.Backend | None,
) -> None:
    for marker in markers:
        match marker.name:
            case "cpu_only" if device_utils.is_cupy_device(backend):
                pytest.xfail("currently only runs on CPU")
            case "embedded_only" if not is_embedded(backend):
                pytest.skip("stencil runs only on embedded backend")
            case "embedded_remap_error" if is_embedded(backend):
                # https://github.com/GridTools/gt4py/issues/1583
                pytest.xfail("Embedded backend currently fails in remap function.")
            case "embedded_static_args" if is_embedded(backend):
                pytest.xfail(" gt4py _compiled_programs returns error when backend is None.")
            case "uses_as_offset" if is_embedded(backend):
                pytest.xfail("Embedded backend does not support as_offset.")
            case "uses_concat_where" if is_embedded(backend):
                pytest.xfail("Embedded backend does not support concat_where.")
            case "gtfn_too_slow" if is_gtfn_backend(backend):
                pytest.skip("GTFN compilation is too slow for this test.")
            case "skip_value_error":
                if grid.limited_area or grid.geometry_type == base.GeometryType.ICOSAHEDRON:
                    # TODO (@halungge) this still skips too many tests: it matters what connectivity the test uses
                    pytest.skip(
                        "Stencil does not support domain containing skip values. Consider shrinking domain."
                    )


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = dataclasses.field(default_factory=lambda: (slice(None),))


def run_verify_and_benchmark(
    test_func: Callable[[], None],
    verification_func: Callable[[], None],
    benchmark_fixture: Optional[
        Any
    ],  # should be pytest_benchmark.fixture.BenchmarkFixture pytest_benchmark is not typed
) -> None:
    """
    Function to perform verification and benchmarking of test_func (along with normally executing it).

    Args:
        test_func: Function to be run, verified and benchmarked.
        verification_func: Function to be used for verification of test_func.
        benchmark_fixture: pytest-benchmark fixture.

    Note:
        - test_func and verification_func should be provided with bound arguments, i.e. with functools.partial.
    """
    test_func()
    verification_func()

    if benchmark_fixture is not None and benchmark_fixture.enabled:
        benchmark_fixture(test_func)


def _verify_stencil_test(
    self: StencilTest,
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    reference_outputs: dict[str, np.ndarray | tuple[np.ndarray, ...]],
) -> None:
    for out in self.OUTPUTS:
        name, refslice, gtslice = (
            (out.name, out.refslice, out.gtslice)
            if isinstance(out, Output)
            else (out, (slice(None),), (slice(None),))
        )

        input_data_name = input_data[name]  # for mypy
        if isinstance(input_data_name, tuple):
            for i_out_field, out_field in enumerate(input_data_name):
                np.testing.assert_allclose(
                    out_field.asnumpy()[gtslice],
                    reference_outputs[name][i_out_field][refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{name}[{i_out_field}]'",
                )
        else:
            reference_outputs_name = reference_outputs[name]  # for mypy
            assert isinstance(reference_outputs_name, np.ndarray)
            np.testing.assert_allclose(
                input_data_name.asnumpy()[gtslice],
                reference_outputs_name[refslice],
                equal_nan=True,
                err_msg=f"Verification failed for '{name}'",
            )


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


def _test_and_benchmark(
    self: StencilTest,
    grid: base.Grid,
    backend: gtx_backend.Backend | None,
    input_data: dict[str, gtx.Field | tuple[gtx.Field, ...]],
    static_variant: Sequence[str],  # the names of the static parameters
    benchmark: pytest.FixtureRequest,
) -> None:
    if self.MARKERS is not None:
        apply_markers(self.MARKERS, grid, backend)

    connectivities_as_numpy = _ConnectivityConceptFixer(grid)

    reference_outputs = self.reference(
        connectivities_as_numpy,  # TODO(havogt): pass as keyword argument (needs fixes in some tests)
        **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
    )

    input_data = allocate_data(backend, input_data)

    unused_static_params = set(static_variant) - set(input_data.keys())
    if unused_static_params:
        raise ValueError(
            f"Parameter defined in 'STATIC_PARAMS' not in 'input_data': {unused_static_params}"
        )
    static_args = {name: [input_data[name]] for name in static_variant}

    program = self.PROGRAM.with_backend(backend)  # type: ignore[arg-type]  # TODO: gt4py should accept `None` in with_backend
    if backend is not None:
        if isinstance(program, FieldOperator):
            if len(static_args) > 0:
                raise NotImplementedError("'FieldOperator's do not support static arguments yet.")
        else:
            program.compile(offset_provider=grid.connectivities, enable_jit=False, **static_args)  # type: ignore[arg-type]

    test_func = functools.partial(
        program,
        **input_data,  # type: ignore[arg-type]
        offset_provider=grid.connectivities,
    )
    test_func = device_utils.synchronized_function(test_func, backend=backend)

    run_verify_and_benchmark(
        test_func,
        functools.partial(
            _verify_stencil_test,
            self=self,
            input_data=input_data,
            reference_outputs=reference_outputs,
        ),
        benchmark,
    )


class StandardStaticVariants(eve.StrEnum):
    NONE = "none"
    COMPILE_TIME_DOMAIN = "compile_time_domain"
    COMPILE_TIME_VERTICAL = "compile_time_vertical"


class StencilTest:
    """
    Base class to be used for testing stencils.

    Example (pseudo-code):

        >>> class TestMultiplyByTwo(StencilTest):  # doctest: +SKIP
        ...     PROGRAM = multiply_by_two  # noqa: F821
        ...     OUTPUTS = ("some_output",)
        ...
        ...     @pytest.fixture
        ...     def input_data(self):
        ...         return {"some_input": ..., "some_output": ...}
        ...
        ...     @staticmethod
        ...     def reference(some_input, **kwargs):
        ...         return dict(some_output=np.asarray(some_input) * 2)
    """

    @staticmethod
    def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
        static_params = metafunc.cls.STATIC_PARAMS
        if not static_params:
            metafunc.parametrize("static_variant", ((),), ids=["variant=none"], scope="class")
        else:
            metafunc.parametrize(
                "static_variant",
                (() if v is None else v for v in static_params.values()),
                ids=(f"variant={k}" for k in static_params.keys()),
                scope="class",
            )

    PROGRAM: ClassVar[Program | FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    MARKERS: ClassVar[typing.Optional[tuple]] = None
    STATIC_PARAMS: ClassVar[dict[str, Sequence[str]] | None] = None

    reference: ClassVar[Callable[..., dict[str, np.ndarray | tuple[np.ndarray, ...]]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Add two methods for verification and benchmarking. In order to have names that
        # reflect the name of the test we do this dynamically here instead of using regular
        # inheritance.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_and_benchmark)
