# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import hashlib
import typing
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py._core.definitions import is_scalar_type
from gt4py.next import backend as gtx_backend, constructors
from gt4py.next.ffront.decorator import FieldOperator, Program
from typing_extensions import Buffer

from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


@pytest.fixture(scope="session")
def connectivities_as_numpy(grid) -> dict[gtx.Dimension, np.ndarray]:
    return {dim: data_alloc.as_numpy(table) for dim, table in grid.neighbor_tables.items()}


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
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]


def dallclose(a, b, rtol=1.0e-12, atol=0.0, equal_nan=False):
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allocate_data(
    backend: Optional[gtx_backend.Backend], input_data: dict[str, gtx.Field]
) -> dict[str, gtx.Field]:
    _allocate_field = constructors.as_field.partial(allocator=backend)
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
    grid: base.BaseGrid,
    backend: gtx_backend.Backend | None,
):
    for marker in markers:
        match marker.name:
            case "cpu_only" if data_alloc.is_cupy_device(backend):
                pytest.xfail("currently only runs on CPU")
            case "embedded_only" if not is_embedded(backend):
                pytest.skip("stencil runs only on embedded backend")
            case "embedded_remap_error" if is_embedded(backend):
                # https://github.com/GridTools/gt4py/issues/1583
                pytest.xfail("Embedded backend currently fails in remap function.")
            case "embedded_static_args" if is_embedded(backend):
                pytest.xfail(" gt4py _compiled_programs returns error when backend is None.")
            case "infinite_concat_where" if is_embedded(backend):
                pytest.xfail("Embedded backend does not support infinite concat_where.")
            case "uses_as_offset" if is_embedded(backend):
                pytest.xfail("Embedded backend does not support as_offset.")
            case "skip_value_error":
                if grid.limited_area or grid.geometry_type == base.GeometryType.ICOSAHEDRON:
                    # TODO (@halungge) this still skips too many tests: it matters what connectivity the test uses
                    pytest.skip(
                        "Stencil does not support domain containing skip values. Consider shrinking domain."
                    )


@dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))


def run_verify_and_benchmark(
    test_func: Callable[[], None],
    verification_func: Callable[[], None],
    benchmark_fixture: Optional[pytest.FixtureRequest],
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
    self,
    input_data: dict[str, gtx.Field],
    reference_outputs: dict[str, np.ndarray],
) -> None:
    for out in self.OUTPUTS:
        name, refslice, gtslice = (
            (out.name, out.refslice, out.gtslice)
            if isinstance(out, Output)
            else (out, (slice(None),), (slice(None),))
        )

        if isinstance(input_data[name], tuple):
            for i_out_field, out_field in enumerate(input_data[name]):
                np.testing.assert_allclose(
                    out_field.asnumpy()[gtslice],
                    reference_outputs[name][i_out_field][refslice],
                    equal_nan=True,
                    err_msg=f"Verification failed for '{name}[{i_out_field}]'",
                )
        else:
            np.testing.assert_allclose(
                input_data[name].asnumpy()[gtslice],
                reference_outputs[name][refslice],
                equal_nan=True,
                err_msg=f"Verification failed for '{name}'",
            )


def _test_and_benchmark(
    self,
    grid: base.BaseGrid,
    backend: gtx_backend.Backend,
    connectivities_as_numpy: dict[str, np.ndarray],
    input_data: dict[str, gtx.Field],
    benchmark: pytest.FixtureRequest,
) -> None:
    if self.MARKERS is not None:
        apply_markers(self.MARKERS, grid, backend)

    connectivities = connectivities_as_numpy
    reference_outputs = self.reference(
        connectivities,
        **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
    )

    input_data = allocate_data(backend, input_data)

    run_verify_and_benchmark(
        functools.partial(
            self.PROGRAM.with_backend(backend),
            **input_data,
            offset_provider=grid.connectivities,
        ),
        functools.partial(
            _verify_stencil_test,
            self=self,
            input_data=input_data,
            reference_outputs=reference_outputs,
        ),
        benchmark,
    )


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

    PROGRAM: ClassVar[Program | FieldOperator]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    MARKERS: typing.Optional[tuple] = None

    def __init_subclass__(cls, **kwargs):
        # Add two methods for verification and benchmarking. In order to have names that
        # reflect the name of the test we do this dynamically here instead of using regular
        # inheritance.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_and_benchmark)


def reshape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return np.reshape(arr, shape)


def as_1d_connectivity(connectivity: np.ndarray) -> np.ndarray:
    old_shape = connectivity.shape
    return np.arange(old_shape[0] * old_shape[1], dtype=gtx.int32).reshape(old_shape)
