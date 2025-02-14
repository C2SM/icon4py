# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import typing
from dataclasses import dataclass, field
from typing import ClassVar

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py._core.definitions import is_scalar_type
from gt4py.next import backend as gtx_backend, constructors
from gt4py.next.ffront.decorator import Program
from typing_extensions import Buffer

from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc


try:
    import pytest_benchmark
except ModuleNotFoundError:
    pytest_benchmark = None


@pytest.fixture(scope="session")
def connectivities_as_numpy(grid, backend) -> dict[gtx.Dimension, np.ndarray]:
    return {dim: data_alloc.as_numpy(table) for dim, table in grid.connectivities.items()}


def is_python(backend: gtx_backend.Backend | None) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_dace(backend: gtx_backend.Backend | None) -> bool:
    return backend.name.startswith("run_dace_") if backend else False


def is_embedded(backend: gtx_backend.Backend | None) -> bool:
    return backend is None


def is_roundtrip(backend: gtx_backend.Backend | None) -> bool:
    return backend.name == "roundtrip" if backend else False


def extract_backend_name(backend: gtx_backend.Backend | None) -> str:
    return "embedded" if backend is None else backend.name


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]


def dallclose(a, b, rtol=1.0e-12, atol=0.0, equal_nan=False):
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allocate_data(backend, input_data):
    _allocate_field = constructors.as_field.partial(allocator=backend)
    input_data = {
        k: _allocate_field(domain=v.domain, data=v.ndarray) if not is_scalar_type(v) else v
        for k, v in input_data.items()
    }
    return input_data


def apply_markers(
    markers: tuple[pytest.Mark | pytest.MarkDecorator, ...],
    grid: base.BaseGrid,
    backend: gtx_backend.Backend | None,
    is_datatest: bool = False,
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
            case "uses_as_offset" if is_embedded(backend):
                pytest.xfail("Embedded backend does not support as_offset.")
            case "requires_concat_where" if is_embedded(backend):
                pytest.xfail("Stencil requires concat_where.")
            case "skip_value_error":
                if grid.config.limited_area or grid.has_skip_values():
                    # TODO (@halungge) this still skips too many tests: it matters what connectivity the test uses
                    pytest.skip(
                        "Stencil does not support domain containing skip values. Consider shrinking domain."
                    )
            case "datatest" if not is_datatest:
                pytest.skip("need '--datatest' option to run")


@dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))


def _test_validation(
    self,
    grid: base.BaseGrid,
    backend: gtx_backend.Backend,
    connectivities_as_numpy: dict,
    input_data: dict,
):
    if self.MARKERS is not None:
        apply_markers(self.MARKERS, grid, backend)

    connectivities = connectivities_as_numpy
    reference_outputs = self.reference(
        connectivities,
        **{k: v.asnumpy() if isinstance(v, gtx.Field) else v for k, v in input_data.items()},
    )

    input_data = allocate_data(backend, input_data)

    self.PROGRAM.with_backend(backend)(
        **input_data,
        offset_provider=grid.offset_providers,
    )
    for out in self.OUTPUTS:
        name, refslice, gtslice = (
            (out.name, out.refslice, out.gtslice)
            if isinstance(out, Output)
            else (out, (slice(None),), (slice(None),))
        )

        np.testing.assert_allclose(
            input_data[name].asnumpy()[gtslice],
            reference_outputs[name][refslice],
            equal_nan=True,
            err_msg=f"Validation failed for '{name}'",
        )


if pytest_benchmark:

    def _test_execution_benchmark(self, pytestconfig, grid, backend, input_data, benchmark):
        if self.MARKERS is not None:
            apply_markers(self.MARKERS, grid, backend)

        if pytestconfig.getoption(
            "--benchmark-disable"
        ):  # skipping as otherwise program calls are duplicated in tests.
            pytest.skip("Test skipped due to 'benchmark-disable' option.")
        else:
            input_data = allocate_data(backend, input_data)
            benchmark(
                self.PROGRAM.with_backend(backend),
                **input_data,
                offset_provider=grid.offset_providers,
            )

else:

    def _test_execution_benchmark(self, pytestconfig):
        pytest.skip("Test skipped as `pytest-benchmark` is not installed.")


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

    PROGRAM: ClassVar[Program]
    OUTPUTS: ClassVar[tuple[str | Output, ...]]
    MARKERS: typing.Optional[tuple] = None

    def __init_subclass__(cls, **kwargs):
        # Add two methods for verification and benchmarking. In order to have names that
        # reflect the name of the test we do this dynamically here instead of using regular
        # inheritance.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_validation)
        setattr(cls, f"test_{cls.__name__}_benchmark", _test_execution_benchmark)


def reshape(arr: np.ndarray, shape: tuple[int, ...]):
    return np.reshape(arr, shape)


def as_1d_connectivity(connectivity: np.ndarray) -> np.ndarray:
    old_shape = connectivity.shape
    return np.arange(old_shape[0] * old_shape[1], dtype=gtx.int32).reshape(old_shape)
