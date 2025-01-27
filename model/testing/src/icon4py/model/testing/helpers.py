# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
from dataclasses import dataclass, field
from typing import ClassVar

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py._core.definitions import is_scalar_type
from gt4py.next import constructors
from gt4py.next.ffront.decorator import Program
from typing_extensions import Buffer

from icon4py.model.common.utils import data_allocation as data_alloc


try:
    import pytest_benchmark
except ModuleNotFoundError:
    pytest_benchmark = None


@pytest.fixture
def backend(request):
    return request.param


@pytest.fixture
def grid(request):
    return request.param


def is_python(backend) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_embedded(backend) -> bool:
    return backend is None


def is_roundtrip(backend) -> bool:
    return backend.name == "roundtrip" if backend else False


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


@dataclass(frozen=True)
class Output:
    name: str
    refslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))
    gtslice: tuple[slice, ...] = field(default_factory=lambda: (slice(None),))


def _test_validation(self, grid, backend, input_data):
    connectivities = {dim: data_alloc.as_numpy(table) for dim, table in grid.connectivities.items()}
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
