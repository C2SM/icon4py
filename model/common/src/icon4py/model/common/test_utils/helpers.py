# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import numpy.typing as npt
import pytest
from gt4py._core.definitions import is_scalar_type
from gt4py.next import as_field
from gt4py.next import common as gt_common
from gt4py.next import constructors
from gt4py.next.ffront.decorator import Program
from gt4py.next.program_processors.otf_compile_executor import OTFCompileExecutor

from ..grid.base import BaseGrid
from ..grid.icon import IconGrid


try:
    import pytest_benchmark
except ModuleNotFoundError:
    pytest_benchmark = None

from ..grid.simple import SimpleGrid


@pytest.fixture
def backend(request):
    return request.param


def _shape(
    grid,
    *dims: gt_common.Dimension,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
):
    if extend is None:
        extend = {}
    for d in dims:
        if d not in extend.keys():
            extend[d] = 0
    return tuple(grid.size[dim] + extend[dim] for dim in dims)


def random_mask(
    grid: SimpleGrid,
    *dims: gt_common.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
) -> gt_common.Field:
    shape = _shape(grid, *dims, extend=extend)
    arr = np.full(shape, False).flatten()
    arr[: int(arr.size * 0.5)] = True
    np.random.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if dtype:
        arr = arr.astype(dtype)
    return as_field(dims, arr)


def random_field(
    grid,
    *dims,
    low: float = -1.0,
    high: float = 1.0,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
    dtype: Optional[npt.DTypeLike] = None,
) -> gt_common.Field:
    arr = np.random.default_rng().uniform(
        low=low, high=high, size=_shape(grid, *dims, extend=extend)
    )
    if dtype:
        arr = arr.astype(dtype)
    return as_field(dims, arr)


def zero_field(
    grid: BaseGrid,
    *dims: gt_common.Dimension,
    dtype=float,
    extend: Optional[dict[gt_common.Dimension, int]] = None,
) -> gt_common.Field:
    return as_field(dims, np.zeros(shape=_shape(grid, *dims, extend=extend), dtype=dtype))


def constant_field(
    grid: SimpleGrid, value: float, *dims: gt_common.Dimension, dtype=float
) -> gt_common.Field:
    return as_field(
        dims, value * np.ones(shape=tuple(map(lambda x: grid.size[x], dims)), dtype=dtype)
    )


def as_1D_sparse_field(field: gt_common.Field, target_dim: gt_common.Dimension) -> gt_common.Field:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    buffer = field.asnumpy()
    return numpy_to_1D_sparse_field(buffer, target_dim)


def numpy_to_1D_sparse_field(field: np.ndarray, dim: gt_common.Dimension) -> gt_common.Field:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    old_shape = field.shape
    assert len(old_shape) == 2
    new_shape = (old_shape[0] * old_shape[1],)
    return as_field((dim,), field.reshape(new_shape))


def flatten_first_two_dims(*dims: gt_common.Dimension, field: gt_common.Field) -> gt_common.Field:
    """Convert a n-D sparse field to a (n-1)-D flattened (Felix-style) sparse field."""
    buffer = field.asnumpy()
    old_shape = buffer.shape
    assert len(old_shape) >= 2
    flattened_size = old_shape[0] * old_shape[1]
    flattened_shape = (flattened_size,)
    new_shape = flattened_shape + old_shape[2:]
    newarray = buffer.reshape(new_shape)
    return as_field(dims, newarray)


def unflatten_first_two_dims(field: gt_common.Field) -> np.array:
    """Convert a (n-1)-D flattened (Felix-style) sparse field back to a n-D sparse field."""
    old_shape = np.asarray(field).shape
    new_shape = (old_shape[0] // 3, 3) + old_shape[1:]
    return np.asarray(field).reshape(new_shape)


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
    reference_outputs = self.reference(
        grid,
        **{
            k: v.asnumpy() if isinstance(v, gt_common.Field) else np.array(v)
            for k, v in input_data.items()
        },
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

        assert np.allclose(
            input_data[name].asnumpy()[gtslice], reference_outputs[name][refslice], equal_nan=True
        ), f"Validation failed for '{name}'"


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

        >>> class TestMultiplyByTwo(StencilTest): # doctest: +SKIP
        ...    PROGRAM = multiply_by_two  # noqa: F821
        ...    OUTPUTS = ("some_output",)
        ...
        ...    @pytest.fixture
        ...    def input_data(self):
        ...        return {"some_input": ..., "some_output": ...}
        ...
        ...    @staticmethod
        ...    def reference(some_input, **kwargs):
        ...        return dict(some_output=np.asarray(some_input)*2)
    """

    PROGRAM: ClassVar[Program]
    OUTPUTS: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs):
        # Add two methods for verification and benchmarking. In order to have names that
        # reflect the name of the test we do this dynamically here instead of using regular
        # inheritance.
        super().__init_subclass__(**kwargs)
        setattr(cls, f"test_{cls.__name__}", _test_validation)
        setattr(cls, f"test_{cls.__name__}_benchmark", _test_execution_benchmark)


@pytest.fixture
def uses_icon_grid_with_otf(backend, grid):
    """Check whether we are using a compiled backend with the icon_grid.

    Is needed to skip certain stencils where the execution domain needs to be restricted or boundary taken into account.
    """
    if hasattr(backend, "executor") and isinstance(grid, IconGrid):
        if isinstance(backend.executor, OTFCompileExecutor):
            return True
        try:
            from gt4py.next.program_processors.runners import dace_iterator

            if backend in {dace_iterator.run_dace_cpu, dace_iterator.run_dace_gpu}:
                return True
        except ImportError:
            pass
    return False


def reshape(arr: np.array, shape: tuple[int, ...]):
    return np.reshape(arr, shape)
