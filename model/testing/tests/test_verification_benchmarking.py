# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from icon4py.model.testing import helpers


BASE_DTYPE = np.int64


def incr_func(
    field: NDArray[BASE_DTYPE],
    increment: int,
):
    field += increment


def verify_field(
    field: NDArray[BASE_DTYPE],
    increment: int,
    base_value: int,
):
    np.testing.assert_allclose(field, base_value + increment)


def test_verification_benchmarking_infrastructure(benchmark: Optional[pytest.FixtureRequest]):
    base_value = 1
    field = np.array((base_value * np.ones((), dtype=BASE_DTYPE)))

    increment = 6

    helpers.run_verify_and_benchmark(
        functools.partial(incr_func, field=field, increment=increment),
        functools.partial(verify_field, field=field, increment=increment, base_value=base_value),
        benchmark_fixture=None,  # no need to benchmark this test
    )

    current_base_value = field[()]
    assert (
        current_base_value != base_value
    ), "Base values should not be equal. Otherwise, the test did not go through incr_func/ verify_field functions."

    # Expect AssertionError
    with pytest.raises(AssertionError):
        helpers.run_verify_and_benchmark(
            functools.partial(incr_func, field=field, increment=increment),
            functools.partial(
                verify_field, field=field, increment=increment, base_value=base_value
            ),  # base_value should be current_base_value
            benchmark_fixture=None,  # no need to benchmark this test
        )
