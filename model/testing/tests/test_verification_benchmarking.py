# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import numpy as np
from numpy.typing import NDArray

from icon4py.model.testing import helpers

base_type = np.int64


def incr_func(
    field: NDArray[base_type],
    increment: int,
):
    field += increment


def verify_field(
    field: NDArray[base_type],
    increment: int,
    base_value: int,
):
    all_correct = np.all(field == (base_value + increment))
    
    assert type(all_correct) is np.bool_
    assert all_correct, f"Field verification failed"


def test_verification_benchmarking_infrastructure(benchmark):
    base_value = 1
    field = base_value*np.ones((1000, 1000), dtype=base_type)
    
    increment = 6
    
    helpers.run_verify_and_benchmark(
        functools.partial(incr_func, field=field, increment=increment),
        functools.partial(verify_field, field=field, increment=increment, base_value=base_value),
        benchmark,
    )

    current_base_value = np.random.choice(field.flat)
    assert current_base_value != base_value, f"Base values should not be equal. Otherwise, the test did not go through incr_func and/or verify_field functions."
    
    incr_func(field, increment)
    verify_field(field, increment, current_base_value)
