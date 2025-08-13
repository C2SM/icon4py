# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
from unittest import mock

import pytest

from icon4py.model.testing import stencil_tests


def required_args_func(req_arg):
    pass


@pytest.mark.parametrize("benchmark_enabled", [True, False])
def test_run_verify_and_benchmark(benchmark_enabled):
    test_func = mock.Mock(side_effect=required_args_func)
    verification_func = mock.Mock()
    benchmark = mock.Mock(enabled=benchmark_enabled)

    stencil_tests.run_verify_and_benchmark(
        functools.partial(test_func, req_arg=mock.Mock()),
        verification_func,
        benchmark_fixture=benchmark,
    )

    test_func.assert_called_once()
    verification_func.assert_called_once()
    if benchmark_enabled:
        benchmark.assert_called_once()
    else:
        benchmark.assert_not_called()


def test_run_verify_and_benchmark_no_fixture():
    test_func = mock.Mock()
    verification_func = mock.Mock()

    stencil_tests.run_verify_and_benchmark(
        test_func,
        verification_func,
        benchmark_fixture=None,  # No benchmark fixture provided
    )

    test_func.assert_called_once()
    verification_func.assert_called_once()
