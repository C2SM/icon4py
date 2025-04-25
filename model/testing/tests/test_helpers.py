# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from unittest import mock

import pytest

from icon4py.model.testing import helpers


@pytest.mark.parametrize("benchmark_enabled", [True, False])
def test_verification_benchmarking_infrastructure(benchmark_enabled):
    test_func = mock.Mock()
    verification_func = mock.Mock()
    benchmark = mock.Mock(enabled=benchmark_enabled)

    helpers.run_verify_and_benchmark(
        test_func,
        verification_func,
        benchmark_fixture=benchmark,
    )

    test_func.assert_called_once()
    verification_func.assert_called_once()
    if benchmark_enabled:
        benchmark.assert_called_once()
    else:
        benchmark.assert_not_called()

    failing_verification_func = mock.Mock(side_effect=AssertionError("Verification failed."))
    with pytest.raises(AssertionError):
        helpers.run_verify_and_benchmark(
            test_func,
            failing_verification_func,
            benchmark_fixture=None,
        )
