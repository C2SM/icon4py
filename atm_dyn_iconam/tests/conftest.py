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

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmarks"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarks
        return
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


@pytest.fixture
def benchmark_rounds():
    return 3
