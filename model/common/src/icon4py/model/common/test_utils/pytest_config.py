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
from gt4py.next.program_processors.runners.gtfn_cpu import run_gtfn
from gt4py.next.program_processors.runners.roundtrip import executor


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")


def pytest_addoption(parser):
    """Add custom commandline options for pytest.

    This function adds two options:
        1. --datatest: A flag indicating if tests that use serialized data are being run.
           Such tests might be slower since data might be downloaded from online storage.
        2. --backends: An option to specify a comma-separated list of backends, such as cpu or gpu.

        Makes sure the options are set only once even when running tests of several model packages in one session.
    """
    try:
        parser.addoption(
            "--datatest",
            action="store_true",
            help="running tests that use serialized data, can be slow since data might be downloaded from online storage",
            default=False,
        )
    except ValueError:
        pass

    try:
        parser.addoption(
            "--backend",
            action="store",
            default=None,
            help="Comma separated list of backends: cpu, gpu",
        )
    except ValueError:
        pass


def pytest_runtest_setup(item):
    for _ in item.iter_markers(name="datatest"):
        if not item.config.getoption("--datatest"):
            pytest.skip("need '--datatest' option to run")


def pytest_generate_tests(metafunc):
    # parametrise backends
    if "backend" in metafunc.fixturenames:
        backend_option = metafunc.config.getoption("backend")

        params = [executor]  # default
        if backend_option == "cpu":
            params.append(run_gtfn)
        elif backend_option == "gpu":
            raise NotImplementedError("GPU support is not implemented in gt4py yet.")

        metafunc.parametrize("backend", params)
