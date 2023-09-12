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


def pytest_configure(config):
    config.addinivalue_line("markers", "datatest: this test uses binary data")


def pytest_addoption(parser):
    """Add --datatest commandline option for pytest.

    Makes sure the option is set only once even when running tests of several model packages in one session.
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


def pytest_runtest_setup(item):
    for _ in item.iter_markers(name="datatest"):
        if not item.config.getoption("--datatest"):
            pytest.skip("need '--datatest' option to run")
