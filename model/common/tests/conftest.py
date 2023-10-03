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

"""
Initialize pytest.

Workaround for pytest not discovering those configuration function, when they are added to the
diffusion_test/conftest.py folder
"""
from icon4py.model.common.test_utils.helpers import backend, mesh  # noqa: F401 # fixtures
from icon4py.model.common.test_utils.pytest_config import (  # noqa: F401 # pytest config
    pytest_addoption,
    pytest_configure,
    pytest_generate_tests,
    pytest_runtest_setup,
)
