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
import importlib
import sys
from pathlib import Path


def import_testutils():
    testutils = (
        Path(__file__).parent.__str__()
        + "/../../atm_dyn_iconam/tests/test_utils/__init__.py"
    )
    spec = importlib.util.spec_from_file_location("helpers", testutils)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helpers = import_testutils()

from helpers.fixtures import (  # noqa F401
    data_provider,
    get_grid_files,
    grid_savepoint,
    r04b09_dsl_gridfile,
    setup_icon_data,
)
