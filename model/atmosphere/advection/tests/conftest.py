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

from icon4py.model.common.test_utils.grid_utils import (  # noqa : F401  # fixtures from test_utils
    grid,
)
from icon4py.model.common.test_utils.helpers import (  # noqa : F401  # fixtures from test_utils
    backend,
)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--backend") == "gtfn_gpu":
        skip_marker = pytest.mark.skip(reason="FIXME: advection tests need to be fixed for GPU. Skipping")
        for item in items:
            item.add_marker(skip_marker)