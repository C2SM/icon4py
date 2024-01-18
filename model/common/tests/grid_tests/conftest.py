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

from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401
    damping_height,
    data_provider,
    datapath,
    decomposition_info,
    download_ser_data,
    experiment,
    grid_savepoint,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)

from .utils import MCH_GRID_FILE


@pytest.fixture
def grid_file():
    return MCH_GRID_FILE
