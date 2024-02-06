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

from icon4py.model.common.decomposition.definitions import SingleNodeRun
from icon4py.model.common.test_utils.datatest_utils import (
    SERIALIZED_DATA_PATH,
    create_icon_serial_data_provider,
    get_datapath_for_experiment,
    get_processor_properties_for_run,
    get_ranked_data_path,
)


def get_icon_grid(on_gpu: bool):
    processor_properties = get_processor_properties_for_run(SingleNodeRun())
    ranked_path = get_ranked_data_path(SERIALIZED_DATA_PATH, processor_properties)
    data_path = get_datapath_for_experiment(ranked_path)
    icon_data_provider = create_icon_serial_data_provider(data_path, processor_properties)
    grid_savepoint = icon_data_provider.from_savepoint_grid()
    return grid_savepoint.construct_icon_grid(on_gpu)


@pytest.fixture
def grid(request):
    return request.param
