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

from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    SingleNodeExchange,
    create_exchange,
)
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401 # import fixtures form test_utils
    processor_props,
)


def test_create_single_node_runtime_without_mpi(processor_props):  # noqa: F811 # fixture
    decomposition_info = DecompositionInfo(klevels=10)
    exchange = create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, SingleNodeExchange)
