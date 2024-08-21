# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    SingleNodeExchange,
    create_exchange,
)
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401 # import fixtures form test_utils
    experiment,
    processor_props,
)


def test_create_single_node_runtime_without_mpi(processor_props):  # noqa: F811 # fixture
    decomposition_info = DecompositionInfo(klevels=10)
    exchange = create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, SingleNodeExchange)
