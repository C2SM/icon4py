# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from icon4py.model.common.decomposition.definitions import (
    DecompositionInfo,
    SingleNodeExchange,
    create_exchange,
)
from icon4py.model.testing.fixtures.datatest import (  # import fixtures form test_utils
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
def test_create_single_node_runtime_without_mpi(icon_grid, processor_props):
    decomposition_info = DecompositionInfo(
        klevels=10,
        num_cells=icon_grid.num_cells,
        num_edges=icon_grid.num_edges,
        num_vertices=icon_grid.num_vertices,
    )
    exchange = create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, SingleNodeExchange)
