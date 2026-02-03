# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import simple
from icon4py.model.testing import definitions as test_defs
from icon4py.model.testing.fixtures import processor_props

from ...grid import utils as grid_utils


@pytest.mark.parametrize("processor_props", [False], indirect=True)
def test_create_single_node_runtime_without_mpi(processor_props):  # fixture
    decomposition_info = definitions.DecompositionInfo()
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    assert isinstance(exchange, definitions.SingleNodeExchange)


def get_neighbor_tables_for_simple_grid() -> dict[str, data_alloc.NDArray]:
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }
    return neighbor_tables


offsets = [dims.E2C, dims.E2V, dims.C2E, dims.C2E2C, dims.V2C, dims.V2E, dims.C2V, dims.E2C2V]


@pytest.mark.parametrize("dim", grid_utils.main_horizontal_dims())
def test_decomposition_info_single_node_empty_halo(
    dim: gtx.Dimension,
    processor_props: definitions.ProcessProperties,
) -> None:
    if not processor_props.is_single_rank():
        pytest.xfail()

    manager = grid_utils.run_grid_manager(
        test_defs.Grids.MCH_CH_R04B09_DSL, keep_skip_values=True, backend=None
    )

    decomposition_info = manager.decomposition_info
    for level in (
        definitions.DecompositionFlag.FIRST_HALO_LEVEL,
        definitions.DecompositionFlag.SECOND_HALO_LEVEL,
        definitions.DecompositionFlag.THIRD_HALO_LEVEL,
    ):
        assert decomposition_info.get_halo_size(dim, level) == 0
        assert np.count_nonzero(decomposition_info.halo_level_mask(dim, level)) == 0
    assert (
        decomposition_info.get_halo_size(dim, definitions.DecompositionFlag.OWNED)
        == manager.grid.size[dim]
    )


@pytest.mark.parametrize(
    "flag, expected",
    [
        (definitions.DecompositionFlag.OWNED, False),
        (definitions.DecompositionFlag.SECOND_HALO_LEVEL, True),
        (definitions.DecompositionFlag.THIRD_HALO_LEVEL, True),
        (definitions.DecompositionFlag.FIRST_HALO_LEVEL, True),
        (definitions.DecompositionFlag.UNDEFINED, False),
    ],
)
def test_decomposition_info_is_distributed(flag, expected) -> None:
    mesh = simple.simple_grid(allocator=None, num_levels=10)
    decomp = definitions.DecompositionInfo()
    decomp.set_dimension(
        dims.CellDim,
        np.arange(mesh.num_cells),
        np.arange(mesh.num_cells),
        np.ones((mesh.num_cells,)) * flag,
    )
    assert decomp.is_distributed() == expected
