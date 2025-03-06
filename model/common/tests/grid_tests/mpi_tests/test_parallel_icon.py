# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import re

import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.testing.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)

from .. import utils


try:
    import mpi4py  # F401:  import mpi4py to check for optional mpi dependency
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_props(processor_props):  # noqa: F811  # fixture
    """dummy test to check whether the MPI initialization and GHEX setup works."""
    import ghex.context as ghex

    assert processor_props.comm

    assert isinstance(
        processor_props.comm, mpi4py.MPI.Comm
    ), "comm needs to be an instance of MPI.Comm"
    ghex.make_context(processor_props.comm)


LOCAL_IDX_2 = {
    dims.CellDim: {0: 10454, 1: 10454},
    dims.EdgeDim: {0: 15830, 1: 15754},
    dims.VertexDim: {0: 5375, 1: 5296},
}

LOCAL_IDX_4 = {
    dims.CellDim: {0: 5238, 1: 5222, 2: 5231, 3: 5230},
    dims.EdgeDim: {0: 7929, 1: 7838, 2: 7955, 3: 7889},
    dims.VertexDim: {0: 2688, 1: 2612, 2: 2723, 3: 2656},
}

LOCAL_IDX = {4: LOCAL_IDX_4, 2: LOCAL_IDX_2}


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_distributed_local(processor_props, dim, icon_grid, caplog):  # noqa: F811  # fixture
    caplog.set_level(logging.INFO)
    check_comm_size(processor_props)
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    # local still runs entire field:
    assert icon_grid.start_index(domain) == 0
    print(
        f"rank {processor_props.rank}/{processor_props.comm_size} dim = {dim}  LOCAL : ({icon_grid.start_index(domain)}, {icon_grid.end_index(domain)})"
    )
    assert (
        icon_grid.end_index(domain)
        == LOCAL_IDX[processor_props.comm_size][dim][processor_props.rank]
    )


HALO_IDX_4 = {
    dims.CellDim: {
        0: (5238, 5340, 5446),
        1: (5222, 5325, 5433),
        2: (5231, 5334, 5442),
        3: (5230, 5334, 5444),
    },
    dims.EdgeDim: {
        0: (7929, 7966, 8173),
        1: (7838, 7940, 8150),
        2: (7955, 7955, 8165),
        3: (7889, 7954, 8167),
    },
    dims.VertexDim: {
        0: (2688, 2727, 2834),
        1: (2612, 2717, 2825),
        2: (2723, 2723, 2832),
        3: (2656, 2723, 2832),
    },
}
HALO_IDX_2 = {
    dims.CellDim: {
        0: (10454, 10531, 10611),
        1: (10454, 10531, 10612),
    },
    dims.EdgeDim: {
        0: (15830, 15830, 15986),
        1: (15754, 15830, 15987),
    },
    dims.VertexDim: {
        0: (5375, 5375, 5455),
        1: (5296, 5375, 5456),
    },
}

HALO_IDX = {4: HALO_IDX_4, 2: HALO_IDX_2}


@pytest.mark.datatest
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi
@pytest.mark.parametrize("dim", utils.horizontal_dim())
@pytest.mark.parametrize("marker", [h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2])
def test_distributed_halo(processor_props, dim, marker, icon_grid):  # noqa: F811  # fixture
    check_comm_size(processor_props)
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    domain = h_grid.domain(dim)(marker)
    start_index = icon_grid.start_index(domain)
    end_index = icon_grid.end_index(domain)
    rank = processor_props.rank
    print(
        f"rank {rank}/{processor_props.comm_size} dim = {dim}  {marker} : ({start_index}, {end_index})"
    )

    assert start_index == HALO_IDX[processor_props.comm_size][dim][rank][num - 1]
    assert end_index == HALO_IDX[processor_props.comm_size][dim][rank][num]
