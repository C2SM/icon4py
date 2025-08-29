# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib

import numpy as np
import pytest
from gt4py import next as gtx

import icon4py.model.common.grid.gridfile
import icon4py.model.testing.grid_utils as grid_utils
from icon4py.model.common import exceptions, dimension as dims
from icon4py.model.common.decomposition import halo, mpi_decomposition, definitions as defs
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid, gridfile
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    test_utils as test_helpers,
    definitions as test_defs,
)

from .. import utils
from ..fixtures import *
from ...decomposition import utils as decomp_utils


try:
    import mpi4py  # noqa F401:  import mpi4py to check for optional mpi dependency

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

log = logging.getLogger(__file__)
vertical_config = v_grid.VerticalGridConfig(num_levels=1)


def run_gridmananger_for_multinode(
    file: pathlib.Path,
    vertical_config: v_grid.VerticalGridConfig,
    run_properties: defs.ProcessProperties,
    decomposer: halo.Decomposer,
) -> gm.GridManager:
    manager = _grid_manager(file, vertical_config)
    manager(
        keep_skip_values=True, backend=None, run_properties=run_properties, decomposer=decomposer
    )
    return manager


def _grid_manager(file: pathlib.Path, vertical_config: v_grid.VerticalGridConfig):
    manager = gm.GridManager(str(file), vertical_config)
    return manager


def run_grid_manager_for_singlenode(
    file: pathlib.Path, vertical_config: v_grid.VerticalGridConfig
) -> gm.GridManager:
    manager = _grid_manager(file, vertical_config)
    manager(
        keep_skip_values=True,
        run_properties=defs.SingleNodeProcessProperties(),
        decomposer=halo.SingleNodeDecomposer(),
        backend=None,
    )
    return manager


@pytest.mark.xfail
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        # (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT)
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dims())
def test_start_end_index(
    caplog, backend, processor_props, grid_file, experiment, dim, icon_grid
):  # fixture
    caplog.set_level(logging.INFO)
    file = grid_utils.resolve_full_grid_file_name(grid_file)

    partitioner = halo.SimpleMetisDecomposer()
    manager = gm.GridManager(
        file,
        v_grid.VerticalGridConfig(1),
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
    )
    single_node_grid = gm.GridManager(
        file,
        v_grid.VerticalGridConfig(1),
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
    ).grid

    for domain in utils.global_grid_domains(dim):
        assert grid.start_index(domain) == single_node_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == single_node_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(processor_props):
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL.name)
    manager = gm.GridManager(file, vertical_config, gridfile.ToZeroBasedIndexTransformation())
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            backend=None,
            run_properties=processor_props,
            decomposer=halo.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_distributed_fields(processor_props, caplog):
    caplog.set_level(logging.INFO)
    print(f"myrank - {processor_props.rank}: running with processor_props =  {processor_props}")
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL.name)
    grid_manager = run_grid_manager_for_singlenode(file, vertical_config)
    single_node_grid = grid_manager.grid
    global_cell_area = grid_manager.geometry[gridfile.GeometryName.CELL_AREA]
    global_edge_lat = grid_manager.coordinates[dims.EdgeDim]["lat"]
    global_vertex_lon = grid_manager.coordinates[dims.VertexDim]["lon"]

    multinode = run_gridmananger_for_multinode(
        file,
        vertical_config,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    decomposition_info = multinode.decomposition_info

    local_cell_area = multinode.geometry[gridfile.GeometryName.CELL_AREA]
    local_edge_lat = multinode.coordinates[dims.EdgeDim]["lat"]
    local_vertex_lon = multinode.coordinates[dims.VertexDim]["lon"]
    print(
        f"rank = {processor_props.rank} has size(cell_area): {local_cell_area.ndarray.shape}, "
        f"has size(edge_length): {local_edge_lat.ndarray.shape}, has size(vertex_length): {local_vertex_lon.ndarray.shape}"
    )
    # the local number of cells must be at most the global number of cells (analytically computed)
    assert (
        local_cell_area.asnumpy().shape[0] <= single_node_grid.global_properties.num_cells
    ), "local field is larger than global field"
    # global read: read the same (global fields)

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.CellDim,
        global_cell_area.asnumpy(),
        local_cell_area.asnumpy(),
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_edge_lat.asnumpy(),
        local_edge_lat.asnumpy(),
    )
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.VertexDim,
        global_vertex_lon.asnumpy(),
        local_vertex_lon.asnumpy(),
    )


def gather_field(field: np.ndarray, comm: mpi4py.MPI.Comm) -> tuple:
    local_sizes = np.array(comm.gather(field.size, root=0))
    if comm.rank == 0:
        recv_buffer = np.empty(sum(local_sizes), dtype=field.dtype)
        log.debug(
            f"rank: {comm.rank} - setup receive buffer with size {sum(local_sizes)} on rank 0"
        )
    else:
        recv_buffer = None
    comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    if comm.rank == 0:
        log.debug(f"fields gathered:")
        log.debug(f"field sizes {local_sizes}")

    return local_sizes, recv_buffer


def assert_gathered_field_against_global(
    decomposition_info: defs.DecompositionInfo,
    processor_props: defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
):
    assert (
        local_field.shape
        == decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.ALL).shape
    )
    owned_entries = local_field[
        decomposition_info.local_index(dim, defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props.comm)
    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.OWNED),
        processor_props.comm,
    )
    if processor_props.rank == 0:
        print(f"rank = {processor_props.rank}: asserting gathered fields")
        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match  {gathered_sizes}"
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)
        sorted_[gathered_global_indices] = gathered_field
        assert test_helpers.dallclose(
            sorted_, global_reference_field
        ), f"Gathered field values do not match for dim {dim}.- "
