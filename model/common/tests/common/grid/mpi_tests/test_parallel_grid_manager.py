# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import operator
import pathlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from gt4py import next as gtx
from gt4py.next import typing as gtx_typing

import icon4py.model.common.grid.gridfile
from icon4py.model.common import dimension as dims, exceptions
from icon4py.model.common.decomposition import definitions as defs, halo, mpi_decomposition
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes,
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation.interpolation_fields import compute_geofac_div
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions,
    definitions as test_defs,
    grid_utils,
    test_utils as test_helpers,
)

from ...decomposition import utils as decomp_utils
from .. import utils
from ..fixtures import backend, experiment, grid_savepoint, icon_grid, processor_props, data_provider, download_ser_data, ranked_data_path


try:
    import mpi4py

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
        keep_skip_values=True, allocator=None, run_properties=run_properties, decomposer=decomposer
    )
    return manager


def _grid_manager(file: pathlib.Path, vertical_config: v_grid.VerticalGridConfig) -> gm.GridManager:
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
        allocator=None,
    )
    return manager


#@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment",
    [
        (test_defs.Experiments.EXCLAIM_APE),
        # (test_defs.Experiments.MCH_CH_R04B09)
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dims())
def test_start_end_index(
    caplog: Any,
    backend: gtx_typing.Backend | None,
    processor_props: defs.ProcessProperties,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    icon_grid: base.Grid,
) -> None:
    #decomp_utils.dummy_four_ranks(3)
    caplog.set_level(logging.INFO)
    grid_file = experiment.grid
    file = grid_utils.resolve_full_grid_file_name(grid_file)

    partitioner = halo.SimpleMetisDecomposer()
    manager = gm.GridManager(
        file,
        v_grid.VerticalGridConfig(1),
        icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation(),
    )
    manager(backend, keep_skip_values=True, decomposer=partitioner, run_properties=processor_props)
    grid = manager.grid

    domains = (h_grid.domain(dim)(z) for z in h_grid.VERTEX_AND_CELL_ZONES)
    for domain in domains:
        assert icon_grid.start_index(domain) == grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert icon_grid.end_index(domain) == grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(processor_props: defs.ProcessProperties) -> None:
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    manager = gm.GridManager(file, vertical_config, gridfile.ToZeroBasedIndexTransformation())
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            allocator=None,
            run_properties=processor_props,
            decomposer=halo.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_fields_distribute_and_gather(processor_props: defs.ProcessProperties, caplog: Any) -> None:
    caplog.set_level(logging.INFO)
    print(f"myrank - {processor_props.rank}: running with processor_props =  {processor_props}")
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    single_node = run_grid_manager_for_singlenode(file, vertical_config)
    single_node_grid = single_node.grid
    global_cell_area = single_node.geometry_fields[gridfile.GeometryName.CELL_AREA]
    global_edge_lat = single_node.coordinates[dims.EdgeDim]["lat"]
    global_vertex_lon = single_node.coordinates[dims.VertexDim]["lon"]

    multinode = run_gridmananger_for_multinode(
        file=file,
        vertical_config=vertical_config,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    decomposition_info = multinode.decomposition_info

    local_cell_area = multinode.geometry_fields[gridfile.GeometryName.CELL_AREA]
    local_edge_lat = multinode.coordinates[dims.EdgeDim]["lat"]
    local_vertex_lon = multinode.coordinates[dims.VertexDim]["lon"]
    print(
        f"rank = {processor_props.rank} has size(cell_area): {local_cell_area.ndarray.shape}, "
        f"has size(edge_length): {local_edge_lat.ndarray.shape}, has size(vertex_length): {local_vertex_lon.ndarray.shape}"
    )
    global_num_cells = single_node_grid.config.num_cells

    # the local number of cells must be at most the global number of cells (analytically computed)
    assert (
        local_cell_area.asnumpy().shape[0] <= global_num_cells
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
    constant_dims = field.shape[1:]
    constant_length = functools.reduce(operator.mul, constant_dims) if len(constant_dims) > 0 else 1

    local_sizes = np.array(comm.gather(field.size, root=0))

    if comm.rank == 0:
        recv_buffer = np.empty(np.sum(local_sizes), dtype=field.dtype)
        log.debug(
            f"rank:{comm} - {comm.rank} - setup receive buffer with size {sum(local_sizes)} on rank 0"
        )
    else:
        recv_buffer = None

    comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    if comm.rank == 0:
        log.debug("fields gathered:")
        log.debug(f"field sizes {local_sizes}")
        local_first_dim = tuple(size / constant_length for size in local_sizes)
        gathered_field = recv_buffer.reshape((-1, *constant_dims))  # type: ignore [union-attr]
    else:
        gathered_field = None
        local_first_dim = field.shape
    return local_first_dim, gathered_field


def assert_gathered_field_against_global(
    decomposition_info: defs.DecompositionInfo,
    processor_props: defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
) -> None:
    assert (
        local_field.shape[0]
        == decomposition_info.global_index(dim, defs.DecompositionInfo.EntryType.ALL).shape[0]
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
        print(f"rank = {processor_props.rank}: asserting gathered fields: ")
        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match  {gathered_sizes}"
        print(
            f"rank = {processor_props.rank}: Checking field size: --- gathered sizes {gathered_sizes}"
        )
        print(
            f"rank = {processor_props.rank}:                      --- gathered field has size {gathered_sizes}"
        )
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)  # type: ignore [attr-defined]
        sorted_[gathered_global_indices] = gathered_field
        assert test_helpers.dallclose(
            sorted_, global_reference_field
        ), f"Gathered field values do not match for dim {dim}.- "
        print(
            f"rank = {processor_props.rank}:  comparing fields (samples) "
            f"\n      -- gathered {sorted_[:6]}  "
            f"\n    -- global ref {global_reference_field[:6]}"
        )


# TODO add test including halo access:
#  Will uses
#    - geofac_div
#    - geofac_n2s


# TODO (halungge): fix non contiguous dimension for embedded in gt4py
@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_halo_neighbor_access_c2e(
    processor_props: defs.ProcessProperties, backend: gtx_typing.Backend | None
) -> None:
    #    processor_props = decomp_utils.DummyProps(1)
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    print(f"running on {processor_props.comm}")
    single_node = run_grid_manager_for_singlenode(file, vertical_config)
    single_node_grid = single_node.grid
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid,
        coordinates=single_node.coordinates,
        decomposition_info=single_node.decomposition_info,
        extra_fields=single_node.geometry_fields,
        metadata=geometry_attributes.attrs,
    )

    print(
        f"rank = {processor_props.rank} : single node grid has size {single_node.decomposition_info.get_horizontal_size()!r}"
    )
    reference = data_alloc.zero_field(single_node_grid, dims.CellDim, dims.C2EDim)
    single_node_edge_length = single_node_geometry.get(geometry_attributes.EDGE_LENGTH)
    single_node_cell_area = single_node_geometry.get(geometry_attributes.CELL_AREA)
    single_node_edge_orientation = single_node_geometry.get(
        geometry_attributes.CELL_NORMAL_ORIENTATION
    )
    compute_geofac_div.with_backend(None)(
        primal_edge_length=single_node_edge_length,
        area=single_node_cell_area,
        edge_orientation=single_node_edge_orientation,
        out=reference,
        offset_provider={"C2E": single_node_grid.get_connectivity("C2E")},
    )
    print(
        f"rank = {processor_props.rank} : single node computed field reference has size  {reference.asnumpy().shape}"
    )
    multinode_grid_manager = run_gridmananger_for_multinode(
        file=file,
        vertical_config=vertical_config,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' (1 : {decomposition_info.get_halo_size(dims.CellDim, defs.DecompositionFlag.FIRST_HALO_LINE)}), (2: {decomposition_info.get_halo_size(dims.CellDim, defs.DecompositionFlag.SECOND_HALO_LINE)})"
    )
    distributed_coordinates = multinode_grid_manager.coordinates
    distributed_geometry = geometry.GridGeometry(
        backend=backend,
        grid=distributed_grid,
        coordinates=distributed_coordinates,
        decomposition_info=decomposition_info,
        extra_fields=extra_geometry_fields,
        metadata=geometry_attributes.attrs,
    )

    edge_length = distributed_geometry.get(geometry_attributes.EDGE_LENGTH)
    cell_area = distributed_geometry.get(geometry_attributes.CELL_AREA)
    edge_orientation = distributed_geometry.get(geometry_attributes.CELL_NORMAL_ORIENTATION)

    geofac_div = data_alloc.zero_field(distributed_grid, dims.CellDim, dims.C2EDim)
    compute_geofac_div.with_backend(None)(
        primal_edge_length=edge_length,
        area=cell_area,
        edge_orientation=edge_orientation,
        out=geofac_div,
        offset_provider={"C2E": distributed_grid.get_connectivity("C2E")},
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.CellDim,
        global_reference_field=reference.asnumpy(),
        local_field=geofac_div.asnumpy(),
    )

    print(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize(
    "field_offset",
    [dims.C2V, dims.E2V, dims.V2C, dims.E2C, dims.C2E, dims.V2E, dims.C2E2C, dims.V2E2V],
)
def test_local_connectivities(
    processor_props: defs.ProcessProperties,
    caplog: Iterator,
    field_offset: gtx.FieldOffset,
) -> None:
    caplog.set_level(logging.INFO)  # type: ignore [attr-defined]
    grid = utils.run_grid_manager(
        test_defs.Grids.R02B04_GLOBAL, keep_skip_values=True, backend=None
    ).grid
    partitioner = halo.SimpleMetisDecomposer()
    face_face_connectivity = grid.get_connectivity(dims.C2E2C).ndarray
    neighbor_tables = grid.get_neighbor_tables()
    labels = partitioner(face_face_connectivity, num_partitions=processor_props.comm_size)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=neighbor_tables,
        run_properties=processor_props,
        num_levels=1,
    )

    decomposition_info = halo_generator(labels)

    connectivity = gm.construct_local_connectivity(
        field_offset, decomposition_info, connectivity=grid.get_connectivity(field_offset).ndarray
    )
    # there is an neighbor list for each index of the target dimension on the node
    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], defs.DecompositionInfo.EntryType.ALL
        ).size
    )
    # all neighbor indices are valid local indices
    assert np.max(connectivity) == np.max(
        decomposition_info.local_index(field_offset.source, defs.DecompositionInfo.EntryType.ALL)
    )
    # TODO what else to assert?
    # - outer halo entries have SKIP_VALUE neighbors (depends on offsets)
