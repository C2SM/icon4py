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
from typing import Any

import numpy as np
import pytest
from gt4py import next as gtx
from gt4py.next import common as gtx_common, typing as gtx_typing

from icon4py.model.common import dimension as dims, exceptions
from icon4py.model.common.decomposition import definitions as decomp_defs, halo, mpi_decomposition
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes,
    geometry_stencils,
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.interpolation import interpolation_fields
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils

from ..fixtures import backend, processor_props
from . import utils


try:
    import mpi4py

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

log = logging.getLogger(__file__)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(processor_props: decomp_defs.ProcessProperties) -> None:
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    manager = gm.GridManager(file, utils.NUM_LEVELS, gridfile.ToZeroBasedIndexTransformation())
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            allocator=None,
            run_properties=processor_props,
            decomposer=halo.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


def _get_neighbor_tables(grid: base.Grid) -> dict:
    return {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_fields_distribute_and_gather(
    processor_props: decomp_defs.ProcessProperties, caplog: Any
) -> None:
    caplog.set_level(logging.INFO)
    print(f"myrank - {processor_props.rank}: running with processor_props =  {processor_props}")
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    single_node = utils.run_grid_manager_for_singlenode(file)
    single_node_grid = single_node.grid
    global_cell_area = single_node.geometry_fields[gridfile.GeometryName.CELL_AREA]
    global_edge_lat = single_node.coordinates[dims.EdgeDim]["lat"]
    global_vertex_lon = single_node.coordinates[dims.VertexDim]["lon"]

    multinode = utils.run_gridmananger_for_multinode(
        file=file,
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
    decomposition_info: decomp_defs.DecompositionInfo,
    processor_props: decomp_defs.ProcessProperties,  # F811 # fixture
    dim: gtx.Dimension,
    global_reference_field: np.ndarray,
    local_field: np.ndarray,
) -> None:
    print(
        f" rank= {processor_props.rank}/{processor_props.comm_size}----exchanging field of main dim {dim}"
    )
    assert (
        local_field.shape[0]
        == decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.ALL).shape[
            0
        ]
    )
    owned_entries = local_field[
        decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props.comm)
    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED),
        processor_props.comm,
    )
    rtol = 1e-12
    if processor_props.rank == 0:
        print(f"rank = {processor_props.rank}: asserting gathered fields: ")
        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match  {gathered_sizes}"
        print(
            f"rank = {processor_props.rank}: Checking field size: --- gathered sizes {gathered_sizes} = {sum(gathered_sizes)}"
        )
        print(
            f"rank = {processor_props.rank}:                      --- gathered field has size {gathered_sizes}"
        )
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)  # type: ignore [attr-defined]
        sorted_[gathered_global_indices] = gathered_field
        print(
            f" rank = {processor_props.rank}: SHAPES: global reference field {global_reference_field.shape}, gathered = {gathered_field.shape}"
        )

        mismatch = np.where(
            np.abs(sorted_ - global_reference_field) / np.abs(global_reference_field) > rtol
        )
        print(f"rank = {processor_props.rank}: mismatch found in {mismatch}")

        np.testing.assert_allclose(sorted_, global_reference_field, rtol=rtol, verbose=True)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_neighbor_access_c2e(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    print(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_node = utils.run_grid_manager_for_singlenode(file)
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
    # has to be computed in gt4py-embedded
    interpolation_fields.compute_geofac_div.with_backend(None)(
        primal_edge_length=single_node_edge_length,
        area=single_node_cell_area,
        edge_orientation=single_node_edge_orientation,
        out=reference,
        offset_provider={"C2E": single_node_grid.get_connectivity("C2E")},
    )
    print(
        f"rank = {processor_props.rank} : single node computed field reference has size  {reference.asnumpy().shape}"
    )
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
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
    # has to be computed in gt4py-embedded
    interpolation_fields.compute_geofac_div.with_backend(None)(
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
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_access_e2c2v(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    print(f"running on {processor_props.comm}")
    single_node = utils.run_grid_manager_for_singlenode(file)
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
    reference_u = single_node_geometry.get(geometry_attributes.EDGE_NORMAL_VERTEX_U).asnumpy()
    reference_v = single_node_geometry.get(geometry_attributes.EDGE_NORMAL_VERTEX_V).asnumpy()
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    decomposition_info = multinode_grid_manager.decomposition_info
    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'EdgeDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
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
    vertex_lat = distributed_geometry.get(geometry_attributes.VERTEX_LAT)
    vertex_lon = distributed_geometry.get(geometry_attributes.VERTEX_LON)
    x = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_X)
    y = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_Y)
    z = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_Z)

    u0 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v0 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    u1 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v1 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    u2 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v2 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    u3 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v3 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)

    geometry_stencils.zonal_and_meridional_component_of_edge_field_at_vertex.with_backend(backend)(
        vertex_lat,
        vertex_lon,
        x,
        y,
        z,
        out=(u0, v0, u1, v1, u2, v2, u3, v3),
        offset_provider={"E2C2V": distributed_grid.get_connectivity(dims.E2C2V)},
    )
    u_component = np.vstack((u0.asnumpy(), u1.asnumpy(), u2.asnumpy(), u3.asnumpy())).T
    v_component = np.vstack((v0.asnumpy(), v1.asnumpy(), v2.asnumpy(), v3.asnumpy())).T

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_u,
        local_field=u_component,
    )
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_v,
        local_field=v_component,
    )


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_access_e2c(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    print(f"running on {processor_props.comm}")
    single_node = utils.run_grid_manager_for_singlenode(file)
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
    reference_u = single_node_geometry.get(geometry_attributes.EDGE_NORMAL_CELL_U).asnumpy()
    reference_v = single_node_geometry.get(geometry_attributes.EDGE_NORMAL_CELL_V).asnumpy()
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'EdgeDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
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
    cell_lat = distributed_geometry.get(geometry_attributes.CELL_LAT)
    cell_lon = distributed_geometry.get(geometry_attributes.CELL_LON)
    x = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_X)
    y = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_Y)
    z = distributed_geometry.get(geometry_attributes.EDGE_NORMAL_Z)

    u0 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v0 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    u1 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)
    v1 = data_alloc.random_field(distributed_grid, dims.EdgeDim, allocator=backend)

    geometry_stencils.zonal_and_meridional_component_of_edge_field_at_cell_center.with_backend(
        backend
    )(
        cell_lat,
        cell_lon,
        x,
        y,
        z,
        out=(u0, v0, u1, v1),
        offset_provider={"E2C": distributed_grid.get_connectivity(dims.E2C)},
    )
    u_component = np.vstack((u0.asnumpy(), u1.asnumpy())).T
    v_component = np.vstack((v0.asnumpy(), v1.asnumpy())).T

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_u,
        local_field=u_component,
    )
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_v,
        local_field=v_component,
    )


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_neighbor_access_e2v(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    print(f"running on {processor_props.comm}")
    file = grid_utils.resolve_full_grid_file_name(grid)
    single_node = utils.run_grid_manager_for_singlenode(file)
    single_node_grid = single_node.grid
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid,
        coordinates=single_node.coordinates,
        decomposition_info=single_node.decomposition_info,
        extra_fields=single_node.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    reference_tangent_x = single_node_geometry.get(geometry_attributes.EDGE_TANGENT_X).asnumpy()
    reference_tangent_y = single_node_geometry.get(geometry_attributes.EDGE_TANGENT_Y).asnumpy()
    print(
        f"rank = {processor_props.rank} : single node computed field reference has size  {reference_tangent_x.shape}"
    )
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'EdgeDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    distributed_coordinates = multinode_grid_manager.coordinates
    vertex_lat = distributed_coordinates[dims.VertexDim]["lat"]
    vertex_lon = distributed_coordinates[dims.VertexDim]["lon"]
    tangent_orientation = multinode_grid_manager.geometry_fields.get(
        gridfile.GeometryName.TANGENT_ORIENTATION
    )

    tangent_x = data_alloc.zero_field(distributed_grid, dims.EdgeDim)
    tangent_y = data_alloc.zero_field(distributed_grid, dims.EdgeDim)
    tangent_z = data_alloc.zero_field(distributed_grid, dims.EdgeDim)

    geometry_stencils.cartesian_coordinates_of_edge_tangent.with_backend(backend)(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        edge_orientation=tangent_orientation,
        domain={dims.EdgeDim: (0, distributed_grid.num_edges)},
        offset_provider=distributed_grid.connectivities,
        out=(tangent_x, tangent_y, tangent_z),
    )

    # only the computation of the tangent uses neighbor access
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_tangent_x,
        local_field=tangent_x.asnumpy(),
    )
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.EdgeDim,
        global_reference_field=reference_tangent_y,
        local_field=tangent_y.asnumpy(),
    )

    print(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_neighbor_access_v2e(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    print(f"running on {processor_props.comm}")
    single_node = utils.run_grid_manager_for_singlenode(file)
    single_node_grid = single_node.grid
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid,
        coordinates=single_node.coordinates,
        decomposition_info=single_node.decomposition_info,
        extra_fields=single_node.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    single_node_dual_edge_length = single_node_geometry.get(geometry_attributes.DUAL_EDGE_LENGTH)
    single_node_edge_orientation = single_node_geometry.get(
        geometry_attributes.VERTEX_EDGE_ORIENTATION
    )
    single_node_dual_area = single_node_geometry.get(geometry_attributes.DUAL_AREA)
    single_node_owner_mask = gtx.as_field(
        (dims.VertexDim,),
        data=single_node.decomposition_info.owner_mask(dims.VertexDim),  # type: ignore  [arg-type]
        dtype=bool,
    )
    reference = data_alloc.zero_field(single_node_grid, dims.VertexDim, dims.V2EDim)
    lateral_boundary_start = single_node_grid.start_index(
        h_grid.vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end = single_node_grid.start_index(h_grid.vertex_domain(h_grid.Zone.END))

    interpolation_fields.compute_geofac_rot.with_backend(None)(
        single_node_dual_edge_length,
        single_node_edge_orientation,
        single_node_dual_area,
        single_node_owner_mask,
        out=reference,
        domain={dims.VertexDim: (lateral_boundary_start, horizontal_end)},
        offset_provider={"V2E": single_node_grid.get_connectivity(dims.V2E)},
    )

    print(
        f"rank = {processor_props.rank} : single node computed field reference has size  {reference.asnumpy().shape}"
    )
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'EdgeDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}),"
        f" (2: {decomposition_info.get_halo_size(dims.EdgeDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    distributed_coordinates = multinode_grid_manager.coordinates
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    distributed_geometry = geometry.GridGeometry(
        backend=backend,
        grid=distributed_grid,
        coordinates=distributed_coordinates,
        decomposition_info=decomposition_info,
        extra_fields=extra_geometry_fields,
        metadata=geometry_attributes.attrs,
    )

    dual_edge_length = distributed_geometry.get(geometry_attributes.DUAL_EDGE_LENGTH)
    edge_orientation = distributed_geometry.get(geometry_attributes.VERTEX_EDGE_ORIENTATION)
    dual_area = distributed_geometry.get(geometry_attributes.DUAL_AREA)
    geofac_rot = data_alloc.zero_field(distributed_grid, dims.VertexDim, dims.V2EDim)
    onwner_mask = gtx.as_field((dims.VertexDim,), decomposition_info.owner_mask(dims.VertexDim))  # type: ignore [arg-type]
    interpolation_fields.compute_geofac_rot.with_backend(None)(
        dual_edge_length=dual_edge_length,
        edge_orientation=edge_orientation,
        dual_area=dual_area,
        owner_mask=onwner_mask,
        out=geofac_rot,
        offset_provider={"V2E": distributed_grid.get_connectivity(dims.V2E)},
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.VertexDim,
        global_reference_field=reference.asnumpy(),
        local_field=geofac_rot.asnumpy(),
    )

    print(f"rank = {processor_props.rank}/{processor_props.comm_size} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_halo_neighbor_access_c2e2c(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    center_weight = 0.3
    xp = data_alloc.import_array_ns(allocator=backend)
    start_zone = h_grid.cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    print(f"running on {processor_props.comm}")
    single_node = utils.run_grid_manager_for_singlenode(file)
    single_node_grid = single_node.grid
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid,
        coordinates=single_node.coordinates,
        decomposition_info=single_node.decomposition_info,
        extra_fields=single_node.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    reference = interpolation_fields._compute_c_bln_avg(
        single_node_grid.get_connectivity(dims.C2E2C).ndarray,
        single_node_geometry.get(geometry_attributes.CELL_LAT).ndarray,
        single_node_geometry.get(geometry_attributes.CELL_LON).ndarray,
        center_weight,
        horizontal_start=single_node_grid.start_index(start_zone),
        array_ns=xp,
    )

    print(
        f"rank = {processor_props.rank} : single node computed field reference has size  {reference.shape}"
    )
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    distributed_coordinates = multinode_grid_manager.coordinates
    extra_geometry_fields = multinode_grid_manager.geometry_fields
    distributed_geometry = geometry.GridGeometry(
        backend=backend,
        grid=distributed_grid,
        coordinates=distributed_coordinates,
        decomposition_info=decomposition_info,
        extra_fields=extra_geometry_fields,
        metadata=geometry_attributes.attrs,
    )

    c_bln_avg = interpolation_fields._compute_c_bln_avg(
        distributed_grid.get_connectivity(dims.C2E2C).ndarray,
        distributed_geometry.get(geometry_attributes.CELL_LAT).ndarray,
        distributed_geometry.get(geometry_attributes.CELL_LON).ndarray,
        center_weight,
        horizontal_start=distributed_grid.start_index(start_zone),
        array_ns=xp,
    )

    print(
        f"rank = {processor_props.rank}/{processor_props.comm_size} - computed field has shape =({c_bln_avg.shape})"
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.CellDim,
        global_reference_field=reference,
        local_field=c_bln_avg,
    )

    print(f"rank = {processor_props.rank}/{processor_props.comm_size} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_halo_neighbor_access_v2c(
    processor_props: decomp_defs.ProcessProperties, backend: gtx_typing.Backend
) -> None:
    file = grid_utils.resolve_full_grid_file_name(test_defs.Grids.R02B04_GLOBAL)
    print(f"running on {processor_props.comm}")
    single_node = utils.run_grid_manager_for_singlenode(file)
    single_node_grid = single_node.grid

    buffer = np.repeat(
        single_node.coordinates[dims.CellDim]["lat"].asnumpy()[:, None],
        repeats=utils.NUM_LEVELS,
        axis=1,
    )
    full_cell_k_field = gtx.as_field(
        (dims.CellDim, dims.KDim),
        data=buffer,  # type: ignore [arg-type]
        dtype=float,
        allocator=backend,
    )
    print(
        f"rank = {processor_props.rank}  / {processor_props.comm_size}: single node input field has size  {full_cell_k_field.asnumpy().shape}"
    )
    buffer = single_node.coordinates[dims.VertexDim]["lat"].asnumpy() + 1.0
    full_coef = gtx.as_field(
        (dims.VertexDim, dims.V2CDim),
        data=np.repeat((buffer / np.max(buffer))[:, None], 6, axis=1),  # type: ignore [arg-type]
        dtype=float,
        allocator=backend,
    )

    reference = data_alloc.zero_field(
        single_node_grid,
        dims.VertexDim,
        dims.KDim,
        dtype=full_cell_k_field.dtype,
        allocator=backend,
    )
    _compute_cell_2_vertex_interpolation(
        full_cell_k_field,
        full_coef,
        out=reference,
        offset_provider={"V2C": single_node_grid.get_connectivity(dims.V2C)},
    )

    print(
        f"rank = {processor_props.rank}/ {processor_props.comm_size} : single node computed field reference has size  {reference.asnumpy().shape}"
    )
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    decomposition_info = multinode_grid_manager.decomposition_info

    print(f"rank = {processor_props.rank} : {decomposition_info.get_horizontal_size()!r}")
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    print(
        f"rank = {processor_props.rank}: halo size for 'VertexDim' "
        f"(1 : {decomposition_info.get_halo_size(dims.VertexDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {decomposition_info.get_halo_size(dims.VertexDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    my_global_cells = decomposition_info.global_index(dims.CellDim)
    distributed_cell_k_buffer = (
        full_cell_k_field.asnumpy()[my_global_cells, :]
        .ravel(order="K")
        .reshape(distributed_grid.num_cells, utils.NUM_LEVELS)
    )
    # validate the input data distribution
    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.CellDim,
        global_reference_field=full_cell_k_field.asnumpy(),
        local_field=distributed_cell_k_buffer,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: input field shape = ([{distributed_cell_k_buffer.shape})"
    )

    distributed_cell_k_field = gtx.as_field(
        (dims.CellDim, dims.KDim),
        data=distributed_cell_k_buffer,  # type: ignore [arg-type]
        dtype=distributed_cell_k_buffer.dtype,
        allocator=backend,
    )

    my_global_vertices = decomposition_info.global_index(dims.VertexDim)

    distributed_coef_buffer = (
        full_coef.asnumpy()[my_global_vertices, :]
        .ravel(order="K")
        .reshape((distributed_grid.num_vertices, 6))
    )
    distributed_coef = gtx.as_field(
        (dims.VertexDim, dims.V2CDim),
        data=distributed_coef_buffer,  # type: ignore  [arg-type]
        allocator=backend,
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dims.VertexDim,
        global_reference_field=full_coef.asnumpy(),
        local_field=distributed_coef_buffer,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: coefficient shape = ([{distributed_coef_buffer.shape})"
    )
    output = data_alloc.zero_field(distributed_grid, dims.VertexDim, dims.KDim, allocator=backend)
    _compute_cell_2_vertex_interpolation(
        distributed_cell_k_field,
        distributed_coef,
        out=output,
        offset_provider={"V2C": distributed_grid.get_connectivity(dims.V2C)},
    )

    assert_gathered_field_against_global(
        decomposition_info,
        processor_props,
        dim=dims.VertexDim,
        global_reference_field=reference.asnumpy(),
        local_field=output.asnumpy(),
    )
    nn = np.where(single_node_grid.get_connectivity(dims.V2C).asnumpy() == 3855)
    if processor_props.rank == 0:
        print(nn)
        print(single_node_grid.get_connectivity(dims.V2C).asnumpy()[nn[0]])

    print(f"rank={processor_props.rank}/{processor_props.comm_size}: ")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", (test_defs.Grids.R02B04_GLOBAL,))
def test_validate_skip_values_in_distributed_connectivities(
    processor_props: decomp_defs.ProcessProperties, grid: test_defs.GridDescription
) -> None:
    file = grid_utils.resolve_full_grid_file_name(grid)
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=halo.SimpleMetisDecomposer(),
    )
    distributed_grid = multinode_grid_manager.grid
    for k, c in distributed_grid.connectivities.items():
        if gtx_common.is_neighbor_connectivity(c):
            skip_values_in_table = np.count_nonzero(c.asnumpy() == c.skip_value)
            found_skips = skip_values_in_table > 0
            assert (
                found_skips == (c.skip_value is not None)
            ), f"rank={processor_props.rank} / {processor_props.comm_size}: {k} - # of skip values found in table = {skip_values_in_table},  skip value is {c.skip_value}"
            if skip_values_in_table > 0:
                assert (
                    c in icon.CONNECTIVITIES_ON_BOUNDARIES or icon.CONNECTIVITIES_ON_PENTAGONS
                ), f"rank={processor_props.rank} / {processor_props.comm_size}: {k} has skip found in table"
