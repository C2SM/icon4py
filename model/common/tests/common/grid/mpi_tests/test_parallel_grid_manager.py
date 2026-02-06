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
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomp_defs,
    mpi_decomposition,
)
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes,
    geometry_stencils,
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes,
    interpolation_factory,
    interpolation_fields,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils, test_utils

from ..fixtures import backend, global_grid_descriptor, processor_props
from . import utils


try:
    import mpi4py

    mpi_decomposition.init_mpi()
except ImportError:
    pytest.skip("Skipping parallel on single node installation", allow_module_level=True)

log = logging.getLogger(__file__)


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(
    processor_props: decomp_defs.ProcessProperties,
    global_grid_descriptor: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(global_grid_descriptor)
    manager = gm.GridManager(
        grid_file=file,
        config=v_grid.VerticalGridConfig(num_levels=utils.NUM_LEVELS),
        transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            allocator=None,
            run_properties=processor_props,
            decomposer=decomp.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


def _get_neighbor_tables(grid: base.Grid) -> dict:
    return {
        k: v.ndarray
        for k, v in grid.connectivities.items()
        if gtx_common.is_neighbor_connectivity(v)
    }


def gather_field(field: np.ndarray, props: decomp_defs.ProcessProperties) -> tuple:
    constant_dims = tuple(field.shape[1:])
    print(f"gather_field on rank={props.rank} - gathering field of local shape {field.shape}")
    constant_length = functools.reduce(operator.mul, constant_dims, 1)
    local_sizes = np.array(props.comm.gather(field.size, root=0))
    if props.rank == 0:
        recv_buffer = np.empty(np.sum(local_sizes), dtype=field.dtype)
        print(
            f"gather_field on rank = {props.rank} - setup receive buffer with size {sum(local_sizes)} on rank 0"
        )
    else:
        recv_buffer = None

    props.comm.Gatherv(sendbuf=field, recvbuf=(recv_buffer, local_sizes), root=0)
    if props.rank == 0:
        local_first_dim = tuple(sz // constant_length for sz in local_sizes)
        print(
            f" gather_field on rank = 0: computed local dims {local_first_dim} - constant dims {constant_dims}"
        )
        gathered_field = recv_buffer.reshape((-1, *constant_dims))  # type: ignore [union-attr]
    else:
        gathered_field = None
        local_first_dim = field.shape
    return local_first_dim, gathered_field


def check_local_global_field(
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

    # Compare full local field, including halos, against global reference field.
    # TODO(msimberg): Is halo always expected to be populated?
    global_indices_local_field = decomposition_info.global_index(
        dim,
        decomp_defs.DecompositionInfo.EntryType.OWNED,  # ALL if checking halos
    )
    local_indices_local_field = decomposition_info.local_index(
        dim,
        decomp_defs.DecompositionInfo.EntryType.OWNED,  # ALL if checking halos
    )
    local_field_from_global_field = global_reference_field[global_indices_local_field]
    local_field_from_local_field = local_field[local_indices_local_field]
    np.testing.assert_allclose(
        local_field_from_global_field, local_field_from_local_field, atol=0.0, verbose=True
    )

    # Compare owned local field, excluding halos, against global reference
    # field, by gathering owned entries to the first rank. This ensures that in
    # total we have the full global field distributed on all ranks.
    owned_entries = local_field[
        decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
    ]
    gathered_sizes, gathered_field = gather_field(owned_entries, processor_props)

    global_index_sizes, gathered_global_indices = gather_field(
        decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED),
        processor_props,
    )

    if processor_props.rank == 0:
        print(f"rank = {processor_props.rank}: asserting gathered fields: ")

        assert np.all(
            gathered_sizes == global_index_sizes
        ), f"gathered field sizes do not match:  {dim} {gathered_sizes} - {global_index_sizes}"
        print(
            f"rank = {processor_props.rank}: Checking field size on dim ={dim}: --- gathered sizes {gathered_sizes} = {sum(gathered_sizes)}"
        )
        print(
            f"rank = {processor_props.rank}:                      --- gathered field has size {gathered_sizes}"
        )
        sorted_ = np.zeros(global_reference_field.shape, dtype=gtx.float64)  # type: ignore [attr-defined]
        sorted_[gathered_global_indices] = gathered_field
        print(
            f" rank = {processor_props.rank}: SHAPES: global reference field {global_reference_field.shape}, gathered = {gathered_field.shape}"
        )

        # We expect an exact match, since the starting point is the same (grid
        # file) and we are doing the exact same computations in single rank and
        # multi rank mode.
        np.testing.assert_allclose(sorted_, global_reference_field, atol=0.0, verbose=True)


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, dim",
    [
        # TODO(msimberg): Get dim out of field?
        (geometry_attributes.CELL_AREA, dims.CellDim),
        (geometry_attributes.EDGE_LENGTH, dims.EdgeDim),
        (geometry_attributes.VERTEX_LAT, dims.VertexDim),
        (geometry_attributes.EDGE_NORMAL_VERTEX_U, dims.EdgeDim),
        (geometry_attributes.EDGE_NORMAL_VERTEX_V, dims.EdgeDim),
        (geometry_attributes.EDGE_NORMAL_CELL_U, dims.EdgeDim),
        (geometry_attributes.EDGE_NORMAL_CELL_V, dims.EdgeDim),
        (geometry_attributes.EDGE_TANGENT_X, dims.EdgeDim),
        (geometry_attributes.EDGE_TANGENT_Y, dims.EdgeDim),
    ],
)
def test_geometry_fields_compare_single_multi_rank(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    # TODO(msimberg): Maybe use regular grid fixture and skip local area grids?
    global_grid_descriptor: test_defs.GridDescription,
    attrs_name: str,
    dim: gtx.Dimension,
) -> None:
    # TODO(msimberg): Add fixture for "always single rank" grid manager
    file = grid_utils.resolve_full_grid_file_name(global_grid_descriptor)
    print(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_node_grid_manager = utils.run_grid_manager_for_singlenode(file)
    # TODO(msimberg): Add fixture for "always single rank" geometry
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid_manager.grid,
        coordinates=single_node_grid_manager.coordinates,
        decomposition_info=single_node_grid_manager.decomposition_info,
        extra_fields=single_node_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    print(
        f"rank = {processor_props.rank} : single node grid has size {single_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )

    # TODO(msimberg): Use regular grid manager fixture (should anyway be multi rank by default)
    multi_node_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
    )
    print(
        f"rank = {processor_props.rank} : {multi_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    # TODO(msimberg): Use regular geometry fixture
    multi_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=multi_node_grid_manager.grid,
        coordinates=multi_node_grid_manager.coordinates,
        decomposition_info=multi_node_grid_manager.decomposition_info,
        extra_fields=multi_node_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )

    check_local_global_field(
        decomposition_info=multi_node_grid_manager.decomposition_info,
        processor_props=processor_props,
        dim=dim,
        global_reference_field=single_node_geometry.get(attrs_name).asnumpy(),
        local_field=multi_node_geometry.get(attrs_name).asnumpy(),
    )

    print(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, dim",
    [
        (interpolation_attributes.GEOFAC_DIV, dims.CellDim),
        (interpolation_attributes.GEOFAC_ROT, dims.VertexDim),
        (interpolation_attributes.C_BLN_AVG, dims.CellDim),
    ],
)
def test_interpolation_fields_compare_single_multi_rank(
    processor_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    # TODO(msimberg): Maybe use regular grid fixture and skip local area grids?
    global_grid_descriptor: test_defs.GridDescription,
    attrs_name: str,
    dim: gtx.Dimension,
) -> None:
    # TODO(msimberg): Add fixture for "always single rank" grid manager
    file = grid_utils.resolve_full_grid_file_name(global_grid_descriptor)
    print(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
    single_node_grid_manager = utils.run_grid_manager_for_singlenode(file)
    # TODO(msimberg): Add fixture for "always single rank" geometry
    single_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=single_node_grid_manager.grid,
        coordinates=single_node_grid_manager.coordinates,
        decomposition_info=single_node_grid_manager.decomposition_info,
        extra_fields=single_node_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    # TODO(msimberg): Add fixture for "always single rank" interpolation factory
    single_node_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=single_node_grid_manager.grid,
        decomposition_info=single_node_grid_manager.decomposition_info,
        geometry_source=single_node_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    print(
        f"rank = {processor_props.rank} : single node grid has size {single_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )

    # TODO(msimberg): Use regular grid manager fixture (should anyway be multi rank by default)
    multi_node_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
    )
    print(
        f"rank = {processor_props.rank} : {multi_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    print(
        f"rank = {processor_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    # TODO(msimberg): Use regular geometry fixture
    multi_node_geometry = geometry.GridGeometry(
        backend=backend,
        grid=multi_node_grid_manager.grid,
        coordinates=multi_node_grid_manager.coordinates,
        decomposition_info=multi_node_grid_manager.decomposition_info,
        extra_fields=multi_node_grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
    )
    # TODO(msimberg): Use regular interpolation factory fixture
    multi_node_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_node_grid_manager.grid,
        decomposition_info=multi_node_grid_manager.decomposition_info,
        geometry_source=multi_node_geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=mpi_decomposition.GHexMultiNodeExchange(
            processor_props, multi_node_grid_manager.decomposition_info
        ),
    )

    field_ref = single_node_interpolation.get(attrs_name).asnumpy()
    field = multi_node_interpolation.get(attrs_name).asnumpy()

    check_local_global_field(
        decomposition_info=multi_node_grid_manager.decomposition_info,
        processor_props=processor_props,
        dim=dim,
        global_reference_field=field_ref,
        local_field=field,
    )

    print(f"rank = {processor_props.rank} - DONE")


# @pytest.mark.mpi
# @pytest.mark.parametrize("processor_props", [True], indirect=True)
# @pytest.mark.parametrize("attrs_name, dim", [(metrics_attributes.DDXT_Z_HALF_E, dims.EdgeDim)])
# def test_metrics_fields_compare_single_multi_rank(
#     processor_props: decomp_defs.ProcessProperties,
#     backend: gtx_typing.Backend | None,
#     # TODO(msimberg): Maybe use regular grid fixture and skip local area grids?
#     global_grid_descriptor: test_defs.GridDescription,
#     attrs_name: str,
#     dim: gtx.Dimension,
# ) -> None:
#     # TODO(msimberg): Add fixture for "always single rank" grid manager
#     file = grid_utils.resolve_full_grid_file_name(global_grid_descriptor)
#     print(f"running on {processor_props.comm} with {processor_props.comm_size} ranks")
#     single_node_grid_manager = utils.run_grid_manager_for_singlenode(file)
#     # TODO(msimberg): Add fixture for "always single rank" geometry
#     single_node_geometry = geometry.GridGeometry(
#         backend=backend,
#         grid=single_node_grid_manager.grid,
#         coordinates=single_node_grid_manager.coordinates,
#         decomposition_info=single_node_grid_manager.decomposition_info,
#         extra_fields=single_node_grid_manager.geometry_fields,
#         metadata=geometry_attributes.attrs,
#     )
#     # TODO(msimberg): Add fixture for "always single rank" interpolation factory
#     single_node_interpolation = interpolation_factory.InterpolationFieldsFactory(
#         grid=single_node_grid_manager.grid,
#         decomposition_info=single_node_grid_manager.decomposition_info,
#         geometry_source=single_node_geometry,
#         backend=backend,
#         metadata=interpolation_attributes.attrs,
#         exchange=decomp_defs.SingleNodeExchange(),
#     )
#     # TODO metrics factory
#     print(
#         f"rank = {processor_props.rank} : single node grid has size {single_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
#     )

#     # TODO(msimberg): Use regular grid manager fixture (should anyway be multi rank by default)
#     multi_node_grid_manager = utils.run_gridmananger_for_multinode(
#         file=file,
#         run_properties=processor_props,
#         decomposer=decomp.MetisDecomposer(),
#     )
#     print(
#         f"rank = {processor_props.rank} : {multi_node_grid_manager.decomposition_info.get_horizontal_size()!r}"
#     )
#     print(
#         f"rank = {processor_props.rank}: halo size for 'CellDim' "
#         f"(1: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
#         f"(2: {multi_node_grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
#     )
#     # TODO(msimberg): Use regular geometry fixture
#     multi_node_geometry = geometry.GridGeometry(
#         backend=backend,
#         grid=multi_node_grid_manager.grid,
#         coordinates=multi_node_grid_manager.coordinates,
#         decomposition_info=multi_node_grid_manager.decomposition_info,
#         extra_fields=multi_node_grid_manager.geometry_fields,
#         metadata=geometry_attributes.attrs,
#     )
#     # TODO(msimberg): Use regular interpolation factory fixture
#     multi_node_interpolation = interpolation_factory.InterpolationFieldsFactory(
#         grid=multi_node_grid_manager.grid,
#         decomposition_info=multi_node_grid_manager.decomposition_info,
#         geometry_source=multi_node_geometry,
#         backend=backend,
#         metadata=interpolation_attributes.attrs,
#         exchange=mpi_decomposition.GHexMultiNodeExchange(
#             processor_props, multi_node_grid_manager.decomposition_info
#         ),
#     )

#     field_ref = single_node_interpolation.get(attrs_name).asnumpy()
#     field = multi_node_interpolation.get(attrs_name).asnumpy()

#     check_local_global_field(
#         decomposition_info=multi_node_grid_manager.decomposition_info,
#         processor_props=processor_props,
#         dim=dim,
#         global_reference_field=field_ref,
#         local_field=field,
#     )

#     print(f"rank = {processor_props.rank} - DONE")


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_validate_skip_values_in_distributed_connectivities(
    processor_props: decomp_defs.ProcessProperties,
    global_grid_descriptor: test_defs.GridDescription,
) -> None:
    file = grid_utils.resolve_full_grid_file_name(global_grid_descriptor)
    multinode_grid_manager = utils.run_gridmananger_for_multinode(
        file=file,
        run_properties=processor_props,
        decomposer=decomp.MetisDecomposer(),
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


@pytest.mark.mpi
@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.parametrize("grid", [test_defs.Grids.MCH_CH_R04B09_DSL])
def test_limited_area_raises(
    processor_props: decomp_defs.ProcessProperties,
    grid: test_defs.GridDescription,
) -> None:
    with pytest.raises(
        NotImplementedError, match="Limited-area grids are not supported in distributed runs"
    ):
        _ = utils.run_gridmananger_for_multinode(
            file=grid_utils.resolve_full_grid_file_name(grid),
            run_properties=processor_props,
            decomposer=decomp.MetisDecomposer(),
        )
