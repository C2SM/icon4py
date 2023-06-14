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
import logging
from pathlib import Path

import ghex
import numpy as np
import pytest

from atm_dyn_iconam.tests.test_utils.serialbox_utils import IconSerialDataProvider
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.driver.io_utils import (
    SerializationType,
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_static_fields,
)
from icon4py.driver.parallel_setup import (
    DecompositionInfo,
    Exchange,
    get_processor_properties,
)


"""
running tests with mpi:

mpirun -np 2 python -m pytest --with-mpi tests/mpi_tests/test_parallel_setup.py

mpirun -np 2 pytest -v --with-mpi tests/mpi_tests/


"""
props = get_processor_properties()


@pytest.mark.mpi
def test_processor_properties_from_comm_world(mpi):
    props = get_processor_properties()
    assert props.rank < mpi.COMM_WORLD.Get_size()
    assert props.comm_name == mpi.COMM_WORLD.Get_name()


# TODO s [magdalena] extract fixture, more useful asserts...





@pytest.mark.skipif(
    props.comm_size > 2, reason="input files available for 1 or 2 nodes"
)
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (CellDim, (10448, 10448), (10611, 10612)),
        (EdgeDim, (15820, 15738), (16065, 16067)),
        (VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
def test_decomposition_info_masked(mpi, datapath, dim, owned, total):
    my_rank = props.rank
    decomposition_info = read_decomp_info(datapath, props, SerializationType.SB)
    all_indices = decomposition_info.global_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]
    assert all_indices.shape[0] == my_total

    owned_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned

    halo_indices = decomposition_info.global_index(
        dim, DecompositionInfo.EntryType.HALO
    )
    assert halo_indices.shape[0] == my_total - my_owned
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)

@pytest.mark.skipif(
    props.comm_size > 2, reason="input files available for 1 or 2 nodes"
)
@pytest.mark.parametrize(
    ("dim, owned, total"),
    (
        (CellDim, (10448, 10448), (10611, 10612)),
        (EdgeDim, (15820, 15738), (16065, 16067)),
        (VertexDim, (5373, 5290), (5455, 5456)),
    ),
)
def test_decomposition_info_local_index(mpi, datapath, dim, owned, total):
    props = get_processor_properties()

    my_rank = props.rank
    decomposition_info = read_decomp_info(datapath, props, SerializationType.SB)
    all_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.ALL)
    my_total = total[my_rank]
    my_owned = owned[my_rank]

    assert all_indices.shape[0] == my_total
    assert np.array_equal(all_indices, np.arange(0, my_total))
    halo_indices = decomposition_info.local_index(dim, DecompositionInfo.EntryType.HALO)
    assert halo_indices.shape[0] == my_total - my_owned
    assert halo_indices.shape[0] < all_indices.shape[0]
    assert np.alltrue(halo_indices <= np.max(all_indices))

    owned_indices = decomposition_info.local_index(
        dim, DecompositionInfo.EntryType.OWNED
    )
    assert owned_indices.shape[0] == my_owned
    assert owned_indices.shape[0] <= all_indices.shape[0]
    assert np.alltrue(owned_indices <= np.max(all_indices))
    _assert_index_partitioning(all_indices, halo_indices, owned_indices)



def _assert_index_partitioning(all_indices, halo_indices, owned_indices):
    owned_list = owned_indices.tolist()
    halos_list = halo_indices.tolist()
    all_list = all_indices.tolist()
    assert set(owned_list) & set(halos_list) == set()
    assert set(owned_list) & set(all_list) == set(owned_list)
    assert set(halos_list) & set(all_list) == set(halos_list)
    assert set(halos_list) | set(owned_list) == set(all_list)



@pytest.mark.mpi
def test_decomposition_info_matches_gridsize(datapath):
    decomposition_info = read_decomp_info(datapath, props, SerializationType.SB)
    icon_grid = read_icon_grid(datapath, props.rank)
    assert \
    decomposition_info.global_index(dim=CellDim, entry_type=DecompositionInfo.EntryType.ALL).shape[
        0] == icon_grid.num_cells()
    assert decomposition_info.global_index(VertexDim, DecompositionInfo.EntryType.ALL).shape[
               0] == icon_grid.num_vertices()
    assert decomposition_info.global_index(EdgeDim, DecompositionInfo.EntryType.ALL).shape[
               0] == icon_grid.num_edges()

#@pytest.mark.mpi
@pytest.mark.skip
def test_parallel_diffusion(r04b09_diffusion_config, parallel_props, step_date_init, caplog):
    caplog.set_level(logging.INFO)
    experiment_name = "mch_ch_r04b09_dsl"
    path = Path(
        f"/home/magdalena/data/exclaim/dycore/{experiment_name}/mpitasks2/{experiment_name}/ser_data"
    )

    icon_grid = read_icon_grid(path, rank=parallel_props.rank)
    decomp_info = read_decomp_info(
        path,
        parallel_props,
    )
    context = ghex.context(ghex.mpi_comm(parallel_props.comm), True)
    #assert context.size() == 2

    diffusion_params = DiffusionParams(r04b09_diffusion_config)
    diffusion_initial_data = IconSerialDataProvider(
        "icon_pydycore", str(path), True, mpi_rank=parallel_props.rank
    ).from_savepoint_diffusion_init(linit=True, date=step_date_init)
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        path, rank=parallel_props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(
        path, rank=parallel_props.rank
    )

    dtime = diffusion_initial_data.get_metadata("dtime").get("dtime")
    exchange = Exchange(context, decomp_info)
    print(f"exchange setup: for {exchange.get_size()} nodes")

    diffusion = Diffusion(exchange)
    diffusion.init(
        icon_grid,
        r04b09_diffusion_config,
        diffusion_params,
        vertical_geometry,
        metric_state,
        interpolation_state,
    )
    print("diffusion initialized")
    diffusion.run(
        diagnostic_state=diffusion_initial_data.construct_diagnostics(),
        prognostic_state=diffusion_initial_data.construct_prognostics(),
        dtime=dtime,
        tangent_orientation=edge_geometry.tangent_orientation,
        inverse_primal_edge_lengths=edge_geometry.inverse_primal_edge_lengths,
        inverse_dual_edge_length=edge_geometry.inverse_dual_edge_lengths,
        inverse_vert_vert_lengths=edge_geometry.inverse_vertex_vertex_lengths,
        primal_normal_vert=edge_geometry.primal_normal_vert,
        dual_normal_vert=edge_geometry.dual_normal_vert,
        edge_areas=edge_geometry.edge_areas,
        cell_areas=cell_geometry.area,
    )
