# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import cffi
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.tools.py2fgen import test_utils
from icon4py.tools.py2fgen.wrappers import common as wrapper_common, grid_wrapper


@pytest.fixture
def grid_init(grid_savepoint):
    # --- Set Up Grid Parameters ---
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    cell_starts = test_utils.array_to_array_info(grid_savepoint._read_int32("c_start_index"))
    cell_ends = test_utils.array_to_array_info(grid_savepoint._read_int32("c_end_index"))
    vertex_starts = test_utils.array_to_array_info(grid_savepoint._read_int32("v_start_index"))
    vertex_ends = test_utils.array_to_array_info(grid_savepoint._read_int32("v_end_index"))
    edge_starts = test_utils.array_to_array_info(grid_savepoint._read_int32("e_start_index"))
    edge_ends = test_utils.array_to_array_info(grid_savepoint._read_int32("e_end_index"))

    c2e = test_utils.array_to_array_info(grid_savepoint._read_int32("c2e"))
    e2c = test_utils.array_to_array_info(grid_savepoint._read_int32("e2c"))
    c2e2c = test_utils.array_to_array_info(grid_savepoint._read_int32("c2e2c"))
    e2c2e = test_utils.array_to_array_info(grid_savepoint._read_int32("e2c2e"))
    e2v = test_utils.array_to_array_info(grid_savepoint._read_int32("e2v"))
    v2e = test_utils.array_to_array_info(grid_savepoint._read_int32("v2e"))
    v2c = test_utils.array_to_array_info(grid_savepoint._read_int32("v2c"))
    e2c2v = test_utils.array_to_array_info(grid_savepoint._read_int32("e2c2v"))
    c2v = test_utils.array_to_array_info(grid_savepoint._read_int32("c2v"))

    # --- Extract Grid Parameters from Savepoint ---
    tangent_orientation = test_utils.array_to_array_info(
        grid_savepoint.tangent_orientation().ndarray
    )
    inverse_primal_edge_lengths = test_utils.array_to_array_info(
        grid_savepoint.inverse_primal_edge_lengths().ndarray
    )
    inv_dual_edge_length = test_utils.array_to_array_info(
        grid_savepoint.inv_dual_edge_length().ndarray
    )
    inv_vert_vert_length = test_utils.array_to_array_info(
        grid_savepoint.inv_vert_vert_length().ndarray
    )
    edge_areas = test_utils.array_to_array_info(grid_savepoint.edge_areas().ndarray)
    f_e = test_utils.array_to_array_info(grid_savepoint.f_e().ndarray)
    cell_areas = test_utils.array_to_array_info(grid_savepoint.cell_areas().ndarray)
    mean_cell_area = grid_savepoint.mean_cell_area()
    primal_normal_vert_x = test_utils.array_to_array_info(
        grid_savepoint.primal_normal_vert_x().ndarray
    )
    primal_normal_vert_y = test_utils.array_to_array_info(
        grid_savepoint.primal_normal_vert_y().ndarray
    )
    dual_normal_vert_x = test_utils.array_to_array_info(grid_savepoint.dual_normal_vert_x().ndarray)
    dual_normal_vert_y = test_utils.array_to_array_info(grid_savepoint.dual_normal_vert_y().ndarray)
    primal_normal_cell_x = test_utils.array_to_array_info(
        grid_savepoint.primal_normal_cell_x().ndarray
    )
    primal_normal_cell_y = test_utils.array_to_array_info(
        grid_savepoint.primal_normal_cell_y().ndarray
    )
    dual_normal_cell_x = test_utils.array_to_array_info(grid_savepoint.dual_normal_cell_x().ndarray)
    dual_normal_cell_y = test_utils.array_to_array_info(grid_savepoint.dual_normal_cell_y().ndarray)
    cell_center_lat = test_utils.array_to_array_info(grid_savepoint.cell_center_lat().ndarray)
    cell_center_lon = test_utils.array_to_array_info(grid_savepoint.cell_center_lon().ndarray)
    edge_center_lat = test_utils.array_to_array_info(grid_savepoint.edge_center_lat().ndarray)
    edge_center_lon = test_utils.array_to_array_info(grid_savepoint.edge_center_lon().ndarray)
    primal_normal_x = test_utils.array_to_array_info(grid_savepoint.primal_normal_v1().ndarray)
    primal_normal_y = test_utils.array_to_array_info(grid_savepoint.primal_normal_v2().ndarray)

    # not running in parallel
    _dummy_int_array = np.array([], dtype=np.int32)
    _dummy_bool_array = np.array([], dtype=np.bool_)
    c_glb_index = test_utils.array_to_array_info(_dummy_int_array)
    e_glb_index = test_utils.array_to_array_info(_dummy_int_array)
    v_glb_index = test_utils.array_to_array_info(_dummy_int_array)
    c_owner_mask = test_utils.array_to_array_info(_dummy_bool_array)
    e_owner_mask = test_utils.array_to_array_info(_dummy_bool_array)
    v_owner_mask = test_utils.array_to_array_info(_dummy_bool_array)

    ffi = cffi.FFI()
    grid_wrapper.grid_init(
        ffi,
        perf_counters=None,
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        edge_areas=edge_areas,
        f_e=f_e,
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        cell_areas=cell_areas,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
        mean_cell_area=mean_cell_area,
        c_glb_index=c_glb_index,
        e_glb_index=e_glb_index,
        v_glb_index=v_glb_index,
        c_owner_mask=c_owner_mask,
        e_owner_mask=e_owner_mask,
        v_owner_mask=v_owner_mask,
        comm_id=None,
        backend=wrapper_common.BackendIntEnum.DEFAULT,
    )


# TODO(): if useful add a grid_init (mock) test like in test_diffusion_wrapper.py
# otherwise rename this to something without the `test_` prefix
