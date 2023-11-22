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

import numpy as np
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.btraj_dreg_stencil_03 import btraj_dreg_stencil_03
from icon4py.model.common.dimension import E2CDim, ECDim, EdgeDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import (
    as_1D_sparse_field,
    constant_field,
    numpy_to_1D_sparse_field,
    random_field,
)


def btraj_dreg_stencil_03_numpy(
    p_vn: np.array,
    p_vt: np.array,
    cell_idx: np.array,
    cell_blk: np.array,
    edge_verts_1_x: np.array,
    edge_verts_2_x: np.array,
    edge_verts_1_y: np.array,
    edge_verts_2_y: np.array,
    pos_on_tplane_e_1_x: np.array,
    pos_on_tplane_e_2_x: np.array,
    pos_on_tplane_e_1_y: np.array,
    pos_on_tplane_e_2_y: np.array,
    primal_normal_cell_x: np.array,
    primal_normal_cell_y: np.array,
    dual_normal_cell_x: np.array,
    dual_normal_cell_y: np.array,
    lvn_sys_pos: np.array,
    p_dt: float,
) -> tuple[np.array]:
    lvn_pos = np.where(p_vn >= 0.0, True, False)
    cell_idx = np.expand_dims(cell_idx, axis=-1)
    cell_blk = np.expand_dims(cell_blk, axis=-1)
    primal_normal_cell_x = np.expand_dims(primal_normal_cell_x, axis=-1)
    dual_normal_cell_x = np.expand_dims(dual_normal_cell_x, axis=-1)
    primal_normal_cell_y = np.expand_dims(primal_normal_cell_y, axis=-1)
    dual_normal_cell_y = np.expand_dims(dual_normal_cell_y, axis=-1)
    edge_verts_1_x = np.expand_dims(edge_verts_1_x, axis=-1)
    edge_verts_1_y = np.expand_dims(edge_verts_1_y, axis=-1)
    edge_verts_2_x = np.expand_dims(edge_verts_2_x, axis=-1)
    edge_verts_2_y = np.expand_dims(edge_verts_2_y, axis=-1)
    pos_on_tplane_e_1_x = np.expand_dims(pos_on_tplane_e_1_x, axis=-1)
    pos_on_tplane_e_1_y = np.expand_dims(pos_on_tplane_e_1_y, axis=-1)
    pos_on_tplane_e_2_x = np.expand_dims(pos_on_tplane_e_2_x, axis=-1)
    pos_on_tplane_e_2_y = np.expand_dims(pos_on_tplane_e_2_y, axis=-1)

    p_cell_idx = np.where(lvn_pos, cell_idx[:, 0], cell_idx[:, 1])
    p_cell_blk = np.where(lvn_pos, cell_blk[:, 0], cell_blk[:, 1])
    p_cell_rel_idx_dsl = np.where(lvn_pos, int32(0), int32(1))

    depart_pts_1_x = np.broadcast_to(edge_verts_1_x, p_vn.shape) - p_vn * p_dt
    depart_pts_1_y = np.broadcast_to(edge_verts_1_y, p_vn.shape) - p_vt * p_dt
    depart_pts_2_x = np.broadcast_to(edge_verts_2_x, p_vn.shape) - p_vn * p_dt
    depart_pts_2_y = np.broadcast_to(edge_verts_2_y, p_vn.shape) - p_vt * p_dt

    pos_on_tplane_e_x = np.where(lvn_pos, pos_on_tplane_e_1_x, pos_on_tplane_e_2_x)
    pos_on_tplane_e_y = np.where(lvn_pos, pos_on_tplane_e_1_y, pos_on_tplane_e_2_y)

    pos_dreg_vert_c_1_x = edge_verts_1_x - pos_on_tplane_e_x
    pos_dreg_vert_c_1_y = edge_verts_1_y - pos_on_tplane_e_y
    pos_dreg_vert_c_2_x = np.where(lvn_sys_pos, depart_pts_1_x, edge_verts_2_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_2_y = np.where(lvn_sys_pos, depart_pts_1_y, edge_verts_2_y) - pos_on_tplane_e_y
    pos_dreg_vert_c_3_x = depart_pts_2_x - pos_on_tplane_e_x
    pos_dreg_vert_c_3_y = depart_pts_2_y - pos_on_tplane_e_y
    pos_dreg_vert_c_4_x = np.where(lvn_sys_pos, edge_verts_2_x, depart_pts_1_x) - pos_on_tplane_e_x
    pos_dreg_vert_c_4_y = np.where(lvn_sys_pos, edge_verts_2_y, depart_pts_1_y) - pos_on_tplane_e_y

    pn_cell_1 = np.where(lvn_pos, primal_normal_cell_x[:, 0], primal_normal_cell_x[:, 1])
    pn_cell_2 = np.where(lvn_pos, primal_normal_cell_y[:, 0], primal_normal_cell_y[:, 1])
    dn_cell_1 = np.where(lvn_pos, dual_normal_cell_x[:, 0], dual_normal_cell_x[:, 1])
    dn_cell_2 = np.where(lvn_pos, dual_normal_cell_y[:, 0], dual_normal_cell_y[:, 1])

    p_coords_dreg_v_1_lon_dsl = pos_dreg_vert_c_1_x * pn_cell_1 + pos_dreg_vert_c_1_y * dn_cell_1
    p_coords_dreg_v_2_lon_dsl = pos_dreg_vert_c_2_x * pn_cell_1 + pos_dreg_vert_c_2_y * dn_cell_1
    p_coords_dreg_v_3_lon_dsl = pos_dreg_vert_c_3_x * pn_cell_1 + pos_dreg_vert_c_3_y * dn_cell_1
    p_coords_dreg_v_4_lon_dsl = pos_dreg_vert_c_4_x * pn_cell_1 + pos_dreg_vert_c_4_y * dn_cell_1
    p_coords_dreg_v_1_lat_dsl = pos_dreg_vert_c_1_x * pn_cell_2 + pos_dreg_vert_c_1_y * dn_cell_2
    p_coords_dreg_v_2_lat_dsl = pos_dreg_vert_c_2_x * pn_cell_2 + pos_dreg_vert_c_2_y * dn_cell_2
    p_coords_dreg_v_3_lat_dsl = pos_dreg_vert_c_3_x * pn_cell_2 + pos_dreg_vert_c_3_y * dn_cell_2
    p_coords_dreg_v_4_lat_dsl = pos_dreg_vert_c_4_x * pn_cell_2 + pos_dreg_vert_c_4_y * dn_cell_2

    return (
        p_cell_idx,
        p_cell_rel_idx_dsl,
        p_cell_blk,
        p_coords_dreg_v_1_lon_dsl,
        p_coords_dreg_v_2_lon_dsl,
        p_coords_dreg_v_3_lon_dsl,
        p_coords_dreg_v_4_lon_dsl,
        p_coords_dreg_v_1_lat_dsl,
        p_coords_dreg_v_2_lat_dsl,
        p_coords_dreg_v_3_lat_dsl,
        p_coords_dreg_v_4_lat_dsl,
    )


def test_btraj_dreg_stencil_03(backend):
    grid = SimpleGrid()

    p_vn = random_field(grid, EdgeDim, KDim)
    p_vt = random_field(grid, EdgeDim, KDim)
    cell_idx = np.asarray(grid.connectivities[E2CDim], dtype=int32)
    cell_idx_new = numpy_to_1D_sparse_field(cell_idx, ECDim)
    cell_blk = constant_field(grid, 1, EdgeDim, E2CDim, dtype=int32)
    cell_blk_new = as_1D_sparse_field(cell_blk, ECDim)

    edge_verts_1_x = random_field(grid, EdgeDim)
    edge_verts_2_x = random_field(grid, EdgeDim)
    edge_verts_1_y = random_field(grid, EdgeDim)
    edge_verts_2_y = random_field(grid, EdgeDim)
    pos_on_tplane_e_1_x = random_field(grid, EdgeDim)
    pos_on_tplane_e_2_x = random_field(grid, EdgeDim)
    pos_on_tplane_e_1_y = random_field(grid, EdgeDim)
    pos_on_tplane_e_2_y = random_field(grid, EdgeDim)
    primal_normal_cell_x = random_field(grid, EdgeDim, E2CDim)
    primal_normal_cell_x_new = as_1D_sparse_field(primal_normal_cell_x, ECDim)
    dual_normal_cell_x = random_field(grid, EdgeDim, E2CDim)
    dual_normal_cell_x_new = as_1D_sparse_field(dual_normal_cell_x, ECDim)
    primal_normal_cell_y = random_field(grid, EdgeDim, E2CDim)
    primal_normal_cell_y_new = as_1D_sparse_field(primal_normal_cell_y, ECDim)
    dual_normal_cell_y = random_field(grid, EdgeDim, E2CDim)
    dual_normal_cell_y_new = as_1D_sparse_field(dual_normal_cell_y, ECDim)
    lvn_sys_pos = constant_field(grid, True, EdgeDim, KDim, dtype=bool)
    p_dt = 2.0
    p_cell_idx = constant_field(grid, 0, EdgeDim, KDim, dtype=int32)
    p_cell_rel_idx_dsl = constant_field(grid, 0, EdgeDim, KDim, dtype=int32)
    p_cell_blk = constant_field(grid, 0, EdgeDim, KDim, dtype=int32)
    p_coords_dreg_v_1_lon_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_2_lon_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_3_lon_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_4_lon_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_1_lat_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_2_lat_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_3_lat_dsl = random_field(grid, EdgeDim, KDim)
    p_coords_dreg_v_4_lat_dsl = random_field(grid, EdgeDim, KDim)

    (
        p_cell_idx_ref,
        p_cell_rel_idx_dsl_ref,
        p_cell_blk_ref,
        p_coords_dreg_v_1_lon_dsl_ref,
        p_coords_dreg_v_2_lon_dsl_ref,
        p_coords_dreg_v_3_lon_dsl_ref,
        p_coords_dreg_v_4_lon_dsl_ref,
        p_coords_dreg_v_1_lat_dsl_ref,
        p_coords_dreg_v_2_lat_dsl_ref,
        p_coords_dreg_v_3_lat_dsl_ref,
        p_coords_dreg_v_4_lat_dsl_ref,
    ) = btraj_dreg_stencil_03_numpy(
        p_vn.asnumpy(),
        p_vt.asnumpy(),
        cell_idx,
        cell_blk.asnumpy(),
        edge_verts_1_x.asnumpy(),
        edge_verts_2_x.asnumpy(),
        edge_verts_1_y.asnumpy(),
        edge_verts_2_y.asnumpy(),
        pos_on_tplane_e_1_x.asnumpy(),
        pos_on_tplane_e_2_x.asnumpy(),
        pos_on_tplane_e_1_y.asnumpy(),
        pos_on_tplane_e_2_y.asnumpy(),
        primal_normal_cell_x.asnumpy(),
        primal_normal_cell_y.asnumpy(),
        dual_normal_cell_x.asnumpy(),
        dual_normal_cell_y.asnumpy(),
        lvn_sys_pos.asnumpy(),
        p_dt,
    )

    btraj_dreg_stencil_03.with_backend(backend)(
        p_vn,
        p_vt,
        cell_idx_new,
        cell_blk_new,
        edge_verts_1_x,
        edge_verts_2_x,
        edge_verts_1_y,
        edge_verts_2_y,
        pos_on_tplane_e_1_x,
        pos_on_tplane_e_2_x,
        pos_on_tplane_e_1_y,
        pos_on_tplane_e_2_y,
        primal_normal_cell_x_new,
        primal_normal_cell_y_new,
        dual_normal_cell_x_new,
        dual_normal_cell_y_new,
        lvn_sys_pos,
        p_dt,
        p_cell_idx,
        p_cell_rel_idx_dsl,
        p_cell_blk,
        p_coords_dreg_v_1_lon_dsl,
        p_coords_dreg_v_2_lon_dsl,
        p_coords_dreg_v_3_lon_dsl,
        p_coords_dreg_v_4_lon_dsl,
        p_coords_dreg_v_1_lat_dsl,
        p_coords_dreg_v_2_lat_dsl,
        p_coords_dreg_v_3_lat_dsl,
        p_coords_dreg_v_4_lat_dsl,
        offset_provider={
            "E2C": grid.get_offset_provider("E2C"),
            "E2EC": grid.get_offset_provider("E2EC"),
        },
    )
    assert np.allclose(p_cell_idx.asnumpy(), p_cell_idx_ref)
    assert np.allclose(p_cell_rel_idx_dsl.asnumpy(), p_cell_rel_idx_dsl_ref)
    assert np.allclose(p_cell_blk.asnumpy(), p_cell_blk_ref)
    assert np.allclose(p_coords_dreg_v_1_lon_dsl.asnumpy(), p_coords_dreg_v_1_lon_dsl_ref)
    assert np.allclose(p_coords_dreg_v_2_lon_dsl.asnumpy(), p_coords_dreg_v_2_lon_dsl_ref)
    assert np.allclose(p_coords_dreg_v_3_lon_dsl.asnumpy(), p_coords_dreg_v_3_lon_dsl_ref)
    assert np.allclose(p_coords_dreg_v_4_lon_dsl.asnumpy(), p_coords_dreg_v_4_lon_dsl_ref)
    assert np.allclose(p_coords_dreg_v_1_lat_dsl.asnumpy(), p_coords_dreg_v_1_lat_dsl_ref)
    assert np.allclose(p_coords_dreg_v_2_lat_dsl.asnumpy(), p_coords_dreg_v_2_lat_dsl_ref)
    assert np.allclose(p_coords_dreg_v_3_lat_dsl.asnumpy(), p_coords_dreg_v_3_lat_dsl_ref)
    assert np.allclose(p_coords_dreg_v_4_lat_dsl.ndarray, p_coords_dreg_v_4_lat_dsl_ref)
