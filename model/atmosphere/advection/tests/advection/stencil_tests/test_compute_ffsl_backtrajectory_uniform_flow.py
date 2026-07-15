# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-form validation of compute_ffsl_backtrajectory on uniform flow.

Runs the stencil on the synthetic equilateral torus patch (SimpleGrid has no
coordinates, so the geometry fields are built from the patch) with constant
p_vn = C > 0, p_vt = 0 and the init-time torus geometry from
weno_least_squares. This validates the geometry AND the stencil together
against paper values derived from the btraj_dreg semantics
(mo_advection_traj.f90 363-773):

- vn >= 0 selects cell 1 as the upwind cell (f90 678-686), so
  p_cell_rel_idx_dsl = 0 and p_cell_idx = E2C[0].
- The departure region is the edge swept upstream by C*dt along the edge
  normal: arrival points are the two edge vertices, departure points are
  those vertices shifted by -C*dt along the primal normal (f90 692-701; the
  tangential shift vanishes for vt = 0).
- The outputs are these four points relative to the upwind cell circumcenter
  (f90 706-731), rotated back to the global frame (f90 739-757; on the plane
  torus primal/dual_normal_cell are the global primal/dual normal
  components, so the rotation from the edge-local frame is exact).
- With lcounterclock=.TRUE. (the miura3 call, mo_advection_hflux.f90
  2260-2265), lvn_sys_pos = vn * tangent_orientation >= 0 orders the
  vertices counterclockwise: (arrival 1, departure 1, departure 2,
  arrival 2) if lvn_sys_pos else (arrival 1, arrival 2, departure 2,
  departure 1) (f90 719-731).

The expected coordinates are computed from the UNWRAPPED patch triangles
(cell-local vertex positions and normals only), independently of the
edge-local-frame arrays produced by Deliverable 3, so agreement pins down
both the frame conventions and the signs. Two closed-form consequences are
asserted on top: the vertex loop is counterclockwise (positive shoelace
area) and the parallelogram area equals edge_length * C * dt.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection import weno_least_squares as weno
from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory import (
    compute_ffsl_backtrajectory,
)
from icon4py.model.common import dimension as dims

# fixture
from icon4py.model.testing.fixtures.datatest import backend

from .. import utils


# uniform normal velocity [m/s], time step [s]: backtrajectory length 0.3 edge lengths
C_VN = 0.3
DT = 1.0
NLEV = 3


@pytest.fixture(scope="module")
def torus_patch() -> utils.TorusPatch:
    return utils.build_torus_patch()


@pytest.mark.level("unit")
def test_compute_ffsl_backtrajectory_uniform_flow(torus_patch, backend):
    n_edges = torus_patch.e2c.shape[0]
    (
        pos_on_tplane_e_x,
        pos_on_tplane_e_y,
        edge_verts_x,
        edge_verts_y,
    ) = weno.compute_ffsl_backtrajectory_geometry_torus(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        vertex_x=torus_patch.vertex_x,
        vertex_y=torus_patch.vertex_y,
        edge_center_x=torus_patch.edge_center_x,
        edge_center_y=torus_patch.edge_center_y,
        primal_normal_x=torus_patch.primal_normal_x,
        primal_normal_y=torus_patch.primal_normal_y,
        dual_normal_x=torus_patch.dual_normal_x,
        dual_normal_y=torus_patch.dual_normal_y,
        e2c=torus_patch.e2c,
        e2v=torus_patch.e2v,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )

    def edge_field(values: np.ndarray) -> gtx.Field:
        return gtx.as_field((dims.EdgeDim,), values, allocator=backend)

    def edge_k_field(values: np.ndarray, dtype=None) -> gtx.Field:
        data = np.broadcast_to(values[:, np.newaxis], (n_edges, NLEV)).copy()
        if dtype is not None:
            data = data.astype(dtype)
        return gtx.as_field((dims.EdgeDim, dims.KDim), data, allocator=backend)

    def edge_e2c_field(values: np.ndarray, dtype=None) -> gtx.Field:
        data = values if dtype is None else values.astype(dtype)
        return gtx.as_field((dims.EdgeDim, dims.E2CDim), data, allocator=backend)

    # on the plane torus the cell-local frames equal the global frame, so
    # primal/dual_normal_cell are the per-edge normals on both E2C slots (cvec2gvec,
    # iconmath mo_math_utilities.f90 343-346)
    def both_cells(per_edge: np.ndarray) -> np.ndarray:
        return np.repeat(per_edge[:, np.newaxis], 2, axis=1)

    # lvn_sys_pos = p_vn * tangent_orientation >= 0 for lcounterclock=.TRUE.
    # (f90 527-537); with p_vn = C > 0 it reduces to tangent_orientation > 0
    lvn_sys_pos = C_VN * torus_patch.tangent_orientation >= 0.0

    outputs = {
        name: edge_k_field(np.zeros(n_edges), dtype=dtype)
        for name, dtype in [
            ("p_cell_idx", np.int32),
            ("p_cell_rel_idx_dsl", np.int32),
            ("p_cell_blk", np.int32),
            ("p_coords_dreg_v_1_lon_dsl", None),
            ("p_coords_dreg_v_2_lon_dsl", None),
            ("p_coords_dreg_v_3_lon_dsl", None),
            ("p_coords_dreg_v_4_lon_dsl", None),
            ("p_coords_dreg_v_1_lat_dsl", None),
            ("p_coords_dreg_v_2_lat_dsl", None),
            ("p_coords_dreg_v_3_lat_dsl", None),
            ("p_coords_dreg_v_4_lat_dsl", None),
        ]
    }
    compute_ffsl_backtrajectory.with_backend(backend)(
        p_vn=edge_k_field(np.full(n_edges, C_VN)),
        p_vt=edge_k_field(np.zeros(n_edges)),
        cell_idx=edge_e2c_field(torus_patch.e2c, dtype=np.int32),
        cell_blk=edge_e2c_field(np.zeros((n_edges, 2)), dtype=np.int32),
        edge_verts_1_x=edge_field(edge_verts_x[:, 0]),
        edge_verts_2_x=edge_field(edge_verts_x[:, 1]),
        edge_verts_1_y=edge_field(edge_verts_y[:, 0]),
        edge_verts_2_y=edge_field(edge_verts_y[:, 1]),
        pos_on_tplane_e_1_x=edge_field(pos_on_tplane_e_x[:, 0]),
        pos_on_tplane_e_2_x=edge_field(pos_on_tplane_e_x[:, 1]),
        pos_on_tplane_e_1_y=edge_field(pos_on_tplane_e_y[:, 0]),
        pos_on_tplane_e_2_y=edge_field(pos_on_tplane_e_y[:, 1]),
        primal_normal_cell_x=edge_e2c_field(both_cells(torus_patch.primal_normal_x)),
        primal_normal_cell_y=edge_e2c_field(both_cells(torus_patch.primal_normal_y)),
        dual_normal_cell_x=edge_e2c_field(both_cells(torus_patch.dual_normal_x)),
        dual_normal_cell_y=edge_e2c_field(both_cells(torus_patch.dual_normal_y)),
        lvn_sys_pos=edge_k_field(lvn_sys_pos, dtype=bool),
        p_dt=DT,
        **outputs,
        horizontal_start=0,
        horizontal_end=gtx.int32(n_edges),
        vertical_start=0,
        vertical_end=gtx.int32(NLEV),
        # only sparse-dimension accesses, but gtfn still asks for the connectivity
        offset_provider={
            "E2C": gtx.as_connectivity(
                (dims.EdgeDim, dims.E2CDim),
                dims.CellDim,
                data=torus_patch.e2c,
                dtype=gtx.int32,
                allocator=backend,
            )
        },
    )

    # closed form from the unwrapped patch triangles: for each edge the two arrival
    # points are the upwind cell's local vertex positions matching E2V (centroid ==
    # circumcenter for equilateral triangles), the departure points sit C*dt upstream
    # along the primal normal
    normal = np.stack((torus_patch.primal_normal_x, torus_patch.primal_normal_y), axis=1)
    arrival = np.empty((n_edges, 2, 2))
    for e in range(n_edges):
        upwind_cell = torus_patch.e2c[e, 0]
        cell_vertex_ids = torus_patch.c2v[upwind_cell].tolist()
        for nv in range(2):
            arrival[e, nv] = torus_patch.local_vertices[
                upwind_cell, cell_vertex_ids.index(torus_patch.e2v[e, nv])
            ]
    departure = arrival - C_VN * DT * normal[:, np.newaxis, :]

    # counterclockwise numbering (f90 719-731)
    expected = np.empty((n_edges, 4, 2))
    expected[:, 0] = arrival[:, 0]
    expected[:, 1] = np.where(lvn_sys_pos[:, np.newaxis], departure[:, 0], arrival[:, 1])
    expected[:, 2] = departure[:, 1]
    expected[:, 3] = np.where(lvn_sys_pos[:, np.newaxis], arrival[:, 1], departure[:, 0])

    assert np.all(outputs["p_cell_rel_idx_dsl"].asnumpy() == 0)
    assert np.all(outputs["p_cell_idx"].asnumpy() == torus_patch.e2c[:, 0:1])
    for k, comp in [(0, "lon"), (1, "lat")]:
        for nv in range(4):
            actual = outputs[f"p_coords_dreg_v_{nv + 1}_{comp}_dsl"].asnumpy()
            np.testing.assert_allclose(
                actual,
                np.broadcast_to(expected[:, nv, k : k + 1], (n_edges, NLEV)),
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"departure region vertex {nv + 1}, component {comp}",
            )

    # closed-form consequences, asserted on the stencil output at level 0: the vertex
    # loop is counterclockwise (positive shoelace area) and the parallelogram area is
    # edge_length * C * dt
    x = np.stack(
        [outputs[f"p_coords_dreg_v_{nv + 1}_lon_dsl"].asnumpy()[:, 0] for nv in range(4)], axis=1
    )
    y = np.stack(
        [outputs[f"p_coords_dreg_v_{nv + 1}_lat_dsl"].asnumpy()[:, 0] for nv in range(4)], axis=1
    )
    signed_area = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    np.testing.assert_allclose(
        signed_area, torus_patch.edge_length * C_VN * DT, rtol=1e-12, atol=1e-14
    )
