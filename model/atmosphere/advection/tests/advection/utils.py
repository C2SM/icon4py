# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import logging

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import serialbox as sb, test_utils


log = logging.getLogger(__name__)


def construct_interpolation_state(
    savepoint: sb.InterpolationSavepoint, backend: gtx_typing.Backend | None
) -> advection_states.AdvectionInterpolationState:
    return advection_states.AdvectionInterpolationState(
        geofac_div=savepoint.geofac_div(),
        rbf_vec_coeff_e=savepoint.rbf_vec_coeff_e(),
        pos_on_tplane_e_1=savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=savepoint.pos_on_tplane_e_y(),
    )


def construct_least_squares_state(
    least_squares_coeffs: data_alloc.NDArray, backend: gtx_typing.Backend | None
) -> advection_states.AdvectionLeastSquaresState:
    return advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=gtx.as_field(
            (dims.CellDim, dims.C2E2CDim),
            least_squares_coeffs[:, 0, :],
            allocator=backend,
        ),
        lsq_pseudoinv_2=gtx.as_field(
            (dims.CellDim, dims.C2E2CDim),
            least_squares_coeffs[:, 1, :],
            allocator=backend,
        ),
    )


def construct_metric_state(
    icon_grid, savepoint: sb.MetricSavepoint, backend: gtx_typing.Backend | None
) -> advection_states.AdvectionMetricState:
    constant_f = data_alloc.constant_field(icon_grid, 1.0, dims.KDim, allocator=backend)
    ddqz_z_full_np = np.reciprocal(savepoint.inv_ddqz_z_full().asnumpy())
    return advection_states.AdvectionMetricState(
        deepatmo_divh=constant_f,
        deepatmo_divzl=constant_f,
        deepatmo_divzu=constant_f,
        ddqz_z_full=gtx.as_field((dims.CellDim, dims.KDim), ddqz_z_full_np, allocator=backend),
    )


def construct_diagnostic_init_state(
    icon_grid,
    savepoint: sb.AdvectionInitSavepoint,
    ntracer: int,
    backend: gtx_typing.Backend | None,
) -> advection_states.AdvectionDiagnosticState:
    return advection_states.AdvectionDiagnosticState(
        airmass_now=savepoint.airmass_now(),
        airmass_new=savepoint.airmass_new(),
        grf_tend_tracer=savepoint.grf_tend_tracer(ntracer),
        hfl_tracer=data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, allocator=backend),
        vfl_tracer=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        ),
    )


def construct_diagnostic_exit_state(
    icon_grid,
    savepoint: sb.AdvectionExitSavepoint,
    ntracer: int,
    backend: gtx_typing.Backend | None,
) -> advection_states.AdvectionDiagnosticState:
    return advection_states.AdvectionDiagnosticState(
        airmass_now=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend),
        airmass_new=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend),
        grf_tend_tracer=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim),
        hfl_tracer=savepoint.hfl_tracer(ntracer),
        vfl_tracer=savepoint.vfl_tracer(ntracer),
    )


def construct_prep_adv(
    savepoint: sb.AdvectionInitSavepoint,
) -> advection_states.AdvectionPrepAdvState:
    return advection_states.AdvectionPrepAdvState(
        vn_traj=savepoint.vn_traj(),
        mass_flx_me=savepoint.mass_flx_me(),
        mass_flx_ic=savepoint.mass_flx_ic(),
    )


def log_dbg(field, name=""):
    log.debug(f"{name}: min={field.min()}, max={field.max()}, mean={field.mean()}")


def log_serialized(
    diagnostic_state: advection_states.AdvectionDiagnosticState,
    prep_adv: advection_states.AdvectionPrepAdvState,
    p_tracer_now: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
):
    log_dbg(diagnostic_state.airmass_now.asnumpy(), "airmass_now")
    log_dbg(diagnostic_state.airmass_new.asnumpy(), "airmass_new")
    log_dbg(diagnostic_state.grf_tend_tracer.asnumpy(), "grf_tend_tracer")
    log_dbg(prep_adv.vn_traj.asnumpy(), "vn_traj")
    log_dbg(prep_adv.mass_flx_me.asnumpy(), "mass_flx_me")
    log_dbg(prep_adv.mass_flx_ic.asnumpy(), "mass_flx_ic")
    log_dbg(p_tracer_now.asnumpy(), "p_tracer_now")
    log.debug(f"dtime: {dtime}")


def verify_advection_fields(
    *,
    grid: icon_grid.IconGrid,
    diagnostic_state: advection_states.AdvectionDiagnosticState,
    diagnostic_state_ref: advection_states.AdvectionDiagnosticState,
    p_tracer_new: fa.CellKField[ta.wpfloat],
    p_tracer_new_ref: fa.CellKField[ta.wpfloat],
    even_timestep: bool,
):
    # cell indices
    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
    start_cell_lateral_boundary_level_2 = grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
    end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    # edge indices
    edge_domain = h_grid.domain(dims.EdgeDim)
    start_edge_lateral_boundary_level_5 = grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )
    end_edge_halo = grid.end_index(edge_domain(h_grid.Zone.HALO))

    hfl_tracer_range = np.arange(start_edge_lateral_boundary_level_5, end_edge_halo)
    vfl_tracer_range = (
        np.arange(start_cell_lateral_boundary_level_2, end_cell_end)
        if even_timestep
        else np.arange(start_cell_nudging, end_cell_local)
    )
    p_tracer_new_range = np.arange(start_cell_lateral_boundary, end_cell_local)

    # log advection output fields
    log_dbg(diagnostic_state.hfl_tracer.asnumpy()[hfl_tracer_range, :], "hfl_tracer")
    log_dbg(diagnostic_state_ref.hfl_tracer.asnumpy()[hfl_tracer_range, :], "hfl_tracer_ref")
    log_dbg(diagnostic_state.vfl_tracer.asnumpy()[vfl_tracer_range, :], "vfl_tracer")
    log_dbg(diagnostic_state_ref.vfl_tracer.asnumpy()[vfl_tracer_range, :], "vfl_tracer_ref")
    log_dbg(p_tracer_new.asnumpy()[p_tracer_new_range, :], "p_tracer_new")
    log_dbg(p_tracer_new_ref.asnumpy()[p_tracer_new_range, :], "p_tracer_new_ref")

    # verify advection output fields
    assert test_utils.dallclose(
        diagnostic_state.hfl_tracer.asnumpy()[hfl_tracer_range, :],
        diagnostic_state_ref.hfl_tracer.asnumpy()[hfl_tracer_range, :],
        rtol=1e-10,
        atol=1e-11,
    )
    assert test_utils.dallclose(
        diagnostic_state.vfl_tracer.asnumpy()[vfl_tracer_range, :],
        diagnostic_state_ref.vfl_tracer.asnumpy()[vfl_tracer_range, :],
        rtol=1e-10,
    )
    assert test_utils.dallclose(
        p_tracer_new.asnumpy()[p_tracer_new_range, :],
        p_tracer_new_ref.asnumpy()[p_tracer_new_range, :],
        atol=1e-16,
    )


# ---- synthetic periodic torus patch (pure numpy, no grid files) ----


@dataclasses.dataclass(frozen=True)
class TorusPatch:
    edge_length: float
    domain_length: float
    domain_height: float
    vertex_x: np.ndarray
    vertex_y: np.ndarray
    c2v: np.ndarray
    c2e2c: np.ndarray
    c2e2c2e2c: np.ndarray
    cell_center_x: np.ndarray
    cell_center_y: np.ndarray
    # unwrapped vertex coordinates relative to the cell center (centroid), (n_cells, 3, 2)
    local_vertices: np.ndarray
    # edges: E2V ordered by ascending vertex id, E2C by ascending cell id; the primal
    # normal points from cell 1 to cell 2 and the tangent (dual normal) is
    # tangent_orientation * normalize(v2 - v1), mirroring the ICON conventions
    # (mo_advection_traj.f90 660-673 and the icon4py torus geometry stencils)
    e2v: np.ndarray
    e2c: np.ndarray
    c2e: np.ndarray
    edge_center_x: np.ndarray
    edge_center_y: np.ndarray
    primal_normal_x: np.ndarray
    primal_normal_y: np.ndarray
    dual_normal_x: np.ndarray
    dual_normal_y: np.ndarray
    tangent_orientation: np.ndarray


def build_torus_patch(nx: int = 8, ny: int = 8, edge_length: float = 1.0) -> TorusPatch:
    """Periodic equilateral-triangle torus: nx*ny quads, each split into 2 triangles.

    Vertex rows are offset by half an edge length alternately (ny must be even
    for periodicity in y). Cell centers are the triangle centroids.
    """
    assert ny % 2 == 0
    dx = edge_length
    dy = edge_length * np.sqrt(3.0) / 2.0
    domain_length = nx * dx
    domain_height = ny * dy

    def vertex_id(i: int, j: int) -> int:
        return (j % ny) * nx + (i % nx)

    def vertex_coord(i: int, j: int) -> tuple[float, float]:
        # unwrapped coordinates; consistent across the periodic seam for even ny
        return ((i + 0.5 * (j % 2)) * dx, j * dy)

    vertex_x = np.array([vertex_coord(i, j)[0] for j in range(ny) for i in range(nx)])
    vertex_y = np.array([vertex_coord(i, j)[1] for j in range(ny) for i in range(nx)])

    triangles = []  # (i, j) vertex index pairs, unwrapped
    for j in range(ny):
        for i in range(nx):
            if j % 2 == 0:
                # row j not offset, row j+1 offset by +dx/2
                triangles.append([(i, j), (i + 1, j), (i, j + 1)])  # up
                triangles.append([(i + 1, j), (i, j + 1), (i + 1, j + 1)])  # down
            else:
                # row j offset by +dx/2, row j+1 not offset
                triangles.append([(i, j), (i + 1, j), (i + 1, j + 1)])  # up
                triangles.append([(i, j), (i, j + 1), (i + 1, j + 1)])  # down
    n_cells = len(triangles)

    c2v = np.array([[vertex_id(i, j) for (i, j) in tri] for tri in triangles], dtype=np.int32)
    coords = np.array([[vertex_coord(i, j) for (i, j) in tri] for tri in triangles])
    centers = coords.mean(axis=1)
    local_vertices = coords - centers[:, np.newaxis, :]
    cell_center_x = centers[:, 0] % domain_length
    cell_center_y = centers[:, 1] % domain_height

    # neighbors share exactly two vertices; brute force is fine at this size
    vertex_sets = [frozenset(row) for row in c2v.tolist()]
    c2e2c = np.array(
        [
            [d for d in range(n_cells) if d != c and len(vertex_sets[c] & vertex_sets[d]) == 2]
            for c in range(n_cells)
        ],
        dtype=np.int32,
    )
    assert c2e2c.shape == (n_cells, 3)
    # grid_manager-style butterfly table: c2e2c[c2e2c] (center cell 3x + 6 outer cells)
    c2e2c2e2c = c2e2c[c2e2c].reshape(n_cells, 9)

    edges = _build_patch_edges(
        c2v=c2v,
        coords=coords,
        centers=centers,
        domain_length=domain_length,
        domain_height=domain_height,
    )

    return TorusPatch(
        edge_length=edge_length,
        domain_length=domain_length,
        domain_height=domain_height,
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        c2v=c2v,
        c2e2c=c2e2c,
        c2e2c2e2c=c2e2c2e2c,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        local_vertices=local_vertices,
        **edges,
    )


def _closest_image(ref: np.ndarray, value: np.ndarray, period: float) -> np.ndarray:
    # minimal-image wrap of value as seen from ref
    delta = value - ref
    return ref + delta - period * np.round(delta / period)


def _build_patch_edges(
    *,
    c2v: np.ndarray,
    coords: np.ndarray,
    centers: np.ndarray,
    domain_length: float,
    domain_height: float,
) -> dict:
    """Edge tables and edge-local frames for the torus patch.

    E2V is ordered by ascending vertex id and E2C by ascending cell id; the
    tangent orientation is then derived from those orderings exactly as the
    grid generator defines it: +1 if ((v2 - v1) x (c2 - c1)) points along +z,
    else -1. The stored tangent (ICON dual normal) is
    tangent_orientation * normalize(v2 - v1) and the primal normal is the
    tangent rotated by +90 degrees (icon4py torus geometry,
    cartesian_coordinates_of_edge_tangent/normal_torus), which always points
    from cell 1 to cell 2 (asserted below).
    """
    n_cells = c2v.shape[0]
    # edge key (vertex id pair, ascending) -> list of (cell, unwrapped v1/v2 coordinates)
    edge_cells: dict = {}
    for c in range(n_cells):
        for k in range(3):
            va, vb = int(c2v[c, k]), int(c2v[c, (k + 1) % 3])
            pa, pb = coords[c, k], coords[c, (k + 1) % 3]
            if vb < va:
                va, vb, pa, pb = vb, va, pb, pa
            edge_cells.setdefault((va, vb), []).append((c, pa, pb))
    assert all(len(v) == 2 for v in edge_cells.values())

    n_edges = len(edge_cells)  # == 3 * n_cells / 2
    e2v = np.empty((n_edges, 2), dtype=np.int32)
    e2c = np.empty((n_edges, 2), dtype=np.int32)
    edge_center_x = np.empty(n_edges)
    edge_center_y = np.empty(n_edges)
    primal_normal = np.empty((n_edges, 2))
    dual_normal = np.empty((n_edges, 2))
    orientation = np.empty(n_edges)

    period = np.array([domain_length, domain_height])
    for e, ((va, vb), cells) in enumerate(sorted(edge_cells.items())):
        (c1, pa, pb), (c2, _, _) = sorted(cells)
        e2v[e] = (va, vb)
        e2c[e] = (c1, c2)
        # unwrapped edge midpoint and tangent direction from cell 1's frame
        midpoint = 0.5 * (pa + pb)
        tangent = (pb - pa) / np.linalg.norm(pb - pa)
        # cell centers moved to the closest periodic image of the edge midpoint
        center_1 = _closest_image(midpoint, centers[c1], period)
        center_2 = _closest_image(midpoint, centers[c2], period)
        cell_diff = center_2 - center_1
        orientation[e] = np.sign(tangent[0] * cell_diff[1] - tangent[1] * cell_diff[0])
        dual_normal[e] = orientation[e] * tangent
        primal_normal[e] = (-dual_normal[e, 1], dual_normal[e, 0])  # rotate by +90 degrees
        assert np.dot(primal_normal[e], cell_diff) > 0.0
        edge_center_x[e] = midpoint[0] % domain_length
        edge_center_y[e] = midpoint[1] % domain_height

    # C2E by inverting E2C (ascending edge id; the order is irrelevant to the consumers)
    c2e_lists: list[list[int]] = [[] for _ in range(n_cells)]
    for e in range(n_edges):
        for c in e2c[e]:
            c2e_lists[c].append(e)
    assert all(len(edges) == 3 for edges in c2e_lists)
    c2e = np.asarray(c2e_lists, dtype=np.int32)

    return dict(
        e2v=e2v,
        e2c=e2c,
        c2e=c2e,
        edge_center_x=edge_center_x,
        edge_center_y=edge_center_y,
        primal_normal_x=primal_normal[:, 0],
        primal_normal_y=primal_normal[:, 1],
        dual_normal_x=dual_normal[:, 0],
        dual_normal_y=dual_normal[:, 1],
        tangent_orientation=orientation,
    )
