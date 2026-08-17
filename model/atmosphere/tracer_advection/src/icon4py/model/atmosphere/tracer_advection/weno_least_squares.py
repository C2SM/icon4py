# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Init-time WENO least-squares coefficient machinery for the miura_weno schemes.

Port of the torus branch of the candidate least-squares setup from ICON
(mo_intp_coeffs_lsq_bln.f90, icon-exclaim branch transport_ajocksch): 9-point
stencil construction, torus moments, and the 27 (quadratic) / 3 (linear)
candidate pseudoinverses. All Fortran line references below are to that file
unless stated otherwise. Also hosts the init-time torus geometry consumed by
the miura3 FFSL backtrajectory ('compute_ffsl_backtrajectory').

Pure init-time numpy/cupy code (no gt4py); assumes a boundary-free torus grid
on a single rank. Unknowns are ordered [x, y, x^2, y^2, xy] (f90 1986-1996).
"""

from typing import Final

import numpy as np

import icon4py.model.common.type_alias as ta
from icon4py.model.common.interpolation import interpolation_fields
from icon4py.model.common.utils import data_allocation as data_alloc


# quadratic lsq_high configuration (mo_interpol_config.f90, lsq_high_ord=2)
LSQ_DIM_C_QUADRATIC: Final[int] = 9
LSQ_DIM_UNK_QUADRATIC: Final[int] = 5
LSQ_WGT_EXP_QUADRATIC: Final[int] = 5

# ICON's lsq_high_set for lsq_high_ord=3 (mo_interpol_config.f90:377-380): 9 unknowns over the
# same 9-point stencil, so the system is square rather than overdetermined, and unweighted.
LSQ_DIM_C_CUBIC: Final[int] = 9
LSQ_DIM_UNK_CUBIC: Final[int] = 9
LSQ_WGT_EXP_CUBIC: Final[int] = 0

# Zero patterns of the 27 quadratic candidate weight sets (f90 1719-1835): after
# the `do i = 1, 27` reset to the full distance weights (1719-1721), candidates
# 4-27 (1-based) zero exactly 4 of the 9 stencil positions (1735-1835);
# candidates 1-3 keep the full weights. 0-based candidate and position indices;
# position order within each tuple follows the Fortran statement order.
CANDIDATE_ZERO_PATTERNS_QUADRATIC: Final[tuple[tuple[int, ...], ...]] = (
    (),  # cand 1
    (),  # cand 2
    (),  # cand 3
    (2, 4, 5, 7),  # cand 4:  positions 3, 5, 6, 8
    (5, 7, 8, 1),  # cand 5:  positions 6, 8, 9, 2
    (8, 1, 2, 4),  # cand 6:  positions 9, 2, 3, 5
    (2, 5, 7, 8),  # cand 7:  positions 3, 6, 8, 9
    (5, 8, 1, 2),  # cand 8:  positions 6, 9, 2, 3
    (8, 2, 4, 5),  # cand 9:  positions 9, 3, 5, 6
    (4, 7, 8, 1),  # cand 10: positions 5, 8, 9, 2
    (7, 1, 2, 4),  # cand 11: positions 8, 2, 3, 5
    (1, 4, 5, 7),  # cand 12: positions 2, 5, 6, 8
    (1, 2, 5, 7),  # cand 13: positions 2, 3, 6, 8
    (2, 4, 7, 8),  # cand 14: positions 3, 5, 8, 9
    (4, 5, 8, 1),  # cand 15: positions 5, 6, 9, 2
    (0, 1, 2, 4),  # cand 16: positions 1, 2, 3, 5
    (0, 1, 2, 8),  # cand 17: positions 1, 2, 3, 9
    (3, 4, 5, 7),  # cand 18: positions 4, 5, 6, 8
    (3, 4, 5, 2),  # cand 19: positions 4, 5, 6, 3
    (6, 7, 8, 1),  # cand 20: positions 7, 8, 9, 2
    (6, 7, 8, 5),  # cand 21: positions 7, 8, 9, 6
    (0, 1, 2, 5),  # cand 22: positions 1, 2, 3, 6
    (0, 1, 2, 7),  # cand 23: positions 1, 2, 3, 8
    (3, 4, 5, 8),  # cand 24: positions 4, 5, 6, 9
    (3, 4, 5, 1),  # cand 25: positions 4, 5, 6, 2
    (6, 7, 8, 2),  # cand 26: positions 7, 8, 9, 3
    (6, 7, 8, 4),  # cand 27: positions 7, 8, 9, 5
)

# Live l_weights_s values (f90 2590-2646): slots 1-3 = 1, slots 4-21 = 0,
# slots 22-27 = 2.991549980478795 (1-based).
L_WEIGHTS_S: Final[np.ndarray] = np.array(
    [1.0] * 3 + [0.0] * 18 + [2.991549980478795] * 6, dtype=ta.wpfloat
)


def _plane_torus_closest_coordinates(
    v0: data_alloc.NDArray, v1: data_alloc.NDArray, period: float
) -> data_alloc.NDArray:
    # port of plane_torus_closest_coordinates (iconmath mo_math_utilities.F90,
    # 720-760), one coordinate at a time: wrap v1 by +/- period where
    # |v0 - v1| > period/2 so it lies in the closest periodic image seen from v0
    array_ns = data_alloc.array_namespace(v1)
    wrap = array_ns.abs(v0 - v1) > 0.5 * period
    return array_ns.where(wrap, array_ns.where(v0 > v1, v1 + period, v1 - period), v1)


def compute_torus_distance_vectors(
    *,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    neighbor_table: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """Distance vectors from each cell center to its neighbors' centers, (n_cells, k, 2).

    Port of f90 1599-1607: the neighbor centers are moved to their closest
    periodic image before taking the difference.
    """
    array_ns = data_alloc.array_namespace(cell_center_x)
    center_x = cell_center_x[:, array_ns.newaxis]
    center_y = cell_center_y[:, array_ns.newaxis]
    neighbor_x = _plane_torus_closest_coordinates(
        center_x, cell_center_x[neighbor_table], domain_length
    )
    neighbor_y = _plane_torus_closest_coordinates(
        center_y, cell_center_y[neighbor_table], domain_height
    )
    return array_ns.stack((neighbor_x - center_x, neighbor_y - center_y), axis=-1)


def compute_ffsl_backtrajectory_geometry_torus(
    *,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    primal_normal_y: data_alloc.NDArray,
    dual_normal_x: data_alloc.NDArray,
    dual_normal_y: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    e2v: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    """Positions of the E2C cell centers and E2V vertices in the edge-local frame.

    Port of calculate_planar_distance_at_edge (mo_intp_coeffs.f90 2278-2406):
    the separation vector from the edge midpoint to the closest periodic image
    of each neighboring cell circumcenter / edge vertex, projected onto the
    edge primal normal (x component) and dual normal, i.e. tangent
    (y component). Returns (pos_on_tplane_e_x, pos_on_tplane_e_y, edge_verts_x,
    edge_verts_y), each (n_edges, 2): ICON's pos_on_tplane_e components 1:2
    (cells) and 3:4 (vertices). Unlike the equilateral shortcut in
    interpolation_fields.compute_pos_on_tplane_e_x_y_torus this is the full
    projection, valid for any planar torus grid.

    The remaining static inputs of 'compute_ffsl_backtrajectory' need no
    torus-specific setup: primal/dual_normal_cell equal the per-edge
    primal/dual normal on both E2C slots because cvec2gvec is the identity on
    the plane torus (iconmath mo_math_utilities.f90 343-346, applied in
    complete_patchinfo, mo_intp_coeffs.f90 1743-1785) - the grid geometry
    already broadcasts EDGE_NORMAL/EDGE_TANGENT to the cell slots. lvn_sys_pos
    is velocity dependent, p_vn * tangent_orientation >= 0 for
    lcounterclock=.TRUE. (mo_advection_traj.f90 527-537), and is computed at
    runtime by 'compute_ffsl_backtrajectory_counterclockwise_indicator'.
    """
    array_ns = data_alloc.array_namespace(cell_center_x)

    def offsets_in_edge_frame(
        point_x: data_alloc.NDArray, point_y: data_alloc.NDArray
    ) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
        # f90 2331-2342 / 2348-2359: separation vector between the edge midpoint and the
        # closest periodic image of the point
        dx = (
            _plane_torus_closest_coordinates(
                edge_center_x[:, array_ns.newaxis], point_x, domain_length
            )
            - edge_center_x[:, array_ns.newaxis]
        )
        dy = (
            _plane_torus_closest_coordinates(
                edge_center_y[:, array_ns.newaxis], point_y, domain_height
            )
            - edge_center_y[:, array_ns.newaxis]
        )
        # f90 2368-2390: rotate into the local (primal normal, dual normal) system
        return (
            dx * primal_normal_x[:, array_ns.newaxis] + dy * primal_normal_y[:, array_ns.newaxis],
            dx * dual_normal_x[:, array_ns.newaxis] + dy * dual_normal_y[:, array_ns.newaxis],
        )

    pos_on_tplane_e_x, pos_on_tplane_e_y = offsets_in_edge_frame(
        cell_center_x[e2c], cell_center_y[e2c]
    )
    edge_verts_x, edge_verts_y = offsets_in_edge_frame(vertex_x[e2v], vertex_y[e2v])
    return pos_on_tplane_e_x, pos_on_tplane_e_y, edge_verts_x, edge_verts_y


def create_stencil_c9(c2e2c: data_alloc.NDArray, c2v: data_alloc.NDArray) -> data_alloc.NDArray:
    """9-point stencil in Fortran position order, (n_cells, 9).

    Port of create_stencil_c9 (f90 334-446). For each direct neighbor jec in
    0..2, position 3*jec holds C2E2C[:, jec] and positions 3*jec+1 / 3*jec+2
    hold its two non-center neighbors in the neighbor's C2E2C order (f90
    383-406), swapped (f90 407-437) if the first outer shares no vertex with
    direct neighbor (jec+1) % 3. Assumes a boundary-free torus grid.
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    if array_ns.any(c2e2c < 0) or array_ns.any(c2v < 0):
        raise ValueError(
            "Found skip values in the connectivities: 'create_stencil_c9' requires "
            "a boundary-free (torus) grid."
        )
    n_cells = c2e2c.shape[0]
    center = array_ns.arange(n_cells, dtype=c2e2c.dtype)
    stencil = array_ns.empty((n_cells, 9), dtype=c2e2c.dtype)

    # f90 383-406: direct neighbors and their non-center neighbors in table order
    for jec in range(3):
        direct = c2e2c[:, jec]
        neighbors = c2e2c[direct]
        is_center = neighbors == center[:, array_ns.newaxis]
        if not array_ns.all(array_ns.sum(is_center, axis=1) == 1):
            raise ValueError(
                f"Direct neighbor {jec} does not point back to the center cell exactly once."
            )
        center_pos = array_ns.argmax(is_center, axis=1)
        stencil[:, 3 * jec] = direct
        stencil[:, 3 * jec + 1] = array_ns.where(center_pos == 0, neighbors[:, 1], neighbors[:, 0])
        stencil[:, 3 * jec + 2] = array_ns.where(center_pos == 2, neighbors[:, 1], neighbors[:, 2])

    # f90 407-437: swap the outer pair if the first outer shares no vertex with
    # direct neighbor (jec+1) % 3
    for jec in range(3):
        next_direct = c2e2c[:, (jec + 1) % 3]
        first_outer = stencil[:, 3 * jec + 1].copy()
        second_outer = stencil[:, 3 * jec + 2].copy()
        shares_vertex = array_ns.any(
            c2v[next_direct][:, :, array_ns.newaxis] == c2v[first_outer][:, array_ns.newaxis, :],
            axis=(1, 2),
        )
        stencil[:, 3 * jec + 1] = array_ns.where(shares_vertex, first_outer, second_outer)
        stencil[:, 3 * jec + 2] = array_ns.where(shares_vertex, second_outer, first_outer)

    return stencil


def compute_lsq_moments_torus(
    *,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    c2v: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
    cubic: bool = False,
) -> data_alloc.NDArray:
    """Cell averages of the monomials, (n_cells, 5) or (n_cells, 9) when 'cubic'.

    The ordering is ICON's (mo_intp_coeffs_lsq_bln.f90:888-896), so the cubic set
    extends the quadratic one rather than reordering it:

        [x, y, x^2, y^2, xy]  +  [x^3, y^3, x^2 y, x y^2]

    Port of the torus moments block (f90 1957-2133): analytic polygon line
    integrals over the cell vertices, with the vertices moved to their closest
    periodic image relative to the cell center.
    """
    array_ns = data_alloc.array_namespace(cell_center_x)
    # f90 1962-1972: distance vectors between cell center and vertices
    vert_x = _plane_torus_closest_coordinates(
        cell_center_x[:, array_ns.newaxis], vertex_x[c2v], domain_length
    )
    vert_y = _plane_torus_closest_coordinates(
        cell_center_y[:, array_ns.newaxis], vertex_y[c2v], domain_height
    )
    dx = vert_x - cell_center_x[:, array_ns.newaxis]
    dy = vert_y - cell_center_y[:, array_ns.newaxis]
    # values at the cyclically next vertex (f90 jecp)
    dxp = array_ns.roll(dx, -1, axis=1)
    dyp = array_ns.roll(dy, -1, axis=1)
    delx = dxp - dx
    dely = dyp - dy

    # reciprocal control volume area (f90 2015-2023)
    z_rcarea = 2.0 / array_ns.sum((dxp + dx) * dely, axis=1)

    # integrands for each edge (f90 2031-2055)
    fx = dxp**2 + dxp * dx + dx**2
    fy = dyp**2 + dyp * dy + dy**2
    fxx = (dxp + dx) * (dxp**2 + dx**2)
    fyy = (dyp + dy) * (dyp**2 + dy**2)
    fxy = dyp * (3.0 * dxp**2 + 2.0 * dxp * dx + dx**2) + dy * (
        dxp**2 + 2.0 * dxp * dx + 3.0 * dx**2
    )

    # f90 2121-2133
    moments = array_ns.empty((c2v.shape[0], 9 if cubic else 5), dtype=ta.wpfloat)
    moments[:, 0] = z_rcarea / 6.0 * array_ns.sum(fx * dely, axis=1)
    moments[:, 1] = -z_rcarea / 6.0 * array_ns.sum(fy * delx, axis=1)
    moments[:, 2] = z_rcarea / 12.0 * array_ns.sum(fxx * dely, axis=1)
    moments[:, 3] = -z_rcarea / 12.0 * array_ns.sum(fyy * delx, axis=1)
    moments[:, 4] = z_rcarea / 24.0 * array_ns.sum(fxy * dely, axis=1)
    if not cubic:
        return moments

    # third-order integrands, f90 960-1014. fxxx/fyyy are written in the delx/dely form
    # the Fortran actually uses, not the algebraically equal MAPLE form it quotes beside it.
    fxxx = 5.0 * dx**4 + 10.0 * dx**3 * delx + 10.0 * dx**2 * delx**2 + 5.0 * dx * delx**3 + delx**4
    fyyy = 5.0 * dy**4 + 10.0 * dy**3 * dely + 10.0 * dy**2 * dely**2 + 5.0 * dy * dely**3 + dely**4
    fxxy = (
        4.0 * dxp**3 * dyp
        + 3.0 * dx * dxp**2 * dyp
        + 2.0 * dx**2 * dxp * dyp
        + dx**3 * dyp
        + dxp**3 * dy
        + 2.0 * dx * dxp**2 * dy
        + 3.0 * dx**2 * dxp * dy
        + 4.0 * dx**3 * dy
    )
    fxyy = (
        6.0 * dxp**2 * dyp**2
        + 3.0 * dx * dxp * dyp**2
        + dx**2 * dyp**2
        + 3.0 * dxp**2 * dy * dyp
        + 4.0 * dx * dxp * dy * dyp
        + 3.0 * dx**2 * dy * dyp
        + dxp**2 * dy**2
        + 3.0 * dx * dxp * dy**2
        + 6.0 * dx**2 * dy**2
    )

    # f90 1039-1047; note 8 and 9 both contract with dely, unlike the y-only moments above
    moments[:, 5] = z_rcarea / 20.0 * array_ns.sum(fxxx * dely, axis=1)
    moments[:, 6] = -z_rcarea / 20.0 * array_ns.sum(fyyy * delx, axis=1)
    moments[:, 7] = z_rcarea / 60.0 * array_ns.sum(fxxy * dely, axis=1)
    moments[:, 8] = z_rcarea / 60.0 * array_ns.sum(fxyy * dely, axis=1)
    return moments


def compute_lsq_moments_hat(
    *,
    lsq_moments: data_alloc.NDArray,
    stencil_c9: data_alloc.NDArray,
    z_dist: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Stencil cell moments translated to the center cell frame, (n_cells, 9, n_unknowns).

    Port of f90 2217-2241 for the quadratic set and of mo_intp_coeffs_lsq_bln.f90:1111-1170
    for the cubic one; the unknown count is taken from 'lsq_moments', so passing the cubic
    moments yields the cubic shifts. Each stencil cell's own moments are shifted by the
    torus-periodic distance vector z_dist from the center cell to that cell.
    """
    array_ns = data_alloc.array_namespace(lsq_moments)
    moments = lsq_moments[stencil_c9]  # (n_cells, 9, n_unknowns)
    dx = z_dist[..., 0]
    dy = z_dist[..., 1]
    moments_hat = array_ns.empty(moments.shape, dtype=ta.wpfloat)
    moments_hat[..., 0] = moments[..., 0] + dx
    moments_hat[..., 1] = moments[..., 1] + dy
    moments_hat[..., 2] = moments[..., 2] + 2.0 * moments[..., 0] * dx + dx**2
    moments_hat[..., 3] = moments[..., 3] + 2.0 * moments[..., 1] * dy + dy**2
    moments_hat[..., 4] = moments[..., 4] + moments[..., 0] * dy + moments[..., 1] * dx + dx * dy
    if moments.shape[-1] == 5:
        return moments_hat

    moments_hat[..., 5] = (
        moments[..., 5] + 3.0 * moments[..., 2] * dx + 3.0 * moments[..., 0] * dx**2 + dx**3
    )
    moments_hat[..., 6] = (
        moments[..., 6] + 3.0 * moments[..., 3] * dy + 3.0 * moments[..., 1] * dy**2 + dy**3
    )
    moments_hat[..., 7] = (
        moments[..., 7]
        + moments[..., 2] * dy
        + 2.0 * moments[..., 4] * dx
        + 2.0 * moments[..., 0] * dx * dy
        + moments[..., 1] * dx**2
        + dx**2 * dy
    )
    moments_hat[..., 8] = (
        moments[..., 8]
        + moments[..., 3] * dx
        + 2.0 * moments[..., 4] * dy
        + 2.0 * moments[..., 1] * dx * dy
        + moments[..., 0] * dy**2
        + dx * dy**2
    )
    return moments_hat


def compute_candidate_weights_quadratic(z_dist: data_alloc.NDArray) -> data_alloc.NDArray:
    """The 27 candidate row-weight sets for the 9-point stencil, (n_cells, 27, 9)."""
    array_ns = data_alloc.array_namespace(z_dist)
    # f90 1614-1622: 1/dist**wgt_exp
    z_norm = array_ns.sqrt(array_ns.sum(z_dist**2, axis=2))
    weights = 1.0 / z_norm**LSQ_WGT_EXP_QUADRATIC
    # f90 1719-1721: every candidate starts from the full distance weights
    candidate_weights = array_ns.repeat(weights[:, array_ns.newaxis, :], 27, axis=1)
    # f90 1735-1835: hard-coded zero patterns
    for cand, positions in enumerate(CANDIDATE_ZERO_PATTERNS_QUADRATIC):
        for pos in positions:
            candidate_weights[:, cand, pos] = 0.0
    # f90 1945-1949: per-candidate max normalization
    candidate_weights /= array_ns.max(candidate_weights, axis=2, keepdims=True)
    return candidate_weights


def _svd_pseudoinverse(
    design: data_alloc.NDArray, weights: data_alloc.NDArray
) -> data_alloc.NDArray:
    # f90 2476-2581: Moore-Penrose inverse V * 1/S * U^T of the weighted design
    # matrix; the row weights multiply the pseudoinverse columns so that it
    # applies to unweighted z_b vectors at runtime
    array_ns = data_alloc.array_namespace(design)
    u, s, v_t = array_ns.linalg.svd(design, full_matrices=False)
    pseudoinv = array_ns.matmul(
        v_t.swapaxes(-1, -2) / s[..., array_ns.newaxis, :], u.swapaxes(-1, -2)
    )
    pseudoinv *= weights[..., array_ns.newaxis, :]
    return pseudoinv


def _moment_increments(
    *,
    stencil_c9: data_alloc.NDArray,
    lsq_moments: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """The stencil offsets and the unweighted design matrix of the fit.

    f90 2294-2308: A[js, ju] = w[js] * (moments_hat[js, ju] - moments[ju]); the returned
    'diff' is that difference, without the row weights, which differ per candidate and per
    reconstruction order. Works for either order: the unknown count comes from
    'lsq_moments'.
    """
    array_ns = data_alloc.array_namespace(lsq_moments)
    z_dist = compute_torus_distance_vectors(
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        neighbor_table=stencil_c9,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    moments_hat = compute_lsq_moments_hat(
        lsq_moments=lsq_moments, stencil_c9=stencil_c9, z_dist=z_dist
    )
    return z_dist, moments_hat - lsq_moments[:, array_ns.newaxis, :]


def _full_stencil_pseudoinverse_quadratic(
    z_dist: data_alloc.NDArray, diff: data_alloc.NDArray
) -> data_alloc.NDArray:
    """The standard pseudoinverse over all 9 stencil rows, (n_cells, 5, 9).

    f90 2582-2589 for the WENO scheme; the same matrix miura3 (ihadv_tracer=3 with
    lsq_high_ord=2) reconstructs from, since it has no candidate sub-stencils.
    """
    array_ns = data_alloc.array_namespace(diff)
    n_cells = diff.shape[0]
    full_weights = interpolation_fields.compute_lsq_weights_c(z_dist, LSQ_WGT_EXP_QUADRATIC)
    full_design = full_weights[:, :, array_ns.newaxis] * diff
    return interpolation_fields.compute_lsq_pseudoinv(
        cell_owner_mask=array_ns.ones(n_cells, dtype=bool),
        z_lsq_mat_c=full_design,
        lsq_weights_c=full_weights,
        start_idx=0,
        min_rlcell_int=n_cells,
        lsq_dim_unk=LSQ_DIM_UNK_QUADRATIC,
        lsq_dim_c=LSQ_DIM_C_QUADRATIC,
    )


def compute_lsq_pseudoinverse_cubic(
    *,
    stencil_c9: data_alloc.NDArray,
    lsq_moments: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """The cubic reconstruction pseudoinverse, (n_cells, 9, 9).

    'lsq_moments' must be the cubic set (compute_lsq_moments_torus(cubic=True)). Applied to
    the unweighted increments z_b = avg(stencil cell) - avg(center), it yields the derivative
    coefficients [x, y, x^2, y^2, xy, x^3, y^3, x^2 y, x y^2] of the conservative cubic fit.

    ICON gives this set wgt_exp = 0, so unlike the quadratic one the rows are unweighted and
    the 9x9 system is square: the "pseudo" inverse is the plain inverse where the stencil is
    not degenerate. It is still computed through the SVD, as ICON does with llsq_svd.
    """
    if lsq_moments.shape[-1] != LSQ_DIM_UNK_CUBIC:
        raise ValueError(
            f"Invalid argument 'lsq_moments': the cubic reconstruction needs "
            f"{LSQ_DIM_UNK_CUBIC} moments, got {lsq_moments.shape[-1]}; pass the moments "
            "computed with 'cubic=True'."
        )
    array_ns = data_alloc.array_namespace(lsq_moments)
    _, diff = _moment_increments(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    # wgt_exp = 0 means every row weight is 1, so the design matrix is 'diff' itself
    weights = array_ns.ones(diff.shape[:2], dtype=ta.wpfloat)
    return _svd_pseudoinverse(diff, weights)


def compute_lsq_pseudoinverse_quadratic(
    *,
    stencil_c9: data_alloc.NDArray,
    lsq_moments: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """The quadratic reconstruction pseudoinverse of miura3, (n_cells, 5, 9).

    Applied to the unweighted increments z_b = avg(stencil cell) - avg(center), it yields
    the derivative coefficients [x, y, x^2, y^2, xy] of the conservative quadratic fit.
    This is the WENO scheme's full-stencil matrix without its l_weights_s correction, so
    the two cannot drift apart.
    """
    z_dist, diff = _moment_increments(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    return _full_stencil_pseudoinverse_quadratic(z_dist, diff)


def compute_weno_pseudoinverse_quadratic(
    *,
    stencil_c9: data_alloc.NDArray,
    lsq_moments: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """The 27 candidate pseudoinverses for the quadratic reconstruction, (n_cells, 27, 5, 9).

    Applied to the unweighted increments z_b = avg(stencil cell) - avg(center),
    candidate k >= 3 yields the derivative coefficients [x, y, x^2, y^2, xy] of
    the conservative quadratic fit on its active rows; candidates 0-2 are the
    full-stencil pseudoinverse subjected to the l_weights_s correction.
    """
    array_ns = data_alloc.array_namespace(lsq_moments)
    z_dist, diff = _moment_increments(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    candidate_weights = compute_candidate_weights_quadratic(z_dist)
    design = candidate_weights[:, :, :, array_ns.newaxis] * diff[:, array_ns.newaxis, :, :]
    pseudoinv = _svd_pseudoinverse(design, candidate_weights)

    # f90 2582-2589: candidates 1-3 are overwritten with the standard
    # full-stencil pseudoinverse (all 9 rows active, full distance weights)
    full_pseudoinv = _full_stencil_pseudoinverse_quadratic(z_dist, diff)
    pseudoinv[:, 0:3] = full_pseudoinv[:, array_ns.newaxis, :, :]

    # f90 2646-2657: literal port of the interleaved correction loop
    # `do i = 4, 27, 3`; with the live L_WEIGHTS_S only i + k in
    # {21, 24}, {22, 25}, {23, 26} (0-based) contribute
    for i in range(3, 27, 3):
        pseudoinv[:, 0] -= pseudoinv[:, i + 0] * L_WEIGHTS_S[i + 0]
        pseudoinv[:, 1] -= pseudoinv[:, i + 1] * L_WEIGHTS_S[i + 1]
        pseudoinv[:, 2] -= pseudoinv[:, i + 2] * L_WEIGHTS_S[i + 2]

    return pseudoinv


def compute_weno_pseudoinverse_linear(
    *,
    c2e2c: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    """The 3 candidate pseudoinverses for the linear reconstruction, (n_cells, 3, 2, 3).

    Unknowns are [x, y]. Unit row weights; candidate i zeroes the row of direct
    neighbor i (f90 1625-1634). With llsq_lin_consv=.FALSE. the moments vanish
    and the design matrix rows are the plain torus-periodic distance vectors
    (f90 2217-2308). No l_weights_s correction (f90 2582 guards on dim_c > 3).
    """
    array_ns = data_alloc.array_namespace(cell_center_x)
    n_cells = c2e2c.shape[0]
    z_dist = compute_torus_distance_vectors(
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        neighbor_table=c2e2c,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    # candidate weights, (n_cells, 3 candidates, 3 rows); the per-candidate max
    # normalization (f90 1945-1949) is a no-op for unit weights
    candidate_weights = array_ns.ones((n_cells, 3, 3), dtype=ta.wpfloat)
    for i in range(3):
        for js in range(3):
            candidate_weights[c2e2c[:, i] == c2e2c[:, js], i, js] = 0.0
    design = candidate_weights[:, :, :, array_ns.newaxis] * z_dist[:, array_ns.newaxis, :, :]
    return _svd_pseudoinverse(design, candidate_weights)


def scatter_to_offsets(
    *,
    values_fortran_order: data_alloc.NDArray,
    stencil_c9: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    c2e2c2e2c: data_alloc.NDArray,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Redistribute Fortran-ordered stencil coefficients onto the connectivity slots.

    values_fortran_order has shape (n_cells, n_cand, n_unk, 9) in
    create_stencil_c9 position order. Fortran positions {0, 3, 6} (the direct
    neighbors) go to the C2E2C slot holding that cell id; outer positions go to
    the first C2E2C2E2C slot holding that cell id (each slot claimed at most
    once). Unmatched butterfly slots (center-cell entries, duplicated direct
    neighbors) get coefficient 0, so summing over both offsets at runtime
    reproduces the Fortran-ordered sum.

    Returns (direct, butterfly) of shapes (n_cells, n_cand, n_unk, 3) and
    (n_cells, n_cand, n_unk, 9).
    """
    array_ns = data_alloc.array_namespace(values_fortran_order)
    if values_fortran_order.ndim != 4 or values_fortran_order.shape[-1] != 9:
        raise ValueError(
            "Invalid argument 'values_fortran_order': expected shape "
            f"(n_cells, n_cand, n_unk, 9), got {values_fortran_order.shape}."
        )
    n_cells = values_fortran_order.shape[0]
    cells = array_ns.arange(n_cells)[:, array_ns.newaxis]

    sorted_stencil = array_ns.sort(stencil_c9, axis=1)
    if array_ns.any(sorted_stencil[:, 1:] == sorted_stencil[:, :-1]):
        raise ValueError("The 9 stencil cells are not distinct for every cell.")

    direct_positions = array_ns.asarray([0, 3, 6])
    outer_positions = array_ns.asarray([1, 2, 4, 5, 7, 8])

    direct_match = (
        stencil_c9[:, direct_positions, array_ns.newaxis] == c2e2c[:, array_ns.newaxis, :]
    )
    if not array_ns.all(array_ns.any(direct_match, axis=2)):
        raise ValueError("A direct stencil position was not found in C2E2C.")
    direct_slot = array_ns.argmax(direct_match, axis=2)  # (n_cells, 3)

    outer_match = (
        stencil_c9[:, outer_positions, array_ns.newaxis] == c2e2c2e2c[:, array_ns.newaxis, :]
    )
    if not array_ns.all(array_ns.any(outer_match, axis=2)):
        raise ValueError("An outer stencil position was not found in C2E2C2E2C.")
    outer_slot = array_ns.argmax(outer_match, axis=2)  # (n_cells, 6), first occurrence
    sorted_slots = array_ns.sort(outer_slot, axis=1)
    if array_ns.any(sorted_slots[:, 1:] == sorted_slots[:, :-1]):
        raise ValueError("A C2E2C2E2C slot was claimed by more than one outer stencil position.")

    direct = array_ns.zeros((*values_fortran_order.shape[:-1], 3), dtype=values_fortran_order.dtype)
    butterfly = array_ns.zeros(
        (*values_fortran_order.shape[:-1], 9), dtype=values_fortran_order.dtype
    )
    direct[cells, :, :, direct_slot] = values_fortran_order[cells, :, :, direct_positions]
    butterfly[cells, :, :, outer_slot] = values_fortran_order[cells, :, :, outer_positions]
    return direct, butterfly
