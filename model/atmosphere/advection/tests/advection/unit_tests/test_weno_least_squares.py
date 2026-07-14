# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the init-time WENO least-squares coefficient machinery.

All tests run on plain numpy: either the SimpleGrid connectivity tables or a
small synthetic periodic equilateral-triangle torus patch built here.
"""

import dataclasses

import numpy as np
import pytest

from icon4py.model.atmosphere.advection import weno_least_squares as weno
from icon4py.model.common.grid import simple
from icon4py.model.common.interpolation import interpolation_fields


# number of quadratic candidates / unknowns / stencil positions
N_CAND = 27
N_UNK = 5
N_STENCIL = 9


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


def _build_torus_patch(nx: int = 8, ny: int = 8, edge_length: float = 1.0) -> TorusPatch:
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
    )


@pytest.fixture(scope="module")
def torus_patch() -> TorusPatch:
    return _build_torus_patch()


def _triangle_average(f, vertices: np.ndarray) -> np.ndarray:
    """Average of f over triangles via the 3-point edge-midpoint rule.

    Exact for polynomials of degree <= 2. vertices has shape (..., 3, 2).
    """
    m01 = 0.5 * (vertices[..., 0, :] + vertices[..., 1, :])
    m12 = 0.5 * (vertices[..., 1, :] + vertices[..., 2, :])
    m20 = 0.5 * (vertices[..., 2, :] + vertices[..., 0, :])
    return (f(m01) + f(m12) + f(m20)) / 3.0


def _patch_moments(patch: TorusPatch) -> np.ndarray:
    return weno.compute_lsq_moments_torus(
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        vertex_x=patch.vertex_x,
        vertex_y=patch.vertex_y,
        c2v=patch.c2v,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )


def _patch_stencil_and_distances(patch: TorusPatch) -> tuple[np.ndarray, np.ndarray]:
    stencil = weno.create_stencil_c9(patch.c2e2c, patch.c2v)
    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        neighbor_table=stencil,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )
    return stencil, z_dist


def _patch_moments_diff(patch: TorusPatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stencil, z_dist, moments_hat - moments): the unweighted design matrix rows."""
    moments = _patch_moments(patch)
    stencil, z_dist = _patch_stencil_and_distances(patch)
    moments_hat = weno.compute_lsq_moments_hat(
        lsq_moments=moments, stencil_c9=stencil, z_dist=z_dist
    )
    return stencil, z_dist, moments_hat - moments[:, np.newaxis, :]


def _patch_pseudoinverse(patch: TorusPatch) -> np.ndarray:
    return weno.compute_weno_pseudoinverse_quadratic(
        stencil_c9=weno.create_stencil_c9(patch.c2e2c, patch.c2v),
        lsq_moments=_patch_moments(patch),
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )


def _full_stencil_pseudoinverse_reference(patch: TorusPatch) -> np.ndarray:
    """Full-stencil pseudoinverse via the generic machinery in interpolation_fields."""
    _, z_dist, diff = _patch_moments_diff(patch)
    n_cells = z_dist.shape[0]
    weights = interpolation_fields.compute_lsq_weights_c(z_dist, N_STENCIL, 5)
    design = weights[:, :, np.newaxis] * diff
    return interpolation_fields.compute_lsq_pseudoinv(
        cell_owner_mask=np.ones(n_cells, dtype=bool),
        z_lsq_mat_c=design,
        lsq_weights_c=weights,
        start_idx=0,
        min_rlcell_int=n_cells,
        lsq_dim_unk=N_UNK,
        lsq_dim_c=N_STENCIL,
    )


def _random_quadratic(rng: np.random.Generator):
    """Random quadratic p(x, y); returns (evaluator, derivative coefficients [b,c,d,e,f])."""
    a, b, c, d, e, f = rng.uniform(-1.0, 1.0, 6)

    def poly(v: np.ndarray) -> np.ndarray:
        x, y = v[..., 0], v[..., 1]
        return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y

    return poly, np.array([b, c, d, e, f])


def _quadratic_cell_average_increments(
    patch: TorusPatch, stencil: np.ndarray, z_dist: np.ndarray, poly
) -> np.ndarray:
    """z_b: cell averages of poly over the stencil cells in the center cell frame,
    minus the center cell average, via exact quadrature (independent of moments)."""
    avg_center = _triangle_average(poly, patch.local_vertices)
    stencil_vertices = patch.local_vertices[stencil] + z_dist[:, :, np.newaxis, :]
    avg_stencil = _triangle_average(poly, stencil_vertices)
    return avg_stencil - avg_center[:, np.newaxis]


# 1. stencil construction on the SimpleGrid tables
def test_create_stencil_c9_simple_grid():
    grid_data = simple.SimpleGridData()
    c2e2c = np.asarray(grid_data.c2e2c_table)
    c2v = np.asarray(grid_data.c2v_table)
    stencil = weno.create_stencil_c9(c2e2c, c2v)
    n_cells = c2e2c.shape[0]

    # direct neighbors at Fortran positions 1, 4, 7
    np.testing.assert_array_equal(stencil[:, [0, 3, 6]], c2e2c)

    # the 6 outer cells are the two-ring minus the direct neighbors minus the center
    for c in range(n_cells):
        two_ring = set(c2e2c[c2e2c[c]].flatten().tolist())
        outer = stencil[c, [1, 2, 4, 5, 7, 8]].tolist()
        assert len(set(outer)) == 6
        assert set(outer) == two_ring - set(c2e2c[c].tolist()) - {c}

    # orientation-swap invariant (f90 create_stencil_c9, lines 407-437): the first
    # outer of direct neighbor jec shares a vertex with direct neighbor (jec+1) % 3
    for jec in range(3):
        next_direct = c2e2c[:, (jec + 1) % 3]
        first_outer = stencil[:, 3 * jec + 1]
        shares_vertex = np.any(
            c2v[next_direct][:, :, np.newaxis] == c2v[first_outer][:, np.newaxis, :],
            axis=(1, 2),
        )
        assert shares_vertex.all()


# 2. torus moments against an exact quadrature reference and the closed form
def test_moments_match_quadrature_and_closed_form(torus_patch):
    moments = _patch_moments(torus_patch)

    monomials = [
        lambda v: v[..., 0],
        lambda v: v[..., 1],
        lambda v: v[..., 0] ** 2,
        lambda v: v[..., 1] ** 2,
        lambda v: v[..., 0] * v[..., 1],
    ]
    reference = np.stack(
        [_triangle_average(f, torus_patch.local_vertices) for f in monomials], axis=1
    )
    np.testing.assert_allclose(moments, reference, rtol=1e-12, atol=1e-14)

    # closed form for an equilateral triangle of side a centered at its centroid:
    # first moments vanish, avg(x^2) = avg(y^2) = a^2/24, avg(xy) = 0 (for our
    # up/down orientations the vertex xy products cancel pairwise)
    a = torus_patch.edge_length
    closed_form = np.array([0.0, 0.0, a**2 / 24.0, a**2 / 24.0, 0.0])
    np.testing.assert_allclose(
        moments, np.broadcast_to(closed_form, moments.shape), rtol=1e-12, atol=1e-14
    )


# 3. quadratic reconstruction exactness for every candidate pseudoinverse
def test_quadratic_candidates_recover_polynomial(torus_patch):
    stencil, z_dist = _patch_stencil_and_distances(torus_patch)
    pseudoinv = _patch_pseudoinverse(torus_patch)
    n_cells = stencil.shape[0]

    # zeroed stencil positions must have exactly zero coefficients
    for cand, positions in enumerate(weno.CANDIDATE_ZERO_PATTERNS_QUADRATIC):
        for pos in positions:
            np.testing.assert_array_equal(pseudoinv[:, cand, :, pos], 0.0)

    rng = np.random.default_rng(42)
    for _ in range(3):
        poly, derivative_coeffs = _random_quadratic(rng)
        z_b = _quadratic_cell_average_increments(torus_patch, stencil, z_dist, poly)
        recovered = np.einsum("ncus,ns->ncu", pseudoinv, z_b)
        expected = np.broadcast_to(derivative_coeffs, (n_cells, N_UNK))

        # candidates 3..26 are untouched by the l_weights_s correction and must
        # recover the 5 derivative coefficients exactly on their active rows
        for cand in range(3, N_CAND):
            np.testing.assert_allclose(recovered[:, cand], expected, rtol=1e-10, atol=1e-11)

        # corrected candidates 0-2 recover (1 - sum of their group's l_weights_s)
        # times the coefficients: only 0-based candidates {21, 24}, {22, 25},
        # {23, 26} carry non-zero weight 2.991549980478795
        factor = 1.0 - 2.0 * 2.991549980478795
        for cand in range(3):
            np.testing.assert_allclose(
                recovered[:, cand], factor * expected, rtol=1e-10, atol=1e-10
            )


# 4. l_weights_s correction arithmetic identity
def test_l_weights_correction_identity(torus_patch):
    pseudoinv = _patch_pseudoinverse(torus_patch)
    reference_full = _full_stencil_pseudoinverse_reference(torus_patch)

    # undoing the correction loop must give back the pre-correction candidates
    # 0-2, which are copies of the full-stencil pseudoinverse
    for k in range(3):
        reconstructed = pseudoinv[:, k].copy()
        for i in range(3, N_CAND, 3):
            reconstructed += pseudoinv[:, i + k] * weno.L_WEIGHTS_S[i + k]
        np.testing.assert_allclose(reconstructed, reference_full, rtol=1e-12, atol=1e-13)


# 5. scatter round-trip through the connectivities
@pytest.mark.parametrize("grid_name", ["torus_patch", "simple_grid"])
def test_scatter_round_trip(torus_patch, grid_name):
    if grid_name == "torus_patch":
        c2e2c = torus_patch.c2e2c
        c2v = torus_patch.c2v
        c2e2c2e2c = torus_patch.c2e2c2e2c
    else:
        # hand-coded tables: butterfly holds 3 direct + 6 outer cells instead of
        # the grid_manager layout (center cell 3x + 6 outer cells)
        grid_data = simple.SimpleGridData()
        c2e2c = np.asarray(grid_data.c2e2c_table)
        c2v = np.asarray(grid_data.c2v_table)
        c2e2c2e2c = np.asarray(grid_data.c2e2c2e2c_table)

    stencil = weno.create_stencil_c9(c2e2c, c2v)
    n_cells = c2e2c.shape[0]
    rng = np.random.default_rng(7)
    values = rng.uniform(-1.0, 1.0, size=(n_cells, N_CAND, N_UNK, N_STENCIL))

    direct, butterfly = weno.scatter_to_offsets(
        values_fortran_order=values, stencil_c9=stencil, c2e2c=c2e2c, c2e2c2e2c=c2e2c2e2c
    )
    assert direct.shape == (n_cells, N_CAND, N_UNK, 3)
    assert butterfly.shape == (n_cells, N_CAND, N_UNK, 9)

    # gather back through the connectivities (claim-first rule for the butterfly)
    gathered = np.empty_like(values)
    for c in range(n_cells):
        for pos in range(N_STENCIL):
            cell_id = stencil[c, pos]
            if pos in (0, 3, 6):
                gathered[c, :, :, pos] = direct[c, :, :, c2e2c[c].tolist().index(cell_id)]
            else:
                gathered[c, :, :, pos] = butterfly[c, :, :, c2e2c2e2c[c].tolist().index(cell_id)]
    np.testing.assert_array_equal(gathered, values)

    # every coefficient lands exactly once; unmatched butterfly slots are zero
    np.testing.assert_allclose(
        direct.sum(axis=-1) + butterfly.sum(axis=-1), values.sum(axis=-1), rtol=1e-12, atol=1e-13
    )


# 6. SVD conditioning go/no-go gate
def test_svd_conditioning_go_no_go(torus_patch):
    _, z_dist, diff = _patch_moments_diff(torus_patch)
    candidate_weights = weno.compute_candidate_weights_quadratic(z_dist)
    design = candidate_weights[:, :, :, np.newaxis] * diff[:, np.newaxis, :, :]
    singular_values = np.linalg.svd(design, compute_uv=False)
    ratio = singular_values[..., -1] / singular_values[..., 0]
    assert ratio.min() > 1e-8, (
        f"Rank-deficient candidate systems: min(s_min/s_max) = {ratio.min():.3e} "
        f"at cell/candidate {np.unravel_index(ratio.argmin(), ratio.shape)}."
    )


# 7. linear candidates recover the exact gradient of a linear field
def test_linear_candidates_recover_gradient(torus_patch):
    pseudoinv = weno.compute_weno_pseudoinverse_linear(
        c2e2c=torus_patch.c2e2c,
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    n_cells = torus_patch.c2e2c.shape[0]
    assert pseudoinv.shape == (n_cells, 3, 2, 3)

    # candidate i has a zero coefficient on the row of direct neighbor i
    for i in range(3):
        np.testing.assert_array_equal(pseudoinv[:, i, :, i], 0.0)

    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        neighbor_table=torus_patch.c2e2c,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    rng = np.random.default_rng(3)
    gradient = rng.uniform(-1.0, 1.0, 2)
    z_b = gradient[0] * z_dist[..., 0] + gradient[1] * z_dist[..., 1]
    recovered = np.einsum("nkus,ns->nku", pseudoinv, z_b)
    np.testing.assert_allclose(
        recovered, np.broadcast_to(gradient, (n_cells, 3, 2)), rtol=1e-12, atol=1e-13
    )


def _linear_weno_blend(pseudoinv: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """Smoothness-weighted blend of the 3 linear candidate gradients (f90 995-1019).

    pseudoinv has shape (n_cells, 3 candidates, 2 [zonal, meridional], 3 rows),
    z_b (n_cells, 3 rows); returns the blended [zonal, meridional] gradient
    (n_cells, 2).
    """
    grads = np.einsum("ncus,ns->ncu", pseudoinv, z_b)  # (n_cells, 3 cand, 2)
    cx = grads[:, :, 0]
    cy = grads[:, :, 1]
    s = 1.0 / ((cx**2 + cy**2) + 1.0e-20) ** 2
    smooth_sum = s.sum(axis=1)
    blended_x = (cx * s).sum(axis=1) / smooth_sum
    blended_y = (cy * s).sum(axis=1) / smooth_sum
    return np.stack((blended_x, blended_y), axis=1)


# 8. on a linear field all 3 candidates coincide, so the WENO blend reduces to
# the plain least-squares gradient (blend-of-equal-candidates == single candidate)
def test_linear_weno_blend_equals_plain_lsq(torus_patch):
    pseudoinv = weno.compute_weno_pseudoinverse_linear(
        c2e2c=torus_patch.c2e2c,
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    n_cells = torus_patch.c2e2c.shape[0]
    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        neighbor_table=torus_patch.c2e2c,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    rng = np.random.default_rng(11)
    gradient = rng.uniform(-1.0, 1.0, 2)
    z_b = gradient[0] * z_dist[..., 0] + gradient[1] * z_dist[..., 1]

    # each candidate recovers the exact gradient, so the 3 smoothness weights coincide
    candidate_grads = np.einsum("ncus,ns->ncu", pseudoinv, z_b)
    np.testing.assert_allclose(
        candidate_grads, np.broadcast_to(gradient, (n_cells, 3, 2)), rtol=1e-12, atol=1e-13
    )

    # the blend equals the exact gradient and the plain single-candidate gradient
    blended = _linear_weno_blend(pseudoinv, z_b)
    np.testing.assert_allclose(
        blended, np.broadcast_to(gradient, (n_cells, 2)), rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(blended, candidate_grads[:, 0], rtol=1e-12, atol=1e-13)

    # pure numpy cross-check: blending identical candidate gradients returns them
    identical = np.broadcast_to(rng.uniform(-1.0, 1.0, 2), (n_cells, 3, 2)).copy()
    cx = identical[:, :, 0]
    cy = identical[:, :, 1]
    s = 1.0 / ((cx**2 + cy**2) + 1.0e-20) ** 2
    smooth_sum = s.sum(axis=1)
    blended_identical = np.stack(
        ((cx * s).sum(axis=1) / smooth_sum, (cy * s).sum(axis=1) / smooth_sum), axis=1
    )
    np.testing.assert_allclose(blended_identical, identical[:, 0], rtol=1e-12, atol=1e-13)
