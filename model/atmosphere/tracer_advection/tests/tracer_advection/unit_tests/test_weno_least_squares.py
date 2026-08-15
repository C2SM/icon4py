# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the init-time WENO least-squares coefficient machinery.

All tests run on plain numpy: either the SimpleGrid connectivity tables or the
small synthetic periodic equilateral-triangle torus patch from ..utils.
"""

import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import weno_least_squares as weno
from icon4py.model.common.grid import simple
from icon4py.model.common.interpolation import interpolation_fields

from ..utils import TorusPatch, build_torus_patch


# number of quadratic candidates / unknowns / stencil positions
N_CAND = 27
N_UNK = 5
N_STENCIL = 9


@pytest.fixture(scope="module")
def torus_patch() -> TorusPatch:
    return build_torus_patch()


def _triangle_average(f, vertices: np.ndarray) -> np.ndarray:
    """Average of f over triangles via the 3-point edge-midpoint rule.

    Exact for polynomials of degree <= 2. vertices has shape (..., 3, 2).
    """
    m01 = 0.5 * (vertices[..., 0, :] + vertices[..., 1, :])
    m12 = 0.5 * (vertices[..., 1, :] + vertices[..., 2, :])
    m20 = 0.5 * (vertices[..., 2, :] + vertices[..., 0, :])
    return (f(m01) + f(m12) + f(m20)) / 3.0


def _triangle_average_cubic(f, vertices: np.ndarray) -> np.ndarray:
    """Average of f over triangles via the 4-point Strang-Fix rule, exact for degree <= 3.

    The 3-point midpoint rule above is only exact to degree 2, so the cubic tests need this
    one. Barycentric points (1/3,1/3,1/3) with weight -27/48 and the three permutations of
    (3/5,1/5,1/5) with weight 25/48. vertices has shape (..., 3, 2).
    """
    v0, v1, v2 = vertices[..., 0, :], vertices[..., 1, :], vertices[..., 2, :]
    centroid = (v0 + v1 + v2) / 3.0
    total = -27.0 / 48.0 * f(centroid)
    for a, b, c in ((0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)):
        total = total + 25.0 / 48.0 * f(a * v0 + b * v1 + c * v2)
    return total


def _random_cubic(rng: np.random.Generator):
    """Random cubic p(x, y); returns (evaluator, the 9 derivative coefficients).

    The coefficients come back in ICON's moment order,
    [x, y, x^2, y^2, xy, x^3, y^3, x^2 y, x y^2].
    """
    a, b, c, d, e, f, g, h, i, j = rng.uniform(-1.0, 1.0, 10)

    def poly(v: np.ndarray) -> np.ndarray:
        x, y = v[..., 0], v[..., 1]
        return (
            a
            + b * x
            + c * y
            + d * x**2
            + e * y**2
            + f * x * y
            + g * x**3
            + h * y**3
            + i * x**2 * y
            + j * x * y**2
        )

    return poly, np.array([b, c, d, e, f, g, h, i, j])


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
    weights = interpolation_fields.compute_lsq_weights_c(z_dist, 5)
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


# 3b. the miura3 (non-WENO) pseudoinverse
def test_quadratic_pseudoinverse_matches_full_stencil_reference(torus_patch):
    computed = weno.compute_lsq_pseudoinverse_quadratic(
        stencil_c9=weno.create_stencil_c9(torus_patch.c2e2c, torus_patch.c2v),
        lsq_moments=_patch_moments(torus_patch),
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    np.testing.assert_allclose(
        computed, _full_stencil_pseudoinverse_reference(torus_patch), rtol=1e-12, atol=1e-13
    )


def test_quadratic_pseudoinverse_recovers_polynomial(torus_patch):
    """The property miura3 rests on: the fit of an exact quadratic is that quadratic."""
    stencil, z_dist = _patch_stencil_and_distances(torus_patch)
    pseudoinv = weno.compute_lsq_pseudoinverse_quadratic(
        stencil_c9=stencil,
        lsq_moments=_patch_moments(torus_patch),
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    n_cells = stencil.shape[0]

    rng = np.random.default_rng(7)
    for _ in range(3):
        poly, derivative_coeffs = _random_quadratic(rng)
        z_b = _quadratic_cell_average_increments(torus_patch, stencil, z_dist, poly)
        recovered = np.einsum("nus,ns->nu", pseudoinv, z_b)
        np.testing.assert_allclose(
            recovered,
            np.broadcast_to(derivative_coeffs, (n_cells, N_UNK)),
            rtol=1e-10,
            atol=1e-11,
        )


# 3c. the cubic (lsq_high_ord=3) reconstruction, which FFSL needs
def _patch_moments_cubic(patch: TorusPatch) -> np.ndarray:
    return weno.compute_lsq_moments_torus(
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        vertex_x=patch.vertex_x,
        vertex_y=patch.vertex_y,
        c2v=patch.c2v,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
        cubic=True,
    )


def test_cubic_moments_extend_the_quadratic_ones(torus_patch):
    """The cubic set must not reorder the quadratic one, only append to it."""
    quadratic = _patch_moments(torus_patch)
    cubic = _patch_moments_cubic(torus_patch)
    assert cubic.shape == (quadratic.shape[0], 9)
    np.testing.assert_array_equal(cubic[:, :5], quadratic)


def test_cubic_moments_match_quadrature(torus_patch):
    """The analytic polygon integrals against a quadrature exact for cubics."""
    moments = _patch_moments_cubic(torus_patch)
    monomials = (
        lambda v: v[..., 0],
        lambda v: v[..., 1],
        lambda v: v[..., 0] ** 2,
        lambda v: v[..., 1] ** 2,
        lambda v: v[..., 0] * v[..., 1],
        lambda v: v[..., 0] ** 3,
        lambda v: v[..., 1] ** 3,
        lambda v: v[..., 0] ** 2 * v[..., 1],
        lambda v: v[..., 0] * v[..., 1] ** 2,
    )
    for unknown, monomial in enumerate(monomials):
        expected = _triangle_average_cubic(monomial, torus_patch.local_vertices)
        np.testing.assert_allclose(moments[:, unknown], expected, rtol=1e-10, atol=1e-12)


def test_cubic_pseudoinverse_recovers_polynomial(torus_patch):
    """The property FFSL's cubic reconstruction rests on."""
    stencil, z_dist = _patch_stencil_and_distances(torus_patch)
    pseudoinv = weno.compute_lsq_pseudoinverse_cubic(
        stencil_c9=stencil,
        lsq_moments=_patch_moments_cubic(torus_patch),
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    assert pseudoinv.shape == (stencil.shape[0], 9, 9)

    rng = np.random.default_rng(11)
    for _ in range(3):
        poly, derivative_coeffs = _random_cubic(rng)
        avg_center = _triangle_average_cubic(poly, torus_patch.local_vertices)
        stencil_vertices = torus_patch.local_vertices[stencil] + z_dist[:, :, np.newaxis, :]
        z_b = _triangle_average_cubic(poly, stencil_vertices) - avg_center[:, np.newaxis]

        recovered = np.einsum("nus,ns->nu", pseudoinv, z_b)
        np.testing.assert_allclose(
            recovered,
            np.broadcast_to(derivative_coeffs, (stencil.shape[0], 9)),
            rtol=1e-7,
            atol=1e-8,
        )


def test_cubic_pseudoinverse_rejects_quadratic_moments(torus_patch):
    with pytest.raises(ValueError, match="needs 9 moments"):
        weno.compute_lsq_pseudoinverse_cubic(
            stencil_c9=weno.create_stencil_c9(torus_patch.c2e2c, torus_patch.c2v),
            lsq_moments=_patch_moments(torus_patch),
            cell_center_x=torus_patch.cell_center_x,
            cell_center_y=torus_patch.cell_center_y,
            domain_length=torus_patch.domain_length,
            domain_height=torus_patch.domain_height,
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


# 9. torus patch edge tables: sanity of the synthetic grid the geometry tests run on
def test_torus_patch_edge_tables(torus_patch):
    n_cells = torus_patch.c2e2c.shape[0]
    n_edges = torus_patch.e2c.shape[0]
    assert n_edges == 3 * n_cells // 2

    # each edge vertex belongs to both adjacent cells
    for e in range(n_edges):
        for c in torus_patch.e2c[e]:
            assert set(torus_patch.e2v[e].tolist()) <= set(torus_patch.c2v[c].tolist())

    # orthonormal edge frames and +-1 orientations
    pn = np.stack((torus_patch.primal_normal_x, torus_patch.primal_normal_y), axis=1)
    dn = np.stack((torus_patch.dual_normal_x, torus_patch.dual_normal_y), axis=1)
    np.testing.assert_allclose(np.linalg.norm(pn, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(dn, axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.sum(pn * dn, axis=1), 0.0, atol=1e-12)
    assert set(np.unique(torus_patch.tangent_orientation).tolist()) == {-1.0, 1.0}


# 10. ffsl backtrajectory torus geometry: edge-frame positions of cells and vertices
def test_ffsl_backtrajectory_geometry_torus(torus_patch):
    pos_x, pos_y, verts_x, verts_y = weno.compute_ffsl_backtrajectory_geometry_torus(
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
    a = torus_patch.edge_length
    n_edges = torus_patch.e2c.shape[0]

    # For an equilateral triangle the centroid (= patch cell center) is the circumcenter
    # and lies on the perpendicular bisector of each edge at distance a/(2*sqrt(3)) from
    # its midpoint (circumradius R = a/sqrt(3), apothem R/2). The primal normal points
    # from cell 1 to cell 2, so the normal components of the two cell offsets have
    # opposite signs and the tangential components vanish.
    apothem = a / (2.0 * np.sqrt(3.0))
    np.testing.assert_allclose(pos_x[:, 0], -apothem, rtol=1e-12)
    np.testing.assert_allclose(pos_x[:, 1], apothem, rtol=1e-12)
    np.testing.assert_allclose(pos_y, 0.0, atol=1e-12)

    # matches the equilateral shortcut of the existing interpolation field (the cell
    # centers sit half a dual edge length = one apothem away from the edge midpoint)
    ref_x, ref_y = interpolation_fields.compute_pos_on_tplane_e_x_y_torus(
        np.full(n_edges, 2.0 * apothem), torus_patch.e2c
    )
    np.testing.assert_allclose(pos_x, ref_x, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(pos_y, ref_y, atol=1e-13)

    # the plane-torus edges are straight segments whose stored center is the midpoint:
    # the vertex offsets are purely tangential with magnitude a/2, ordered along the
    # stored tangent by the orientation, (v2 - v1) . T = tangent_orientation * a
    np.testing.assert_allclose(verts_x, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(verts_y), a / 2.0, rtol=1e-12)
    np.testing.assert_allclose(
        verts_y[:, 1] - verts_y[:, 0], torus_patch.tangent_orientation * a, rtol=1e-12
    )
