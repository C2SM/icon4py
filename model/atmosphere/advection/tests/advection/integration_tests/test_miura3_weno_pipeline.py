# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Full-pipeline cross-check of the miura3 WENO tracer flux (ihadv_tracer=103).

The reference is a literal numpy re-implementation of the Fortran cell loop
(upwind_hflux_miura3_weno, mo_advection_hflux.f90 2443-2546): 9-point z_b
differences, 27-candidate reconstruction with the linear-constraint c0,
per-edge smoothness weighting, and the scatter to the edges owned (upwind)
by the cell. It is written from the Fortran, loops and all, independent of
the gt4py stencils.

The gt4py pipeline under test composes the runtime stencils exactly as the
ThirdOrderMiuraWeno driver does: reconstruct_quadratic_coefficients_weno_candidate
x27 -> accumulate_weno_candidate_flux_weights x27 ->
compute_horizontal_tracer_flux_from_weno_coefficients, on the synthetic
periodic torus patch with REAL candidate pseudoinverses, moments and scatter
tables from weno_least_squares (the same real-coefficient source Task 1's
unit tests use; SimpleGrid has no coordinates).

Three gates:
- the numpy reference itself is validated against a hand-derived closed form
  (zero pseudoinverses -> pure c0 reconstruction -> p_out_e =
  p_cc(upwind) * quad_1 * mass_flx);
- reference vs pipeline on random data at 1e-12 relative, both with the live
  L_WEIGHTS_S (candidates 4-21 inert) and with all-distinct synthetic weights
  (catches any candidate/weight pairing off-by-one);
- on an exactly quadratic tracer field the candidates collapse into two
  exactly-known groups (candidates 4-27 recover the derivative coefficients d,
  candidates 1-3 recover (1 - 2*2.991549980478795) * d because of the init-time
  l_weights_s correction, see test_weno_least_squares), so the WENO flux has a
  closed form computed here from the polynomial alone, without pseudoinverses.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.advection import weno_least_squares as weno
from icon4py.model.atmosphere.advection.stencils.accumulate_weno_candidate_flux_weights import (
    accumulate_weno_candidate_flux_weights,
)
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_weno_coefficients import (
    compute_horizontal_tracer_flux_from_weno_coefficients,
)
from icon4py.model.atmosphere.advection.stencils.reconstruct_quadratic_coefficients_weno_candidate import (
    reconstruct_quadratic_coefficients_weno_candidate,
)
from icon4py.model.common import dimension as dims

# fixture
from icon4py.model.testing.fixtures.datatest import backend

from .. import utils


N_CAND = 27
NLEV = 3
# f90 2509: literal 1d-20 regularization
WENO_EPS = 1e-20
# live l_weights_s value of candidates 22-27 (f90 2590-2646) and the resulting
# recovery factor of the corrected candidates 1-3 on smooth data
LAMBDA_S = 2.991549980478795
CORRECTED_FACTOR = 1.0 - 2.0 * LAMBDA_S


@pytest.fixture(scope="module")
def torus_patch() -> utils.TorusPatch:
    return utils.build_torus_patch()


@pytest.fixture(scope="module")
def patch_coefficients(torus_patch) -> dict:
    """Real init-time coefficients: stencil, moments, 27 pseudoinverses, scatter."""
    stencil_c9 = weno.create_stencil_c9(torus_patch.c2e2c, torus_patch.c2v)
    lsq_moments = weno.compute_lsq_moments_torus(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        vertex_x=torus_patch.vertex_x,
        vertex_y=torus_patch.vertex_y,
        c2v=torus_patch.c2v,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    pseudoinv = weno.compute_weno_pseudoinverse_quadratic(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    direct, butterfly = weno.scatter_to_offsets(
        values_fortran_order=pseudoinv,
        stencil_c9=stencil_c9,
        c2e2c=torus_patch.c2e2c,
        c2e2c2e2c=torus_patch.c2e2c2e2c,
    )
    return dict(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        pseudoinv=pseudoinv,
        direct=direct,
        butterfly=butterfly,
    )


def _upwind_hflux_miura3_weno_reference(
    *,
    p_cc: np.ndarray,  # (n_cells, nlev)
    pseudoinv: np.ndarray,  # (n_cells, 27, 5, 9), Fortran stencil order
    stencil_c9: np.ndarray,  # (n_cells, 9)
    lsq_moments: np.ndarray,  # (n_cells, 5)
    cell_area: np.ndarray,  # (n_cells,)
    c2e: np.ndarray,  # (n_cells, 3)
    upwind_cell: np.ndarray,  # (n_edges, nlev), the ptr_ilc of each edge
    quad: np.ndarray,  # (n_edges, nlev, 6), z_quad_vector_sum
    p_mass_flx_e: np.ndarray,  # (n_edges, nlev)
    l_weights_s: np.ndarray,  # (27,)
) -> np.ndarray:
    """Literal port of the miura3 WENO cell loop (mo_advection_hflux.f90 2443-2546).

    Live path only: quadratic, no limiter (p_itype_hlimit /= ifluxl_sm).
    0-based indices; z_lsq_coeff components 2..6 are coeff[1:6] here.
    """
    n_cells, nlev = p_cc.shape
    p_out_e = np.zeros(p_mass_flx_e.shape)
    for jc in range(n_cells):
        for jk in range(nlev):
            # f90 2447-2449: 9-point differences to the center cell
            z_b = np.empty(9)
            for js in range(9):
                z_b[js] = p_cc[stencil_c9[jc, js], jk] - p_cc[jc, jk]
            # f90 2450-2451
            z_lsq_weighted = np.zeros((6, 3))
            smooth_sum = np.zeros(3)
            # f90 2452-2456: the cell's edges and the ownership flags
            jee = c2e[jc]
            jf = np.empty(3, dtype=bool)
            for ie in range(3):
                jf[ie] = upwind_cell[jee[ie], jk] == jc
            area = cell_area[jc]
            # f90 2458-2512: 27-candidate loop
            for cand in range(N_CAND):
                coeff = np.empty(6)
                for ju in range(5):
                    coeff[1 + ju] = np.dot(pseudoinv[jc, cand, ju, :], z_b)
                # f90 2494-2495: c0 from the linear constraint
                coeff[0] = p_cc[jc, jk] - np.dot(coeff[1:6], lsq_moments[jc])
                # f90 2497-2506: smoothness vector; zlc(4:6) squared in place
                zlc = coeff.copy()
                smooth = np.empty(6)
                smooth[1] = 2.0 * (zlc[1] * zlc[3] + zlc[2] * zlc[5])
                smooth[2] = 2.0 * (zlc[1] * zlc[5] + zlc[2] * zlc[4])
                smooth[5] = 2.0 * zlc[5] * (zlc[3] + zlc[4])
                zlc[3] = zlc[3] ** 2
                zlc[4] = zlc[4] ** 2
                zlc[5] = zlc[5] ** 2
                smooth[3] = 2.0 * (zlc[3] + zlc[5])
                smooth[4] = 2.0 * (zlc[4] + zlc[5])
                smooth[0] = zlc[1] ** 2 + zlc[2] ** 2 + area * (zlc[3] + zlc[4] + zlc[5])
                # f90 2507-2511: per-edge smoothness weighting and accumulation
                for ie in range(3):
                    smoothness = np.dot(smooth, quad[jee[ie], jk, :])
                    smoothness = l_weights_s[cand] / (smoothness + WENO_EPS) ** 2
                    z_lsq_weighted[:, ie] += coeff * smoothness
                    smooth_sum[ie] += smoothness
            # f90 2513-2521: normalize and write the flux to the owned edges
            for ie in range(3):
                if jf[ie]:
                    z_lsq_weighted[:, ie] /= smooth_sum[ie]
                    p_out_e[jee[ie], jk] = (
                        np.dot(z_lsq_weighted[:, ie], quad[jee[ie], jk, :])
                        * p_mass_flx_e[jee[ie], jk]
                    )
    return p_out_e


def _random_inputs(torus_patch, seed: int) -> dict:
    """Random-but-consistent runtime inputs shared by reference and pipeline."""
    rng = np.random.default_rng(seed)
    n_cells = torus_patch.c2e2c.shape[0]
    n_edges = torus_patch.e2c.shape[0]
    rel_idx = rng.integers(0, 2, size=(n_edges, NLEV)).astype(np.int32)
    return dict(
        p_cc=rng.uniform(0.1, 1.0, size=(n_cells, NLEV)),
        rel_idx=rel_idx,
        upwind_cell=np.take_along_axis(torus_patch.e2c, rel_idx, axis=1),  # e2c[e, rel_idx[e, k]]
        quad=rng.uniform(0.1, 1.0, size=(n_edges, NLEV, 6)),
        p_mass_flx_e=rng.uniform(-1.0, 1.0, size=(n_edges, NLEV)),
        cell_area=np.full(n_cells, np.sqrt(3.0) / 4.0 * torus_patch.edge_length**2),
    )


# hand-derived closed form validating the reference itself: with pseudoinv == 0 every
# candidate reduces to the constant reconstruction coeff = (p_cc, 0, ..., 0), all
# smoothness indicators vanish, so the blend is exactly that constant and
# p_out_e = p_cc(upwind cell) * quad_1 * mass_flx on every edge
@pytest.mark.level("integration")
def test_reference_constant_reconstruction_closed_form(torus_patch):
    inputs = _random_inputs(torus_patch, seed=1)
    n_cells = torus_patch.c2e2c.shape[0]
    stencil_c9 = weno.create_stencil_c9(torus_patch.c2e2c, torus_patch.c2v)
    p_out_e = _upwind_hflux_miura3_weno_reference(
        p_cc=inputs["p_cc"],
        pseudoinv=np.zeros((n_cells, N_CAND, 5, 9)),
        stencil_c9=stencil_c9,
        lsq_moments=np.zeros((n_cells, 5)),
        cell_area=inputs["cell_area"],
        c2e=torus_patch.c2e,
        upwind_cell=inputs["upwind_cell"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=weno.L_WEIGHTS_S,
    )
    p_cc_upwind = inputs["p_cc"][inputs["upwind_cell"], np.arange(NLEV)[np.newaxis, :]]
    expected = p_cc_upwind * inputs["quad"][:, :, 0] * inputs["p_mass_flx_e"]
    np.testing.assert_allclose(p_out_e, expected, rtol=1e-14, atol=1e-15)
    assert np.all(p_out_e != 0.0), "vacuous closed-form check: zero fluxes"


def _run_gt4py_pipeline(
    torus_patch,
    backend,
    *,
    p_cc: np.ndarray,
    direct: np.ndarray,  # (n_cells, 27, 5, 3)
    butterfly: np.ndarray,  # (n_cells, 27, 5, 9)
    lsq_moments: np.ndarray,
    cell_area: np.ndarray,
    rel_idx: np.ndarray,  # (n_edges, nlev) int32
    quad: np.ndarray,  # (n_edges, nlev, 6)
    p_mass_flx_e: np.ndarray,
    l_weights_s: np.ndarray,
) -> np.ndarray:
    """Run the runtime stencils in the ThirdOrderMiuraWeno order on the patch."""
    n_cells = torus_patch.c2e2c.shape[0]
    n_edges = torus_patch.e2c.shape[0]
    nlev = p_cc.shape[1]

    def connectivity(table, source_dim, target_dims):
        return gtx.as_connectivity(
            target_dims, source_dim, data=table, dtype=gtx.int32, allocator=backend
        )

    offset_provider = {
        "C2E2C": connectivity(torus_patch.c2e2c, dims.CellDim, (dims.CellDim, dims.C2E2CDim)),
        "C2E2C2E2C": connectivity(
            torus_patch.c2e2c2e2c, dims.CellDim, (dims.CellDim, dims.C2E2C2E2CDim)
        ),
        "E2C": connectivity(torus_patch.e2c, dims.CellDim, (dims.EdgeDim, dims.E2CDim)),
    }

    def cell_field(values):
        return gtx.as_field((dims.CellDim,), values, allocator=backend)

    def cell_k_field(values):
        return gtx.as_field((dims.CellDim, dims.KDim), values, allocator=backend)

    def edge_k_field(values):
        return gtx.as_field((dims.EdgeDim, dims.KDim), values, allocator=backend)

    p_cc_field = cell_k_field(p_cc)
    moments_fields = {
        f"lsq_moments_{u + 1}": cell_field(lsq_moments[:, u].copy()) for u in range(5)
    }
    coeff_fields = {
        f"p_coeff_{c + 1}_dsl": cell_k_field(np.zeros((n_cells, nlev))) for c in range(6)
    }
    quad_fields = {
        f"z_quad_vector_sum_{q + 1}": edge_k_field(quad[:, :, q].copy()) for q in range(6)
    }
    accumulators = {
        **{f"z_lsq_weighted_{q + 1}": edge_k_field(np.zeros((n_edges, nlev))) for q in range(6)},
        "smooth_sum": edge_k_field(np.zeros((n_edges, nlev))),
    }
    rel_idx_field = edge_k_field(rel_idx)
    cell_area_field = cell_field(cell_area)

    cell_domain = dict(
        horizontal_start=0,
        horizontal_end=gtx.int32(n_cells),
        vertical_start=0,
        vertical_end=gtx.int32(nlev),
    )
    edge_domain = dict(
        horizontal_start=0,
        horizontal_end=gtx.int32(n_edges),
        vertical_start=0,
        vertical_end=gtx.int32(nlev),
    )

    for cand in range(N_CAND):
        reconstruct_quadratic_coefficients_weno_candidate.with_backend(backend)(
            p_cc=p_cc_field,
            **{
                f"lsq_pseudoinv_direct_{u + 1}": gtx.as_field(
                    (dims.CellDim, dims.C2E2CDim), direct[:, cand, u, :].copy(), allocator=backend
                )
                for u in range(5)
            },
            **{
                f"lsq_pseudoinv_butterfly_{u + 1}": gtx.as_field(
                    (dims.CellDim, dims.C2E2C2E2CDim),
                    butterfly[:, cand, u, :].copy(),
                    allocator=backend,
                )
                for u in range(5)
            },
            **moments_fields,
            **coeff_fields,
            **cell_domain,
            offset_provider=offset_provider,
        )
        accumulate_weno_candidate_flux_weights.with_backend(backend)(
            **{f"p_coeff_{c + 1}": coeff_fields[f"p_coeff_{c + 1}_dsl"] for c in range(6)},
            cell_area=cell_area_field,
            p_cell_rel_idx_dsl=rel_idx_field,
            **quad_fields,
            **accumulators,
            l_weight_s=float(l_weights_s[cand]),
            **edge_domain,
            offset_provider=offset_provider,
        )

    p_out_e = edge_k_field(np.zeros((n_edges, nlev)))
    compute_horizontal_tracer_flux_from_weno_coefficients.with_backend(backend)(
        **accumulators,
        **{
            f"p_quad_vector_sum_{q + 1}": quad_fields[f"z_quad_vector_sum_{q + 1}"]
            for q in range(6)
        },
        p_mass_flx_e=edge_k_field(p_mass_flx_e),
        p_out_e=p_out_e,
        **edge_domain,
        offset_provider=offset_provider,
    )
    return p_out_e.asnumpy()


# main gate: reference (Fortran order) vs pipeline (scattered offsets) on random data.
# The live L_WEIGHTS_S leaves candidates 4-21 inert (weight 0); the synthetic all-distinct
# weights additionally pin the candidate/weight pairing across all 27 launches.
@pytest.mark.level("integration")
@pytest.mark.parametrize("weights", ["live", "synthetic"])
def test_pipeline_matches_fortran_reference(torus_patch, patch_coefficients, backend, weights):
    inputs = _random_inputs(torus_patch, seed=2)
    l_weights_s = (
        weno.L_WEIGHTS_S
        if weights == "live"
        else np.random.default_rng(3).uniform(0.5, 3.0, N_CAND)
    )

    expected = _upwind_hflux_miura3_weno_reference(
        p_cc=inputs["p_cc"],
        pseudoinv=patch_coefficients["pseudoinv"],
        stencil_c9=patch_coefficients["stencil_c9"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        c2e=torus_patch.c2e,
        upwind_cell=inputs["upwind_cell"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=l_weights_s,
    )
    actual = _run_gt4py_pipeline(
        torus_patch,
        backend,
        p_cc=inputs["p_cc"],
        direct=patch_coefficients["direct"],
        butterfly=patch_coefficients["butterfly"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        rel_idx=inputs["rel_idx"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=l_weights_s,
    )
    assert np.all(expected != 0.0), "vacuous cross-check: zero reference fluxes"
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def _triangle_average(f, vertices: np.ndarray) -> np.ndarray:
    """Average over triangles via the edge-midpoint rule, exact for quadratics."""
    m01 = 0.5 * (vertices[..., 0, :] + vertices[..., 1, :])
    m12 = 0.5 * (vertices[..., 1, :] + vertices[..., 2, :])
    m20 = 0.5 * (vertices[..., 2, :] + vertices[..., 0, :])
    return (f(m01) + f(m12) + f(m20)) / 3.0


def _smoothness_vector(coeff: np.ndarray, area: float) -> np.ndarray:
    """z_lsq_smooth of a coefficient vector [c0, cx, cy, cxx, cyy, cxy] (f90 2497-2506)."""
    c = coeff
    return np.array(
        [
            c[1] ** 2 + c[2] ** 2 + area * (c[3] ** 2 + c[4] ** 2 + c[5] ** 2),
            2.0 * (c[1] * c[3] + c[2] * c[5]),
            2.0 * (c[1] * c[5] + c[2] * c[4]),
            2.0 * (c[3] ** 2 + c[5] ** 2),
            2.0 * (c[4] ** 2 + c[5] ** 2),
            2.0 * c[5] * (c[3] + c[4]),
        ]
    )


# smooth-field consistency: on an exactly quadratic tracer field every candidate
# reconstruction is exact, so the candidates form two exactly-known groups (see module
# docstring) and the WENO flux has a closed form computed from the polynomial alone
@pytest.mark.level("integration")
def test_pipeline_smooth_field_closed_form(torus_patch, patch_coefficients, backend):
    n_edges = torus_patch.e2c.shape[0]
    inputs = _random_inputs(torus_patch, seed=4)
    # constant-in-k upwind selection so each edge has one upwind cell
    rel_idx = np.broadcast_to(
        np.random.default_rng(5).integers(0, 2, size=(n_edges, 1)), (n_edges, NLEV)
    ).astype(np.int32)
    upwind_cell = torus_patch.e2c[np.arange(n_edges), rel_idx[:, 0]]

    # random global quadratic q(x, y); p_cc = exact cell average at each cell's stored
    # (wrapped) center position
    a, b, c, d, e, f = np.random.default_rng(6).uniform(-1.0, 1.0, 6)

    def poly(v: np.ndarray) -> np.ndarray:
        x, y = v[..., 0], v[..., 1]
        return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y

    centers = np.stack((torus_patch.cell_center_x, torus_patch.cell_center_y), axis=1)
    global_vertices = centers[:, np.newaxis, :] + torus_patch.local_vertices
    p_cc_1d = _triangle_average(poly, global_vertices)
    p_cc = np.broadcast_to(p_cc_1d[:, np.newaxis], (centers.shape[0], NLEV)).copy()

    # a cell sees consistent global increments iff no stencil member is a wrapped image
    # (its stored center already is the minimal image seen from the cell); the WENO flux
    # of an edge only depends on its upwind cell's stencil
    stencil_c9 = patch_coefficients["stencil_c9"]
    period = np.array([torus_patch.domain_length, torus_patch.domain_height])
    stencil_offsets = centers[stencil_c9] - centers[:, np.newaxis, :]
    clean_cell = np.all(np.abs(stencil_offsets) < 0.5 * period, axis=(1, 2))
    valid_edge = clean_cell[upwind_cell]
    assert valid_edge.sum() > 20, "too few seam-free edges: enlarge the patch"

    actual = _run_gt4py_pipeline(
        torus_patch,
        backend,
        p_cc=p_cc,
        direct=patch_coefficients["direct"],
        butterfly=patch_coefficients["butterfly"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        rel_idx=rel_idx,
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=weno.L_WEIGHTS_S,
    )

    # closed form per valid edge, from the polynomial only: local derivative
    # coefficients at the upwind center, the two exact candidate groups, and the
    # explicit two-group WENO blend
    lsq_moments = patch_coefficients["lsq_moments"]
    expected = np.empty((valid_edge.sum(), NLEV))
    for row, edge in enumerate(np.flatnonzero(valid_edge)):
        u = upwind_cell[edge]
        xc, yc = centers[u]
        deriv = np.array([b + 2.0 * d * xc + f * yc, c + 2.0 * e * yc + f * xc, d, e, f])
        coeff_q = np.concatenate(([p_cc_1d[u] - deriv @ lsq_moments[u]], deriv))
        deriv_c = CORRECTED_FACTOR * deriv
        coeff_c = np.concatenate(([p_cc_1d[u] - deriv_c @ lsq_moments[u]], deriv_c))
        area = inputs["cell_area"][u]
        for jk in range(NLEV):
            quad_e = inputs["quad"][edge, jk]
            beta_q = _smoothness_vector(coeff_q, area) @ quad_e
            beta_c = _smoothness_vector(coeff_c, area) @ quad_e
            # guard the closed form against accidental near-cancellation of beta
            assert min(abs(beta_q), abs(beta_c)) > 1e-6
            # candidates 1-3: weight 1 each; 22-27: weight LAMBDA_S each; 4-21: weight 0
            w_c = 1.0 / (beta_c + WENO_EPS) ** 2
            w_q = LAMBDA_S / (beta_q + WENO_EPS) ** 2
            blend = (3.0 * w_c * coeff_c + 6.0 * w_q * coeff_q) / (3.0 * w_c + 6.0 * w_q)
            expected[row, jk] = blend @ quad_e * inputs["p_mass_flx_e"][edge, jk]

    np.testing.assert_allclose(actual[valid_edge], expected, rtol=1e-9, atol=1e-11)
