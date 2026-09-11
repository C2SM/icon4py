# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The (1 - S) factor of the quadratic WENO type-VI candidates and the blend bias it causes.

These tests document a known deficiency of the published construction; they do not test a
desired property. The quadratic WENO scheme (Jocksch et al., PPAM 2026, section 2.3) obtains
its three type-VI candidates from the full-stencil pseudoinverse: the Fortran assembles them as
A+_full - sum_{i in group} d_i A+_i (mo_intp_coeffs_lsq_bln.f90 2669-2680 on
transport_ajocksch_capture), and 'weno_least_squares.compute_weno_pseudoinverse_quadratic'
ports that literally. For smooth data every fitted
candidate reproduces the derivatives, so a type-VI candidate returns (1 - S) times them, S the
group's linear-weight sum (5.9831 OPTIMIZED, 8 UNITY). Its smoothness indicator is then
(1 - S)^2 and its nonlinear weight (1 - S)^-4 times what the linear weights assume, at every
resolution, and the normalised blend returns the derivatives short by the constant

    delta = (D + 3 (1 - S)^-3) / (D + 3 (1 - S)^-4) - 1,   D = sum of the fitted candidates' d_j,

-1.6214e-3 (OPTIMIZED) and -4.1647e-4 (UNITY): a first-order numerical diffusion, which is why
the quadratic WENO rows of model/driver/tests/driver/scientific_validation/
test_weno_order_study.py fall to first order on smooth data. The finding and the measurements:
model/atmosphere/tracer_advection/docs/weno_idealized_status.md, section "W6".

If the type-VI assembly is changed (for example to (1 + S) A+_full - sum d_i A+_i, which
removes the bias), these tests fail. That is intended: they pin the published construction the
port reproduces, so a change to it has to update them, the order study's gates and the status
note together.

numpy only, after the W6 review's coefficient-level check (workspace
weno_data/w6_review/mech.py): the port's coefficient functions on two resolutions of one
equilateral torus patch (edge lengths 1 and 1/2), applied to a quadratic field and to a
Gaussian. The smoothness indicator is transcribed from 'accumulate_weno_candidate_flux_weights'
(f90 2996-3008), in double rather than the Fortran's REAL(sp); its quadrature vector is the
area average of the monomials over the departure region of each outflow edge, for a
displacement of 0.11 edge lengths along the domain diagonal. Only h / r_e matters: the fits are
scale free and the 1e-20 regularisation is far below the indicators here.
"""

from typing import Final

import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import weno_least_squares as weno

from ..utils import TorusPatch, build_torus_patch


_OPTIMIZED = weno.WenoLinearWeights.OPTIMIZED
_UNITY = weno.WenoLinearWeights.UNITY

#: the documented values (docs/weno_idealized_status.md, "W6")
_GROUP_WEIGHT_SUM: Final = {_OPTIMIZED: 5.9831, _UNITY: 8.0}
_BLEND_BIAS: Final = {_OPTIMIZED: -1.6214e-3, _UNITY: -4.1647e-4}

#: (quads per direction, edge length): one domain, bisected
_RESOLUTIONS: Final = ((24, 1.0), (48, 0.5))
#: e-folding radius of the Gaussian at the domain centre: h / r_e = 1/6 and 1/12
_GAUSSIAN_RADIUS: Final = 6.0
#: the core cells whose outflow edges are evaluated: centre within 1.2 r_e of the Gaussian's,
#: so that no stencil reaches the periodic seam of the minimum-image Gaussian
_CORE_RADIUS: Final = 1.2
#: departure-region displacement in edge lengths (the order study's CFL number)
_DISPLACEMENT: Final = 0.11
#: f90 3008 (accumulate_weno_candidate_flux_weights._WENO_EPS)
_EPS: Final = 1e-20

#: slots of the three type-VI candidates' groups: candidate k is assembled from slots
#: 3 + k, 6 + k, ..., 24 + k (f90 2670-2680)
_GROUPS: Final = tuple(tuple(range(3 + k, 27, 3)) for k in range(3))


@pytest.fixture(scope="module")
def patches() -> dict[float, TorusPatch]:
    return {h: build_torus_patch(n, n, h) for n, h in _RESOLUTIONS}


def _group_weight_sums(option: weno.WenoLinearWeights) -> np.ndarray:
    weights = weno.linear_weights(option)
    return np.array([weights[list(group)].sum() for group in _GROUPS])


def _predicted_blend_bias(option: weno.WenoLinearWeights) -> float:
    """delta from S and D, for fitted candidates that all return the same derivatives."""
    s = _group_weight_sums(option)[0]
    d = weno.linear_weights(option)[3:].sum()
    return float((d + 3.0 * (1.0 - s) ** -3) / (d + 3.0 * (1.0 - s) ** -4) - 1.0)


def _pseudoinverse(patch: TorusPatch, option: weno.WenoLinearWeights) -> np.ndarray:
    return weno.compute_weno_pseudoinverse_quadratic(
        stencil_c9=weno.create_stencil_c9(patch.c2e2c, patch.c2v),
        lsq_moments=weno.compute_lsq_moments_torus(
            cell_center_x=patch.cell_center_x,
            cell_center_y=patch.cell_center_y,
            vertex_x=patch.vertex_x,
            vertex_y=patch.vertex_y,
            c2v=patch.c2v,
            domain_length=patch.domain_length,
            domain_height=patch.domain_height,
        ),
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
        l_weights_s=weno.linear_weights(option),
    )


# --- the (1 - S) factor on a quadratic field


#: derivative coefficients [x, y, x^2, y^2, xy] of the quadratic, all non-zero
_QUADRATIC: Final = np.array([0.8, -0.6, 0.3, -0.25, 0.4])


def _quadratic_increments(patch: TorusPatch, stencil: np.ndarray) -> np.ndarray:
    """z_b of the quadratic (in every cell's own frame): 3-point cell averages, exact."""
    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        neighbor_table=stencil,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )

    def average(vertices: np.ndarray) -> np.ndarray:
        midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=-2))
        x, y = midpoints[..., 0], midpoints[..., 1]
        b, c, d, e, f = _QUADRATIC
        return np.mean(b * x + c * y + d * x**2 + e * y**2 + f * x * y, axis=-1)

    stencil_average = average(patch.local_vertices[stencil] + z_dist[:, :, np.newaxis, :])
    return stencil_average - average(patch.local_vertices)[:, np.newaxis]


def test_group_weight_sums_and_predicted_bias():
    for option in (_OPTIMIZED, _UNITY):
        sums = _group_weight_sums(option)
        np.testing.assert_array_equal(sums, sums[0])
        print(
            f"\n{option.name}: S = {sums[0]:.6f}, 1 - S = {1.0 - sums[0]:.6f}, "
            f"predicted delta = {_predicted_blend_bias(option):.5e}"
        )
        assert sums[0] == pytest.approx(_GROUP_WEIGHT_SUM[option], abs=5e-5)
        assert _predicted_blend_bias(option) == pytest.approx(_BLEND_BIAS[option], rel=1e-4)


@pytest.mark.parametrize("option", [_OPTIMIZED, _UNITY], ids=lambda option: option.name)
@pytest.mark.parametrize("edge_length", [h for _, h in _RESOLUTIONS])
def test_type_vi_candidates_return_one_minus_s_times_the_derivatives(patches, option, edge_length):
    """Documents the published construction: fails if the type-VI assembly is fixed."""
    patch = patches[edge_length]
    stencil = weno.create_stencil_c9(patch.c2e2c, patch.c2v)
    coefficients = np.einsum(
        "ncus,ns->ncu", _pseudoinverse(patch, option), _quadratic_increments(patch, stencil)
    )
    one_minus_s = 1.0 - _group_weight_sums(option)[0]
    ratio = coefficients[:, :3] / _QUADRATIC
    print(
        f"\n{option.name} h = {edge_length}: type-VI / true derivatives in "
        f"[{ratio.min():.10f}, {ratio.max():.10f}], 1 - S = {one_minus_s:.10f}, max relative "
        f"deviation {np.max(np.abs(ratio / one_minus_s - 1.0)):.1e}"
    )
    # the fitted candidates reproduce the quadratic ...
    np.testing.assert_allclose(
        coefficients[:, 3:], np.broadcast_to(_QUADRATIC, coefficients[:, 3:].shape), rtol=1e-8
    )
    # ... and the assembled type-VI candidates (1 - S) times it
    np.testing.assert_allclose(
        coefficients[:, :3],
        np.broadcast_to(one_minus_s * _QUADRATIC, coefficients[:, :3].shape),
        rtol=1e-8,
    )


# --- the blend bias delta on a Gaussian


def _dunavant_7() -> tuple[np.ndarray, np.ndarray]:
    """Degree-5 triangle rule: barycentric points (7, 3) and weights summing to 1."""
    a1, b1 = 0.059715871789770, 0.470142064105115
    a2, b2 = 0.797426985353087, 0.101286507323456
    points = [(1 / 3, 1 / 3, 1 / 3)]
    points += [(a1, b1, b1), (b1, a1, b1), (b1, b1, a1), (a2, b2, b2), (b2, a2, b2), (b2, b2, a2)]
    weights = [0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3
    return np.array(points), np.array(weights)


def _offsets_from_domain_centre(patch: TorusPatch) -> np.ndarray:
    """Minimum-image cell-centre offsets from the domain centre, (n_cells, 2)."""
    offsets = []
    for coordinate, period in (
        (patch.cell_center_x, patch.domain_length),
        (patch.cell_center_y, patch.domain_height),
    ):
        delta = coordinate - 0.5 * period
        offsets.append(delta - period * np.round(delta / period))
    return np.stack(offsets, axis=1)


def _polygon_monomial_averages(vertices: np.ndarray) -> np.ndarray:
    """Area averages of [1, x, y, x^2, y^2, xy] over polygons (n, n_vertices, 2)."""
    x, y = vertices[..., 0], vertices[..., 1]
    xn, yn = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum(axis=1)
    integrals = [
        area,
        (cross * (x + xn)).sum(axis=1) / 6.0,
        (cross * (y + yn)).sum(axis=1) / 6.0,
        (cross * (x**2 + x * xn + xn**2)).sum(axis=1) / 12.0,
        (cross * (y**2 + y * yn + yn**2)).sum(axis=1) / 12.0,
        (cross * (x * yn + 2.0 * x * y + 2.0 * xn * yn + xn * y)).sum(axis=1) / 24.0,
    ]
    return np.stack(integrals, axis=1) / area[:, np.newaxis]


def _outflow_departure_regions(
    patch: TorusPatch, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(index into 'cells', quadrature vector (n_edges, 6)) of every outflow edge of 'cells'.

    The departure region of an outflow edge is the parallelogram between the edge and its
    copy displaced upstream, in the upwind cell's frame.
    """
    wind = np.array([patch.domain_length, patch.domain_height])
    wind /= np.linalg.norm(wind)
    displacement = _DISPLACEMENT * patch.edge_length * wind
    vertices = patch.local_vertices[cells]
    upwind, quad_vectors = [], []
    for a, b, opposite in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        edge = vertices[:, b] - vertices[:, a]
        normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
        normal *= np.sign(np.sum((vertices[:, a] - vertices[:, opposite]) * normal, axis=1))[
            :, np.newaxis
        ]
        outflow = np.nonzero(normal @ wind > 0.0)[0]
        region = np.stack(
            [
                vertices[:, a],
                vertices[:, b],
                vertices[:, b] - displacement,
                vertices[:, a] - displacement,
            ],
            axis=1,
        )[outflow]
        upwind.append(outflow)
        quad_vectors.append(_polygon_monomial_averages(region))
    return np.concatenate(upwind), np.concatenate(quad_vectors)


def _smoothness_indicator(
    coefficients: np.ndarray, area: float, quad_vector: np.ndarray
) -> np.ndarray:
    """accumulate_weno_candidate_flux_weights (f90 2996-3007) in double, (n_edges, 27)."""
    c2, c3, c4, c5, c6 = np.moveaxis(coefficients, -1, 0)
    smooth = (
        c2**2 + c3**2 + area * (c4**2 + c5**2 + c6**2),
        2.0 * (c2 * c4 + c3 * c6),
        2.0 * (c2 * c6 + c3 * c5),
        2.0 * (c4**2 + c6**2),
        2.0 * (c5**2 + c6**2),
        2.0 * c6 * (c4 + c5),
    )
    return sum(s * quad_vector[:, np.newaxis, i] for i, s in enumerate(smooth))


def _blend_bias(patch: TorusPatch, option: weno.WenoLinearWeights) -> tuple[float, int]:
    """(delta, number of edges) on the outflow edges of the Gaussian's core cells.

    delta is the projection of the blended gradient minus the gradient blended from the fitted
    candidates alone onto the latter, summed over the edges: the part of the blend the
    type-VI candidates contribute, relative to the derivatives the fitted ones agree on.
    """
    offsets = _offsets_from_domain_centre(patch)
    points, weights = _dunavant_7()
    quadrature_points = offsets[:, np.newaxis, :] + np.einsum(
        "qv,nvd->nqd", points, patch.local_vertices
    )
    averages = np.exp(-np.sum(quadrature_points**2, axis=-1) / _GAUSSIAN_RADIUS**2) @ weights

    cells = np.nonzero(np.hypot(*offsets.T) <= _CORE_RADIUS * _GAUSSIAN_RADIUS)[0]
    stencil = weno.create_stencil_c9(patch.c2e2c, patch.c2v)[cells]
    increments = averages[stencil] - averages[cells, np.newaxis]
    coefficients = np.einsum("ncus,ns->ncu", _pseudoinverse(patch, option)[cells], increments)

    upwind, quad_vector = _outflow_departure_regions(patch, cells)
    coefficients = coefficients[upwind]
    area = np.sqrt(3.0) / 4.0 * patch.edge_length**2
    beta = _smoothness_indicator(coefficients, area, quad_vector)
    alpha = weno.linear_weights(option) / (beta + _EPS) ** 2
    blend = np.einsum("nk,nku->nu", alpha, coefficients) / alpha.sum(axis=1, keepdims=True)
    fitted = np.einsum("nk,nku->nu", alpha[:, 3:], coefficients[:, 3:]) / alpha[:, 3:].sum(
        axis=1, keepdims=True
    )
    gradient_difference = (blend - fitted)[:, :2]
    delta = np.sum(gradient_difference * fitted[:, :2]) / np.sum(fitted[:, :2] ** 2)
    return float(delta), upwind.size


@pytest.mark.parametrize("option", [_OPTIMIZED, _UNITY], ids=lambda option: option.name)
def test_blend_bias_is_the_constant_delta(patches, option):
    """Documents the published construction: fails if the type-VI assembly is fixed."""
    deltas = {}
    for edge_length, patch in patches.items():
        deltas[edge_length], n_edges = _blend_bias(patch, option)
        print(
            f"\n{option.name} h = {edge_length} (h / r_e = {edge_length / _GAUSSIAN_RADIUS:.4f}, "
            f"{n_edges} edges): delta = {deltas[edge_length]:.5e}, "
            f"{deltas[edge_length] / _BLEND_BIAS[option] - 1.0:+.2e} from the documented "
            f"{_BLEND_BIAS[option]:.4e}"
        )
    for delta in deltas.values():
        assert delta == pytest.approx(_BLEND_BIAS[option], rel=1e-2)
    coarse, fine = (deltas[h] for _, h in _RESOLUTIONS)
    assert fine == pytest.approx(coarse, rel=1e-2)
