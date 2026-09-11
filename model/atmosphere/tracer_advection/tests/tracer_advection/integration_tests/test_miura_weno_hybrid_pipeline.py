# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Full-pipeline cross-check of the hybrid quadratic / WENO tracer flux (ihadv_tracer=132).

The reference is a literal numpy re-implementation of the Fortran cell loop
(upwind_hflux_miura_weno_hyb, mo_advection_hflux.f90 3547-3696): the full 9-point fit,
its single-precision residual against ICON's 'lsq_error', the threshold test
``lsqe <= 5e-5 * (p_cc + 1e-10)**2`` with the constants as single-precision literals,
and either the plain quadratic flux or the 27-candidate WENO blend with unit linear
weights (f90 3666) on the edges owned by the cell.

Gates on the reference alone (plain numpy, run these with -n0 and no backend):
- with the threshold pushed to +inf / -1 the reference collapses to the plain quadratic
  flux / the 103 reference of test_miura3_weno_pipeline with unit weights, so the two
  branches are the exact code paths they claim to be;
- the residual is the Fortran's: ICON's 'lsq_error' is the distance-*weighted* design
  matrix but z_b is unweighted, so the residual of an exact linear or quadratic field is
  not round-off but ~(1 - w_outer)^2 * z_b_outer^2, and with c_sel = 5e-5 only cells whose
  stencil is constant to ~0.5% keep the plain branch (a slope of 1% per edge sends every
  cell to WENO). The selection gate therefore uses a nearly constant field (plain) and an
  O(1) field (WENO), and pins the residual of a linear field to that closed form;
- each of those has a mutation twin that shows the assertion can fail.

The gt4py gate composes the runtime stencils as ThirdOrderMiuraWenoHybrid does and
compares fluxes and masks against the reference on random data. The residual's
summation order differs from the Fortran's (direct rows, then butterfly rows), so cells
whose residual lies within a narrow band around the threshold are excluded from the
mask comparison; the band is asserted to be almost empty.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import weno_least_squares as weno
from icon4py.model.atmosphere.tracer_advection.stencils.accumulate_weno_candidate_flux_weights import (
    accumulate_weno_candidate_flux_weights,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_quadratic_coefficients import (
    compute_horizontal_tracer_flux_from_quadratic_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_weno_coefficients import (
    compute_horizontal_tracer_flux_from_weno_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_weno_hybrid_stencil_selection import (
    compute_weno_hybrid_stencil_selection,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_quadratic_coefficients_svd import (
    reconstruct_quadratic_coefficients_svd,
)
from icon4py.model.atmosphere.tracer_advection.stencils.select_horizontal_tracer_flux_by_upwind_cell import (
    select_horizontal_tracer_flux_by_upwind_cell,
)
from icon4py.model.common import dimension as dims, model_backends, type_alias as ta
from icon4py.model.common.initial_condition.analytical import moving_cylinder
from icon4py.model.testing import (
    definitions as test_defs,
    grid_utils as gridtest_utils,
    serialbox as sb,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    process_props,
)

from .. import utils
from ..fixtures import advection_init_savepoint
from .test_jocksch_reference import NUM_LEVELS as CAPTURE_NUM_LEVELS, case, date, experiment
from .test_miura3_weno_pipeline import (
    N_CAND,
    NLEV,
    _random_inputs,
    _upwind_hflux_miura3_weno_reference,
)


#: the Fortran's single-precision literals (f90 3574) as the doubles they are promoted to
THRESHOLD = float(np.float32(5e-5))
EPS = float(np.float32(1e-10))
#: the Fortran's REAL(sp) quantities of the residual path, as the port resolves them
SP = ta.fortran_sp_float
UNIT_WEIGHTS = np.ones(N_CAND)


@pytest.fixture(scope="module")
def torus_patch() -> utils.TorusPatch:
    return utils.build_torus_patch()


@pytest.fixture(scope="module")
def patch_coefficients(torus_patch) -> dict:
    """Real init-time coefficients of both branches and ICON's lsq_error."""
    stencil_c9 = weno.create_stencil_c9(torus_patch.c2e2c, torus_patch.c2v)
    geometry = dict(
        stencil_c9=stencil_c9,
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    lsq_moments = weno.compute_lsq_moments_torus(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        vertex_x=torus_patch.vertex_x,
        vertex_y=torus_patch.vertex_y,
        c2v=torus_patch.c2v,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    pseudoinv_full = weno.compute_lsq_pseudoinverse_quadratic(lsq_moments=lsq_moments, **geometry)
    pseudoinv_weno = weno.compute_weno_pseudoinverse_quadratic(lsq_moments=lsq_moments, **geometry)
    lsq_error = weno.compute_lsq_error_quadratic(lsq_moments=lsq_moments, **geometry)
    scatter = dict(stencil_c9=stencil_c9, c2e2c=torus_patch.c2e2c, c2e2c2e2c=torus_patch.c2e2c2e2c)
    full_direct, full_butterfly = weno.scatter_to_offsets(
        values_fortran_order=pseudoinv_full[:, np.newaxis], **scatter
    )
    weno_direct, weno_butterfly = weno.scatter_to_offsets(
        values_fortran_order=pseudoinv_weno, **scatter
    )
    error_direct, error_butterfly = weno.scatter_to_offsets(
        values_fortran_order=lsq_error[:, np.newaxis], **scatter
    )
    return dict(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        pseudoinv_full=pseudoinv_full,
        pseudoinv_weno=pseudoinv_weno,
        lsq_error=lsq_error,
        full_direct=full_direct[:, 0],
        full_butterfly=full_butterfly[:, 0],
        weno_direct=weno_direct,
        weno_butterfly=weno_butterfly,
        error_direct=error_direct[:, 0],
        error_butterfly=error_butterfly[:, 0],
        butterfly_active=weno.compute_butterfly_slot_mask(**scatter),
    )


def _fit_residual_reference(
    lsq_error: np.ndarray, coeff: np.ndarray, z_b: np.ndarray
) -> np.floating:
    """f90 3564-3568 for one cell: the residual of the fit in the Fortran's REAL(sp).

    lsq_error is (5, 9), coeff the 6 double coefficients [c0, cx, cy, cxx, cyy, cxy], z_b
    the 9 double increments in stencil order; every intermediate is rounded to SP.
    """
    lsq_error = lsq_error.astype(SP)
    zlc = coeff[1:6].astype(SP)
    lsqe = SP(0.0)
    for js in range(9):
        # DOT_PRODUCT(lsq_error(1:5, is), zlc(1:5)) in single precision, u = 1..5
        dot = SP(0.0)
        for ju in range(5):
            dot = SP(dot + lsq_error[ju, js] * zlc[ju])
        lsqe = SP(lsqe + SP(dot - SP(z_b[js])) ** 2)
    return lsqe


def _upwind_hflux_miura_weno_hyb_reference(
    *,
    p_cc: np.ndarray,  # (n_cells, nlev)
    pseudoinv_full: np.ndarray,  # (n_cells, 5, 9), Fortran stencil order
    pseudoinv_weno: np.ndarray,  # (n_cells, 27, 5, 9)
    lsq_error: np.ndarray,  # (n_cells, 5, 9)
    stencil_c9: np.ndarray,  # (n_cells, 9)
    lsq_moments: np.ndarray,  # (n_cells, 5)
    cell_area: np.ndarray,  # (n_cells,)
    c2e: np.ndarray,  # (n_cells, 3)
    upwind_cell: np.ndarray,  # (n_edges, nlev)
    quad: np.ndarray,  # (n_edges, nlev, 6)
    p_mass_flx_e: np.ndarray,  # (n_edges, nlev)
    threshold: float = THRESHOLD,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal port of the hybrid cell loop (f90 3547-3696), no limiter.

    Returns (p_out_e, use_weno (n_cells, nlev), lsqe (n_cells, nlev) in SP).
    """
    n_cells, nlev = p_cc.shape
    p_out_e = np.zeros(p_mass_flx_e.shape)
    use_weno = np.zeros((n_cells, nlev), dtype=bool)
    lsqe_out = np.zeros((n_cells, nlev), dtype=SP)
    # the WENO branch is the 103 loop with unit weights; take it from the 103 reference
    # (per edge, only the fluxes of the edges owned by the cell are written there too)
    p_out_weno = _upwind_hflux_miura3_weno_reference(
        p_cc=p_cc,
        pseudoinv=pseudoinv_weno,
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_area=cell_area,
        c2e=c2e,
        upwind_cell=upwind_cell,
        quad=quad,
        p_mass_flx_e=p_mass_flx_e,
        l_weights_s=UNIT_WEIGHTS,
    )
    for jc in range(n_cells):
        for jk in range(nlev):
            # f90 3547-3549
            z_b = np.empty(9)
            for js in range(9):
                z_b[js] = p_cc[stencil_c9[jc, js], jk] - p_cc[jc, jk]
            # f90 3553-3562: the full-stencil fit, coefficients 2..6
            coeff = np.empty(6)
            for ju in range(5):
                coeff[1 + ju] = np.dot(pseudoinv_full[jc, ju, :], z_b)
            # f90 3564-3568
            lsqe = _fit_residual_reference(lsq_error[jc], coeff, z_b)
            lsqe_out[jc, jk] = lsqe
            # f90 3569-3573
            jee = c2e[jc]
            jf = np.array([upwind_cell[jee[ie], jk] == jc for ie in range(3)])
            # f90 3574: lsqe (single) .le. 5e-5 * (p_cc + 1e-10)**2 (double)
            plain = np.float64(lsqe) <= threshold * (p_cc[jc, jk] + eps) ** 2
            use_weno[jc, jk] = not plain
            if plain:
                # f90 3575-3582: c0 from the linear constraint, same coefficients on all edges
                coeff[0] = p_cc[jc, jk] - np.dot(coeff[1:6], lsq_moments[jc])
                for ie in range(3):
                    if jf[ie]:
                        # f90 3690-3696
                        p_out_e[jee[ie], jk] = (
                            np.dot(coeff, quad[jee[ie], jk, :]) * p_mass_flx_e[jee[ie], jk]
                        )
            else:
                for ie in range(3):
                    if jf[ie]:
                        p_out_e[jee[ie], jk] = p_out_weno[jee[ie], jk]
    return p_out_e, use_weno, lsqe_out


def _reference_kwargs(torus_patch, patch_coefficients, inputs) -> dict:
    return dict(
        p_cc=inputs["p_cc"],
        pseudoinv_full=patch_coefficients["pseudoinv_full"],
        pseudoinv_weno=patch_coefficients["pseudoinv_weno"],
        lsq_error=patch_coefficients["lsq_error"],
        stencil_c9=patch_coefficients["stencil_c9"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        c2e=torus_patch.c2e,
        upwind_cell=inputs["upwind_cell"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
    )


def _plain_quadratic_flux_reference(torus_patch, patch_coefficients, inputs) -> np.ndarray:
    """miura3 (ihadv_tracer=3): the upwind cell's full fit dotted with the quadrature."""
    p_cc = inputs["p_cc"]
    stencil_c9 = patch_coefficients["stencil_c9"]
    z_b = p_cc[stencil_c9] - p_cc[:, np.newaxis, :]  # (n_cells, 9, nlev)
    deriv = np.einsum("nus,nsk->nuk", patch_coefficients["pseudoinv_full"], z_b)
    c0 = p_cc - np.einsum("nuk,nu->nk", deriv, patch_coefficients["lsq_moments"])
    coeff = np.concatenate((c0[:, np.newaxis], deriv), axis=1)  # (n_cells, 6, nlev)
    upwind = inputs["upwind_cell"]
    levels = np.arange(NLEV)[np.newaxis, :]
    coeff_upwind = np.transpose(coeff, (0, 2, 1))[upwind, levels]  # (n_edges, nlev, 6)
    return np.einsum("ekq,ekq->ek", coeff_upwind, inputs["quad"]) * inputs["p_mass_flx_e"]


# --- gates on the reference alone -------------------------------------------------------


@pytest.mark.level("integration")
def test_reference_threshold_limits_are_the_two_branches(torus_patch, patch_coefficients):
    inputs = _random_inputs(torus_patch, seed=11)
    kwargs = _reference_kwargs(torus_patch, patch_coefficients, inputs)

    all_plain, use_weno, _ = _upwind_hflux_miura_weno_hyb_reference(**kwargs, threshold=np.inf)
    assert not use_weno.any()
    np.testing.assert_allclose(
        all_plain,
        _plain_quadratic_flux_reference(torus_patch, patch_coefficients, inputs),
        rtol=1e-13,
        atol=1e-15,
    )

    all_weno, use_weno, _ = _upwind_hflux_miura_weno_hyb_reference(**kwargs, threshold=-1.0)
    assert use_weno.all()
    weno_unit = _upwind_hflux_miura3_weno_reference(
        p_cc=inputs["p_cc"],
        pseudoinv=patch_coefficients["pseudoinv_weno"],
        stencil_c9=patch_coefficients["stencil_c9"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        c2e=torus_patch.c2e,
        upwind_cell=inputs["upwind_cell"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=UNIT_WEIGHTS,
    )
    np.testing.assert_array_equal(all_weno, weno_unit)
    assert np.all(all_weno != 0.0) and np.all(all_plain != 0.0)
    # the branches are genuinely different fluxes
    assert not np.allclose(all_weno, all_plain, rtol=1e-3)


@pytest.mark.level("integration")
def test_reference_weno_branch_uses_unit_weights_not_the_live_set(torus_patch, patch_coefficients):
    # mutation: the hybrid's WENO branch is 103 with d_j = 1 (f90 3666), not with L_WEIGHTS_S
    inputs = _random_inputs(torus_patch, seed=11)
    kwargs = _reference_kwargs(torus_patch, patch_coefficients, inputs)
    all_weno, _, _ = _upwind_hflux_miura_weno_hyb_reference(**kwargs, threshold=-1.0)
    weno_live = _upwind_hflux_miura3_weno_reference(
        p_cc=inputs["p_cc"],
        pseudoinv=patch_coefficients["pseudoinv_weno"],
        stencil_c9=patch_coefficients["stencil_c9"],
        lsq_moments=patch_coefficients["lsq_moments"],
        cell_area=inputs["cell_area"],
        c2e=torus_patch.c2e,
        upwind_cell=inputs["upwind_cell"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        l_weights_s=weno.L_WEIGHTS_S,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(all_weno, weno_live, rtol=1e-6)


def _clean_cells(torus_patch, stencil_c9) -> np.ndarray:
    """Cells none of whose stencil members is a wrapped periodic image."""
    centers = np.stack((torus_patch.cell_center_x, torus_patch.cell_center_y), axis=1)
    period = np.array([torus_patch.domain_length, torus_patch.domain_height])
    offsets = centers[stencil_c9] - centers[:, np.newaxis, :]
    return np.all(np.abs(offsets) < 0.5 * period, axis=(1, 2))


def _row_weights(torus_patch, stencil_c9: np.ndarray) -> np.ndarray:
    """The max-normalised 1/dist**5 row weights of the full fit, (n_cells, 9)."""
    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=torus_patch.cell_center_x,
        cell_center_y=torus_patch.cell_center_y,
        neighbor_table=stencil_c9,
        domain_length=torus_patch.domain_length,
        domain_height=torus_patch.domain_height,
    )
    weights = 1.0 / np.linalg.norm(z_dist, axis=2) ** 5
    return weights / weights.max(axis=1, keepdims=True)


@pytest.mark.level("integration")
def test_reference_residual_of_a_linear_field_is_the_weight_mismatch(
    torus_patch, patch_coefficients
):
    # for a field the fit reproduces exactly (here: linear in x), A c = z_b row by row, so
    # the Fortran's residual against the weighted rows is sum_js ((w_js - 1) z_b_js)^2
    inputs = _random_inputs(torus_patch, seed=12)
    stencil_c9 = patch_coefficients["stencil_c9"]
    clean = _clean_cells(torus_patch, stencil_c9)
    slope = 1e-2
    p_cc = np.broadcast_to(
        (1.0 + slope * torus_patch.cell_center_x)[:, np.newaxis],
        (torus_patch.c2e2c.shape[0], NLEV),
    ).copy()
    _, use_weno, lsqe = _upwind_hflux_miura_weno_hyb_reference(
        **{**_reference_kwargs(torus_patch, patch_coefficients, inputs), "p_cc": p_cc}
    )
    z_b = p_cc[stencil_c9, 0] - p_cc[:, 0, np.newaxis]
    expected = np.sum(((_row_weights(torus_patch, stencil_c9) - 1.0) * z_b) ** 2, axis=1)
    np.testing.assert_allclose(lsqe[clean, 0], expected[clean], rtol=1e-5)
    # ... which at a 1% slope per edge is far above c_sel * q^2: every cell goes WENO
    assert use_weno[clean].all()


@pytest.mark.level("integration")
def test_reference_linear_field_residual_detects_unweighted_rows(torus_patch, patch_coefficients):
    # mutation: the paper's formula (unweighted rows) has a round-off residual instead
    inputs = _random_inputs(torus_patch, seed=12)
    stencil_c9 = patch_coefficients["stencil_c9"]
    clean = _clean_cells(torus_patch, stencil_c9)
    p_cc = np.broadcast_to(
        (1.0 + 1e-2 * torus_patch.cell_center_x)[:, np.newaxis],
        (torus_patch.c2e2c.shape[0], NLEV),
    ).copy()
    unweighted = patch_coefficients["lsq_error"] / _row_weights(torus_patch, stencil_c9)[
        :, np.newaxis, :
    ].astype(SP)
    _, use_weno, lsqe = _upwind_hflux_miura_weno_hyb_reference(
        **{
            **_reference_kwargs(torus_patch, patch_coefficients, inputs),
            "p_cc": p_cc,
            "lsq_error": unweighted.astype(SP),
        }
    )
    z_b = p_cc[stencil_c9, 0] - p_cc[:, 0, np.newaxis]
    expected = np.sum(((_row_weights(torus_patch, stencil_c9) - 1.0) * z_b) ** 2, axis=1)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(lsqe[clean, 0], expected[clean], rtol=1e-5)
    with pytest.raises(AssertionError):
        assert use_weno[clean].all()


@pytest.mark.level("integration")
def test_reference_selects_plain_where_constant_and_weno_where_not(torus_patch, patch_coefficients):
    inputs = _random_inputs(torus_patch, seed=12)
    n_cells = torus_patch.c2e2c.shape[0]
    # constant up to 1e-4: the residual is ~1e-8, c_sel * q^2 is 5e-5 -> plain everywhere
    rng = np.random.default_rng(13)
    p_cc_flat = 1.0 + 1e-4 * rng.uniform(-1.0, 1.0, size=(n_cells, NLEV))
    _, use_weno, lsqe = _upwind_hflux_miura_weno_hyb_reference(
        **{**_reference_kwargs(torus_patch, patch_coefficients, inputs), "p_cc": p_cc_flat}
    )
    assert not use_weno.any()
    assert np.all(lsqe > 0.0)
    # O(1) variation: WENO everywhere
    _, use_weno, _ = _upwind_hflux_miura_weno_hyb_reference(
        **_reference_kwargs(torus_patch, patch_coefficients, inputs)
    )
    assert use_weno.all()
    # a constant half: plain there, WENO where the stencil sees the variation (kept at
    # least 0.5 away from the constant so no residual can fall under the threshold)
    p_cc_half = np.where(
        (torus_patch.cell_center_x < 0.5 * torus_patch.domain_length)[:, np.newaxis],
        0.5 * inputs["p_cc"],
        1.0,
    )
    _, use_weno, _ = _upwind_hflux_miura_weno_hyb_reference(
        **{**_reference_kwargs(torus_patch, patch_coefficients, inputs), "p_cc": p_cc_half}
    )
    stencil_values = p_cc_half[patch_coefficients["stencil_c9"]]
    varies = (stencil_values.max(axis=1) - stencil_values.min(axis=1)) > 0.0
    assert varies.any() and not varies.all()
    np.testing.assert_array_equal(use_weno, varies)


@pytest.mark.level("integration")
def test_reference_constant_selection_detects_a_wrong_threshold(torus_patch, patch_coefficients):
    # mutation: a threshold four orders of magnitude tighter makes the 1e-4 ripple trip
    # the WENO branch, so the plain-everywhere assertion is not vacuous
    inputs = _random_inputs(torus_patch, seed=12)
    n_cells = torus_patch.c2e2c.shape[0]
    p_cc_flat = 1.0 + 1e-4 * np.random.default_rng(13).uniform(-1.0, 1.0, size=(n_cells, NLEV))
    _, use_weno, _ = _upwind_hflux_miura_weno_hyb_reference(
        **{**_reference_kwargs(torus_patch, patch_coefficients, inputs), "p_cc": p_cc_flat},
        threshold=5e-9,
    )
    with pytest.raises(AssertionError):
        assert not use_weno.any()


# --- the selection mask's sensitivity to the residual's precision -----------------------


def _selection_residual(
    *,
    p_cc: np.ndarray,  # (n_cells,)
    lsq_error: np.ndarray,  # (n_cells, 5, 9)
    pseudoinv_full: np.ndarray,  # (n_cells, 5, 9)
    stencil_c9: np.ndarray,  # (n_cells, 9)
    sp: type,
) -> tuple[np.ndarray, np.ndarray]:
    """f90 3547-3574 on one level, vectorised, with the residual path in the kind ``sp``.

    The fit (3553-3562) is double; lsq_error, zlc and every partial sum of the residual
    (3564-3568) are rounded to ``sp`` in the Fortran's order. Returns the residual ``lsqe``
    widened to double and the double threshold ``5e-5 * (p_cc + 1e-10)**2`` it is compared
    with (3574); the mask is ``lsqe > threshold``.
    """
    z_b = p_cc[stencil_c9] - p_cc[:, np.newaxis]
    coeff = np.einsum("nus,ns->nu", pseudoinv_full, z_b)
    lsq_error_sp = lsq_error.astype(sp)
    zlc = coeff.astype(sp)
    dot = np.zeros(z_b.shape, dtype=sp)
    for ju in range(5):
        dot = (dot + lsq_error_sp[:, ju, :] * zlc[:, ju : ju + 1]).astype(sp)
    residual = (dot - z_b.astype(sp)).astype(sp)
    lsqe = np.zeros(p_cc.shape, dtype=sp)
    for js in range(9):
        lsqe = (lsqe + residual[:, js] * residual[:, js]).astype(sp)
    return lsqe.astype(np.float64), THRESHOLD * (p_cc + EPS) ** 2


def _selection_mask(**kwargs) -> np.ndarray:
    """The hybrid's WENO mask (f90 3574), see _selection_residual."""
    lsqe, threshold = _selection_residual(**kwargs)
    return lsqe > threshold


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "cylinder_center", [(None, None), (0.0, 0.0)], ids=["domain_centre", "origin"]
)
def test_cylinder_selection_mask_is_the_same_in_single_and_double_precision(cylinder_center):
    """The residual's precision does not change the hybrid's mask on the cylinder's initial state.

    The port evaluates the residual in ta.fortran_sp_float, currently double, where the
    Fortran uses REAL(sp). On the moving-cylinder experiment of the driver tests (the 20 x 22
    torus with 5 km edges, the cylinder of radius 25 km at the domain centre or, as on
    Jocksch's grid, at the origin) the mask is identical either way: cells whose stencil is
    constant have a residual of exactly zero in both kinds, and cells that see the cylinder's
    edge have a residual of O(1) against a threshold of 5e-5, so nothing lands within
    single-precision round-off of the threshold. Measured 2026-09-11: 0 of 880 cells differ
    for both centres, hence the zero-count assertion.
    """
    patch = utils.build_torus_patch(nx=20, ny=22, edge_length=5000.0)
    stencil_c9 = weno.create_stencil_c9(patch.c2e2c, patch.c2v)
    geometry = dict(
        stencil_c9=stencil_c9,
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )
    lsq_moments = weno.compute_lsq_moments_torus(
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        vertex_x=patch.vertex_x,
        vertex_y=patch.vertex_y,
        c2v=patch.c2v,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )
    coefficients = dict(
        lsq_error=weno.compute_lsq_error_quadratic(lsq_moments=lsq_moments, **geometry),
        pseudoinv_full=weno.compute_lsq_pseudoinverse_quadratic(
            lsq_moments=lsq_moments, **geometry
        ),
        stencil_c9=stencil_c9,
    )
    cylinder = moving_cylinder.sample_cylinder(
        config=moving_cylinder.MovingCylinderConfig(
            center_x=cylinder_center[0], center_y=cylinder_center[1], radius=25000.0
        ),
        cell_center_x=patch.cell_center_x,
        cell_center_y=patch.cell_center_y,
        domain_length=patch.domain_length,
        domain_height=patch.domain_height,
    )
    # a disc of ~176 cells (the exact count depends on where the centroids fall)
    assert 150 < int(cylinder.sum()) < 200

    mask_single = _selection_mask(p_cc=cylinder, sp=np.float32, **coefficients)
    mask_double = _selection_mask(p_cc=cylinder, sp=np.float64, **coefficients)
    n_differ = int(np.sum(mask_single != mask_double))
    print(
        f"\nhybrid selection on the cylinder ({cylinder_center}): WENO on "
        f"{int(mask_double.sum())} of {mask_double.size} cells in double, "
        f"{int(mask_single.sum())} in single, {n_differ} cells differ"
    )
    assert mask_double.any() and not mask_double.all(), "vacuous: one branch only"
    assert n_differ == 0


#: the steps of the ihadv132 captures whose 'advection-init' tracer the mask is evaluated on
#: (step 1 carries the initial condition, the later ones the Fortran's evolved field)
EVOLVED_STEPS = [1, 2, 10, 50, 100]
#: the 'tag' of test_defs.Experiments.jocksch_cylinder: the capture on the original
#: generated grid file, where the Fortran's cylinder around the origin is the 48-cell
#: quarter-disc in the corner, and the one on the '_centred' file (the same grid shifted
#: to the origin), where it is the full 176-cell disc of the paper
GRID_VARIANTS = [pytest.param("", id="quarter_disc"), pytest.param("_centred", id="full_disc")]


@pytest.fixture
def grid_variant(request: pytest.FixtureRequest) -> str:
    """The capture's grid-file tag ('' or '_centred'); set by parametrization.

    A test that does not parametrise it gets the quarter-disc capture ('').
    """
    return getattr(request, "param", "")


@pytest.fixture
def experiment_description(
    case: tuple[int, int], grid_variant: str
) -> test_defs.ExperimentDescription:
    """Overrides test_jocksch_reference's fixture: the same (ihadv, hlim) case with the tag."""
    return test_defs.Experiments.jocksch_cylinder(*case, grid_variant)


@pytest.mark.datatest
@pytest.mark.parametrize("case", [pytest.param((132, 0), id="ihadv132_hlim0")], indirect=True)
@pytest.mark.parametrize("grid_variant", GRID_VARIANTS, indirect=True)
@pytest.mark.parametrize("step", EVOLVED_STEPS)
def test_evolved_selection_mask_is_the_same_in_single_and_double_precision(
    case: tuple[int, int],
    grid_variant: str,
    step: int,
    *,
    experiment: test_defs.Experiment,
    advection_init_savepoint: sb.AdvectionInitSavepoint,
) -> None:
    """The residual's precision does not change the hybrid's mask on the Fortran's evolved fields.

    The check above is decided by construction: on the initial cylinder every stencil is
    either constant (residual exactly 0 in both kinds) or O(1) against the 5e-5 threshold.
    Here the same numpy selection runs on the tracer of the 'advection-init' savepoints
    of the two ihadv132_hlim0 captures (test_jocksch_reference.py): on the original
    generated grid file the Fortran's cylinder around the origin is the 48-cell
    quarter-disc in the corner (icon-ajocksch/CAPTURE_NOTES.md, 'Grid'), on the '_centred'
    file the full 176-cell disc of the paper; step 1 is that initial condition, the later
    steps the Fortran's own hybrid solution after step - 1 steps, whose cells behind the
    disc carry the scheme's ripples at every magnitude. The coefficients are the port's
    (lsq_error and the full quadratic pseudoinverse of weno_least_squares on the grid file
    the capture used), the field one level of the savepoint (the ten levels are asserted
    identical).

    The assertion is measured, not banded: single precision moves the residual by
    ``|lsqe_sp - lsqe_wp|`` and the double residual is ``|lsqe_wp - threshold|`` away from
    the threshold, so the mask cannot flip where the first is smaller than the second.
    Both are printed relative to the cell's threshold (the largest perturbation and the
    smallest margin over the cells) and their per-cell ratio is asserted below 1
    everywhere, together with the zero count of differing cells. The two global numbers
    are not comparable with each other: outside the disc p_cc = 0 makes the threshold
    5e-25 while a neighbour's O(1) jump perturbs the residual by ~1e-7, so the largest
    relative perturbation is 1e16-1e19 at every step (2.9e16 to 1.7e19 over the steps and
    captures below), on cells whose margin is larger still.

    Measured 2026-09-11 (WENO-selected cells in single / double precision, cells that
    differ, of 880; then the largest per-cell perturbation-to-margin ratio and the smallest
    margin relative to the threshold). Quarter-disc: step 1: 78 / 78, 0, 1.5e-7, 1.0;
    step 2: 89 / 89, 0, 2.4e-7, 1.0; step 10: 227 / 227, 0, 6.5e-7, 1.3e-1; step 50:
    624 / 624, 0, 9.3e-6, 8.7e-3; step 100: 666 / 666, 0, 2.3e-6, 3.8e-3. Full disc:
    step 1: 136 / 136, 0, 1.5e-7, 1.0; step 2: 168 / 168, 0, 2.4e-7, 1.0; step 10:
    456 / 456, 0, 1.0e-6, 2.0e-1; step 50: 820 / 820, 0, 2.4e-6, 4.6e-2; step 100:
    852 / 852, 0, 5.5e-6, 2.8e-2. Single precision moves no residual by more than 1e-5
    of its distance to the threshold, so the selection of the evolved field is not a
    round-off decision either.
    """
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=CAPTURE_NUM_LEVELS,
        keep_skip_values=True,
        allocator=model_backends.get_allocator(None),
    )
    grid = grid_manager.grid
    domain_length, domain_height = grid.grid_params.domain_length, grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None
    c2e2c = grid.get_connectivity("C2E2C").asnumpy()
    c2v = grid.get_connectivity("C2V").asnumpy()
    cell_center_x = grid_manager.coordinates[dims.CellDim]["x"].asnumpy()
    cell_center_y = grid_manager.coordinates[dims.CellDim]["y"].asnumpy()
    stencil_c9 = weno.create_stencil_c9(c2e2c, c2v)
    geometry = dict(
        stencil_c9=stencil_c9,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    lsq_moments = weno.compute_lsq_moments_torus(
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        vertex_x=grid_manager.coordinates[dims.VertexDim]["x"].asnumpy(),
        vertex_y=grid_manager.coordinates[dims.VertexDim]["y"].asnumpy(),
        c2v=c2v,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    coefficients = dict(
        lsq_error=weno.compute_lsq_error_quadratic(lsq_moments=lsq_moments, **geometry),
        pseudoinv_full=weno.compute_lsq_pseudoinverse_quadratic(
            lsq_moments=lsq_moments, **geometry
        ),
        stencil_c9=stencil_c9,
    )

    tracer = advection_init_savepoint.tracer(0).asnumpy()
    assert tracer.shape == (stencil_c9.shape[0], CAPTURE_NUM_LEVELS)
    assert (tracer == tracer[:, :1]).all(), "the capture's ten levels are identical columns"
    p_cc = tracer[:, 0]
    if step == 1:
        n_disc = int(np.sum(p_cc == 1.0))
        assert n_disc == (176 if grid_variant == "_centred" else 48), "not the expected IC"

    lsqe_single, threshold = _selection_residual(p_cc=p_cc, sp=np.float32, **coefficients)
    lsqe_double, _ = _selection_residual(p_cc=p_cc, sp=np.float64, **coefficients)
    mask_single = lsqe_single > threshold
    mask_double = lsqe_double > threshold
    n_differ = int(np.sum(mask_single != mask_double))
    perturbation = np.abs(lsqe_single - lsqe_double)
    margin = np.abs(lsqe_double - threshold)
    print(
        f"\nhybrid selection on the ihadv132_hlim0{grid_variant} capture, step {step:3d}: "
        f"WENO on {int(mask_single.sum())} of {mask_single.size} cells in single, "
        f"{int(mask_double.sum())} in double, {n_differ} cells differ; "
        f"max |lsqe_sp - lsqe_wp| / threshold = {float((perturbation / threshold).max()):.3e}, "
        f"min |lsqe_wp - threshold| / threshold = {float((margin / threshold).min()):.3e}, "
        f"max |lsqe_sp - lsqe_wp| / |lsqe_wp - threshold| = "
        f"{float((perturbation / margin).max()):.3e}"
    )
    if step == 1:
        assert mask_double.any() and not mask_double.all(), "vacuous: one branch only"
    assert n_differ == 0
    assert (perturbation < margin).all()


# --- the gt4py pipeline -----------------------------------------------------------------


def _run_gt4py_hybrid_pipeline(
    torus_patch,
    backend,
    *,
    patch_coefficients: dict,
    p_cc: np.ndarray,
    cell_area: np.ndarray,
    rel_idx: np.ndarray,
    quad: np.ndarray,
    p_mass_flx_e: np.ndarray,
    threshold: float = THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the runtime stencils in the ThirdOrderMiuraWenoHybrid order on the patch."""
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

    def direct_field(values, dtype=None):
        return gtx.as_field((dims.CellDim, dims.C2E2CDim), values.copy(), allocator=backend)

    def butterfly_field(values):
        return gtx.as_field((dims.CellDim, dims.C2E2C2E2CDim), values.copy(), allocator=backend)

    p_cc_field = cell_k_field(p_cc)
    moments_fields = {
        f"lsq_moments_{u + 1}": cell_field(patch_coefficients["lsq_moments"][:, u].copy())
        for u in range(5)
    }
    coeff_fields = {
        f"p_coeff_{c + 1}_dsl": cell_k_field(np.zeros((n_cells, nlev))) for c in range(6)
    }
    quad_fields = {
        f"p_quad_vector_sum_{q + 1}": edge_k_field(quad[:, :, q].copy()) for q in range(6)
    }
    accumulators = {
        **{f"z_lsq_weighted_{q + 1}": edge_k_field(np.zeros((n_edges, nlev))) for q in range(6)},
        "smooth_sum": edge_k_field(np.zeros((n_edges, nlev))),
    }
    rel_idx_field = edge_k_field(rel_idx)
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

    # the full fit and the selection mask
    reconstruct_quadratic_coefficients_svd.with_backend(backend)(
        p_cc=p_cc_field,
        **{
            f"lsq_pseudoinv_direct_{u + 1}": direct_field(patch_coefficients["full_direct"][:, u])
            for u in range(5)
        },
        **{
            f"lsq_pseudoinv_butterfly_{u + 1}": butterfly_field(
                patch_coefficients["full_butterfly"][:, u]
            )
            for u in range(5)
        },
        **moments_fields,
        **coeff_fields,
        **cell_domain,
        offset_provider=offset_provider,
    )
    use_weno = cell_k_field(np.zeros((n_cells, nlev), dtype=bool))
    compute_weno_hybrid_stencil_selection.with_backend(backend)(
        p_cc=p_cc_field,
        **{f"p_coeff_{c}": coeff_fields[f"p_coeff_{c}_dsl"] for c in (2, 3, 4, 5, 6)},
        **{
            f"lsq_error_direct_{u + 1}": direct_field(patch_coefficients["error_direct"][:, u])
            for u in range(5)
        },
        **{
            f"lsq_error_butterfly_{u + 1}": butterfly_field(
                patch_coefficients["error_butterfly"][:, u]
            )
            for u in range(5)
        },
        lsq_butterfly_active=butterfly_field(
            patch_coefficients["butterfly_active"].astype(np.int32)
        ),
        use_weno=use_weno,
        selection_threshold=float(np.float32(threshold)),
        selection_eps=EPS,
        **cell_domain,
        offset_provider=offset_provider,
    )

    # the plain branch
    p_out_e = edge_k_field(np.zeros((n_edges, nlev)))
    compute_horizontal_tracer_flux_from_quadratic_coefficients.with_backend(backend)(
        **{f"p_coeff_{c + 1}": coeff_fields[f"p_coeff_{c + 1}_dsl"] for c in range(6)},
        p_cell_rel_idx_dsl=rel_idx_field,
        **quad_fields,
        p_mass_flx_e=edge_k_field(p_mass_flx_e),
        p_out_e=p_out_e,
        **edge_domain,
        offset_provider=offset_provider,
    )

    # the WENO branch with unit weights
    for cand in range(N_CAND):
        reconstruct_quadratic_coefficients_svd.with_backend(backend)(
            p_cc=p_cc_field,
            **{
                f"lsq_pseudoinv_direct_{u + 1}": direct_field(
                    patch_coefficients["weno_direct"][:, cand, u]
                )
                for u in range(5)
            },
            **{
                f"lsq_pseudoinv_butterfly_{u + 1}": butterfly_field(
                    patch_coefficients["weno_butterfly"][:, cand, u]
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
            cell_area=cell_field(cell_area),
            p_cell_rel_idx_dsl=rel_idx_field,
            **{
                f"z_quad_vector_sum_{q + 1}": quad_fields[f"p_quad_vector_sum_{q + 1}"]
                for q in range(6)
            },
            **accumulators,
            l_weight_s=1.0,
            **edge_domain,
            offset_provider=offset_provider,
        )
    p_flux_weno = edge_k_field(np.zeros((n_edges, nlev)))
    compute_horizontal_tracer_flux_from_weno_coefficients.with_backend(backend)(
        **accumulators,
        **quad_fields,
        p_mass_flx_e=edge_k_field(p_mass_flx_e),
        p_out_e=p_flux_weno,
        **edge_domain,
        offset_provider=offset_provider,
    )

    # per edge, the upwind cell's choice
    select_horizontal_tracer_flux_by_upwind_cell.with_backend(backend)(
        use_first=use_weno,
        p_cell_rel_idx_dsl=rel_idx_field,
        p_flux_first=p_flux_weno,
        p_flux_second=p_out_e,
        p_out_e=p_out_e,
        **edge_domain,
        offset_provider=offset_provider,
    )
    return p_out_e.asnumpy(), use_weno.asnumpy()


def _mixed_tracer_field(torus_patch, rng: np.random.Generator) -> np.ndarray:
    """O(1) variation on one half, a 1e-4 ripple on the other: both branches get exercised."""
    n_cells = torus_patch.c2e2c.shape[0]
    varying = rng.uniform(0.1, 1.0, size=(n_cells, NLEV))
    ripple = 1.0 + 1e-4 * rng.uniform(-1.0, 1.0, size=(n_cells, NLEV))
    left = (torus_patch.cell_center_x < 0.5 * torus_patch.domain_length)[:, np.newaxis]
    return np.where(left, varying, ripple)


@pytest.mark.level("integration")
def test_pipeline_matches_fortran_reference(torus_patch, patch_coefficients, backend):
    inputs = _random_inputs(torus_patch, seed=14)
    inputs["p_cc"] = _mixed_tracer_field(torus_patch, np.random.default_rng(15))

    expected, use_weno_ref, lsqe_ref = _upwind_hflux_miura_weno_hyb_reference(
        **_reference_kwargs(torus_patch, patch_coefficients, inputs)
    )
    assert use_weno_ref.any() and not use_weno_ref.all(), "vacuous: one branch only"

    actual, use_weno = _run_gt4py_hybrid_pipeline(
        torus_patch,
        backend,
        patch_coefficients=patch_coefficients,
        p_cc=inputs["p_cc"],
        cell_area=inputs["cell_area"],
        rel_idx=inputs["rel_idx"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
    )

    # the residual is summed in another order than the Fortran's: cells within a narrow
    # band around the threshold may legitimately land on the other side
    thresholds = THRESHOLD * (inputs["p_cc"] + EPS) ** 2
    decided = np.abs(lsqe_ref.astype(np.float64) - thresholds) > 1e-5 * thresholds
    assert decided.mean() > 0.99, "too many undecided cells for the mask comparison"
    np.testing.assert_array_equal(use_weno[decided], use_weno_ref[decided])

    upwind = inputs["upwind_cell"]
    decided_edge = decided[upwind, np.arange(NLEV)[np.newaxis, :]]
    assert np.all(expected[decided_edge] != 0.0), "vacuous cross-check: zero reference fluxes"
    np.testing.assert_allclose(actual[decided_edge], expected[decided_edge], rtol=1e-12, atol=1e-14)


@pytest.mark.level("integration")
@pytest.mark.parametrize("threshold", [np.inf, -1.0])
def test_pipeline_threshold_limits(torus_patch, patch_coefficients, backend, threshold):
    # both stencil branches agree with the reference's branches exactly when forced
    inputs = _random_inputs(torus_patch, seed=16)
    expected, use_weno_ref, _ = _upwind_hflux_miura_weno_hyb_reference(
        **_reference_kwargs(torus_patch, patch_coefficients, inputs), threshold=threshold
    )
    actual, use_weno = _run_gt4py_hybrid_pipeline(
        torus_patch,
        backend,
        patch_coefficients=patch_coefficients,
        p_cc=inputs["p_cc"],
        cell_area=inputs["cell_area"],
        rel_idx=inputs["rel_idx"],
        quad=inputs["quad"],
        p_mass_flx_e=inputs["p_mass_flx_e"],
        threshold=threshold,
    )
    np.testing.assert_array_equal(use_weno, use_weno_ref)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
