# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-check of Jocksch's cell-local positive-definite limiter (his itype_hlimit=4).

Two numpy references of the limiter block of his kernels (mo_advection_hflux.f90
3013-3032 in upwind_hflux_miura3_weno, identical in 102/132/202/203), written as the
Fortran cell loop over the edges the cell is the upwind cell of:

- 'literal': ``z_b = MAX(z_b, 0)`` as he wrote it, which takes the flux sign as the
  outflow sign;
- 'oriented': the outflow is ``flux * orientation`` with the orientation of the edge
  normal relative to the cell, ``sign(geofac_div)``; this is what the icon4py stencils
  implement.

Gates on the references alone (plain numpy):
- where every upwind cell sees its edge normals pointing outward (all vn >= 0 on the
  patch, whose normals point from E2C[0] to E2C[1]) the two references coincide bit for
  bit, and with random wind directions they do not: his clamp zeroes every physical
  vn < 0 flux, and his flux_out goes negative on those cells, so r_m does too;
- the oriented reference keeps every cell's updated mass non-negative for any
  reconstructed fluxes, the unclamped scaling alone does not;
- units: ``flux_out`` is ``geofac_div * dt * flux`` with the flux per unit edge length,
  the same convention as ICON's hflx_limiter_pd and icon4py's PositiveDefinite, so the
  limiter is exercised with the fluxes the flux stencils produce.

The gt4py gate runs the two stencils on the patch against the oriented reference.
"""

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection_horizontal
from icon4py.model.atmosphere.tracer_advection.stencils.apply_cell_local_positive_definite_horizontal_flux_factor import (
    apply_cell_local_positive_definite_horizontal_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_cell_local_positive_definite_horizontal_flux_factor import (
    compute_cell_local_positive_definite_horizontal_flux_factor,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.testing.fixtures.datatest import backend

from .. import utils


NLEV = 3
DTIME = 0.3


@pytest.fixture(scope="module")
def torus_patch() -> utils.TorusPatch:
    return utils.build_torus_patch()


def _geofac_div(torus_patch) -> np.ndarray:
    """edge_length * orientation / area on the C2E slots, (n_cells, 3).

    The patch's normals point from E2C[0] to E2C[1], so the orientation is +1 for the
    first cell of an edge and -1 for the second (utils._build_patch_edges asserts this).
    """
    n_cells = torus_patch.c2e.shape[0]
    area = np.sqrt(3.0) / 4.0 * torus_patch.edge_length**2
    orientation = np.where(
        torus_patch.e2c[torus_patch.c2e, 0] == np.arange(n_cells)[:, np.newaxis], 1.0, -1.0
    )
    return orientation * torus_patch.edge_length / area


def _random_inputs(torus_patch, seed: int, *, vn_sign: str) -> dict:
    """Reconstructed fluxes, wind, tracer and density; vn_sign in {'positive', 'mixed'}."""
    rng = np.random.default_rng(seed)
    n_cells = torus_patch.c2e.shape[0]
    n_edges = torus_patch.e2c.shape[0]
    p_vn = rng.uniform(0.1, 1.0, size=(n_edges, NLEV))
    if vn_sign == "mixed":
        p_vn *= rng.choice([-1.0, 1.0], size=(n_edges, NLEV))
    # a reconstructed flux is q_R * mass flux with q_R of either sign near zero
    q_r = rng.uniform(-0.3, 1.0, size=(n_edges, NLEV))
    return dict(
        p_vn=p_vn,
        p_mflx_tracer_h=q_r * p_vn,
        p_cc=rng.uniform(0.0, 0.2, size=(n_cells, NLEV)),
        p_rhodz_now=rng.uniform(0.5, 1.5, size=(n_cells, NLEV)),
        geofac_div=_geofac_div(torus_patch),
    )


def _upwind_cell(torus_patch, p_vn: np.ndarray) -> np.ndarray:
    """The backtrajectory rule: vn >= 0 -> E2C[0], else E2C[1]; (n_edges, nlev)."""
    return np.where(p_vn >= 0.0, torus_patch.e2c[:, 0:1], torus_patch.e2c[:, 1:2])


def _limiter_reference(
    torus_patch,
    *,
    p_mflx_tracer_h: np.ndarray,
    p_vn: np.ndarray,
    p_cc: np.ndarray,
    p_rhodz_now: np.ndarray,
    geofac_div: np.ndarray,
    dtime: float,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal port of f90 3013-3032; returns (limited flux, r_m).

    variant: 'literal' (his MAX(z_b, 0)), 'oriented' (clamp on flux * orientation) or
    'unclamped' (the scaling without the reconstruction limiter, for the mutation test).
    """
    n_cells, nlev = p_cc.shape
    upwind_cell = _upwind_cell(torus_patch, p_vn)
    p_out_e = p_mflx_tracer_h.copy()
    r_m_out = np.empty((n_cells, nlev))
    for jc in range(n_cells):
        for jk in range(nlev):
            jee = torus_patch.c2e[jc]
            jf = np.array([upwind_cell[jee[ie], jk] == jc for ie in range(3)])
            # f90 3015-3025
            flux_out = 0.0
            z_b = np.zeros(3)
            for ie in range(3):
                if jf[ie]:
                    z_b[ie] = p_mflx_tracer_h[jee[ie], jk]
                    if variant == "literal":
                        z_b[ie] = max(z_b[ie], 0.0)
                    elif variant == "oriented":
                        orientation = np.sign(geofac_div[jc, ie])
                        z_b[ie] = orientation * max(orientation * z_b[ie], 0.0)
                    flux_out = flux_out + geofac_div[jc, ie] * dtime * z_b[ie]
            # f90 3026-3027
            r_m = min(1.0, (p_cc[jc, jk] * p_rhodz_now[jc, jk]) / (flux_out + constants.DBL_EPS))
            r_m_out[jc, jk] = r_m
            # f90 3028-3032
            for ie in range(3):
                if jf[ie]:
                    p_out_e[jee[ie], jk] = r_m * z_b[ie]
    return p_out_e, r_m_out


def _reference_kwargs(torus_patch, inputs: dict, variant: str) -> dict:
    return dict(
        torus_patch=torus_patch,
        p_mflx_tracer_h=inputs["p_mflx_tracer_h"],
        p_vn=inputs["p_vn"],
        p_cc=inputs["p_cc"],
        p_rhodz_now=inputs["p_rhodz_now"],
        geofac_div=inputs["geofac_div"],
        dtime=DTIME,
        variant=variant,
    )


def _updated_mass(torus_patch, inputs: dict, flux: np.ndarray) -> np.ndarray:
    """q rho - dt * div(flux) per cell, the horizontal update's new tracer mass."""
    divergence = np.sum(inputs["geofac_div"][:, :, np.newaxis] * flux[torus_patch.c2e], axis=1)
    return inputs["p_cc"] * inputs["p_rhodz_now"] - DTIME * divergence


# --- gates on the references alone ------------------------------------------------------


@pytest.mark.level("integration")
def test_oriented_reference_equals_his_code_where_normals_point_his_way(torus_patch):
    inputs = _random_inputs(torus_patch, seed=1, vn_sign="positive")
    literal, r_m_literal = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "literal"))
    oriented, r_m_oriented = _limiter_reference(
        **_reference_kwargs(torus_patch, inputs, "oriented")
    )
    np.testing.assert_array_equal(oriented, literal)
    np.testing.assert_array_equal(r_m_oriented, r_m_literal)
    # the limiter did something: some fluxes were clamped and some cells scaled
    assert np.any(literal != inputs["p_mflx_tracer_h"])
    assert np.any(r_m_literal < 1.0) and np.any(r_m_literal == 1.0)


@pytest.mark.level("integration")
def test_his_clamp_zeroes_the_inward_normal_fluxes(torus_patch):
    # on an edge whose normal points into its upwind cell (vn < 0) the outflow is -flux:
    # his clamp zeroes the physical fluxes (q_R > 0, flux < 0; the capture's finding on
    # the 20 x 22 torus) and keeps the unphysical ones, the oriented clamp the reverse
    inputs = _random_inputs(torus_patch, seed=2, vn_sign="mixed")
    literal, _ = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "literal"))
    oriented, _ = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "oriented"))
    negative_vn = inputs["p_vn"] < 0.0
    physical = negative_vn & (inputs["p_mflx_tracer_h"] < 0.0)
    unphysical = negative_vn & (inputs["p_mflx_tracer_h"] > 0.0)
    assert physical.any() and unphysical.any()
    np.testing.assert_array_equal(literal[physical], 0.0)
    np.testing.assert_array_equal(oriented[unphysical], 0.0)
    assert np.all(oriented[physical] < 0.0)
    # the unphysical fluxes his clamp keeps enter flux_out with geofac_div < 0, so the
    # cell's r_m comes out negative and flips their sign
    _, r_m_literal = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "literal"))
    assert np.any(r_m_literal < 0.0)
    assert np.any(literal[unphysical] < 0.0)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(oriented, literal)


@pytest.mark.level("integration")
@pytest.mark.parametrize("vn_sign", ["positive", "mixed"])
def test_oriented_reference_keeps_the_mass_non_negative(torus_patch, vn_sign):
    inputs = _random_inputs(torus_patch, seed=3, vn_sign=vn_sign)
    oriented, r_m = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "oriented"))
    upwind_cell = _upwind_cell(torus_patch, inputs["p_vn"])
    # every flux leaves its upwind cell: flux * orientation(upwind cell) >= 0
    is_first = upwind_cell == torus_patch.e2c[:, 0:1]
    outflow = np.where(is_first, oriented, -oriented)
    assert np.all(outflow >= 0.0)
    # and no cell is emptied beyond its content
    mass = _updated_mass(torus_patch, inputs, oriented)
    assert np.all(mass >= -1e-15)
    assert np.any(r_m < 1.0), "vacuous: no cell needed scaling"
    # the unlimited fluxes do empty cells (the reason for the limiter)
    assert np.any(_updated_mass(torus_patch, inputs, inputs["p_mflx_tracer_h"]) < 0.0)


@pytest.mark.level("integration")
def test_scaling_without_the_clamp_does_not_keep_the_mass_non_negative(torus_patch):
    # mutation: dropping the reconstruction limiter lets negative q_R fluxes act as inflow
    # in r_m's budget and the outflow scaling no longer bounds the loss
    inputs = _random_inputs(torus_patch, seed=3, vn_sign="mixed")
    unclamped, _ = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "unclamped"))
    with pytest.raises(AssertionError):
        assert np.all(_updated_mass(torus_patch, inputs, unclamped) >= -1e-15)


# --- the gt4py stencils -----------------------------------------------------------------


def _run_gt4py_limiter(torus_patch, backend, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Run the two limiter stencils in the CellLocalPositiveDefinite order on the patch."""
    n_cells = torus_patch.c2e.shape[0]
    n_edges = torus_patch.e2c.shape[0]

    def connectivity(table, source_dim, target_dims):
        return gtx.as_connectivity(
            target_dims, source_dim, data=table, dtype=gtx.int32, allocator=backend
        )

    offset_provider = {
        "C2E": connectivity(torus_patch.c2e, dims.EdgeDim, (dims.CellDim, dims.C2EDim)),
        "E2C": connectivity(torus_patch.e2c, dims.CellDim, (dims.EdgeDim, dims.E2CDim)),
    }
    cell_k = lambda values: gtx.as_field((dims.CellDim, dims.KDim), values, allocator=backend)  # noqa: E731 [lambda-assignment]
    edge_k = lambda values: gtx.as_field((dims.EdgeDim, dims.KDim), values, allocator=backend)  # noqa: E731 [lambda-assignment]

    p_mflx_tracer_h = edge_k(inputs["p_mflx_tracer_h"].copy())
    p_vn = edge_k(inputs["p_vn"])
    r_m = cell_k(np.zeros((n_cells, NLEV)))
    compute_cell_local_positive_definite_horizontal_flux_factor.with_backend(backend)(
        geofac_div=gtx.as_field(
            (dims.CellDim, dims.C2EDim), inputs["geofac_div"], allocator=backend
        ),
        p_cc=cell_k(inputs["p_cc"]),
        p_rhodz_now=cell_k(inputs["p_rhodz_now"]),
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_vn=p_vn,
        r_m=r_m,
        p_dtime=DTIME,
        dbl_eps=constants.DBL_EPS,
        horizontal_start=0,
        horizontal_end=gtx.int32(n_cells),
        vertical_start=0,
        vertical_end=gtx.int32(NLEV),
        offset_provider=offset_provider,
    )
    apply_cell_local_positive_definite_horizontal_flux_factor.with_backend(backend)(
        r_m=r_m,
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_vn=p_vn,
        horizontal_start=0,
        horizontal_end=gtx.int32(n_edges),
        vertical_start=0,
        vertical_end=gtx.int32(NLEV),
        offset_provider=offset_provider,
    )
    return p_mflx_tracer_h.asnumpy(), r_m.asnumpy()


@pytest.mark.level("integration")
@pytest.mark.parametrize("vn_sign", ["positive", "mixed"])
def test_stencils_match_oriented_reference(torus_patch, backend, vn_sign):
    inputs = _random_inputs(torus_patch, seed=4, vn_sign=vn_sign)
    expected, r_m_expected = _limiter_reference(
        **_reference_kwargs(torus_patch, inputs, "oriented")
    )
    actual, r_m = _run_gt4py_limiter(torus_patch, backend, inputs)
    np.testing.assert_allclose(r_m, r_m_expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)
    assert np.any(r_m < 1.0) and np.any(actual != inputs["p_mflx_tracer_h"])


@pytest.mark.level("integration")
def test_stencils_match_his_code_where_normals_point_his_way(torus_patch, backend):
    inputs = _random_inputs(torus_patch, seed=5, vn_sign="positive")
    expected, r_m_expected = _limiter_reference(**_reference_kwargs(torus_patch, inputs, "literal"))
    actual, r_m = _run_gt4py_limiter(torus_patch, backend, inputs)
    np.testing.assert_allclose(r_m, r_m_expected, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-16)


def test_orientation_convention_check_accepts_the_oriented_geofac_div(torus_patch):
    # _geofac_div is built with ICON's convention: positive exactly on the E2C[0] side
    tracer_advection_horizontal.check_cell_edge_orientation_convention(
        geofac_div=_geofac_div(torus_patch), c2e=torus_patch.c2e, e2c=torus_patch.e2c
    )


def test_orientation_convention_check_rejects_a_flipped_edge(torus_patch):
    geofac_div = _geofac_div(torus_patch)
    e2c = torus_patch.e2c.copy()
    e2c[7] = e2c[7, ::-1]  # one edge with its normal pointing into E2C[0]
    with pytest.raises(ValueError, match="edge orientation convention"):
        tracer_advection_horizontal.check_cell_edge_orientation_convention(
            geofac_div=geofac_div, c2e=torus_patch.c2e, e2c=e2c
        )
