# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the fused tracer advection program (after horizontal limiter)."""

from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection.stencils.compute_fused_tracer_advection import (
    compute_tracer_advection_after_horizontal_limiter,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.testing import stencil_tests

from .test_compute_fused_tracer_advection_terms import (
    apply_positive_definite_horizontal_multiplicative_flux_factor_numpy,
)
from .test_compute_ppm_quadratic_face_values import compute_ppm_quadratic_face_values_numpy
from .test_compute_ppm_quartic_face_values import compute_ppm_quartic_face_values_numpy
from .test_compute_ppm_slope import compute_ppm_slope_numpy
from .test_compute_vertical_parabola_limiter_condition import (
    compute_vertical_parabola_limiter_condition_numpy,
)
from .test_integrate_tracer_horizontally import integrate_tracer_horizontally_numpy
from .test_integrate_tracer_vertically import integrate_tracer_vertically_numpy
from .test_limit_vertical_parabola_semi_monotonically import (
    limit_vertical_parabola_semi_monotonically_numpy,
)
from .test_limit_vertical_slope_semi_monotonically import (
    limit_vertical_slope_semi_monotonically_numpy,
)


def _gather_at_khalf(field: np.ndarray, offset: float, num_levels: int) -> np.ndarray:
    """Gather a CellK field at KHalf positions with fractional K-half shifts.

    `field(KHalf + offset)` for offset in {0.5, 1.5, ...} reads `field[kh + j]`
    (j = offset - 0.5) and for negative offsets `field[kh - j - 1]`... expressed
    directly here as index arithmetic on full (ncells, num_levels) arrays,
    returning a (ncells, num_levels + 1) KHalf array. Out-of-range reads are
    clipped; the stencil's concat_where guards never consume the clipped values.

    The result is a fresh contiguous array, so callers may patch it in place.
    """
    kh = np.arange(num_levels + 1)
    j = int(abs(offset) - 0.5)
    if offset > 0:
        idx = np.clip(kh + j, 0, num_levels - 1)
    else:
        idx = np.clip(kh - j - 1, 0, num_levels - 1)
    return field[:, idx]


def _ppm_sum_neighbor_contributions_numpy(
    *,
    mask1: np.ndarray,
    mask2: np.ndarray,
    js: np.ndarray,
    field: np.ndarray,
    elev: int,
    num_levels: int,
) -> np.ndarray:
    """numpy port of `_sum_neighbor_contributions` (fractional flux).

    The guard fallback `f_k = field(KHalf + 0.5)` only overrides the `j + 1` edge
    columns excluded by the guard; since every gather is a fresh array, that fixup
    is applied in place on the edge columns instead of via a full-array `np.where`.
    """
    ncells = field.shape[0]
    f_k = _gather_at_khalf(field, 0.5, num_levels)
    contrib = np.zeros((ncells, num_levels + 1))
    for j in range(5):
        # plus side: concat_where(KHalfDim < elev - (j - 1), field(KHalf + j + 0.5), f_k)
        plus_val = _gather_at_khalf(field, j + 0.5, num_levels)
        plus_val[:, elev - j + 1 :] = f_k[:, elev - j + 1 :]
        contrib += np.where(mask1 & (js == j), plus_val, 0.0)
        # minus side: concat_where(KHalfDim > j, field(KHalf - j - 0.5), f_k)
        minus_val = _gather_at_khalf(field, -(j + 0.5), num_levels)
        minus_val[:, : j + 1] = f_k[:, : j + 1]
        contrib += np.where(mask2 & (js == j), minus_val, 0.0)
    return contrib


def _ppm_sum_neighbor_contributions_all_numpy(
    *,
    mask1: np.ndarray,
    mask2: np.ndarray,
    js: np.ndarray,
    field1: np.ndarray,
    field2: np.ndarray,
    elev: int,
    num_levels: int,
) -> np.ndarray:
    """numpy port of `_sum_neighbor_contributions_all` (integer flux).

    The guard only zeroes the contributions it excludes, so it folds directly
    into the selection mask instead of a nested `np.where`.
    """
    ncells = field1.shape[0]
    kh = np.arange(num_levels + 1)
    contrib = np.zeros((ncells, num_levels + 1))
    for j in range(5):
        plus_idx = np.clip(kh + j, 0, num_levels - 1)
        minus_idx = np.clip(kh - j - 1, 0, num_levels - 1)
        prod_plus = field1[:, plus_idx] * field2[:, plus_idx]
        prod_minus = field1[:, minus_idx] * field2[:, minus_idx]
        contrib += np.where(mask1 & (js >= j) & (kh < elev + 1 - j)[None, :], prod_plus, 0.0)
        contrib += np.where(mask2 & (js >= j) & (kh > j)[None, :], prod_minus, 0.0)
    return contrib


def _ppm_courant_number_numpy(
    *,
    p_mflx_contra_v: np.ndarray,
    p_cellmass_now: np.ndarray,
    slevp1_ti: int,
    nlev: int,
    wp_eps: float,
    p_dtime: float,
    num_levels: int,
) -> np.ndarray:
    """numpy port of `_compute_ppm4gpu_courant_number` (below + above)."""
    ncells = p_cellmass_now.shape[0]
    kh = np.arange(num_levels + 1)

    def below(z_mass: np.ndarray, z_cfl_in: np.ndarray) -> np.ndarray:
        z_cfl = z_cfl_in.copy()
        z_mass_pos = z_mass > 0.0
        cellmass = {}
        gt = {}
        z_m = z_mass.copy()
        for j, guard in enumerate([nlev - 1, nlev - 2, nlev - 3, nlev - 4]):
            cellmass[j] = np.where(
                kh <= guard, _gather_at_khalf(p_cellmass_now, j + 0.5, num_levels), 0.0
            )
        cellmass[4] = np.where(
            kh <= nlev - 4, _gather_at_khalf(p_cellmass_now, 4.5, num_levels), 0.0
        )
        prev = np.ones_like(z_m, dtype=bool)
        for j in range(4):
            gt[j] = prev & z_mass_pos & (kh <= nlev - j - 1) & (z_m >= cellmass[j])
            z_m = z_m - np.where(gt[j], cellmass[j], 0.0)
            prev = gt[j]
        for j in range(4):
            z_cfl = z_cfl + np.where(gt[j], 1.0, 0.0)
        jks = np.where(kh < nlev + 1, _gather_at_khalf(p_cellmass_now, 0.5, num_levels), 0.0)
        for j in range(4):
            jks = np.where(gt[j], cellmass[j + 1], jks)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_cflfrac = np.where(z_mass_pos, z_m / jks, 0.0)
        z_cfl = z_cfl + np.where(z_cflfrac < 1.0, z_cflfrac, 1.0 - wp_eps)
        return z_cfl

    def above(z_mass: np.ndarray, z_cfl_in: np.ndarray) -> np.ndarray:
        z_cfl = z_cfl_in.copy()
        z_mass_neg = z_mass <= 0.0
        cellmass = {}
        gt = {}
        z_m = z_mass.copy()
        for j, guard in enumerate([slevp1_ti + 1, slevp1_ti + 2, slevp1_ti + 3, slevp1_ti + 4]):
            cellmass[j] = np.where(
                kh >= guard, _gather_at_khalf(p_cellmass_now, -(j + 0.5), num_levels), 0.0
            )
        cellmass[4] = np.where(
            kh >= slevp1_ti + 4, _gather_at_khalf(p_cellmass_now, -4.5, num_levels), 0.0
        )
        prev = np.ones_like(z_m, dtype=bool)
        for j in range(4):
            gt[j] = prev & z_mass_neg & (kh >= slevp1_ti + j + 1) & (np.abs(z_m) >= cellmass[j])
            z_m = z_m + np.where(gt[j], cellmass[j], 0.0)
            prev = gt[j]
        for j in range(4):
            z_cfl = z_cfl - np.where(gt[j], 1.0, 0.0)
        jks = np.where(kh > 0, _gather_at_khalf(p_cellmass_now, -0.5, num_levels), 0.0)
        for j in range(4):
            jks = np.where(gt[j], cellmass[j + 1], jks)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_cflfrac = np.where(z_mass_neg, z_m / jks, 0.0)
        z_cfl = z_cfl + np.where(np.abs(z_cflfrac) < 1.0, z_cflfrac, wp_eps - 1.0)
        return z_cfl

    z_mass = p_dtime * p_mflx_contra_v
    zero = np.zeros((ncells, num_levels + 1))
    return below(z_mass, zero) + above(z_mass, zero)


def _ppm4gpu_flux_numpy(
    *,
    p_cc: np.ndarray,
    p_cellmass_now: np.ndarray,
    p_mflx_contra_v: np.ndarray,
    p_cellhgt_mc_now: np.ndarray,
    slev: int,
    slevp1_ti: int,
    elev: int,
    wp_eps: float,
    p_dtime: float,
    itype_vlimit: int,
    num_levels: int,
) -> np.ndarray:
    """numpy port of `_compute_ppm4gpu_flux` (THIRD_ORDER_PPM vertical flux)."""
    ncells = p_cc.shape[0]
    kh = np.arange(num_levels + 1)

    z_cfl = np.where(
        (kh[None, :] > 0) & (kh[None, :] < elev + 1),
        _ppm_courant_number_numpy(
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellmass_now=p_cellmass_now,
            slevp1_ti=slevp1_ti,
            nlev=elev,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            num_levels=num_levels,
        ),
        0.0,
    )
    z_slope_raw = np.zeros((ncells, num_levels))
    z_slope_raw[:, 1 : num_levels - 1] = compute_ppm_slope_numpy(p_cc, p_cellhgt_mc_now, elev)
    z_slope = np.zeros_like(z_slope_raw)
    z_slope[:, 1 : num_levels - 1] = limit_vertical_slope_semi_monotonically_numpy(
        p_cc, z_slope_raw, elev, num_levels
    )
    if itype_vlimit != 1:  # == VerticalAdvectionType.FIRST_ORDER_UPWIND in the stencil
        z_slope = z_slope_raw

    # faces: kh=0 -> p_cc[0]; 1 <= kh <= elev -> quadratic/quartic; kh >= elev+1 -> p_cc[kh-1]
    p_face = np.empty((ncells, num_levels + 1))
    p_face[:, 0] = p_cc[:, 0]
    p_face[:, elev + 1 :] = p_cc[:, -1:]
    quadratic = compute_ppm_quadratic_face_values_numpy(p_cc, p_cellhgt_mc_now)
    quartic = np.zeros((ncells, num_levels))
    # `compute_ppm_quartic_face_values_numpy` mirrors the standalone stencil test, which
    # allocates `p_cellhgt_mc_now` with one extra K-extended level (the stencil reads
    # `p_cellhgt_mc_now(KHalf + 1.5)`). Here the quartic values are only consumed for
    # faces kh = 2..elev - 1, where those reads stay within the allocated field; the
    # extra level only affects the unused top entry, so replicating the top level is safe.
    p_cellhgt_mc_now_k_ext = np.concatenate((p_cellhgt_mc_now, p_cellhgt_mc_now[:, -1:]), axis=1)
    quartic[:, 2:] = compute_ppm_quartic_face_values_numpy(p_cc, p_cellhgt_mc_now_k_ext, z_slope)
    face_kh = np.arange(1, elev + 1)
    p_face[:, 1 : elev + 1] = np.where(
        (face_kh > 1) & (face_kh < elev), quartic[:, 1 : elev + 1], quadratic[:, 1 : elev + 1]
    )

    l_limit = compute_vertical_parabola_limiter_condition_numpy(p_face, p_cc)
    z_face_up, z_face_low = limit_vertical_parabola_semi_monotonically_numpy(l_limit, p_face, p_cc)
    z_delta_q = 0.5 * (z_face_up - z_face_low)
    z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)

    # fractional flux
    js_frac = np.floor(np.abs(z_cfl))
    z_cflfrac = np.abs(z_cfl) - js_frac
    mask1 = (z_cfl > 0.0) & (z_cflfrac != 0.0)
    mask2 = (z_cfl < 0.0) & (z_cflfrac != 0.0)
    wsign = np.where(z_cfl > 0.0, 1.0, -1.0)
    in_slev_bounds = kh[None, :] - js_frac >= slev
    p_cc_jks = _ppm_sum_neighbor_contributions_numpy(
        mask1=mask1, mask2=mask2, js=js_frac, field=p_cc, elev=elev, num_levels=num_levels
    )
    mass_jks = _ppm_sum_neighbor_contributions_numpy(
        mask1=mask1, mask2=mask2, js=js_frac, field=p_cellmass_now, elev=elev, num_levels=num_levels
    )
    delta_jks = _ppm_sum_neighbor_contributions_numpy(
        mask1=mask1, mask2=mask2, js=js_frac, field=z_delta_q, elev=elev, num_levels=num_levels
    )
    a1_jks = _ppm_sum_neighbor_contributions_numpy(
        mask1=mask1, mask2=mask2, js=js_frac, field=z_a1, elev=elev, num_levels=num_levels
    )
    z_q_int = (
        p_cc_jks
        + wsign * (delta_jks * (1.0 - z_cflfrac))
        - a1_jks * (1.0 - 3.0 * z_cflfrac + 2.0 * z_cflfrac * z_cflfrac)
    )
    p_upflux = np.where(in_slev_bounds, wsign * mass_jks * z_cflfrac * z_q_int / p_dtime, 0.0)

    # integer flux
    js_int = np.floor(np.abs(z_cfl)) - 1.0
    wsign_int = np.where(z_cfl > 0.0, 1.0, -1.0)
    in_slev_bounds_int = kh[None, :] - js_int >= slev
    prod_jks = _ppm_sum_neighbor_contributions_all_numpy(
        mask1=z_cfl > 0.0,
        mask2=z_cfl < 0.0,
        js=js_int,
        field1=p_cc,
        field2=p_cellmass_now,
        elev=elev,
        num_levels=num_levels,
    )
    p_upflux = p_upflux + np.where(in_slev_bounds_int, wsign_int * prod_jks / p_dtime, 0.0)

    return np.where((kh[None, :] > 0) & (kh[None, :] < elev + 1), p_upflux, 0.0)


@pytest.mark.uses_concat_where
class TestComputeTracerAdvectionAfterHorizontalLimiter(stencil_tests.StencilTest):
    PROGRAM = compute_tracer_advection_after_horizontal_limiter
    OUTPUTS = (
        "p_mflx_tracer_h",
        "p_mflx_tracer_v",
        "p_tracer_new",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        r_m: np.ndarray,
        p_mflx_tracer_h_unlimited: np.ndarray,
        p_tracer_now: np.ndarray,
        p_tracer_after_vertical: np.ndarray,
        p_mflx_tracer_v: np.ndarray,
        rhodz_ast2: np.ndarray,
        rhodz_now: np.ndarray,
        rhodz_new: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        p_cellhgt_mc_now: np.ndarray,
        deepatmo_divh: np.ndarray,
        deepatmo_divzl: np.ndarray,
        deepatmo_divzu: np.ndarray,
        geofac_div: np.ndarray,
        k: np.ndarray,
        do_vertical_first: bool,
        ivadv_tracer: int,
        ihadv_tracer: int,
        itype_hlimit: int,
        itype_vlimit: int,
        slev: int,
        slevp1_ti: int,
        elev: int,
        wp_eps: float,
        iadv_slev_jt: int,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)

        tracer_now_for_h = p_tracer_now
        rhodz_for_h_now = rhodz_now
        rhodz_for_h_new = rhodz_ast2
        p_mflx_tracer_h = p_mflx_tracer_h_unlimited.copy()
        p_tracer_new = tracer_now_for_h.copy()
        if do_vertical_first:
            tracer_now_for_h = p_tracer_after_vertical
            rhodz_for_h_now = rhodz_ast2
            rhodz_for_h_new = rhodz_new

        if itype_hlimit == 4:
            p_mflx_tracer_h = apply_positive_definite_horizontal_multiplicative_flux_factor_numpy(
                r_m, p_mflx_tracer_h_unlimited, connectivities
            )

        if ihadv_tracer != 0:
            p_tracer_new = integrate_tracer_horizontally_numpy(
                connectivities,
                p_mflx_tracer_h,
                deepatmo_divh,
                tracer_now_for_h,
                rhodz_for_h_now,
                rhodz_for_h_new,
                geofac_div,
                p_dtime,
            )

        p_mflx_tracer_v_computed = np.zeros_like(p_mflx_tracer_v)
        if ivadv_tracer == 1:
            p_mflx_tracer_v_computed[:, 1:-1] = (
                np.where(
                    p_mflx_contra_v[:, 1:-1] >= 0.0,
                    p_tracer_new[:, 1:],
                    p_tracer_new[:, :-1],
                )
                * p_mflx_contra_v[:, 1:-1]
            )
        elif ivadv_tracer == 3:
            p_mflx_tracer_v_computed = _ppm4gpu_flux_numpy(
                p_cc=p_tracer_new,
                p_cellmass_now=rhodz_ast2,
                p_mflx_contra_v=p_mflx_contra_v,
                p_cellhgt_mc_now=p_cellhgt_mc_now,
                slev=slev,
                slevp1_ti=slevp1_ti,
                elev=elev,
                wp_eps=wp_eps,
                p_dtime=p_dtime,
                itype_vlimit=itype_vlimit,
                num_levels=grid.num_levels,
            )

        if do_vertical_first:
            p_mflx_tracer_v_out = p_mflx_tracer_v
        else:
            p_mflx_tracer_v_out = p_mflx_tracer_v_computed
            p_tracer_new = integrate_tracer_vertically_numpy(
                p_tracer_new,
                rhodz_ast2,
                p_mflx_tracer_v_computed,
                deepatmo_divzl,
                deepatmo_divzu,
                rhodz_new,
                k,
                p_dtime,
                ivadv_tracer,
                iadv_slev_jt,
            )

        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mflx_tracer_v=p_mflx_tracer_v_out,
            p_tracer_new=p_tracer_new,
        )

    @stencil_tests.input_data_fixture(params=[(True, 0), (False, 1), (True, 3), (False, 3)])
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper,
        grid: base.Grid,
        request: pytest.FixtureRequest,
    ) -> dict:
        r_m = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_h_unlimited = data_alloc.random_field(dims.EdgeDim, dims.KDim)
        p_tracer_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_tracer_after_vertical = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_tracer_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        rhodz_ast2 = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        rhodz_new = data_alloc.random_field(dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(dims.CellDim, dims.KHalfDim)
        p_cellhgt_mc_now = data_alloc.random_field(dims.CellDim, dims.KDim)
        deepatmo_divh = data_alloc.random_field(dims.KDim)
        deepatmo_divzl = data_alloc.random_field(dims.KDim)
        deepatmo_divzu = data_alloc.random_field(dims.KDim)
        k = data_alloc.index_field(dims.KDim)
        k_half = data_alloc.index_field(dims.KHalfDim)
        geofac_div = data_alloc.random_field(dims.CellDim, dims.C2EDim)

        p_mflx_tracer_h = data_alloc.zero_field(dims.EdgeDim, dims.KDim)
        p_tracer_new = data_alloc.zero_field(dims.CellDim, dims.KDim)

        do_vertical_first, ivadv_tracer = request.param

        return dict(
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_mflx_tracer_v=p_mflx_tracer_v,
            p_tracer_new=p_tracer_new,
            r_m=r_m,
            p_mflx_tracer_h_unlimited=p_mflx_tracer_h_unlimited,
            p_tracer_now=p_tracer_now,
            p_tracer_after_vertical=p_tracer_after_vertical,
            rhodz_ast2=rhodz_ast2,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            deepatmo_divh=deepatmo_divh,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            slev=gtx.int32(0),
            slevp1_ti=gtx.int32(1),
            elev=gtx.int32(grid.num_levels - 1),
            do_vertical_first=do_vertical_first,
            ivadv_tracer=gtx.int32(ivadv_tracer),
            ihadv_tracer=gtx.int32(2),
            itype_hlimit=gtx.int32(4),
            itype_vlimit=gtx.int32(1),
            iadv_slev_jt=gtx.int32(0),
            geofac_div=geofac_div,
            wp_eps=ta.wpfloat(1e-9),
            p_dtime=ta.wpfloat(5.0),
            start_cell_nudging=gtx.int32(0),
            end_cell_local=gtx.int32(grid.num_cells),
            start_edge_lateral_boundary_level_5=gtx.int32(0),
            end_edge_halo=gtx.int32(grid.num_edges),
            vertical_end=gtx.int32(grid.num_levels),
        )
