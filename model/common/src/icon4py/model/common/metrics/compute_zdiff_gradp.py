# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from types import ModuleType

import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


_LAST_EXACT_V2_PATH: str | None = None
_LAST_DISPATCH_PATH: str | None = None


def _check_finite(
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
) -> data_alloc.NDArray:
    return (
        array_ns.isfinite(z_ifc_e0).all()
        & array_ns.isfinite(z_ifc_e1).all()
        & array_ns.isfinite(z_me).all()
        & array_ns.isfinite(z_aux2).all()
    )


def _check_e3(
    array_ns: ModuleType,
    z_me: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    k_idx_me = array_ns.arange(nlev - 1, dtype=array_ns.int64)[None, :]
    valid_me = (k_idx_me >= fi[:, None] + 1) & (k_idx_me < nlev - 1)
    return ((z_me[:, :-1] >= z_me[:, 1:]) | ~valid_me).all()


def _batched_searchsorted(a, v, array_ns):
    m, n = a.shape
    max_num = max(float(a.max() - a.min()), float(v.max() - v.min())) + 1
    r = max_num * array_ns.arange(m, dtype=a.dtype)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def compute_zdiff_gradp(
    *,
    e2c,
    z_me: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(z_mc)

    nedges = e2c.shape[0]

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0
    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[horizontal_start:, :, :] = (
        array_ns.expand_dims(z_me, axis=1)[horizontal_start:, :, :]
        - z_mc[e2c][horizontal_start:, :, :]
    )
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_asc = z_ifc[:, ::-1].copy()
    z_ifc_e0 = z_ifc_asc[e2c_0]
    z_ifc_e1 = z_ifc_asc[e2c_1]

    fill_high = float(array_ns.max(z_ifc_e0)) + 1.0
    fill_low = float(array_ns.min(z_ifc_e0)) - 1.0

    z_ifc_mask = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
        nlev + 1 - fi[:, None]
    )
    z_me_mask = array_ns.arange(nlev, dtype=array_ns.int64)[None, :] <= fi[:, None]

    z_ifc_e0_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e0)
    z_ifc_e1_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e1)
    z_me_m = array_ns.where(z_me_mask, fill_low, z_me)

    pos_0 = _batched_searchsorted(z_ifc_e0_m, z_me_m, array_ns)
    jk1_0 = array_ns.clip(nlev - pos_0, fi[:, None], nlev - 1)

    pos_1 = _batched_searchsorted(z_ifc_e1_m, z_me_m, array_ns)
    jk1_1 = array_ns.clip(nlev - pos_1, fi[:, None], nlev - 1)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    boundary = array_ns.arange(nedges, dtype=array_ns.int64) >= horizontal_start
    valid_jk = (jk_idx > fi[:, None]) & boundary[:, None]
    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    zdiff_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e0, jk1_0.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 0, :],
    )
    zdiff_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e1, jk1_1.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 1, :],
    )

    vertoffset_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 0, :],
    )
    vertoffset_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 1, :],
    )

    nudging = array_ns.arange(nedges, dtype=array_ns.int64) >= horizontal_start_1
    if nudging.any():
        z_aux2_vec = z_aux2[:, None]

        pos_aux_0 = _batched_searchsorted(z_ifc_e0_m, z_aux2_vec, array_ns)
        jk1_aux_0 = array_ns.clip(nlev - pos_aux_0[:, 0], fi, nlev - 1)

        pos_aux_1 = _batched_searchsorted(z_ifc_e1_m, z_aux2_vec, array_ns)
        jk1_aux_1 = array_ns.clip(nlev - pos_aux_1[:, 0], fi, nlev - 1)

        phase2_mask = valid_jk & (z_me < z_aux2[:, None]) & nudging[:, None]

        zdiff_gradp[:, 0, :] = array_ns.where(
            phase2_mask,
            z_aux2_vec
            - array_ns.take_along_axis(z_mc_e0, jk1_aux_0[:, None].astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 0, :],
        )
        zdiff_gradp[:, 1, :] = array_ns.where(
            phase2_mask,
            z_aux2_vec
            - array_ns.take_along_axis(z_mc_e1, jk1_aux_1[:, None].astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 1, :],
        )

        vertoffset_gradp[:, 0, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_0[:, None] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 0, :],
        )
        vertoffset_gradp[:, 1, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_1[:, None] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp


def _validation_enabled() -> bool:
    return os.environ.get("ICON4PY_VALIDATE_ZDIFF_GRADP", "1") != "0"


def _batched_searchsorted_v2(
    a: data_alloc.NDArray, v: data_alloc.NDArray, array_ns: ModuleType
) -> data_alloc.NDArray:
    a = a.astype(array_ns.float64)
    v = v.astype(array_ns.float64)
    m, n = a.shape
    max_num = array_ns.maximum(a.max() - a.min(), v.max() - v.min()) + 1.0
    r = max_num * array_ns.arange(m, dtype=array_ns.float64)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def _validate_zdiff_gradp_inputs(  # noqa: PLR0917
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    nedges: int,
) -> None:
    finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
    e3_ok = _check_e3(array_ns, z_me, fi, nlev)

    # E1: strict z_ifc decrease over [fi..nlev] in the original (top->bottom)
    # orientation is equivalent to strict increase of the ascending slices up to nlev-fi.
    k_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    valid_ifc = k_idx < (nlev - fi)[:, None]
    e1_ok_0 = ((z_ifc_e0[:, :-1] < z_ifc_e0[:, 1:]) | ~valid_ifc).all()
    e1_ok_1 = ((z_ifc_e1[:, :-1] < z_ifc_e1[:, 1:]) | ~valid_ifc).all()

    # A2 premise: float64 row-offset safety.
    global_max = array_ns.max(
        array_ns.stack([z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()])
    )
    global_min = array_ns.min(
        array_ns.stack([z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()])
    )
    max_num = global_max - global_min + 1.0
    a2_ok = max_num * nedges < 2.0**53

    # Min-level-spacing vs ULP at max_num.
    spacing_0 = array_ns.where(
        valid_ifc,
        z_ifc_e0[:, 1:] - z_ifc_e0[:, :-1],
        array_ns.asarray(array_ns.inf, dtype=z_ifc_e0.dtype),
    )
    spacing_1 = array_ns.where(
        valid_ifc,
        z_ifc_e1[:, 1:] - z_ifc_e1[:, :-1],
        array_ns.asarray(array_ns.inf, dtype=z_ifc_e1.dtype),
    )
    min_spacing = array_ns.min(array_ns.stack([spacing_0.min(), spacing_1.min()]))
    ulp_at_max = array_ns.nextafter(max_num, array_ns.inf) - max_num
    spacing_ok = min_spacing > ulp_at_max

    combined = e1_ok_0
    for check in (e1_ok_1, finite_ok, e3_ok, a2_ok, spacing_ok):
        combined = array_ns.logical_and(combined, check)

    if not bool(combined):
        raise ValueError(
            "compute_zdiff_gradp_v2 input validation failed: strict z_ifc decrease, "
            "z_me monotonicity, finiteness, A2 float-offset premise, or min-spacing-vs-ULP violated."
        )


def _validate_exact_inputs(
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
) -> None:
    finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
    if not bool(finite_ok):
        raise ValueError("Searched arrays contain non-finite values.")


def compute_zdiff_gradp_v2(
    *,
    e2c: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]

    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)
    if hs1 < hs:
        raise ValueError("horizontal_start_1 must be greater than or equal to horizontal_start.")

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_asc = z_ifc[:, ::-1].copy()
    z_ifc_e0 = z_ifc_asc[e2c_0]
    z_ifc_e1 = z_ifc_asc[e2c_1]

    if _validation_enabled():
        _validate_zdiff_gradp_inputs(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev, nedges)
    fill_high = (
        array_ns.max(array_ns.stack([z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()]))
        + 1.0
    )
    fill_low = (
        array_ns.min(array_ns.stack([z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()]))
        - 1.0
    )
    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    fi_sliced = fi[hs:]

    z_ifc_mask = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
        nlev + 1 - fi_sliced[:, None]
    )
    z_me_mask = array_ns.arange(nlev, dtype=array_ns.int64)[None, :] <= fi_sliced[:, None]

    z_ifc_e0_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e0[hs:])
    z_ifc_e1_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e1[hs:])
    z_me_m = array_ns.where(z_me_mask, fill_low, z_me[hs:])

    valid_jk = jk_idx > fi_sliced[:, None]

    # Phase 1, cell 0
    pos_0 = _batched_searchsorted_v2(z_ifc_e0_m, z_me_m, array_ns)
    jk1_0 = array_ns.clip(nlev - pos_0, fi_sliced[:, None], nlev - 1)
    z_mc_e0 = z_mc[e2c_0]
    base_zdiff_c = z_me[hs:] - z_mc_e0[hs:]
    zdiff_gradp[hs:, 0, :] = array_ns.where(
        valid_jk,
        z_me[hs:] - array_ns.take_along_axis(z_mc_e0[hs:], jk1_0.astype(array_ns.int64), axis=1),
        base_zdiff_c,
    )
    vertoffset_gradp[hs:, 0, :] = array_ns.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 0, :],
    )

    # Phase 1, cell 1
    pos_1 = _batched_searchsorted_v2(z_ifc_e1_m, z_me_m, array_ns)
    jk1_1 = array_ns.clip(nlev - pos_1, fi_sliced[:, None], nlev - 1)
    z_mc_e1 = z_mc[e2c_1]
    base_zdiff_c = z_me[hs:] - z_mc_e1[hs:]
    zdiff_gradp[hs:, 1, :] = array_ns.where(
        valid_jk,
        z_me[hs:] - array_ns.take_along_axis(z_mc_e1[hs:], jk1_1.astype(array_ns.int64), axis=1),
        base_zdiff_c,
    )
    vertoffset_gradp[hs:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 1, :],
    )

    # Phase 2
    if hs1 < nedges:
        fi_sliced1 = fi[hs1:]
        z_aux2_v = z_aux2[hs1:, None]

        z_ifc_mask1 = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
            nlev + 1 - fi_sliced1[:, None]
        )

        z_ifc_e0_m1 = array_ns.where(z_ifc_mask1, fill_high, z_ifc_e0[hs1:])
        z_ifc_e1_m1 = array_ns.where(z_ifc_mask1, fill_high, z_ifc_e1[hs1:])

        pos_aux_0 = _batched_searchsorted_v2(z_ifc_e0_m1, z_aux2_v, array_ns)
        jk1_aux_0 = array_ns.clip(nlev - pos_aux_0, fi_sliced1[:, None], nlev - 1)
        jk1_aux_0 = array_ns.where(
            pos_aux_0 >= (nlev + 1 - fi_sliced1)[:, None],
            nlev - 1,
            jk1_aux_0,
        )

        pos_aux_1 = _batched_searchsorted_v2(z_ifc_e1_m1, z_aux2_v, array_ns)
        jk1_aux_1 = array_ns.clip(nlev - pos_aux_1, fi_sliced1[:, None], nlev - 1)
        jk1_aux_1 = array_ns.where(
            pos_aux_1 >= (nlev + 1 - fi_sliced1)[:, None],
            nlev - 1,
            jk1_aux_1,
        )

        phase2_mask = valid_jk[(hs1 - hs) :] & (z_me[hs1:] < z_aux2_v)

        zdiff_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            z_aux2_v
            - array_ns.take_along_axis(z_mc_e0[hs1:], jk1_aux_0.astype(array_ns.int64), axis=1),
            zdiff_gradp[hs1:, 0, :],
        )
        zdiff_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            z_aux2_v
            - array_ns.take_along_axis(z_mc_e1[hs1:], jk1_aux_1.astype(array_ns.int64), axis=1),
            zdiff_gradp[hs1:, 1, :],
        )
        vertoffset_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_0 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 0, :],
        )
        vertoffset_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_1 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp


def compute_zdiff_gradp_exact_v2(
    *,
    e2c: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    global _LAST_EXACT_V2_PATH  # noqa: PLW0603
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]

    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)
    if hs1 < hs:
        raise ValueError("horizontal_start_1 must be greater than or equal to horizontal_start.")

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_e0 = z_ifc[e2c_0, :]
    z_ifc_e1 = z_ifc[e2c_1, :]

    finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
    e3_ok = _check_e3(array_ns, z_me, fi, nlev)

    combined = finite_ok & e3_ok
    if not bool(combined):
        if not bool(finite_ok):
            raise ValueError("Searched arrays contain non-finite values.")
        use_carry = True
    else:
        use_carry = False

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    # Phase 1, cell 0: no carry; fresh scan at every jk.
    succ0 = _exact_query_succ(z_ifc_e0, z_me, array_ns)
    jk1_0 = _exact_phase1_cell0(succ0, fi, array_ns)
    zdiff_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e0, jk1_0, axis=1),
        zdiff_gradp[:, 0, :],
    )
    vertoffset_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 0, :],
    )
    del succ0

    # Phase 1, cell 1: fresh scan when E3 holds, carry fallback otherwise.
    succ1 = _exact_query_succ(z_ifc_e1, z_me, array_ns)
    if use_carry:
        jk1_1 = _exact_carry_loop(succ1, fi, valid_jk, array_ns)
    else:
        jk1_1 = _exact_phase1_cell0(succ1, fi, array_ns)
    zdiff_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e1, jk1_1, axis=1),
        zdiff_gradp[:, 1, :],
    )
    vertoffset_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 1, :],
    )
    del succ1

    # Phase 2: applies to edges [hs1:] only.
    if hs1 < nedges:
        z_aux2_v = z_aux2[:, None].astype(array_ns.float64)

        succ2_0 = _exact_query_succ(z_ifc_e0, z_aux2_v, array_ns)
        jk1_aux_0 = _exact_phase1_cell0(succ2_0, fi, array_ns)
        zdiff_gradp[:, 0, :] = array_ns.where(
            phase2_active,
            z_aux2_v - array_ns.take_along_axis(z_mc_e0, jk1_aux_0.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 0, :],
        )
        vertoffset_gradp[:, 0, :] = array_ns.where(
            phase2_active,
            (jk1_aux_0 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 0, :],
        )
        del succ2_0

        succ2_1 = _exact_query_succ(z_ifc_e1, z_aux2_v, array_ns)
        jk1_aux_1 = _exact_phase1_cell0(succ2_1, fi, array_ns)
        zdiff_gradp[:, 1, :] = array_ns.where(
            phase2_active,
            z_aux2_v - array_ns.take_along_axis(z_mc_e1, jk1_aux_1.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 1, :],
        )
        vertoffset_gradp[:, 1, :] = array_ns.where(
            phase2_active,
            (jk1_aux_1 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 1, :],
        )

    _LAST_EXACT_V2_PATH = "carry" if use_carry else "fast"
    return zdiff_gradp, vertoffset_gradp


def _exact_query_succ(
    z_ifc_k: data_alloc.NDArray, v: data_alloc.NDArray, array_ns: ModuleType
) -> data_alloc.NDArray:
    """Build suffix-minimum successor table for exact bracket matching.

    Args:
        z_ifc_k: (chunk, nlev+1) array with z_ifc in main's k-order (decreasing in a).
        v: (chunk, nq) query values.
        array_ns: array namespace module (numpy or cupy).

    Returns:
        succ: (chunk, nq, nlev) integer array where succ[e, q, t] is the first
            candidate index a >= t satisfying the bracket predicate, or the
            unconditional deepest level nlev-1. Uses int8 for nlev <= 127,
            int16 for 127 < nlev <= 32767; raises ValueError for larger nlev.
    """
    _chunk, nlev_p1 = z_ifc_k.shape
    nlev = nlev_p1 - 1

    a_idx = array_ns.arange(nlev, dtype=array_ns.int64)
    z_ifc_k = z_ifc_k.astype(array_ns.float64)
    v = v.astype(array_ns.float64)

    # Bracket predicate: z_ifc_k[a] >= v >= z_ifc_k[a+1] (z_ifc_k decreasing).
    upper = z_ifc_k[:, None, :-1]
    lower = z_ifc_k[:, None, 1:]
    qv = v[:, :, None]
    bracket = (upper >= qv) & (qv >= lower)
    unconditional = a_idx[None, None, :] == (nlev - 1)
    idx = array_ns.where(bracket | unconditional, a_idx[None, None, :], nlev)

    if nlev <= 127:
        idx = idx.astype(array_ns.int8)
    elif nlev <= 32767:
        idx = idx.astype(array_ns.int16)
    else:
        raise ValueError(
            f"compute_zdiff_gradp_exact successor tables support nlev <= 32767, got {nlev}."
        )

    # Hillis-Steele style doubling scan for the suffix minimum along the
    # candidate axis. cupy does not implement ufunc.accumulate, so we compute
    # succ[jk, t] = min(idx[jk, a] for a in [t, nlev)) explicitly with
    # log2(nlev) elementwise shifted-minimum steps. The RHS slices are fully
    # evaluated before the LHS assignment, so overlapping read/write views are safe.
    S = idx.copy()
    step = 1
    while step < nlev:
        S[..., :-step] = array_ns.minimum(S[..., :-step], S[..., step:])
        step *= 2
    return S


def _exact_phase1_cell0(
    succ: data_alloc.NDArray, fi: data_alloc.NDArray, array_ns: ModuleType
) -> data_alloc.NDArray:
    """Phase-1 cell-0: no carry; gather succ[:, jk, fi] for every jk."""
    fi_idx = fi[:, None, None].astype(array_ns.int64)
    jk1 = array_ns.take_along_axis(succ, fi_idx, axis=2)[:, :, 0]
    return jk1.astype(array_ns.int64)


def _exact_carry_loop(
    succ: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    active: data_alloc.NDArray,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Carry loop replicating main's jk_start lower-bound semantics.

    succ: (chunk, nq, nlev) successor table.
    fi: (chunk,) starting lower bound per edge.
    active: (chunk, nlev) guard controlling whether the carry advances at jk.

    Returns jk1_out: (chunk, nlev) selected candidate for each jk.
    """
    chunk, _, nlev = succ.shape
    single_query = succ.shape[1] == 1
    t = fi.astype(array_ns.int64).copy()
    jk1_out = array_ns.empty((chunk, nlev), dtype=array_ns.int64)
    for jk in range(nlev):
        src = succ[:, 0:1, :] if single_query else succ[:, jk : jk + 1, :]
        jk1 = array_ns.take_along_axis(src, t[:, None, None], axis=2)[:, 0, 0]
        jk1_out[:, jk] = jk1
        t = array_ns.where(active[:, jk], jk1, t)
    return jk1_out


def compute_zdiff_gradp_exact(
    *,
    e2c: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
    chunk_size: int | None = None,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Exact-semantics broadcast variant of compute_zdiff_gradp.

    Replicates main's bracket predicate and jk_start carry exactly for all
    finite inputs with nlev <= 127 (int8 successor tables; int16 fallback for
    larger nlev). Validation is always-on with opt-out via
    ICON4PY_VALIDATE_ZDIFF_GRADP=0.
    """
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]

    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)
    if hs1 < hs:
        raise ValueError("horizontal_start_1 must be greater than or equal to horizontal_start.")

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    # z_ifc in main's k-order (decreasing with k) for both cells.
    z_ifc_e0 = z_ifc[e2c_0, :]
    z_ifc_e1 = z_ifc[e2c_1, :]

    if _validation_enabled():
        _validate_exact_inputs(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)

    # Output assembly identical to v2's D6.
    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    if chunk_size is None:
        mem_per_edge = nlev * 2 * (nlev + 1)
        chunk_size = max(1, min(nedges, (256 * 1024 * 1024) // mem_per_edge))

    for start in range(0, nedges, chunk_size):
        end = min(start + chunk_size, nedges)
        chunk = slice(start, end)
        z_ifc_k0 = z_ifc_e0[chunk, :].astype(array_ns.float64)
        z_ifc_k1 = z_ifc_e1[chunk, :].astype(array_ns.float64)
        z_me_c = z_me[chunk, :].astype(array_ns.float64)
        fi_c = fi[chunk]
        valid_jk_c = valid_jk[chunk, :]

        # Phase 1, cell 0: fresh scan at every jk.
        succ0 = _exact_query_succ(z_ifc_k0, z_me_c, array_ns)
        jk1_0 = _exact_phase1_cell0(succ0, fi_c, array_ns)
        z_mc_e0 = z_mc[e2c_0[chunk]]
        zdiff_gradp[chunk, 0, :] = array_ns.where(
            valid_jk_c,
            z_me_c - array_ns.take_along_axis(z_mc_e0, jk1_0, axis=1),
            zdiff_gradp[chunk, 0, :],
        )
        vertoffset_gradp[chunk, 0, :] = array_ns.where(
            valid_jk_c,
            (jk1_0 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[chunk, 0, :],
        )

        # Phase 1, cell 1: carry t = jk_start between jk iterations.
        succ1 = _exact_query_succ(z_ifc_k1, z_me_c, array_ns)
        jk1_1 = _exact_carry_loop(succ1, fi_c, valid_jk_c, array_ns)
        z_mc_e1 = z_mc[e2c_1[chunk]]
        zdiff_gradp[chunk, 1, :] = array_ns.where(
            valid_jk_c,
            z_me_c - array_ns.take_along_axis(z_mc_e1, jk1_1, axis=1),
            zdiff_gradp[chunk, 1, :],
        )
        vertoffset_gradp[chunk, 1, :] = array_ns.where(
            valid_jk_c,
            (jk1_1 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[chunk, 1, :],
        )

        # Phase 2: applies to edges [hs1:] only.
        if hs1 < nedges:
            phase2_active_c = phase2_active[chunk, :]
            z_aux2_v = z_aux2[chunk, None].astype(array_ns.float64)

            succ2_0 = _exact_query_succ(z_ifc_k0, z_aux2_v, array_ns)
            jk1_aux_0 = _exact_carry_loop(succ2_0, fi_c, phase2_active_c, array_ns)
            zdiff_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                z_aux2_v - array_ns.take_along_axis(z_mc_e0, jk1_aux_0, axis=1),
                zdiff_gradp[chunk, 0, :],
            )
            vertoffset_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_0 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 0, :],
            )

            succ2_1 = _exact_query_succ(z_ifc_k1, z_aux2_v, array_ns)
            jk1_aux_1 = _exact_carry_loop(succ2_1, fi_c, phase2_active_c, array_ns)
            zdiff_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                z_aux2_v - array_ns.take_along_axis(z_mc_e1, jk1_aux_1, axis=1),
                zdiff_gradp[chunk, 1, :],
            )
            vertoffset_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_1 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 1, :],
            )

    return zdiff_gradp, vertoffset_gradp
