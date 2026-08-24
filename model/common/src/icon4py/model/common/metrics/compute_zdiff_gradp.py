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
    return os.environ.get("ICON4PY_VALIDATE_ZDIFF_GRADP") == "1"


def _batched_searchsorted_v2(
    a: data_alloc.NDArray, v: data_alloc.NDArray, array_ns: ModuleType
) -> data_alloc.NDArray:
    a = a.astype(array_ns.float64)
    v = v.astype(array_ns.float64)
    m, n = a.shape
    max_num = max(a.max() - a.min(), v.max() - v.min()) + 1.0
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
) -> None:
    # Strict z_ifc decrease over [fi..nlev] in the original (top->bottom) orientation
    # is equivalent to strict increase of the ascending slices z_ifc_e* up to nlev-fi.
    k_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    valid_ifc = k_idx < (nlev - fi)[:, None]
    if not ((z_ifc_e0[:, :-1] < z_ifc_e0[:, 1:]) | ~valid_ifc).all():
        raise ValueError("Strict z_ifc decrease over [fi..nlev] violated for cell 0.")
    if not ((z_ifc_e1[:, :-1] < z_ifc_e1[:, 1:]) | ~valid_ifc).all():
        raise ValueError("Strict z_ifc decrease over [fi..nlev] violated for cell 1.")

    if not (
        array_ns.isfinite(z_ifc_e0).all()
        and array_ns.isfinite(z_ifc_e1).all()
        and array_ns.isfinite(z_me).all()
        and array_ns.isfinite(z_aux2).all()
    ):
        raise ValueError("Searched arrays contain non-finite values.")

    k_idx_me = array_ns.arange(nlev - 1, dtype=array_ns.int64)[None, :]
    valid_me = (k_idx_me >= fi[:, None] + 1) & (k_idx_me < nlev - 1)
    if not ((z_me[:, :-1] >= z_me[:, 1:]) | ~valid_me).all():
        raise ValueError("z_me non-increasing over [fi+1, nlev-1] violated.")


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

    fill_high = max(z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()) + 1.0
    fill_low = min(z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()) - 1.0

    if _validation_enabled():
        _validate_zdiff_gradp_inputs(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev)

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
