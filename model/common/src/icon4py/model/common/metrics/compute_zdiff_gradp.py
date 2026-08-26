# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


_LAST_EXACT_V2_PATH: str | None = None
_LAST_EXACT_V3_PATH: str | None = None
_LAST_EXACT_V4_PATH: str | None = None
_LAST_EXACT_V5_PATH: str | None = None
_EXACT_V5_KERNEL_CACHE: dict[int, Any] = {}
_LAST_DISPATCH_PATH: str | None = None
_EXACT_V2_MAX_TABLE_BYTES: int = 1 << 30


def _check_finite(
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
) -> data_alloc.NDArray:
    # Stack the four reductions into one device bool so only one host sync
    # is needed when validation is enabled (matches the exact-variant contract).
    finite_0 = array_ns.isfinite(z_ifc_e0).all()
    finite_1 = array_ns.isfinite(z_ifc_e1).all()
    finite_me = array_ns.isfinite(z_me).all()
    finite_aux2 = array_ns.isfinite(z_aux2).all()
    return array_ns.stack([finite_0, finite_1, finite_me, finite_aux2]).all()


def _check_e3(
    array_ns: ModuleType,
    z_me: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    k_idx_me = array_ns.arange(nlev - 1, dtype=array_ns.int64)[None, :]
    valid_me = (k_idx_me >= fi[:, None] + 1) & (k_idx_me < nlev - 1)
    return ((z_me[:, :-1] >= z_me[:, 1:]) | ~valid_me).all()


def _batched_searchsorted(
    a: data_alloc.NDArray, v: data_alloc.NDArray, array_ns: ModuleType
) -> data_alloc.NDArray:
    m, n = a.shape
    max_num = max(float(a.max() - a.min()), float(v.max() - v.min())) + 1
    r = max_num * array_ns.arange(m, dtype=a.dtype)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def compute_zdiff_gradp(
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


def _compute_v2_validation(  # noqa: PLR0917
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    nedges: int,
    *,
    tie_free: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Return a (2,) device boolean array [finite_ok, full_ok] for v2's full validation set.

    ``tie_free`` is a 0-d device bool from the fast path's precomputed
    searchsorted positions; it is folded into ``full_ok``.  This keeps
    validation to a single stacked sync.
    """
    finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
    e3_ok = _check_e3(array_ns, z_me, fi, nlev)

    k_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    valid_ifc = k_idx < (nlev - fi)[:, None]
    e1_ok_0 = ((z_ifc_e0[:, :-1] < z_ifc_e0[:, 1:]) | ~valid_ifc).all()
    e1_ok_1 = ((z_ifc_e1[:, :-1] < z_ifc_e1[:, 1:]) | ~valid_ifc).all()

    global_max = array_ns.max(
        array_ns.stack([z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()])
    )
    global_min = array_ns.min(
        array_ns.stack([z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()])
    )
    max_num = global_max - global_min + 1.0
    a2_ok = max_num * nedges < 2.0**53

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

    full_ok = e1_ok_0
    for check in (e1_ok_1, e3_ok, a2_ok, spacing_ok, tie_free):
        full_ok = array_ns.logical_and(full_ok, check)

    # Single stacked array: two bool() reads on the default stream share one sync.
    return array_ns.stack([finite_ok, full_ok])


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


def _interior_tie_free_from_positions(  # noqa: PLR0917
    a: data_alloc.NDArray,
    v: data_alloc.NDArray,
    pos: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Return a 0-d bool: no query exactly equals an interior interface.

    Same predicate as ``_interior_tie_free`` but uses the searchsorted
    positions ``pos`` already computed by the v2 fast path, avoiding a
    second batched searchsorted pass.
    """
    last = nlev
    right_idx = array_ns.clip(pos, 0, last)
    right = array_ns.take_along_axis(a, right_idx, axis=1)
    left_idx = array_ns.clip(pos - 1, 0, last)
    left = array_ns.take_along_axis(a, left_idx, axis=1)
    eq_right = v == right
    eq_left = (pos > 0) & (v == left)
    max_interior = (nlev - fi - 1)[:, None]
    interior_right = eq_right & (pos >= 1) & (pos <= max_interior)
    interior_left = eq_left & ((pos - 1) >= 2) & ((pos - 1) <= max_interior)
    return array_ns.logical_not(array_ns.any(interior_right | interior_left))


def _compute_tie_free_from_bundle(
    bundle: dict[str, data_alloc.NDArray | None],
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Return the 0-d interior-tie flag from a precomputed v2 bundle."""
    tie_free_0 = _interior_tie_free_from_positions(
        bundle["z_ifc_e0_m"],
        bundle["z_me_m"],
        bundle["pos_0"],
        bundle["fi_sliced"],
        nlev,
        array_ns,
    )
    tie_free_1 = _interior_tie_free_from_positions(
        bundle["z_ifc_e1_m"],
        bundle["z_me_m"],
        bundle["pos_1"],
        bundle["fi_sliced"],
        nlev,
        array_ns,
    )
    if bundle["z_aux2_v"] is not None:
        assert bundle["z_ifc_e0_m1"] is not None
        assert bundle["z_ifc_e1_m1"] is not None
        assert bundle["pos_aux_0"] is not None
        assert bundle["pos_aux_1"] is not None
        tie_free_aux_0 = _interior_tie_free_from_positions(
            bundle["z_ifc_e0_m1"],
            bundle["z_aux2_v"],
            bundle["pos_aux_0"],
            bundle["fi_sliced1"],
            nlev,
            array_ns,
        )
        tie_free_aux_1 = _interior_tie_free_from_positions(
            bundle["z_ifc_e1_m1"],
            bundle["z_aux2_v"],
            bundle["pos_aux_1"],
            bundle["fi_sliced1"],
            nlev,
            array_ns,
        )
        return tie_free_0 & tie_free_1 & tie_free_aux_0 & tie_free_aux_1
    return tie_free_0 & tie_free_1


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


def _compute_zdiff_gradp_v2_bundle(  # noqa: PLR0917
    array_ns: ModuleType,
    z_ifc_e0: data_alloc.NDArray,
    z_ifc_e1: data_alloc.NDArray,
    z_me: data_alloc.NDArray,
    z_aux2: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    hs: int,
    hs1: int,
) -> dict[str, data_alloc.NDArray | None]:
    """Compute fill bounds, masks, and searchsorted positions for v2's fast path.

    Returns a dict that can be passed back to ``compute_zdiff_gradp_v2`` via
    ``_precomputed`` so dispatch (and future callers) avoid recomputing these
    quantities.  Phase-2 entries are ``None`` when ``hs1 >= nedges``.
    """
    fill_high = (
        array_ns.max(array_ns.stack([z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()]))
        + 1.0
    )
    fill_low = (
        array_ns.min(array_ns.stack([z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()]))
        - 1.0
    )

    nedges = z_ifc_e0.shape[0]
    fi_sliced = fi[hs:]
    z_ifc_mask = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
        nlev + 1 - fi_sliced[:, None]
    )
    z_me_mask = array_ns.arange(nlev, dtype=array_ns.int64)[None, :] <= fi_sliced[:, None]

    z_ifc_e0_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e0[hs:])
    z_ifc_e1_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e1[hs:])
    z_me_m = array_ns.where(z_me_mask, fill_low, z_me[hs:])

    pos_0 = _batched_searchsorted_v2(z_ifc_e0_m, z_me_m, array_ns)
    pos_1 = _batched_searchsorted_v2(z_ifc_e1_m, z_me_m, array_ns)

    if hs1 < nedges:
        fi_sliced1 = fi[hs1:]
        z_aux2_v = z_aux2[hs1:, None]
        z_ifc_mask1 = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
            nlev + 1 - fi_sliced1[:, None]
        )
        z_ifc_e0_m1 = array_ns.where(z_ifc_mask1, fill_high, z_ifc_e0[hs1:])
        z_ifc_e1_m1 = array_ns.where(z_ifc_mask1, fill_high, z_ifc_e1[hs1:])
        pos_aux_0 = _batched_searchsorted_v2(z_ifc_e0_m1, z_aux2_v, array_ns)
        pos_aux_1 = _batched_searchsorted_v2(z_ifc_e1_m1, z_aux2_v, array_ns)
    else:
        fi_sliced1 = None
        z_aux2_v = None
        z_ifc_e0_m1 = z_ifc_e1_m1 = None
        pos_aux_0 = pos_aux_1 = None

    return {
        "fill_high": fill_high,
        "fill_low": fill_low,
        "fi_sliced": fi_sliced,
        "fi_sliced1": fi_sliced1,
        "z_ifc_mask": z_ifc_mask,
        "z_me_mask": z_me_mask,
        "z_ifc_e0_m": z_ifc_e0_m,
        "z_ifc_e1_m": z_ifc_e1_m,
        "z_me_m": z_me_m,
        "z_aux2_v": z_aux2_v,
        "z_ifc_e0_m1": z_ifc_e0_m1,
        "z_ifc_e1_m1": z_ifc_e1_m1,
        "pos_0": pos_0,
        "pos_1": pos_1,
        "pos_aux_0": pos_aux_0,
        "pos_aux_1": pos_aux_1,
    }


def compute_zdiff_gradp_v2(  # noqa: PLR0915
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
    _precomputed_validation_ok: bool = False,
    _precomputed: dict[str, data_alloc.NDArray | None] | None = None,
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

    if _precomputed is None:
        bundle = _compute_zdiff_gradp_v2_bundle(
            array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev, hs, hs1
        )
    else:
        bundle = _precomputed

    if _validation_enabled() and not _precomputed_validation_ok:
        # Interior-tie check derived from the same positions the fast path
        # will use for assembly, so validation adds no extra searchsorted.
        tie_free = _compute_tie_free_from_bundle(bundle, nlev, array_ns)
        combined = _compute_v2_validation(
            array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev, nedges, tie_free=tie_free
        )
        if not bool(combined[0] & combined[1]):
            raise ValueError(
                "compute_zdiff_gradp_v2 input validation failed: strict z_ifc decrease, "
                "z_me monotonicity, finiteness, A2 float-offset premise, min-spacing-vs-ULP, "
                "or exact interior tie (v2 would pick the shallower/deeper index differently from main) violated."
            )
    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    fi_sliced = bundle["fi_sliced"]
    assert fi_sliced is not None
    valid_jk = jk_idx > fi_sliced[:, None]

    # Phase 1, cell 0
    pos_0 = bundle["pos_0"]
    assert pos_0 is not None

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
    pos_1 = bundle["pos_1"]
    assert pos_1 is not None

    jk1_1 = array_ns.clip(nlev - pos_1, fi_sliced[:, None], nlev - 1)
    z_mc_e1 = z_mc[e2c_1]
    zdiff_gradp[hs:, 1, :] = array_ns.where(
        valid_jk,
        z_me[hs:] - array_ns.take_along_axis(z_mc_e1[hs:], jk1_1.astype(array_ns.int64), axis=1),
        zdiff_gradp[hs:, 1, :],
    )
    vertoffset_gradp[hs:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 1, :],
    )

    # Phase 2
    if hs1 < nedges:
        fi_sliced1 = bundle["fi_sliced1"]
        z_aux2_v = bundle["z_aux2_v"]
        assert fi_sliced1 is not None
        assert z_aux2_v is not None

        pos_aux_0 = bundle["pos_aux_0"]
        assert pos_aux_0 is not None

        jk1_aux_0 = array_ns.clip(nlev - pos_aux_0, fi_sliced1[:, None], nlev - 1)
        jk1_aux_0 = array_ns.where(
            pos_aux_0 >= (nlev + 1 - fi_sliced1)[:, None],
            nlev - 1,
            jk1_aux_0,
        )

        pos_aux_1 = bundle["pos_aux_1"]
        assert pos_aux_1 is not None
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


def _exact_v2_chunk_size(nlev: int, itemsize: int) -> int:
    bytes_per_edge = 2 * nlev * nlev * itemsize
    return max(1, _EXACT_V2_MAX_TABLE_BYTES // bytes_per_edge)


def _exact_v2_edge_chunks(start: int, end: int, chunk_size: int) -> Iterator[slice]:
    for s in range(start, end, chunk_size):
        yield slice(s, min(s + chunk_size, end))


def compute_zdiff_gradp_exact_v2(  # noqa: PLR0915
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
    """Exact variant with E3 dispatch and 1 GiB-capped auto-chunking.

    The fast path builds successor tables over edge chunks so that one
    ``(chunk, nlev, nlev)`` int8 table plus its doubling-scan copy stays under
    ``_EXACT_V2_MAX_TABLE_BYTES`` (~1 GiB). Results are identical to the
    unchunked gather. Validation follows ``_validation_enabled()``: when enabled
    finiteness raises before compute and E3 selects fast vs. carry path; when
    disabled the fast path is taken without any check.
    """
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

    use_carry = False
    if _validation_enabled():
        finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
        e3_ok = _check_e3(array_ns, z_me, fi, nlev)

        combined = finite_ok & e3_ok
        if not bool(combined):
            if not bool(finite_ok):
                raise ValueError("Searched arrays contain non-finite values.")
            use_carry = True
    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    itemsize = 1 if nlev <= 127 else 2
    chunk_size = _exact_v2_chunk_size(nlev, itemsize)

    # Phase 1, cell 0: no carry; fresh scan at every jk.
    for chunk in _exact_v2_edge_chunks(hs, nedges, chunk_size):
        z_ifc_k0 = z_ifc_e0[chunk, :].astype(array_ns.float64)
        z_me_c = z_me[chunk, :].astype(array_ns.float64)
        fi_c = fi[chunk]
        valid_jk_c = valid_jk[chunk, :]
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

    # Phase 1, cell 1: fresh scan when E3 holds, carry fallback otherwise.
    for chunk in _exact_v2_edge_chunks(hs, nedges, chunk_size):
        z_ifc_k1 = z_ifc_e1[chunk, :].astype(array_ns.float64)
        z_me_c = z_me[chunk, :].astype(array_ns.float64)
        fi_c = fi[chunk]
        valid_jk_c = valid_jk[chunk, :]
        succ1 = _exact_query_succ(z_ifc_k1, z_me_c, array_ns)
        if use_carry:
            jk1_1 = _exact_carry_loop(succ1, fi_c, valid_jk_c, array_ns)
        else:
            jk1_1 = _exact_phase1_cell0(succ1, fi_c, array_ns)
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
        for chunk in _exact_v2_edge_chunks(hs1, nedges, chunk_size):
            z_ifc_k0 = z_ifc_e0[chunk, :].astype(array_ns.float64)
            z_ifc_k1 = z_ifc_e1[chunk, :].astype(array_ns.float64)
            fi_c = fi[chunk]
            phase2_active_c = phase2_active[chunk, :]
            z_aux2_v = z_aux2[chunk, None].astype(array_ns.float64)

            succ2_0 = _exact_query_succ(z_ifc_k0, z_aux2_v, array_ns)
            jk1_aux_0 = _exact_phase1_cell0(succ2_0, fi_c, array_ns)
            z_mc_e0 = z_mc[e2c_0[chunk]]
            zdiff_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                z_aux2_v
                - array_ns.take_along_axis(z_mc_e0, jk1_aux_0.astype(array_ns.int64), axis=1),
                zdiff_gradp[chunk, 0, :],
            )
            vertoffset_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_0 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 0, :],
            )

            succ2_1 = _exact_query_succ(z_ifc_k1, z_aux2_v, array_ns)
            jk1_aux_1 = _exact_phase1_cell0(succ2_1, fi_c, array_ns)
            z_mc_e1 = z_mc[e2c_1[chunk]]
            zdiff_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                z_aux2_v
                - array_ns.take_along_axis(z_mc_e1, jk1_aux_1.astype(array_ns.int64), axis=1),
                zdiff_gradp[chunk, 1, :],
            )
            vertoffset_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_1 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 1, :],
            )

    _LAST_EXACT_V2_PATH = "carry" if use_carry else "fast"
    return zdiff_gradp, vertoffset_gradp


_EXACT_V3_FAST_BLOCK_SIZE: int = 256

_EXACT_V3_FIRST_MATCH_KERNEL_SRC = r"""
extern "C" __global__ void first_match_kernel(
    const double* __restrict__ z_ifc,
    const double* __restrict__ queries,
    const long long* __restrict__ fi,
    long long* __restrict__ out,
    int nedges,
    int nlev,
    int nq
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nedges * nq;
    if (tid >= total) return;

    int e = tid / nq;
    int q = tid - e * nq;
    long long fi_e = fi[e];
    double query = queries[e * nq + q];

    for (long long i = fi_e; i < nlev; ++i) {
        if (i == nlev - 1) {
            out[tid] = nlev - 1;
            return;
        }
        double top = z_ifc[e * (nlev + 1) + i];
        double bot = z_ifc[e * (nlev + 1) + i + 1];
        if (top >= query && query >= bot) {
            out[tid] = i;
            return;
        }
    }

    out[tid] = nlev - 1;
}
"""


def _compile_exact_v3_kernel(array_ns: ModuleType) -> Any:
    """Compile and cache the exact_v3 first-match RawKernel (cupy only)."""
    if array_ns.__name__ != "cupy":
        raise TypeError("_compile_exact_v3_kernel requires cupy.")
    kernel = getattr(_compile_exact_v3_kernel, "_kernel", None)
    if kernel is None:
        kernel = array_ns.RawKernel(_EXACT_V3_FIRST_MATCH_KERNEL_SRC, "first_match_kernel")
        _compile_exact_v3_kernel._kernel = kernel  # type: ignore[attr-defined]
    return kernel


def _launch_exact_v3_first_match_kernel(
    array_ns: ModuleType,
    z_ifc_k: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """Launch the cupy first-match scan kernel for one cell/query pair."""
    kernel = _compile_exact_v3_kernel(array_ns)
    nedges = int(z_ifc_k.shape[0])
    nq = int(queries.shape[1])
    out = array_ns.empty((nedges, nq), dtype=array_ns.int64)
    total = nedges * nq
    block = _EXACT_V3_FAST_BLOCK_SIZE
    grid = (total + block - 1) // block
    kernel(
        (grid,),
        (block,),
        (z_ifc_k, queries, fi, out, nedges, nlev, nq),
    )
    return out


def _first_match_scan_reference(
    z_ifc_k: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
) -> np.ndarray:
    """Numpy reference for the exact_v3 per-edge first-match scan.

    This implements the same predicate as the cupy RawKernel:

        jk1[e, q] = min { i >= fi[e] : i == nlev-1
                                   or (z_ifc_k[e, i] >= query[e, q]
                                       >= z_ifc_k[e, i+1]) }

    The scan starts at ``fi[e]``, uses the same inclusive bracket comparisons
    as ``_exact_query_succ``, and treats the deepest level ``i == nlev-1`` as
    an unconditional member.  Consequently it returns exactly the value that
    the successor-table fast path gathers as ``succ[q, fi]``.
    """
    z_ifc_k = np.asarray(z_ifc_k, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    fi = np.asarray(fi, dtype=np.int64)
    nedges, nlev_p1 = z_ifc_k.shape
    nlev = nlev_p1 - 1
    if queries.ndim == 1:
        queries = queries[:, np.newaxis]
    if queries.shape[0] != nedges:
        raise ValueError("queries must have shape (nedges, nq) or (nedges,).")

    out = np.full((nedges, queries.shape[1]), nlev - 1, dtype=np.int64)
    found = np.zeros((nedges, queries.shape[1]), dtype=bool)

    for i in range(nlev):
        active = (i >= fi[:, np.newaxis]) & ~found
        if i == nlev - 1:
            out[active] = nlev - 1
            break
        bracket = (z_ifc_k[:, i][:, np.newaxis] >= queries) & (
            queries >= z_ifc_k[:, i + 1][:, np.newaxis]
        )
        newly = active & bracket
        out[newly] = i
        found |= newly

    return out


def _exact_v3_assemble_cell(  # noqa: PLR0917
    zdiff_gradp: data_alloc.NDArray,
    vertoffset_gradp: data_alloc.NDArray,
    jk1: data_alloc.NDArray,
    z_mc_ec: data_alloc.NDArray,
    query_v: data_alloc.NDArray,
    active: data_alloc.NDArray,
    jk_idx: data_alloc.NDArray,
    cell: int,
    array_ns: ModuleType,
) -> None:
    """Apply the D6 output-assembly pattern for one cell."""
    zdiff_gradp[:, cell, :] = array_ns.where(
        active,
        query_v - array_ns.take_along_axis(z_mc_ec, jk1.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, cell, :],
    )
    vertoffset_gradp[:, cell, :] = array_ns.where(
        active,
        (jk1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, cell, :],
    )


def compute_zdiff_gradp_exact_v3(
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
    """Exact variant with a cupy first-match-scan fast path and no tables.

    Semantics are identical to ``compute_zdiff_gradp_exact_v2``.  On numpy
    the implementation delegates to ``compute_zdiff_gradp_exact_v2``.  On cupy
    the fast path evaluates main's bracket predicate by a per-(edge, query)
    linear scan starting at ``fi[e]``.

    Correctness proof for the cupy fast path:

    - The kernel predicate is exactly the table predicate from
      ``_exact_query_succ``:
      ``z_ifc_k[e, i] >= query[e, q] >= z_ifc_k[e, i+1]``.
    - The scan starts at ``i = fi[e]``, matching the suffix-minimum gather
      ``succ[q, fi]`` used by ``_exact_phase1_cell0``.
    - ``i == nlev-1`` is returned unconditionally, matching the
      ``unconditional`` column in ``_exact_query_succ``.
    - Therefore the kernel returns ``min{i >= fi[e] : predicate holds}`` for
      every (e, q).  This is byte-identical to ``succ[q, fi]``.
    - Phase-1 cell-0 and both phase-2 queries are fresh in main as well,
      so the fresh value is the correct value.
    - Under E3, phase-1 cell-1's carried ``jk_start`` lower bound never
      exceeds the fresh first-match (D6 E3 proof), so the fresh value equals
      the carry value.
    - Output assembly reuses the D6 pattern from exact_v2, so the final
      fields are bitwise-identical to exact_v2 (and therefore to main-golden
      on E3-valid inputs).
    - If E3 does not hold, the function falls back to exact_v2's carry
      machinery, preserving exactness.
    """
    global _LAST_EXACT_V3_PATH  # noqa: PLW0603
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]

    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)
    if hs1 < hs:
        raise ValueError("horizontal_start_1 must be greater than or equal to horizontal_start.")

    # Numpy path: reuse exact_v2 machinery unchanged.
    if array_ns.__name__ != "cupy":
        out = compute_zdiff_gradp_exact_v2(
            e2c=e2c,
            z_me=z_me,
            z_mc=z_mc,
            z_ifc=z_ifc,
            flat_idx=flat_idx,
            topography=topography,
            nlev=nlev,
            horizontal_start=horizontal_start,
            horizontal_start_1=horizontal_start_1,
        )
        _LAST_EXACT_V3_PATH = _LAST_EXACT_V2_PATH
        return out

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_e0 = z_ifc[e2c_0, :].astype(array_ns.float64)
    z_ifc_e1 = z_ifc[e2c_1, :].astype(array_ns.float64)

    # Validation follows the same opt-out flag as other variants. When enabled,
    # finiteness raises before compute and E3 selects fast vs. carry path.
    if _validation_enabled():
        finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
        e3_ok = _check_e3(array_ns, z_me, fi, nlev)
        combined = finite_ok & e3_ok
        if not bool(combined):
            if not bool(finite_ok):
                raise ValueError("Searched arrays contain non-finite values.")
            _LAST_EXACT_V3_PATH = "carry"
            return compute_zdiff_gradp_exact_v2(
                e2c=e2c,
                z_me=z_me,
                z_mc=z_mc,
                z_ifc=z_ifc,
                flat_idx=flat_idx,
                topography=topography,
                nlev=nlev,
                horizontal_start=horizontal_start,
                horizontal_start_1=horizontal_start_1,
            )
    _LAST_EXACT_V3_PATH = "fast"

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    z_me_f64 = z_me.astype(array_ns.float64)
    z_aux2_v = z_aux2[:, None].astype(array_ns.float64)

    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    jk1_0 = _launch_exact_v3_first_match_kernel(array_ns, z_ifc_e0, z_me_f64, fi, nlev)
    _exact_v3_assemble_cell(
        zdiff_gradp,
        vertoffset_gradp,
        jk1_0,
        z_mc_e0,
        z_me_f64,
        valid_jk,
        jk_idx,
        cell=0,
        array_ns=array_ns,
    )
    del jk1_0

    jk1_1 = _launch_exact_v3_first_match_kernel(array_ns, z_ifc_e1, z_me_f64, fi, nlev)
    _exact_v3_assemble_cell(
        zdiff_gradp,
        vertoffset_gradp,
        jk1_1,
        z_mc_e1,
        z_me_f64,
        valid_jk,
        jk_idx,
        cell=1,
        array_ns=array_ns,
    )
    del jk1_1

    if hs1 < nedges:
        jk1_aux_0 = _launch_exact_v3_first_match_kernel(array_ns, z_ifc_e0, z_aux2_v, fi, nlev)
        _exact_v3_assemble_cell(
            zdiff_gradp,
            vertoffset_gradp,
            jk1_aux_0,
            z_mc_e0,
            z_aux2_v,
            phase2_active,
            jk_idx,
            cell=0,
            array_ns=array_ns,
        )
        del jk1_aux_0

        jk1_aux_1 = _launch_exact_v3_first_match_kernel(array_ns, z_ifc_e1, z_aux2_v, fi, nlev)
        _exact_v3_assemble_cell(
            zdiff_gradp,
            vertoffset_gradp,
            jk1_aux_1,
            z_mc_e1,
            z_aux2_v,
            phase2_active,
            jk_idx,
            cell=1,
            array_ns=array_ns,
        )
        del jk1_aux_1

    return zdiff_gradp, vertoffset_gradp


def compute_zdiff_gradp_dispatch(
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
    global _LAST_DISPATCH_PATH  # noqa: PLW0603
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

    z_ifc_asc = z_ifc[:, ::-1].copy()
    z_ifc_e0 = z_ifc_asc[e2c_0]
    z_ifc_e1 = z_ifc_asc[e2c_1]

    if _validation_enabled():
        # Compute the fast-path bundle once; derive the interior-tie check from
        # the same searchsorted positions so dispatch does not pay for a second
        # batched searchsorted pass.
        bundle = _compute_zdiff_gradp_v2_bundle(
            array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev, hs, hs1
        )
        tie_free = _compute_tie_free_from_bundle(bundle, nlev, array_ns)
        combined = _compute_v2_validation(
            array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2, fi, nlev, nedges, tie_free=tie_free
        )
        finite_ok = combined[0]
        full_ok = combined[1]
        if not bool(finite_ok):
            raise ValueError("Searched arrays contain non-finite values.")
        if bool(full_ok):
            _LAST_DISPATCH_PATH = "fast"
            return compute_zdiff_gradp_v2(
                e2c=e2c,
                z_me=z_me,
                z_mc=z_mc,
                z_ifc=z_ifc,
                flat_idx=flat_idx,
                topography=topography,
                nlev=nlev,
                horizontal_start=horizontal_start,
                horizontal_start_1=horizontal_start_1,
                _precomputed_validation_ok=True,
                _precomputed=bundle,
            )
        _LAST_DISPATCH_PATH = "exact"
        return compute_zdiff_gradp_exact_v2(
            e2c=e2c,
            z_me=z_me,
            z_mc=z_mc,
            z_ifc=z_ifc,
            flat_idx=flat_idx,
            topography=topography,
            nlev=nlev,
            horizontal_start=horizontal_start,
            horizontal_start_1=horizontal_start_1,
        )

    # Validation OFF: unsafe-legacy timing mode; always use the v2 fast path.
    _LAST_DISPATCH_PATH = "fast"
    return compute_zdiff_gradp_v2(
        e2c=e2c,
        z_me=z_me,
        z_mc=z_mc,
        z_ifc=z_ifc,
        flat_idx=flat_idx,
        topography=topography,
        nlev=nlev,
        horizontal_start=horizontal_start,
        horizontal_start_1=horizontal_start_1,
        _precomputed_validation_ok=True,
    )


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
    if nlev <= 127:
        idx_dtype = array_ns.int8
    elif nlev <= 32767:
        idx_dtype = array_ns.int16
    else:
        raise ValueError(
            f"compute_zdiff_gradp_exact successor tables support nlev <= 32767, got {nlev}."
        )
    idx = array_ns.where(
        bracket | unconditional,
        a_idx[None, None, :].astype(idx_dtype),
        array_ns.asarray(nlev, dtype=idx_dtype),
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


def _exact_v4_first_match(
    z_ifc_col: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Return main's first-match index for each (edge, query).

    For edge ``e`` and query ``q`` the result is the smallest candidate
    index ``i >= fi[e]`` such that ``i == nlev - 1`` or
    ``z_ifc_col[e, i] >= queries[e, q] >= z_ifc_col[e, i + 1]``.  This is a
    literal broadcast transcription of the reference loop's bracket predicate
    (``_main_reference``): build the gated candidate mask, replace ungated
    entries by ``nlev``, and take the minimum along the candidate axis.

    The dominant transient is one int32 index plus one boolean gate per
    element (five bytes/element); ``_exact_v2_chunk_size(nlev, 4)`` models
    two int32 tables per edge (eight bytes/element) for margin.  The cap
    holds for ICON nlev (<=137); beyond the ``nlev^2 > MAX_TABLE/8`` chunk
    floor the soft cap is shared with ``compute_zdiff_gradp_exact_v2``.
    """
    _chunk, nlev_p1 = z_ifc_col.shape
    nlev = nlev_p1 - 1
    i_idx = array_ns.arange(nlev, dtype=array_ns.int32)
    nlev_fill = array_ns.asarray(nlev, dtype=array_ns.int32)

    z_ifc_col = z_ifc_col.astype(array_ns.float64)
    queries = queries.astype(array_ns.float64)

    upper = z_ifc_col[:, None, :-1]
    lower = z_ifc_col[:, None, 1:]
    qv = queries[:, :, None]

    bracket = (upper >= qv) & (qv >= lower)
    unconditional = i_idx[None, None, :] == (nlev - 1)  # gated only if fi <= nlev-1
    fi_mask = i_idx[None, None, :] >= fi[:, None, None]

    gated = fi_mask & (bracket | unconditional)
    idx = array_ns.where(gated, i_idx[None, None, :], nlev_fill)
    return idx.min(axis=-1).astype(array_ns.int64)


def compute_zdiff_gradp_exact_v4(
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
    """Exact variant: premise-free broadcast first-match.

    The fast path evaluates the reference loop's bracket predicate for every
    candidate index ``i >= flat_idx`` and takes the first match as the
    minimum gated index.  Because the query is constant per edge in phase 2
    and phase 1 cell 0 carries nothing, the fresh first-match equals main's
    output there unconditionally.  Under E3 the same holds for phase 1 cell 1
    (D6-E3 proof), so an E3 check selects fast versus carry; when E3 fails the
    implementation delegates to ``compute_zdiff_gradp_exact_v2``'s carry loop,
    preserving exactness on all finite inputs.

    Validation follows ``_validation_enabled()``: when enabled, finiteness and
    E3 are checked with one stacked device sync on the fast path (a second
    sync occurs on the carry path, where the finite and E3 bits are read
    separately); when disabled the fast path is taken without any check and
    the defined ``nlev - 1`` fallback applies on non-finite input.
    """
    global _LAST_EXACT_V4_PATH  # noqa: PLW0603
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

    z_ifc_e0 = z_ifc[e2c_0, :]
    z_ifc_e1 = z_ifc[e2c_1, :]

    if _validation_enabled():
        finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
        e3_ok = _check_e3(array_ns, z_me, fi, nlev)
        combined = finite_ok & e3_ok
        if not bool(combined):
            if not bool(finite_ok):
                raise ValueError("Searched arrays contain non-finite values.")
            _LAST_EXACT_V4_PATH = "carry"
            return compute_zdiff_gradp_exact_v2(
                e2c=e2c,
                z_me=z_me,
                z_mc=z_mc,
                z_ifc=z_ifc,
                flat_idx=flat_idx,
                topography=topography,
                nlev=nlev,
                horizontal_start=horizontal_start,
                horizontal_start_1=horizontal_start_1,
            )

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    # int32 transient: boolean gate (1 byte) + int32 index (4 bytes), bounded
    # by sizing chunks as if the work array were 2 * nlev * nlev * int32.
    chunk_size = _exact_v2_chunk_size(nlev, 4)

    for chunk in _exact_v2_edge_chunks(hs, nedges, chunk_size):
        jk1_0 = _exact_v4_first_match(z_ifc_e0[chunk, :], z_me[chunk, :], fi[chunk], array_ns)
        z_mc_e0 = z_mc[e2c_0[chunk]]
        zdiff_gradp[chunk, 0, :] = array_ns.where(
            valid_jk[chunk, :],
            z_me[chunk, :] - array_ns.take_along_axis(z_mc_e0, jk1_0, axis=1),
            zdiff_gradp[chunk, 0, :],
        )
        vertoffset_gradp[chunk, 0, :] = array_ns.where(
            valid_jk[chunk, :],
            (jk1_0 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[chunk, 0, :],
        )

    for chunk in _exact_v2_edge_chunks(hs, nedges, chunk_size):
        jk1_1 = _exact_v4_first_match(z_ifc_e1[chunk, :], z_me[chunk, :], fi[chunk], array_ns)
        z_mc_e1 = z_mc[e2c_1[chunk]]
        zdiff_gradp[chunk, 1, :] = array_ns.where(
            valid_jk[chunk, :],
            z_me[chunk, :] - array_ns.take_along_axis(z_mc_e1, jk1_1, axis=1),
            zdiff_gradp[chunk, 1, :],
        )
        vertoffset_gradp[chunk, 1, :] = array_ns.where(
            valid_jk[chunk, :],
            (jk1_1 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[chunk, 1, :],
        )

    if hs1 < nedges:
        for chunk in _exact_v2_edge_chunks(hs1, nedges, chunk_size):
            z_aux2_v = z_aux2[chunk, None]
            jk1_aux_0 = _exact_v4_first_match(z_ifc_e0[chunk, :], z_aux2_v, fi[chunk], array_ns)
            jk1_aux_1 = _exact_v4_first_match(z_ifc_e1[chunk, :], z_aux2_v, fi[chunk], array_ns)
            z_mc_e0 = z_mc[e2c_0[chunk]]
            z_mc_e1 = z_mc[e2c_1[chunk]]
            zdiff_gradp[chunk, 0, :] = array_ns.where(
                phase2_active[chunk, :],
                z_aux2_v - array_ns.take_along_axis(z_mc_e0, jk1_aux_0, axis=1),
                zdiff_gradp[chunk, 0, :],
            )
            vertoffset_gradp[chunk, 0, :] = array_ns.where(
                phase2_active[chunk, :],
                (jk1_aux_0 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 0, :],
            )
            zdiff_gradp[chunk, 1, :] = array_ns.where(
                phase2_active[chunk, :],
                z_aux2_v - array_ns.take_along_axis(z_mc_e1, jk1_aux_1, axis=1),
                zdiff_gradp[chunk, 1, :],
            )
            vertoffset_gradp[chunk, 1, :] = array_ns.where(
                phase2_active[chunk, :],
                (jk1_aux_1 - jk_idx).astype(gtx.int32),
                vertoffset_gradp[chunk, 1, :],
            )

    _LAST_EXACT_V4_PATH = "fast"
    return zdiff_gradp, vertoffset_gradp


_EXACT_V5_ELEMENTWISE_KERNEL_SRC = r"""
int e = i / nq;
int k = i - e * nq;
int fi_e = FI[e];
double v = Q[e * nq + k];
for (int a = fi_e; a < nlev; ++a) {
    if (a == nlev - 1) {
        out = nlev - 1;
        return;
    }
    double top = D[e * (nlev + 1) + a];
    double bot = D[e * (nlev + 1) + a + 1];
    if (top >= v && v >= bot) {
        out = a;
        return;
    }
}
out = nlev - 1;
"""


def _get_exact_v5_elementwise_kernel(array_ns: ModuleType, nlev: int) -> Any:
    """Return cached cupy ElementwiseKernel for exact_v5 first-match scan.

    The kernel is keyed by ``nlev`` because the scalar ``nlev`` appears in
    the source and the launch bounds; the per-edge ``fi`` is passed as a
    raw array so the same kernel works for any ``nedges``.
    """
    if nlev not in _EXACT_V5_KERNEL_CACHE:
        _EXACT_V5_KERNEL_CACHE[nlev] = array_ns.ElementwiseKernel(
            "raw float64 D, raw float64 Q, raw int32 FI, int32 nedges, int32 nlev, int32 nq",
            "int32 out",
            _EXACT_V5_ELEMENTWISE_KERNEL_SRC,
            f"exact_v5_first_match_nlev{nlev}",
        )
    return _EXACT_V5_KERNEL_CACHE[nlev]


def _launch_exact_v5_first_match_kernel(
    array_ns: ModuleType,
    z_ifc_k: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """Launch the cupy ElementwiseKernel first-match scan for one cell/query pair.

    Returns a (nedges, nq) int32 array; the caller casts to int64 at the
    ``take_along_axis`` call.
    """
    kernel = _get_exact_v5_elementwise_kernel(array_ns, nlev)
    nedges = int(z_ifc_k.shape[0])
    nq = int(queries.shape[1])
    z_ifc_k = array_ns.ascontiguousarray(z_ifc_k.astype(array_ns.float64))
    queries = array_ns.ascontiguousarray(queries.astype(array_ns.float64))
    fi_int32 = fi.astype(array_ns.int32)
    out_flat = kernel(z_ifc_k, queries, fi_int32, nedges, nlev, nq, size=nedges * nq)
    return out_flat.reshape(nedges, nq)


def compute_zdiff_gradp_exact_v5(
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
    """Exact variant with a cupy ElementwiseKernel first-match fast path.

    Semantics are identical to ``compute_zdiff_gradp_exact_v4``: premise-free,
    with finiteness/E3 gated by ``_validation_enabled()`` and E3-false falling
    back to ``compute_zdiff_gradp_exact_v2``'s carry machinery.  On numpy the
    implementation delegates to ``compute_zdiff_gradp_exact_v4``.  On cupy the
    fast path evaluates main's bracket predicate by a single fused
    ``ElementwiseKernel`` per (phase, cell) that computes the fresh first-match
    directly, avoiding the successor-table materialisation of exact_v2/v4.

    The cupy kernel correctness argument mirrors ``compute_zdiff_gradp_exact_v3``:
    the per-thread scan starts at ``fi[e]``, uses the inclusive bracket
    ``z_ifc_k[e, a] >= query >= z_ifc_k[e, a + 1]``, and treats ``a == nlev - 1``
    as an unconditional member.  This returns ``min{a >= fi[e] : predicate}`` for
    every (edge, query), which equals the successor-table gather ``succ[q, fi]``
    used by exact_v2/v4.  Phase 1 cell 0 and both phase 2 cells have no carry in
    main; phase 1 cell 1 equals main's carry under E3 (D6-E3 proof), so the E3
    dispatch selects fast versus carry exactly as exact_v4 does.
    """
    global _LAST_EXACT_V5_PATH  # noqa: PLW0603
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]

    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)
    if hs1 < hs:
        raise ValueError("horizontal_start_1 must be greater than or equal to horizontal_start.")

    # Numpy path: reuse exact_v4's readable broadcast fast path verbatim.
    if array_ns.__name__ != "cupy":
        out = compute_zdiff_gradp_exact_v4(
            e2c=e2c,
            z_me=z_me,
            z_mc=z_mc,
            z_ifc=z_ifc,
            flat_idx=flat_idx,
            topography=topography,
            nlev=nlev,
            horizontal_start=horizontal_start,
            horizontal_start_1=horizontal_start_1,
        )
        _LAST_EXACT_V5_PATH = _LAST_EXACT_V4_PATH
        return out

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    # Gather the decreasing-k column per cell; advanced indexing yields a new
    # C-contiguous array, but force contiguity explicitly for the raw kernel.
    z_ifc_e0 = array_ns.ascontiguousarray(z_ifc[e2c_0, :].astype(array_ns.float64))
    z_ifc_e1 = array_ns.ascontiguousarray(z_ifc[e2c_1, :].astype(array_ns.float64))

    if _validation_enabled():
        finite_ok = _check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)
        e3_ok = _check_e3(array_ns, z_me, fi, nlev)
        combined = finite_ok & e3_ok
        if not bool(combined):
            if not bool(finite_ok):
                raise ValueError("Searched arrays contain non-finite values.")
            _LAST_EXACT_V5_PATH = "carry"
            return compute_zdiff_gradp_exact_v2(
                e2c=e2c,
                z_me=z_me,
                z_mc=z_mc,
                z_ifc=z_ifc,
                flat_idx=flat_idx,
                topography=topography,
                nlev=nlev,
                horizontal_start=horizontal_start,
                horizontal_start_1=horizontal_start_1,
            )

    _LAST_EXACT_V5_PATH = "fast"

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    z_me_f64 = z_me.astype(array_ns.float64)
    z_aux2_v = z_aux2[:, None].astype(array_ns.float64)

    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    jk1_0 = _launch_exact_v5_first_match_kernel(array_ns, z_ifc_e0, z_me_f64, fi, nlev)
    _exact_v3_assemble_cell(
        zdiff_gradp,
        vertoffset_gradp,
        jk1_0,
        z_mc_e0,
        z_me_f64,
        valid_jk,
        jk_idx,
        cell=0,
        array_ns=array_ns,
    )
    del jk1_0

    jk1_1 = _launch_exact_v5_first_match_kernel(array_ns, z_ifc_e1, z_me_f64, fi, nlev)
    _exact_v3_assemble_cell(
        zdiff_gradp,
        vertoffset_gradp,
        jk1_1,
        z_mc_e1,
        z_me_f64,
        valid_jk,
        jk_idx,
        cell=1,
        array_ns=array_ns,
    )
    del jk1_1

    if hs1 < nedges:
        jk1_aux_0 = _launch_exact_v5_first_match_kernel(array_ns, z_ifc_e0, z_aux2_v, fi, nlev)
        _exact_v3_assemble_cell(
            zdiff_gradp,
            vertoffset_gradp,
            jk1_aux_0,
            z_mc_e0,
            z_aux2_v,
            phase2_active,
            jk_idx,
            cell=0,
            array_ns=array_ns,
        )
        del jk1_aux_0

        jk1_aux_1 = _launch_exact_v5_first_match_kernel(array_ns, z_ifc_e1, z_aux2_v, fi, nlev)
        _exact_v3_assemble_cell(
            zdiff_gradp,
            vertoffset_gradp,
            jk1_aux_1,
            z_mc_e1,
            z_aux2_v,
            phase2_active,
            jk_idx,
            cell=1,
            array_ns=array_ns,
        )
        del jk1_aux_1

    return zdiff_gradp, vertoffset_gradp


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
