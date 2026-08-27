# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compute ``zdiff_gradp`` and ``vertoffset_gradp`` via batched searchsorted.

``z_ifc`` (CELL_HEIGHT_ON_HALF_LEVEL) is the SLEVE vertical coordinate built by
``_compute_SLEVE_coordinate_from_vcta_and_topography`` (vertical.py:558) from a
monotone ``vct_a`` table plus smoothed topography.  The grid builder then runs
``_check_and_correct_layer_thickness`` (vertical.py:625), which enforces minimum
layer thicknesses and therefore guarantees ``z_ifc[:, k] > z_ifc[:, k + 1]``
(strictly decreasing in ``k``) for every production grid.  Because ``z_mc`` is
the midpoint of two strictly decreasing half-level values, it is strictly
decreasing in ``k``; ``z_me`` is a non-negative linear combination of two
non-increasing level sequences, so it is non-increasing in ``k`` per edge.  With
these invariants, a single batched ``searchsorted`` on the ascending half-level
column gives the same first-match index as the reference loop, and the result is
non-decreasing in ``jk``, so ``clip(searchsorted_result, fi, nlev-1)`` reproduces
the reference ``jk_start`` carry without a sequential loop.
"""

from __future__ import annotations

from types import ModuleType

import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


def _batched_searchsorted(
    array_ns: ModuleType,
    a: data_alloc.NDArray,
    v: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Batched ``searchsorted`` over the rows of ``a`` with values ``v``.

    ``a`` has shape ``(m, n)`` and each row is sorted ascending.  ``v`` has shape
    ``(m, q)``.  The implementation offsets each row by a large constant so that
    a single call to ``searchsorted`` on the flattened array resolves all rows
    independently.  Float64 is used internally to avoid precision issues in the
    offset computation.
    """
    a = a.astype(array_ns.float64)
    v = v.astype(array_ns.float64)
    m, n = a.shape
    max_num = max(a.max() - a.min(), v.max() - v.min()) + 1.0
    r = max_num * array_ns.arange(m, dtype=array_ns.float64)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel(), side="right").reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def _cumulative_max(array_ns: ModuleType, a: data_alloc.NDArray) -> data_alloc.NDArray:
    """Inclusive forward maximum scan over the last axis using Hillis-Steele."""
    out = a.copy()
    step = 1
    n = a.shape[-1]
    while step < n:
        out[..., step:] = array_ns.maximum(out[..., step:], out[..., :-step])
        step *= 2
    return out


def _cumulative_or(array_ns: ModuleType, a: data_alloc.NDArray) -> data_alloc.NDArray:
    """Inclusive forward logical-or scan over the last axis using Hillis-Steele."""
    out = a.copy()
    step = 1
    n = a.shape[-1]
    while step < n:
        out[..., step:] = array_ns.logical_or(out[..., step:], out[..., :-step])
        step *= 2
    return out


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
    """Compute ``zdiff_gradp`` and ``vertoffset_gradp`` matching main exactly.

    This implementation relies on two grid invariants proved in the module
    docstring: ``z_ifc`` is strictly decreasing in ``k`` (vertical.py:558 and
    vertical.py:625), and ``z_me`` is non-increasing in ``k`` per edge.  Under
    those invariants, ``searchsorted`` returns the reference first-match index
    and the result is non-decreasing in ``jk``; clipping it to ``[fi, nlev-1]``
    reproduces the reference ``jk_start`` carry without an explicit loop.
    """
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

    fill_high = (
        max(
            float(array_ns.max(z_ifc_e0)),
            float(array_ns.max(z_ifc_e1)),
            float(array_ns.max(z_me)),
            float(array_ns.max(z_aux2)),
        )
        + 1.0
    )
    fill_low = (
        min(
            float(array_ns.min(z_ifc_e0)),
            float(array_ns.min(z_ifc_e1)),
            float(array_ns.min(z_me)),
            float(array_ns.min(z_aux2)),
        )
        - 1.0
    )

    z_ifc_mask = array_ns.arange(nlev + 1, dtype=array_ns.int64)[None, :] >= (
        nlev + 1 - fi[:, None]
    )
    z_me_mask = array_ns.arange(nlev, dtype=array_ns.int64)[None, :] <= fi[:, None]

    z_ifc_e0_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e0)
    z_ifc_e1_m = array_ns.where(z_ifc_mask, fill_high, z_ifc_e1)
    z_me_m = array_ns.where(z_me_mask, fill_low, z_me)

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    boundary = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    valid_jk = (jk_idx > fi[:, None]) & boundary[:, None]

    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    # Phase 1, cell 0: independent first-match at every level.
    pos_0 = _batched_searchsorted(array_ns, z_ifc_e0_m, z_me_m)
    raw_jk1_0 = nlev - pos_0
    # If the query lies above the unmasked top, the reference loop falls through
    # to the unconditional nlev-1 bracket.
    jk1_0 = array_ns.where(raw_jk1_0 < fi[:, None], nlev - 1, raw_jk1_0)
    jk1_0 = array_ns.clip(jk1_0, fi[:, None], nlev - 1)
    zdiff_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e0, jk1_0.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 0, :],
    )
    vertoffset_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 0, :],
    )

    # Phase 1, cell 1: replicate the reference jk_start carry.  When E3 holds,
    # the per-level searchsorted result is non-decreasing and the scan is a
    # no-op; when E3 is violated, the scan reproduces the reference fall-through.
    pos_1 = _batched_searchsorted(array_ns, z_ifc_e1_m, z_me_m)
    raw_jk1_1 = nlev - pos_1
    raw_too_high_1 = raw_jk1_1 < fi[:, None]
    # Masked levels (jk <= fi) are set to fi so they do not influence the scan.
    jk1_scan_1 = array_ns.where(
        z_me_mask,
        fi[:, None],
        array_ns.clip(raw_jk1_1, fi[:, None], nlev - 1),
    )
    cum_max_1 = _cumulative_max(array_ns, jk1_scan_1)
    broken_1 = jk1_scan_1 < cum_max_1
    broken_cum_1 = _cumulative_or(array_ns, broken_1)
    jk1_1 = array_ns.where(raw_too_high_1 | broken_cum_1, nlev - 1, jk1_scan_1)
    zdiff_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e1, jk1_1.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 1, :],
    )
    vertoffset_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 1, :],
    )

    # Phase 2: overwrite where z_me is below z_aux2 and the edge is in the
    # nudging zone.  The query is constant per edge, so no level-to-level carry.
    nudging = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    if bool(nudging.any()):
        z_aux2_vec = z_aux2[:, None]
        phase2_mask = valid_jk & (z_me < z_aux2_vec) & nudging[:, None]

        pos_aux_0 = _batched_searchsorted(array_ns, z_ifc_e0_m, z_aux2_vec)
        jk1_aux_0 = array_ns.clip(nlev - pos_aux_0, fi[:, None], nlev - 1)
        # If the constant query lies above the unmasked top of the ascending
        # column, the reference loop falls through to the unconditional nlev-1.
        jk1_aux_0 = array_ns.where(
            pos_aux_0 >= (nlev + 1 - fi)[:, None],
            nlev - 1,
            jk1_aux_0,
        )

        pos_aux_1 = _batched_searchsorted(array_ns, z_ifc_e1_m, z_aux2_vec)
        jk1_aux_1 = array_ns.clip(nlev - pos_aux_1, fi[:, None], nlev - 1)
        jk1_aux_1 = array_ns.where(
            pos_aux_1 >= (nlev + 1 - fi)[:, None],
            nlev - 1,
            jk1_aux_1,
        )

        zdiff_gradp[:, 0, :] = array_ns.where(
            phase2_mask,
            z_aux2_vec
            - array_ns.take_along_axis(z_mc_e0, jk1_aux_0.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 0, :],
        )
        zdiff_gradp[:, 1, :] = array_ns.where(
            phase2_mask,
            z_aux2_vec
            - array_ns.take_along_axis(z_mc_e1, jk1_aux_1.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 1, :],
        )
        vertoffset_gradp[:, 0, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_0 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 0, :],
        )
        vertoffset_gradp[:, 1, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_1 - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp
