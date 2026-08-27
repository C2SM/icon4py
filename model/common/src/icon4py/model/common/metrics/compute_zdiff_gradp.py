# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


def _first_match_batched(
    array_ns: Any,
    z_ifc_col: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """All-at-once first-match for independent queries (no carry).

    For each edge ``e`` and query ``q`` returns the smallest candidate
    ``a >= fi[e]`` such that ``a == nlev - 1`` or
    ``z_ifc_col[e, a] >= queries[e, q] >= z_ifc_col[e, a + 1]``.
    """
    cand = array_ns.arange(nlev, dtype=array_ns.int32)
    upper = z_ifc_col[:, None, :-1]
    lower = z_ifc_col[:, None, 1:]
    qv = queries[:, :, None]
    bracket = (upper >= qv) & (qv >= lower)
    unconditional = cand[None, None, :] == (nlev - 1)
    in_range = cand[None, None, :] >= fi[:, None, None]
    gated = in_range & (bracket | unconditional)
    return array_ns.argmax(gated, axis=-1).astype(array_ns.int32)


def _carry_first_match(
    array_ns: Any,
    z_ifc_col: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    active: data_alloc.NDArray,
    *,
    nlev: int,
) -> data_alloc.NDArray:
    """Sequential carry: replicate main's ``jk_start`` update literally.

    ``queries`` is either ``(nedges,)`` for a constant per-edge query (phase 2)
    or ``(nedges, nlev)`` for a level-varying query (phase 1 cell 1).  At each
    level ``jk`` the search starts at the current carried lower bound ``t``;
    ``t`` is only advanced when ``active[:, jk]`` is true.
    """
    nedges = z_ifc_col.shape[0]
    cand = array_ns.arange(nlev, dtype=array_ns.int32)
    upper = z_ifc_col[:, :-1]
    lower = z_ifc_col[:, 1:]
    unconditional = cand == (nlev - 1)
    t = fi.astype(array_ns.int32).copy()
    jk1 = array_ns.empty((nedges, nlev), dtype=array_ns.int32)
    constant_query = queries.ndim == 1
    for k in range(nlev):
        q = queries if constant_query else queries[:, k]
        qv = q[:, None]
        bracket = (upper >= qv) & (qv >= lower)
        in_range = cand[None, :] >= t[:, None]
        gated = in_range & (bracket | unconditional[None, :])
        first = array_ns.argmax(gated, axis=1).astype(array_ns.int32)
        t = array_ns.where(active[:, k], first, t)
        jk1[:, k] = t
    return jk1


def _successor_table(
    array_ns: Any,
    z_ifc_col: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """Build a suffix-minimum successor table for exact bracket matching.

    Returns ``S`` with shape ``(nedges, nq, nlev)`` where ``S[e, q, t]`` is
    the first candidate ``a >= t`` such that the bracket predicate holds for
    query ``q`` or ``a == nlev - 1``.  The table is built with a Hillis-Steele
    doubling suffix-min scan, so it is fully parallel along the candidate
    axis and needs only ``log2(nlev)`` reduction steps.
    """
    if queries.ndim == 1:
        queries = queries[:, None]

    cand = array_ns.arange(nlev, dtype=array_ns.int32)
    upper = z_ifc_col[:, None, :-1]
    lower = z_ifc_col[:, None, 1:]
    qv = queries[:, :, None]
    bracket = (upper >= qv) & (qv >= lower)
    unconditional = cand[None, None, :] == (nlev - 1)
    idx = array_ns.where(bracket | unconditional, cand[None, None, :], nlev)

    if nlev <= 127:
        idx = idx.astype(array_ns.int8)
    elif nlev <= 32767:
        idx = idx.astype(array_ns.int16)
    else:
        idx = idx.astype(array_ns.int32)

    S = idx.copy()
    step = 1
    while step < nlev:
        S[..., :-step] = array_ns.minimum(S[..., :-step], S[..., step:])
        step *= 2
    return S


def _carry_gather_scan(
    array_ns: Any,
    S: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    active: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Carry scan using a precomputed successor table.

    At each level ``jk`` the next carried lower bound is gathered from the
    table and the carry only advances where ``active`` is true.  This replaces
    the per-step ``argmax`` in the sequential carry with a constant-time gather.
    """
    _, nq, nlev = S.shape
    single_query = nq == 1
    t = fi.astype(array_ns.int64).copy()
    jk1 = array_ns.empty((S.shape[0], nlev), dtype=array_ns.int64)
    for jk in range(nlev):
        src = S[:, 0:1, :] if single_query else S[:, jk : jk + 1, :]
        gathered = array_ns.take_along_axis(src, t[:, None, None], axis=2)[:, 0, 0]
        t = array_ns.where(active[:, jk], gathered, t)
        jk1[:, jk] = t
    return jk1


def _zdiff_gradp_chunk_size(nedges: int, nlev: int) -> int:
    """Memory-capped edge chunk size for successor-table based variants.

    Caps peak successor-table memory per chunk to roughly 256 MiB.  The item
    size is chosen to fit the largest level index needed while avoiding an
    otherwise unnecessary nlev ceiling.
    """
    itemsize = 1 if nlev <= 127 else (2 if nlev <= 32767 else 4)
    mem_per_edge = nlev * 2 * (nlev + 1) * itemsize
    return max(1, min(nedges, (256 * 1024 * 1024) // mem_per_edge))


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

    This is a premise-free, dispatch-free implementation: it evaluates main's
    bracket predicate directly and carries ``jk_start`` between levels exactly
    as the reference loop does.  It is correct for all finite inputs, including
    non-monotone ``z_ifc``, interior ties, and non-increasing ``z_me``.
    """
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]
    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=array_ns.int32)

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_e0 = z_ifc[e2c_0]
    z_ifc_e1 = z_ifc[e2c_1]
    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]

    # Phase 1, cell 0: independent first-match scan at every level.
    jk1_0 = _first_match_batched(array_ns, z_ifc_e0, z_me, fi, nlev)
    zdiff_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e0, jk1_0.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 0, :],
    )
    vertoffset_gradp[:, 0, :] = array_ns.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(array_ns.int32),
        vertoffset_gradp[:, 0, :],
    )

    # Phase 1, cell 1: carry the lower bound between levels.
    jk1_1 = _carry_first_match(array_ns, z_ifc_e1, z_me, fi, valid_jk, nlev=nlev)
    zdiff_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        z_me - array_ns.take_along_axis(z_mc_e1, jk1_1.astype(array_ns.int64), axis=1),
        zdiff_gradp[:, 1, :],
    )
    vertoffset_gradp[:, 1, :] = array_ns.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(array_ns.int32),
        vertoffset_gradp[:, 1, :],
    )

    # Phase 2: overwrite where z_me is below z_aux2 and the edge is in the
    # nudging zone.  The carry starts fresh from fi for each cell column.
    if hs1 < nedges:
        phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

        jk1_aux_0 = _carry_first_match(array_ns, z_ifc_e0, z_aux2, fi, phase2_active, nlev=nlev)
        zdiff_gradp[:, 0, :] = array_ns.where(
            phase2_active,
            z_aux2[:, None]
            - array_ns.take_along_axis(z_mc_e0, jk1_aux_0.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 0, :],
        )
        vertoffset_gradp[:, 0, :] = array_ns.where(
            phase2_active,
            (jk1_aux_0 - jk_idx).astype(array_ns.int32),
            vertoffset_gradp[:, 0, :],
        )

        jk1_aux_1 = _carry_first_match(array_ns, z_ifc_e1, z_aux2, fi, phase2_active, nlev=nlev)
        zdiff_gradp[:, 1, :] = array_ns.where(
            phase2_active,
            z_aux2[:, None]
            - array_ns.take_along_axis(z_mc_e1, jk1_aux_1.astype(array_ns.int64), axis=1),
            zdiff_gradp[:, 1, :],
        )
        vertoffset_gradp[:, 1, :] = array_ns.where(
            phase2_active,
            (jk1_aux_1 - jk_idx).astype(array_ns.int32),
            vertoffset_gradp[:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp


def compute_zdiff_gradp_fast(
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
    """Premise-free, faster variant of ``compute_zdiff_gradp``.

    Cell 0 keeps the independent batched first-match from
    ``_first_match_batched``.  Cell 1 and the phase-2 columns replace the
    sequential per-level ``argmax`` with a precomputed successor table and a
    Hillis-Steele doubling suffix-min scan.  The carry then becomes a gather
    scan that touches only ``O(E)`` data per level.  Edges are processed in
    memory-capped chunks so the ``(E, nlev, nlev)`` cell-1 table never blows
    up host/device memory at full scale.

    The function is correct for all finite inputs, including non-monotone
    ``z_ifc``, interior ties, and non-increasing ``z_me``.
    """
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]
    hs = int(horizontal_start)
    hs1 = int(horizontal_start_1)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0

    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[hs:, :, :] = array_ns.expand_dims(z_me, axis=1)[hs:, :, :] - z_mc[e2c][hs:, :, :]
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=array_ns.int32)

    fi = flat_idx.astype(array_ns.int64)
    e2c_0 = e2c[:, 0].astype(array_ns.int64)
    e2c_1 = e2c[:, 1].astype(array_ns.int64)

    z_ifc_e0 = z_ifc[e2c_0]
    z_ifc_e1 = z_ifc[e2c_1]
    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    edge_hs_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs
    edge_hs1_mask = array_ns.arange(nedges, dtype=array_ns.int64) >= hs1
    valid_jk = (jk_idx > fi[:, None]) & edge_hs_mask[:, None]
    if hs1 < nedges:
        phase2_active = valid_jk & (z_me < z_aux2[:, None]) & edge_hs1_mask[:, None]

    chunk_size = _zdiff_gradp_chunk_size(nedges, nlev)

    for start in range(0, nedges, chunk_size):
        end = min(start + chunk_size, nedges)
        chunk = slice(start, end)

        # Phase 1, cell 0: independent first-match scan at every level.
        jk1_0 = _first_match_batched(array_ns, z_ifc_e0[chunk], z_me[chunk], fi[chunk], nlev)
        zdiff_gradp[chunk, 0, :] = array_ns.where(
            valid_jk[chunk],
            z_me[chunk]
            - array_ns.take_along_axis(z_mc_e0[chunk], jk1_0.astype(array_ns.int64), axis=1),
            zdiff_gradp[chunk, 0, :],
        )
        vertoffset_gradp[chunk, 0, :] = array_ns.where(
            valid_jk[chunk],
            (jk1_0 - jk_idx).astype(array_ns.int32),
            vertoffset_gradp[chunk, 0, :],
        )

        # Phase 1, cell 1: carry the lower bound between levels via the table.
        S1 = _successor_table(array_ns, z_ifc_e1[chunk], z_me[chunk], nlev)
        jk1_1 = _carry_gather_scan(array_ns, S1, fi[chunk], valid_jk[chunk])
        zdiff_gradp[chunk, 1, :] = array_ns.where(
            valid_jk[chunk],
            z_me[chunk]
            - array_ns.take_along_axis(z_mc_e1[chunk], jk1_1.astype(array_ns.int64), axis=1),
            zdiff_gradp[chunk, 1, :],
        )
        vertoffset_gradp[chunk, 1, :] = array_ns.where(
            valid_jk[chunk],
            (jk1_1 - jk_idx).astype(array_ns.int32),
            vertoffset_gradp[chunk, 1, :],
        )
        del S1

        # Phase 2: overwrite where z_me is below z_aux2 and the edge is in the
        # nudging zone.  The carry starts fresh from fi for each cell column.
        if hs1 < nedges:
            phase2_active_c = phase2_active[chunk]
            z_aux2_c = z_aux2[chunk]

            S2_0 = _successor_table(array_ns, z_ifc_e0[chunk], z_aux2_c, nlev)
            jk1_aux_0 = _carry_gather_scan(array_ns, S2_0, fi[chunk], phase2_active_c)
            zdiff_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                z_aux2_c[:, None]
                - array_ns.take_along_axis(
                    z_mc_e0[chunk], jk1_aux_0.astype(array_ns.int64), axis=1
                ),
                zdiff_gradp[chunk, 0, :],
            )
            vertoffset_gradp[chunk, 0, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_0 - jk_idx).astype(array_ns.int32),
                vertoffset_gradp[chunk, 0, :],
            )
            del S2_0

            S2_1 = _successor_table(array_ns, z_ifc_e1[chunk], z_aux2_c, nlev)
            jk1_aux_1 = _carry_gather_scan(array_ns, S2_1, fi[chunk], phase2_active_c)
            zdiff_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                z_aux2_c[:, None]
                - array_ns.take_along_axis(
                    z_mc_e1[chunk], jk1_aux_1.astype(array_ns.int64), axis=1
                ),
                zdiff_gradp[chunk, 1, :],
            )
            vertoffset_gradp[chunk, 1, :] = array_ns.where(
                phase2_active_c,
                (jk1_aux_1 - jk_idx).astype(array_ns.int32),
                vertoffset_gradp[chunk, 1, :],
            )
            del S2_1

    return zdiff_gradp, vertoffset_gradp
