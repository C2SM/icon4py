# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""GT4Py first-match implementation of ``compute_zdiff_gradp``.

This variant is mechanically equivalent to ``compute_zdiff_gradp_exact_v2``:
zero assumed premises, exact match to main's element-by-element semantics for
any finite input.  The GT4Py field operator computes the first-match
suffix-minimum ``M(start, k) = min { a >= start : a == nlev-1 or bracket(a) }``
for a dense (Edge, K) query against sparse (Edge, Cand) interface columns.

Where main carries ``jk_start`` between query levels (phase-1 cell 1 and both
phase-2 cells), the driver re-invokes the operator once per level with a
single-column query and ``start`` equal to the current carried lower bound.
This is exact for non-monotonic z_ifc columns (disjoint brackets) because the
carry is never reconstructed from coarser per-level information: each level
queries ``M(start, k)`` directly.  The cost is O(nlev) small operator launches;
the variant is intended as a correctness reference, not a GPU performance path.
"""

from typing import Any

import gt4py.next as gtx
from gt4py.next import min_over, where

from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_zdiff_gradp import _check_finite, _validation_enabled
from icon4py.model.common.utils import data_allocation as data_alloc


EdgeKField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64]
EdgeCandField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.CandDim], gtx.float64]
EdgeCandIntField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.CandDim], gtx.int32]
EdgeIntField = gtx.Field[gtx.Dims[dims.EdgeDim], gtx.int32]
EdgeKIntField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.int32]


@gtx.field_operator
def _zdiff_first_match(  # noqa: PLR0917
    query: EdgeKField,
    upper: EdgeCandField,
    lower: EdgeCandField,
    cand_idx: EdgeCandIntField,
    start: EdgeIntField,
    nlev: gtx.int32,
) -> EdgeKIntField:
    """First-match bracket bound for one (cell, query) pair.

    For each edge ``e`` and query level ``k`` returns the smallest candidate
    index ``a >= start[e]`` that satisfies the bracket predicate, with the
    deepest level ``a == nlev - 1`` as an unconditional member.  This is
    exactly main's ``jk_start`` lower-bound scan for one query level.
    """
    deepest = nlev - 1
    in_range = cand_idx >= start
    bracket = (upper >= query) & (query >= lower)
    unconditional = cand_idx == deepest
    gated = in_range & (unconditional | bracket)
    return min_over(where(gated, cand_idx, nlev), axis=dims.CandDim)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def _zdiff_first_match_program(  # noqa: PLR0917
    query: EdgeKField,
    upper: EdgeCandField,
    lower: EdgeCandField,
    cand_idx: EdgeCandIntField,
    start: EdgeIntField,
    nlev: gtx.int32,
    out_first: EdgeKIntField,
) -> None:
    """GT4Py program wrapper for ``_zdiff_first_match``.

    A program is required for compiled (gtfn) backends; for the embedded
    backend the program delegates to the field operator directly.
    """
    _zdiff_first_match(
        query,
        upper,
        lower,
        cand_idx,
        start,
        nlev,
        out=out_first,
    )


def _run_first_match(  # noqa: PLR0917
    backend: Any,
    query: data_alloc.NDArray,
    upper: data_alloc.NDArray,
    lower: data_alloc.NDArray,
    cand_idx: data_alloc.NDArray,
    start: data_alloc.NDArray,
    nlev: int,
    out_first: gtx.Field,
    cand_connectivity: gtx.Connectivity,
) -> data_alloc.NDArray:
    query_f = gtx.as_field((dims.EdgeDim, dims.KDim), query, allocator=backend)  # type: ignore[arg-type]
    upper_f = gtx.as_field((dims.EdgeDim, dims.CandDim), upper, allocator=backend)  # type: ignore[arg-type]
    lower_f = gtx.as_field((dims.EdgeDim, dims.CandDim), lower, allocator=backend)  # type: ignore[arg-type]
    cand_idx_f = gtx.as_field((dims.EdgeDim, dims.CandDim), cand_idx, allocator=backend)  # type: ignore[arg-type]
    start_f = gtx.as_field((dims.EdgeDim,), start, allocator=backend)  # type: ignore[arg-type]

    if backend is None:
        _zdiff_first_match_program(
            query_f,
            upper_f,
            lower_f,
            cand_idx_f,
            start_f,
            gtx.int32(nlev),
            out_first,
            offset_provider={"Cand": cand_connectivity},  # type: ignore[dict-item]
        )
    else:
        _zdiff_first_match_program.with_backend(backend)(
            query_f,
            upper_f,
            lower_f,
            cand_idx_f,
            start_f,
            gtx.int32(nlev),
            out_first,
            offset_provider={"Cand": cand_connectivity},  # type: ignore[dict-item]
        )
    return out_first.ndarray.copy()  # type: ignore[attr-defined]


def compute_zdiff_gradp_gt4py(  # noqa: PLR0915
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
    backend: Any = None,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Exact-semantics ``compute_zdiff_gradp`` via a GT4Py first-match operator.

    Args:
        e2c: edge-to-cell connectivity, shape ``(nedges, 2)``.
        z_me: edge geometric height on full levels, shape ``(nedges, nlev)``.
        z_mc: cell geometric height on full levels, shape ``(ncells, nlev)``.
        z_ifc: cell geometric height on half levels, shape ``(ncells, nlev + 1)``.
        flat_idx: flat index per edge, shape ``(nedges,)``.
        topography: surface height per cell, shape ``(ncells,)``.
        nlev: number of full levels.
        horizontal_start: start index for phase-1 base write.
        horizontal_start_1: start index for phase-2 nudging update.
        backend: GT4Py backend for the kernel (``None`` selects embedded).

    Returns:
        ``(zdiff_gradp, vertoffset_gradp)`` with the same shapes and semantics
        as the other ``compute_zdiff_gradp_*`` variants.
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

    # z_ifc in main's k-order (decreasing with k) for both cells.
    z_ifc_e0 = z_ifc[e2c_0, :].astype(array_ns.float64)
    z_ifc_e1 = z_ifc[e2c_1, :].astype(array_ns.float64)

    if _validation_enabled():
        if not bool(_check_finite(array_ns, z_ifc_e0, z_ifc_e1, z_me, z_aux2)):
            raise ValueError("Searched arrays contain non-finite values.")

    # Sparse candidate columns: candidate a corresponds to level index a.
    upper0 = z_ifc_e0[:, :-1]
    lower0 = z_ifc_e0[:, 1:]
    upper1 = z_ifc_e1[:, :-1]
    lower1 = z_ifc_e1[:, 1:]

    cand_idx = array_ns.broadcast_to(
        array_ns.arange(nlev, dtype=array_ns.int32), (nedges, nlev)
    ).copy()
    fi_i32 = flat_idx.astype(array_ns.int32)

    # Reusable output fields for the first-match operator.
    out_first = gtx.as_field(
        (dims.EdgeDim, dims.KDim),
        array_ns.zeros((nedges, nlev), dtype=array_ns.int32),
        allocator=backend,
    )
    out_first_1col = gtx.as_field(
        (dims.EdgeDim, dims.KDim),
        array_ns.zeros((nedges, 1), dtype=array_ns.int32),
        allocator=backend,
    )
    # GT4Py requires an offset-provider entry for the local ``CandDim`` even
    # though the operator only reduces over it (no neighbor access).  The
    # connectivity values are level indices; they are not read as a neighbor
    # table because the reduction is axis-only, but the declaration must be
    # present for the local dimension to resolve.
    cand_connectivity = gtx.as_connectivity(
        [dims.EdgeDim, dims.CandDim],
        dims.EdgeDim,
        cand_idx,
        allocator=backend,
    )
    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    valid_jk = jk_idx > fi[:, None]

    # Phase-1 queries: z_me for every (edge, level).
    query1 = z_me.astype(array_ns.float64)
    first_0 = _run_first_match(
        backend,
        query1,
        upper0,
        lower0,
        cand_idx,
        fi_i32,
        nlev,
        out_first,
        cand_connectivity,
    )

    # Cell-1 phase-1 needs main's jk_start carry semantics.  Re-invoke the
    # first-match operator once per level with start equal to the current
    # carried lower bound; this is exact even for non-monotonic z_ifc columns
    # because each level queries M(start, k) directly.
    current_start = fi_i32.copy()
    first_1 = array_ns.empty((nedges, nlev), dtype=array_ns.int32)
    for jk in range(nlev):
        query_col = query1[:, jk : jk + 1]
        out_col = _run_first_match(
            backend,
            query_col,
            upper1,
            lower1,
            cand_idx,
            current_start,
            nlev,
            out_first_1col,
            cand_connectivity,
        )
        first_1[:, jk] = out_col[:, 0]
        current_start = array_ns.where(valid_jk[:, jk], out_col[:, 0], current_start).astype(
            array_ns.int32
        )

    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    # Phase-1 assembly.
    zdiff_gradp[hs:, 0, :] = array_ns.where(
        valid_jk[hs:, :],
        z_me[hs:]
        - array_ns.take_along_axis(z_mc_e0[hs:], first_0[hs:].astype(array_ns.int64), axis=1),
        zdiff_gradp[hs:, 0, :],
    )
    vertoffset_gradp[hs:, 0, :] = array_ns.where(
        valid_jk[hs:, :],
        (first_0[hs:] - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 0, :],
    )

    zdiff_gradp[hs:, 1, :] = array_ns.where(
        valid_jk[hs:, :],
        z_me[hs:]
        - array_ns.take_along_axis(z_mc_e1[hs:], first_1[hs:].astype(array_ns.int64), axis=1),
        zdiff_gradp[hs:, 1, :],
    )
    vertoffset_gradp[hs:, 1, :] = array_ns.where(
        valid_jk[hs:, :],
        (first_1[hs:] - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 1, :],
    )

    # Phase-2 queries: z_aux2 is constant per edge, but main still carries
    # jk_start between active levels, so both cell columns are driven by the
    # same per-level re-invocation scheme as phase-1 cell 1.
    if hs1 < nedges:
        z_aux2_v_full = (
            array_ns.broadcast_to(z_aux2[:, None], (nedges, nlev)).copy().astype(array_ns.float64)
        )
        active2 = valid_jk & (z_me < z_aux2[:, None])

        current_start = fi_i32.copy()
        first_aux_0 = array_ns.empty((nedges, nlev), dtype=array_ns.int32)
        for jk in range(nlev):
            query_col = z_aux2_v_full[:, jk : jk + 1]
            out_col = _run_first_match(
                backend,
                query_col,
                upper0,
                lower0,
                cand_idx,
                current_start,
                nlev,
                out_first_1col,
                cand_connectivity,
            )
            first_aux_0[:, jk] = out_col[:, 0]
            current_start = array_ns.where(active2[:, jk], out_col[:, 0], current_start).astype(
                array_ns.int32
            )

        current_start = fi_i32.copy()
        first_aux_1 = array_ns.empty((nedges, nlev), dtype=array_ns.int32)
        for jk in range(nlev):
            query_col = z_aux2_v_full[:, jk : jk + 1]
            out_col = _run_first_match(
                backend,
                query_col,
                upper1,
                lower1,
                cand_idx,
                current_start,
                nlev,
                out_first_1col,
                cand_connectivity,
            )
            first_aux_1[:, jk] = out_col[:, 0]
            current_start = array_ns.where(active2[:, jk], out_col[:, 0], current_start).astype(
                array_ns.int32
            )

        phase2_mask = active2[hs1:, :]

        zdiff_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            z_aux2[hs1:, None]
            - array_ns.take_along_axis(
                z_mc_e0[hs1:], first_aux_0[hs1:].astype(array_ns.int64), axis=1
            ),
            zdiff_gradp[hs1:, 0, :],
        )
        vertoffset_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            (first_aux_0[hs1:] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 0, :],
        )

        zdiff_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            z_aux2[hs1:, None]
            - array_ns.take_along_axis(
                z_mc_e1[hs1:], first_aux_1[hs1:].astype(array_ns.int64), axis=1
            ),
            zdiff_gradp[hs1:, 1, :],
        )
        vertoffset_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            (first_aux_1[hs1:] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp
