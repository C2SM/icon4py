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
any finite input (including ties and E3 violations). It uses a fused GT4Py
field operator that broadcasts a dense (Edge, K) query against sparse
(Edge, Cand) interface columns and reduces the candidate axis with a first-match
suffix minimum.
"""

from types import ModuleType
from typing import Any

import gt4py.next as gtx
from gt4py.next import max_over, min_over, where

from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_zdiff_gradp import _check_finite, _validation_enabled
from icon4py.model.common.utils import data_allocation as data_alloc


EdgeKField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64]
EdgeCandField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.CandDim], gtx.float64]
EdgeCandIntField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.CandDim], gtx.int32]
EdgeIntField = gtx.Field[gtx.Dims[dims.EdgeDim], gtx.int32]
EdgeKIntField = gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.int32]


@gtx.field_operator
def _zdiff_match_bounds(  # noqa: PLR0917
    query: EdgeKField,
    upper: EdgeCandField,
    lower: EdgeCandField,
    cand_idx: EdgeCandIntField,
    fi: EdgeIntField,
    nlev: gtx.int32,
) -> tuple[EdgeKIntField, EdgeKIntField]:
    """First-match and last-real-match bracket bounds for one (cell, query) pair.

    For each edge ``e`` and query level ``jk`` returns the smallest candidate
    index ``a >= fi[e]`` that satisfies the bracket predicate, plus the largest
    candidate index satisfying the bracket predicate *without* the unconditional
    deepest-level fallback. These two bounds are enough to reproduce main's
    ``jk_start`` carry semantics in the driver.
    """
    deepest = nlev - 1
    in_range = cand_idx >= fi
    bracket = (upper >= query) & (query >= lower)
    unconditional = cand_idx == deepest
    gated = in_range & (unconditional | bracket)
    first_match = min_over(where(gated, cand_idx, nlev), axis=dims.CandDim)
    last_real_match = max_over(where(in_range & bracket, cand_idx, -1), axis=dims.CandDim)
    return first_match, last_real_match


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def _zdiff_match_bounds_program(  # noqa: PLR0917
    query: EdgeKField,
    upper: EdgeCandField,
    lower: EdgeCandField,
    cand_idx: EdgeCandIntField,
    fi: EdgeIntField,
    nlev: gtx.int32,
    out_first: EdgeKIntField,
    out_last: EdgeKIntField,
) -> None:
    """GT4Py program wrapper for ``_zdiff_match_bounds``.

    A program is required for compiled (gtfn) backends; for the embedded
    backend the program delegates to the field operator directly.
    """
    _zdiff_match_bounds(
        query,
        upper,
        lower,
        cand_idx,
        fi,
        nlev,
        out=(out_first, out_last),
    )


def _apply_carry_phase1(  # noqa: PLR0917
    first: data_alloc.NDArray,
    last: data_alloc.NDArray,
    active: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Apply main's jk_start carry for phase-1 cell-1.

    Active levels are contiguous from ``fi + 1`` to ``nlev - 1``.
    """
    result = first.copy()
    nedges = result.shape[0]
    result[array_ns.arange(nedges), fi] = fi
    for k in range(1, nlev):
        start = result[:, k - 1]
        selected = array_ns.where(
            start <= first[:, k],
            first[:, k],
            array_ns.where(start <= last[:, k], start, nlev - 1),
        )
        result[:, k] = array_ns.where(active[:, k], selected, result[:, k])
    return result


def _apply_carry_phase2(  # noqa: PLR0917
    first: data_alloc.NDArray,
    last: data_alloc.NDArray,
    active: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType,
) -> data_alloc.NDArray:
    """Apply main's jk_start carry for phase-2.

    The carry advances only on active levels; inactive levels propagate the
    last active result so that later active levels see the correct start.
    """
    result = first.copy()
    nedges = result.shape[0]
    result[array_ns.arange(nedges), fi] = fi
    for k in range(1, nlev):
        start = result[:, k - 1]
        selected = array_ns.where(
            start <= first[:, k],
            first[:, k],
            array_ns.where(start <= last[:, k], start, nlev - 1),
        )
        result[:, k] = array_ns.where(active[:, k], selected, start)
    return result


def _run_match_bounds(  # noqa: PLR0917
    backend: Any,
    query: data_alloc.NDArray,
    upper: data_alloc.NDArray,
    lower: data_alloc.NDArray,
    cand_idx: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
    out_first: gtx.Field,
    out_last: gtx.Field,
    connectivity: gtx.Connectivity,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """Wrap inputs as GT4Py fields and invoke the match-bounds program."""
    query_f = gtx.as_field((dims.EdgeDim, dims.KDim), query, allocator=backend)  # type: ignore[arg-type]
    upper_f = gtx.as_field((dims.EdgeDim, dims.CandDim), upper, allocator=backend)  # type: ignore[arg-type]
    lower_f = gtx.as_field((dims.EdgeDim, dims.CandDim), lower, allocator=backend)  # type: ignore[arg-type]
    cand_idx_f = gtx.as_field((dims.EdgeDim, dims.CandDim), cand_idx, allocator=backend)  # type: ignore[arg-type]
    fi_f = gtx.as_field((dims.EdgeDim,), fi, allocator=backend)  # type: ignore[arg-type]

    if backend is None:
        _zdiff_match_bounds_program(
            query_f,
            upper_f,
            lower_f,
            cand_idx_f,
            fi_f,
            gtx.int32(nlev),
            out_first,
            out_last,
            offset_provider={"Cand": connectivity},  # type: ignore[dict-item]
        )
    else:
        _zdiff_match_bounds_program.with_backend(backend)(
            query_f,
            upper_f,
            lower_f,
            cand_idx_f,
            fi_f,
            gtx.int32(nlev),
            out_first,
            out_last,
            offset_provider={"Cand": connectivity},  # type: ignore[dict-item]
        )
    return out_first.ndarray.copy(), out_last.ndarray.copy()  # type: ignore[attr-defined]


def compute_zdiff_gradp_gt4py(
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

    # Reusable output fields and identity connectivity for the candidate axis.
    out_first = gtx.as_field(
        (dims.EdgeDim, dims.KDim),
        array_ns.zeros((nedges, nlev), dtype=array_ns.int32),
        allocator=backend,
    )
    out_last = gtx.as_field(
        (dims.EdgeDim, dims.KDim),
        array_ns.zeros((nedges, nlev), dtype=array_ns.int32),
        allocator=backend,
    )
    cand_connectivity = gtx.as_connectivity(
        [dims.EdgeDim, dims.CandDim],
        dims.EdgeDim,
        cand_idx,
        allocator=backend,
    )

    # Phase-1 queries: z_me for every (edge, level).
    query1 = z_me.astype(array_ns.float64)
    first_0, _last_0 = _run_match_bounds(
        backend,
        query1,
        upper0,
        lower0,
        cand_idx,
        fi_i32,
        nlev,
        out_first,
        out_last,
        cand_connectivity,
    )
    first_1, last_1 = _run_match_bounds(
        backend,
        query1,
        upper1,
        lower1,
        cand_idx,
        fi_i32,
        nlev,
        out_first,
        out_last,
        cand_connectivity,
    )

    jk_idx = array_ns.arange(nlev, dtype=array_ns.int64)[None, :]
    valid_jk = jk_idx > fi[:, None]

    # Cell-1 phase-1 needs main's jk_start carry semantics.
    jk1_1 = _apply_carry_phase1(first_1, last_1, valid_jk, fi, nlev, array_ns)

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
        - array_ns.take_along_axis(z_mc_e1[hs:], jk1_1[hs:].astype(array_ns.int64), axis=1),
        zdiff_gradp[hs:, 1, :],
    )
    vertoffset_gradp[hs:, 1, :] = array_ns.where(
        valid_jk[hs:, :],
        (jk1_1[hs:] - jk_idx).astype(gtx.int32),
        vertoffset_gradp[hs:, 1, :],
    )

    # Phase-2 queries: z_aux2 is constant per edge, broadcast to (Edge, K).
    if hs1 < nedges:
        z_aux2_v = (
            array_ns.broadcast_to(z_aux2[:, None], (nedges, nlev)).copy().astype(array_ns.float64)
        )
        first_aux_0, last_aux_0 = _run_match_bounds(
            backend,
            z_aux2_v,
            upper0,
            lower0,
            cand_idx,
            fi_i32,
            nlev,
            out_first,
            out_last,
            cand_connectivity,
        )
        first_aux_1, last_aux_1 = _run_match_bounds(
            backend,
            z_aux2_v,
            upper1,
            lower1,
            cand_idx,
            fi_i32,
            nlev,
            out_first,
            out_last,
            cand_connectivity,
        )

        active2 = valid_jk & (z_me < z_aux2[:, None])
        jk1_aux_0 = _apply_carry_phase2(first_aux_0, last_aux_0, active2, fi, nlev, array_ns)
        jk1_aux_1 = _apply_carry_phase2(first_aux_1, last_aux_1, active2, fi, nlev, array_ns)

        phase2_mask = active2[hs1:, :]

        zdiff_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            z_aux2[hs1:, None]
            - array_ns.take_along_axis(
                z_mc_e0[hs1:], jk1_aux_0[hs1:].astype(array_ns.int64), axis=1
            ),
            zdiff_gradp[hs1:, 0, :],
        )
        vertoffset_gradp[hs1:, 0, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_0[hs1:] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 0, :],
        )

        zdiff_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            z_aux2[hs1:, None]
            - array_ns.take_along_axis(
                z_mc_e1[hs1:], jk1_aux_1[hs1:].astype(array_ns.int64), axis=1
            ),
            zdiff_gradp[hs1:, 1, :],
        )
        vertoffset_gradp[hs1:, 1, :] = array_ns.where(
            phase2_mask,
            (jk1_aux_1[hs1:] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[hs1:, 1, :],
        )

    return zdiff_gradp, vertoffset_gradp
