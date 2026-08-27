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
non-increasing level sequences, so it is non-increasing in ``k`` per edge.
With these invariants, a single batched ``searchsorted`` (``side="left"``,
GPU-accelerated in cupy) on the ascending half-level column gives the reference
first-match index up to an exact-tie correction: when the query equals
``z_ifc[e, jk+1]`` (the lower bracket boundary) ``side="left"`` returns ``jk+1``
but the Fortran loop takes ``jk``.  The correction is a uniform vectorized
``where`` (no branch, no dispatch) and the result is non-decreasing in ``jk``,
so ``clip(corrected_result, fi, nlev-1)`` reproduces the reference ``jk_start``
carry without a sequential loop.
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
    # The offset per row must exceed the largest cross-row span so that no
    # query from row i lands in row i-1's value range.  This matters when
    # fill values place queries outside the row's own [a.min, a.max].
    max_num = max(a.max() - a.min(), v.max() - v.min(), a.max() - v.min(), v.max() - a.min()) + 1.0
    r = max_num * array_ns.arange(m, dtype=array_ns.float64)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def _first_match(  # noqa: PLR0917
    array_ns: ModuleType,
    z_ifc_asc_masked: data_alloc.NDArray,
    z_ifc_col: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """First bracket index ``jk`` with ``z_ifc[e, jk] >= q >= z_ifc[e, jk+1]``.

    ``z_ifc_asc_masked`` is the ascending half-level column with levels above
    ``fi`` filled to ``fill_high`` so they sort after any real query.  ``z_ifc_col``
    is the original descending column (used for the tie check).  ``fi`` has shape
    ``(m,)`` and ``queries`` has shape ``(m, q)``.  Returns shape ``(m, q)``.
    """
    pos = _batched_searchsorted(array_ns, z_ifc_asc_masked, queries)
    jk1 = nlev - pos
    # Exact-tie correction: side="left" returns jk+1 when the query equals
    # z_ifc[e, jk+1] (the lower bracket boundary); the Fortran first-match
    # takes jk.  Detect ties against the original descending column.
    z_ifc_at_jk1 = array_ns.take_along_axis(z_ifc_col, jk1.astype(array_ns.int64), axis=1)
    jk1 = array_ns.where(queries == z_ifc_at_jk1, jk1 - 1, jk1)
    # Query above the unmasked top: the Fortran loop falls through to nlev-1.
    jk1 = array_ns.where(pos >= (nlev + 1) - fi[:, None], nlev - 1, jk1)
    return array_ns.clip(jk1, fi[:, None], nlev - 1)


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
    """Compute ``zdiff_gradp`` and ``vertoffset_gradp`` matching the reference.

    Relies on the two grid invariants proved in the module docstring: ``z_ifc``
    is strictly decreasing in ``k`` (vertical.py:558 and vertical.py:625), and
    ``z_me`` is non-increasing in ``k`` per edge.  Under those invariants,
    ``searchsorted`` returns the reference first-match index (up to the uniform
    tie correction in :func:`_first_match`) and the result is non-decreasing in
    ``jk``, so clipping it to ``[fi, nlev-1]`` reproduces the reference
    ``jk_start`` carry without an explicit loop.
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
    z_ifc_col_0 = z_ifc[e2c_0]
    z_ifc_col_1 = z_ifc[e2c_1]

    fill_high = max(z_ifc_e0.max(), z_ifc_e1.max(), z_me.max(), z_aux2.max()) + 1.0
    fill_low = min(z_ifc_e0.min(), z_ifc_e1.min(), z_me.min(), z_aux2.min()) - 1.0

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
    jk1_0 = _first_match(array_ns, z_ifc_e0_m, z_ifc_col_0[hs:], z_me_m, fi_sliced, nlev)
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

    # Phase 1, cell 1: under E3 the searchsorted result is non-decreasing in
    # jk, so the same clip as cell 0 reproduces the reference jk_start carry.
    jk1_1 = _first_match(array_ns, z_ifc_e1_m, z_ifc_col_1[hs:], z_me_m, fi_sliced, nlev)
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

        jk1_aux_0 = _first_match(
            array_ns, z_ifc_e0_m1, z_ifc_col_0[hs1:], z_aux2_v, fi_sliced1, nlev
        )
        jk1_aux_1 = _first_match(
            array_ns, z_ifc_e1_m1, z_ifc_col_1[hs1:], z_aux2_v, fi_sliced1, nlev
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
