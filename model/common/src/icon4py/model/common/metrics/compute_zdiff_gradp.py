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

With these invariants, a single batched ``searchsorted`` on the ascending
half-level column gives the same first-match index as the reference loop.
The result is non-decreasing in ``jk``, so ``clip(searchsorted_result, fi,
nlev-1)`` reproduces the reference ``jk_start`` lower-bound update without a sequential
loop.

Premise: the queries (``z_me`` and ``z_aux2``) are strictly between consecutive
``z_ifc`` boundaries, i.e. no query exactly equals a boundary.  This holds for
all production grids: ``z_ifc`` layers are separated by millions to billions of
ULPs (enforced by ``_check_and_correct_layer_thickness``), so each midpoint
``z_mc[c, j] = 0.5 * (z_ifc[c, j] + z_ifc[c, j+1])`` is a distinct float
strictly between its boundaries, and ``z_me`` (a convex combination of two such
midpoints) is strictly between them as well.  An exact tie would require
disabling the layer-thickness correction or corrupting the inputs after grid
construction; the test suite marks such synthetic inputs as ``xfail``.
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
    offset computation; the offset must exceed the largest cross-row span so
    that no query from one row lands in another's value range.
    """
    a = a.astype(array_ns.float64)
    v = v.astype(array_ns.float64)
    m, n = a.shape
    max_num = (
        array_ns.max(
            array_ns.stack(
                [a.max() - a.min(), v.max() - v.min(), a.max() - v.min(), v.max() - a.min()]
            )
        ).astype(array_ns.float64)
        + 1.0
    )
    r = max_num * array_ns.arange(m, dtype=array_ns.float64)[:, None]
    p = array_ns.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * array_ns.arange(m, dtype=p.dtype)[:, None]


def _first_match(
    array_ns: ModuleType,
    z_ifc_asc_masked: data_alloc.NDArray,
    queries: data_alloc.NDArray,
    fi: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """First bracket index ``jk`` with ``z_ifc[e, jk] >= q >= z_ifc[e, jk+1]``.

    ``z_ifc_asc_masked`` is the ascending half-level column with levels above
    ``fi`` filled to ``fill_high`` so they sort after any real query.  ``fi`` has
    shape ``(m,)`` and ``queries`` has shape ``(m, q)``.  Assumes queries are
    strictly between boundaries (no exact ties); see the module docstring.
    Returns shape ``(m, q)``.
    """
    pos = _batched_searchsorted(array_ns, z_ifc_asc_masked, queries)
    jk1 = nlev - pos
    # Query above the unmasked top: the Fortran loop falls through to
    # nlev-1 (searchsorted returns the count of elements <= query).
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
    ``searchsorted`` returns the reference first-match index.
    The result is non-decreasing in ``jk``, so clipping it to ``[fi, nlev-1]`` reproduces the
    reference ``jk_start`` lower-bound update without an explicit loop.
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
    jk1_0 = _first_match(array_ns, z_ifc_e0_m, z_me_m, fi_sliced, nlev)
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
    # jk, so the same clip as cell 0 reproduces the reference jk_start lower-bound update.
    jk1_1 = _first_match(array_ns, z_ifc_e1_m, z_me_m, fi_sliced, nlev)
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

        jk1_aux_0 = _first_match(array_ns, z_ifc_e0_m1, z_aux2_v, fi_sliced1, nlev)
        jk1_aux_1 = _first_match(array_ns, z_ifc_e1_m1, z_aux2_v, fi_sliced1, nlev)

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
