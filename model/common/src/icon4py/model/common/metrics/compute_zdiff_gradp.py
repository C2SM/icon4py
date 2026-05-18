# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as _np

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


try:
    import cupy as _cp
except ImportError:
    _cp = None


def _get_xp(array):
    if _cp is not None and isinstance(array, _cp.ndarray):
        return _cp
    return _np


def _batched_searchsorted(a, v, xp):
    """2D searchsorted via row-offset trick (Divakar).

    Finds insertion positions of v[i] in a[i] for all rows i,
    using a single 1D searchsorted call.

    Requires: each row of ``a`` is sorted ascending.
    Works with both numpy and cupy.
    """
    m, n = a.shape
    max_num = _np.maximum(float(a.max() - a.min()), float(v.max() - v.min())) + 1
    r = max_num * xp.arange(m, dtype=a.dtype)[:, None]
    p = xp.searchsorted((a + r).ravel(), (v + r).ravel()).reshape(v.shape)
    return p - n * xp.arange(m, dtype=p.dtype)[:, None]


def compute_zdiff_gradp(
    e2c,
    z_mc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(z_mc)
    xp = _get_xp(z_mc)

    nedges = e2c.shape[0]
    z_me = array_ns.sum(z_mc[e2c] * array_ns.expand_dims(c_lin_e, axis=-1), axis=1)

    exchange.exchange(dims.EdgeDim, z_me, stream=decomposition.BLOCK)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0
    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    zdiff_gradp[horizontal_start:, :, :] = (
        array_ns.expand_dims(z_me, axis=1)[horizontal_start:, :, :]
        - z_mc[e2c][horizontal_start:, :, :]
    )
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    fi = flat_idx.astype(xp.int64)
    e2c_0 = e2c[:, 0].astype(xp.int64)
    e2c_1 = e2c[:, 1].astype(xp.int64)

    z_ifc_asc = z_ifc[:, ::-1].copy()
    z_ifc_e0 = z_ifc_asc[e2c_0]
    z_ifc_e1 = z_ifc_asc[e2c_1]

    fill_high = float(xp.max(z_ifc_e0)) + 1.0
    fill_low = float(xp.min(z_ifc_e0)) - 1.0

    z_ifc_mask = xp.arange(nlev + 1, dtype=xp.int64)[None, :] >= (nlev + 1 - fi[:, None])
    z_me_mask = xp.arange(nlev, dtype=xp.int64)[None, :] <= fi[:, None]

    z_ifc_e0_m = xp.where(z_ifc_mask, fill_high, z_ifc_e0)
    z_ifc_e1_m = xp.where(z_ifc_mask, fill_high, z_ifc_e1)
    z_me_m = xp.where(z_me_mask, fill_low, z_me)

    pos_0 = _batched_searchsorted(z_ifc_e0_m, z_me_m, xp)
    jk1_0 = xp.clip(nlev - pos_0, fi[:, None], nlev - 1)

    pos_1 = _batched_searchsorted(z_ifc_e1_m, z_me_m, xp)
    jk1_1 = xp.clip(nlev - pos_1, fi[:, None], nlev - 1)

    jk_idx = xp.arange(nlev, dtype=xp.int64)[None, :]
    boundary = xp.arange(nedges, dtype=xp.int64) >= horizontal_start
    valid_jk = (jk_idx > fi[:, None]) & boundary[:, None]
    z_mc_e0 = z_mc[e2c_0]
    z_mc_e1 = z_mc[e2c_1]

    zdiff_gradp[:, 0, :] = xp.where(
        valid_jk,
        z_me - xp.take_along_axis(z_mc_e0, jk1_0.astype(xp.int64), axis=1),
        zdiff_gradp[:, 0, :],
    )
    zdiff_gradp[:, 1, :] = xp.where(
        valid_jk,
        z_me - xp.take_along_axis(z_mc_e1, jk1_1.astype(xp.int64), axis=1),
        zdiff_gradp[:, 1, :],
    )

    vertoffset_gradp[:, 0, :] = xp.where(
        valid_jk,
        (jk1_0 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 0, :],
    )
    vertoffset_gradp[:, 1, :] = xp.where(
        valid_jk,
        (jk1_1 - jk_idx).astype(gtx.int32),
        vertoffset_gradp[:, 1, :],
    )

    nudging = xp.arange(nedges, dtype=xp.int64) >= horizontal_start_1
    if nudging.any():
        z_aux2_vec = z_aux2[:, None]

        pos_aux_0 = _batched_searchsorted(z_ifc_e0_m, z_aux2_vec, xp)
        jk1_aux_0 = xp.clip(nlev - pos_aux_0[:, 0], fi, nlev - 1)

        pos_aux_1 = _batched_searchsorted(z_ifc_e1_m, z_aux2_vec, xp)
        jk1_aux_1 = xp.clip(nlev - pos_aux_1[:, 0], fi, nlev - 1)

        phase2_mask = valid_jk & (z_me < z_aux2[:, None]) & nudging[:, None]

        zdiff_gradp[:, 0, :] = xp.where(
            phase2_mask,
            z_aux2_vec - xp.take_along_axis(z_mc_e0, jk1_aux_0[:, None].astype(xp.int64), axis=1),
            zdiff_gradp[:, 0, :],
        )
        zdiff_gradp[:, 1, :] = xp.where(
            phase2_mask,
            z_aux2_vec - xp.take_along_axis(z_mc_e1, jk1_aux_1[:, None].astype(xp.int64), axis=1),
            zdiff_gradp[:, 1, :],
        )

        vertoffset_gradp[:, 0, :] = xp.where(
            phase2_mask,
            (jk1_aux_0[:, None] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 0, :],
        )
        vertoffset_gradp[:, 1, :] = xp.where(
            phase2_mask,
            (jk1_aux_1[:, None] - jk_idx).astype(gtx.int32),
            vertoffset_gradp[:, 1, :],
        )

    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 1, :], stream=decomposition.BLOCK)

    return zdiff_gradp, vertoffset_gradp
