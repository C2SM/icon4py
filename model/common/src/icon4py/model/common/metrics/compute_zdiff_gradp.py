# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


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
    nedges = e2c.shape[0]
    z_me = array_ns.sum(z_mc[e2c] * array_ns.expand_dims(c_lin_e, axis=-1), axis=1)

    exchange.exchange(dims.EdgeDim, z_me, stream=decomposition.BLOCK)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0  # extrapol_dist
    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    jk_field = array_ns.arange(nlev, dtype=gtx.int32)
    zdiff_gradp[horizontal_start:, :, :] = (
        array_ns.expand_dims(z_me, axis=1)[horizontal_start:, :, :]
        - z_mc[e2c][horizontal_start:, :, :]
    )
    vertidx_gradp = array_ns.expand_dims(
        array_ns.expand_dims(jk_field, axis=0).repeat(2, axis=0), axis=0
    ).repeat(nedges, axis=0)
    vertoffset_gradp = array_ns.expand_dims(
        array_ns.expand_dims(jk_field, axis=0).repeat(2, axis=0), axis=0
    ).repeat(nedges, axis=0)

    for je in range(horizontal_start, nedges):
        fi = int(flat_idx[je])
        njk = nlev - fi - 1
        if njk <= 0:
            continue

        for side in range(2):
            ci = e2c[je, side]
            z_ifc_rev = z_ifc[ci, fi : nlev + 1][::-1]
            pos = np.searchsorted(z_ifc_rev, z_me[je, fi + 1 : nlev])
            jk1_arr = np.clip(nlev - pos, fi, nlev - 1)
            vertidx_gradp[je, side, fi + 1 : nlev] = jk1_arr
            zdiff_gradp[je, side, fi + 1 : nlev] = z_me[je, fi + 1 : nlev] - z_mc[ci, jk1_arr]

    for je in range(horizontal_start_1, nedges):
        fi = int(flat_idx[je])
        njk = nlev - fi - 1
        if njk <= 0:
            continue
        je_slice = slice(fi + 1, nlev)
        je_vals = z_me[je, je_slice]
        mask = je_vals < z_aux2[je]
        if not mask.any():
            continue

        for side in range(2):
            ci = e2c[je, side]
            z_ifc_rev = z_ifc[ci, fi : nlev + 1][::-1]
            pos = np.searchsorted(z_ifc_rev, z_aux2[je])
            jk1_aux = np.clip(nlev - pos, fi, nlev - 1)
            vertidx_gradp[je, side, je_slice][mask] = jk1_aux
            zdiff_gradp[je, side, je_slice][mask] = z_aux2[je] - z_mc[ci, jk1_aux]

    vertoffset_gradp = vertidx_gradp - vertoffset_gradp

    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 1, :], stream=decomposition.BLOCK)

    return zdiff_gradp, vertoffset_gradp
