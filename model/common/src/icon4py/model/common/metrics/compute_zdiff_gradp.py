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
    vertoffset_gradp = array_ns.zeros((nedges, 2, nlev), dtype=gtx.int32)

    z_ifc_asc = z_ifc[:, ::-1]
    e2c_0 = e2c[:, 0]
    e2c_1 = e2c[:, 1]

    for je in range(horizontal_start, nedges):
        fi = int(flat_idx[je])
        njk = nlev - fi - 1
        if njk <= 0:
            continue

        jk_slice = slice(fi + 1, nlev)
        z_me_slice = z_me[je, jk_slice]
        n_prefix = nlev - fi
        jk_field_slice = jk_field[jk_slice]

        ci0 = e2c_0[je]
        pos0 = np.searchsorted(z_ifc_asc[ci0, :n_prefix], z_me_slice)
        jk1_0 = np.clip(nlev - pos0, fi, nlev - 1)
        zdiff_gradp[je, 0, jk_slice] = z_me_slice - z_mc[ci0, jk1_0]
        vertoffset_gradp[je, 0, jk_slice] = jk1_0 - jk_field_slice

        ci1 = e2c_1[je]
        pos1 = np.searchsorted(z_ifc_asc[ci1, :n_prefix], z_me_slice)
        jk1_1 = np.clip(nlev - pos1, fi, nlev - 1)
        zdiff_gradp[je, 1, jk_slice] = z_me_slice - z_mc[ci1, jk1_1]
        vertoffset_gradp[je, 1, jk_slice] = jk1_1 - jk_field_slice

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

        n_prefix = nlev - fi
        jk_field_slice = jk_field[je_slice]

        ci0 = e2c_0[je]
        pos0 = np.searchsorted(z_ifc_asc[ci0, :n_prefix], z_aux2[je])
        jk1_0 = np.clip(nlev - pos0, fi, nlev - 1)
        zdiff_gradp[je, 0, je_slice][mask] = z_aux2[je] - z_mc[ci0, jk1_0]
        vertoffset_gradp[je, 0, je_slice][mask] = jk1_0 - jk_field_slice[mask]

        ci1 = e2c_1[je]
        pos1 = np.searchsorted(z_ifc_asc[ci1, :n_prefix], z_aux2[je])
        jk1_1 = np.clip(nlev - pos1, fi, nlev - 1)
        zdiff_gradp[je, 1, je_slice][mask] = z_aux2[je] - z_mc[ci1, jk1_1]
        vertoffset_gradp[je, 1, je_slice][mask] = jk1_1 - jk_field_slice[mask]

    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 1, :], stream=decomposition.BLOCK)

    return zdiff_gradp, vertoffset_gradp
