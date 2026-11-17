# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


def _first_bracketing_level(
    z_ifc_e: data_alloc.NDArray,
    target: data_alloc.NDArray,
    lower: data_alloc.NDArray,
    shape: tuple[int, ...],
    nlev: int,
) -> data_alloc.NDArray:
    """For each entry, the first level `jk1 >= lower` with `z_ifc_e[jk1] >= target >= z_ifc_e[jk1 + 1]`, or `nlev - 1`.

    Loops over the levels only, every step works on whole arrays so no values are read back to the host.
    """
    array_ns = data_alloc.array_namespace(z_ifc_e)
    level = array_ns.full(shape, nlev - 1, dtype=gtx.int32)
    found = array_ns.zeros(shape, dtype=bool)
    for jk1 in range(nlev - 1):
        hit = (
            ~found
            & (jk1 >= lower)
            & (target <= z_ifc_e[:, :, jk1 : jk1 + 1])
            & (target >= z_ifc_e[:, :, jk1 + 1 : jk1 + 2])
        )
        level = array_ns.where(hit, jk1, level)
        found |= hit
    return level


def compute_zdiff_gradp(
    *,
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
    z_mc_e = z_mc[e2c]
    z_me = array_ns.sum(z_mc_e * array_ns.expand_dims(c_lin_e, axis=-1), axis=1)

    exchange.exchange(dims.EdgeDim, z_me, stream=decomposition.BLOCK)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0  # extrapol_dist
    zdiff_gradp = array_ns.zeros_like(z_mc_e)
    jk_field = array_ns.arange(nlev, dtype=gtx.int32)
    zdiff_gradp[horizontal_start:, :, :] = (
        array_ns.expand_dims(z_me, axis=1)[horizontal_start:, :, :]
        - z_mc_e[horizontal_start:, :, :]
    )
    vertidx_gradp = array_ns.broadcast_to(jk_field, (nedges, 2, nlev)).copy()

    # Shapes: edges x neighbor cells x levels.
    z_ifc_e = z_ifc[e2c]
    edge = array_ns.arange(nedges)[:, None, None]
    jk = jk_field[None, None, :]
    lower = flat_idx[:, None, None]
    below_flat = jk > lower

    """
    The search in `_first_bracketing_level` can also be implemented with gt4py
    code, shown here for the first neighbor cell.

    First part, once for all edges:

    >>> z_ifc_off_koff = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    >>> z_ifc_off = as_field((EdgeDim, KDim,), z_ifc[e2c[:, 0], :])
    >>> _compute_z_ifc_off_koff(
    >>>     z_ifc_off=z_ifc_off,
    >>>     domain={EdgeDim: (horizontal_start, nedges), KDim: (0, nlev)},
    >>>     out=z_ifc_off_koff,
    >>>     offset_provider={"Koff": icon_grid.get_offset_provider("Koff")}
    >>> )

    Second part, for each edge `je` and level `jk` below `flat_idx[je]`:

    >>> param_2 = as_field((KDim,), array_ns.asarray([False] * nlev))
    >>> param_3 = as_field((KDim,), array_ns.arange(nlev))
    >>> z_ifc_off_e = as_field((KDim,), z_ifc[e2c[je, 0], :])
    >>> _compute_param.with_backend(backend)(
    >>>     z_me_jk=z_me[je, jk],
    >>>     z_ifc_off=z_ifc_off_e,
    >>>     z_ifc_off_koff=as_field((KDim,), z_ifc_off_koff.ndarray[je, :]),
    >>>     lower=int(flat_idx[je]),
    >>>     nlev=nlev - 1,
    >>>     out=(param_3, param_2),
    >>>     offset_provider={}
    >>> )
    >>> zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], array_ns.where(param_2.ndarray)[0][0]]
    """

    # Level of each neighbor cell that contains the edge height. The search for the second
    # neighbor in ICON starts from the level found for the previous jk, which gives the same
    # result since the heights decrease with the level.
    z_me_e = z_me[:, None, :]
    level = _first_bracketing_level(z_ifc_e, z_me_e, lower, z_mc_e.shape, nlev)
    mask = (edge >= horizontal_start) & below_flat
    vertidx_gradp = array_ns.where(mask, level, vertidx_gradp)
    zdiff_gradp = array_ns.where(
        mask, z_me_e - array_ns.take_along_axis(z_mc_e, level, axis=2), zdiff_gradp
    )

    # Below the extrapolation height, use the level containing z_aux2, which is the same for all jk.
    z_aux2_e = z_aux2[:, None, None]
    level = _first_bracketing_level(z_ifc_e, z_aux2_e, lower, (nedges, 2, 1), nlev)
    mask = (edge >= horizontal_start_1) & below_flat & (z_me_e < z_aux2_e)
    vertidx_gradp = array_ns.where(mask, level, vertidx_gradp)
    zdiff_gradp = array_ns.where(
        mask, z_aux2_e - array_ns.take_along_axis(z_mc_e, level, axis=2), zdiff_gradp
    )

    vertoffset_gradp = vertidx_gradp - jk_field

    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 1, :], stream=decomposition.BLOCK)

    return zdiff_gradp, vertoffset_gradp
