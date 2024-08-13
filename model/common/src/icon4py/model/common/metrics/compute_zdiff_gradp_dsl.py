# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def compute_zdiff_gradp_dsl(
    e2c,
    z_me: np.array,
    z_mc: np.array,
    z_ifc: np.array,
    flat_idx: np.array,
    z_aux2: np.array,
    nlev: int,
    horizontal_start: int,
    horizontal_start_1: int,
    nedges: int,
) -> np.array:
    zdiff_gradp = np.zeros_like(z_mc[e2c])
    zdiff_gradp[horizontal_start:, :, :] = (
        np.expand_dims(z_me, axis=1)[horizontal_start:, :, :] - z_mc[e2c][horizontal_start:, :, :]
    )
    """
    First part for loop implementation with gt4py code

    >>> z_ifc_off_koff = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    >>> z_ifc_off = as_field((EdgeDim, KDim,), z_ifc[e2c[:, 0], :])
    >>> _compute_z_ifc_off_koff(
    >>>     z_ifc_off=z_ifc_off,
    >>>     domain={EdgeDim: (horizontal_start, nedges), KDim: (0, nlev)},
    >>>     out=z_ifc_off_koff,
    >>>     offset_provider={"Koff": icon_grid.get_offset_provider("Koff")}
    >>> )
    """

    for je in range(horizontal_start, nedges):
        for jk in range(int(flat_idx[je]) + 1, nlev):
            """
            Second part for loop implementation with gt4py code
            >>> param_2 = as_field((KDim,), np.asarray([False] * nlev))
            >>> param_3 = as_field((KDim,), np.arange(nlev))
            >>> z_ifc_off_e = as_field((KDim,), z_ifc[e2c[je, 0], :])
            >>> _compute_param.with_backend(backend)(
            >>>     z_me_jk=z_me[je, jk],
            >>>     z_ifc_off=z_ifc_off_e,
            >>>     z_ifc_off_koff=as_field((KDim,), z_ifc_off_koff.asnumpy()[je, :]),
            >>>     lower=int(flat_idx[je]),
            >>>     nlev=nlev - 1,
            >>>     out=(param_3, param_2),
            >>>     offset_provider={}
            >>> )
            >>> zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], np.where(param_2.asnumpy())[0][0]]
            """

            param = [False] * nlev
            for jk1 in range(int(flat_idx[je]), nlev):
                if (
                    jk1 == nlev - 1
                    or z_me[je, jk] <= z_ifc[e2c[je, 0], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 0], jk1 + 1]
                ):
                    param[jk1] = True

            zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], np.where(param)[0][0]]

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            for jk1 in range(jk_start, nlev):
                if (
                    jk1 == nlev - 1
                    or z_me[je, jk] <= z_ifc[e2c[je, 1], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 1], jk1 + 1]
                ):
                    zdiff_gradp[je, 1, jk] = z_me[je, jk] - z_mc[e2c[je, 1], jk1]
                    jk_start = jk1
                    break

    for je in range(horizontal_start_1, nedges):
        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if (
                        jk1 == nlev - 1
                        or z_aux2[je] <= z_ifc[e2c[je, 0], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 0], jk1 + 1]
                    ):
                        zdiff_gradp[je, 0, jk] = z_aux2[je] - z_mc[e2c[je, 0], jk1]
                        jk_start = jk1
                        break

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if (
                        jk1 == nlev - 1
                        or z_aux2[je] <= z_ifc[e2c[je, 1], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 1], jk1 + 1]
                    ):
                        zdiff_gradp[je, 1, jk] = z_aux2[je] - z_mc[e2c[je, 1], jk1]
                        jk_start = jk1
                        break

    return zdiff_gradp
