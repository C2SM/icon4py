# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


def compute_z1_z2_z3(z_ifc, i1, i2, i3, i4):
    z1 = 0.5 * (z_ifc[:, i2] - z_ifc[:, i1])
    z2 = 0.5 * (z_ifc[:, i2] + z_ifc[:, i3]) - z_ifc[:, i1]
    z3 = 0.5 * (z_ifc[:, i3] + z_ifc[:, i4]) - z_ifc[:, i1]
    return z1, z2, z3


def compute_wgtfacq_c_dsl(
    z_ifc: np.array,
    nlev: int,
) -> np.array:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        z_ifc: Field[CellDim, KDim] (half levels), geometric height at the vertical interface of cells.
        nlev: int, last k level
    Returns:
    Field[CellDim, KDim] (full levels)
    """
    wgtfacq_c = np.zeros((z_ifc.shape[0], nlev + 1))
    wgtfacq_c_dsl = np.zeros((z_ifc.shape[0], nlev))
    z1, z2, z3 = compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)

    wgtfacq_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    wgtfacq_c[:, 1] = (z1 - wgtfacq_c[:, 2] * (z1 - z3)) / (z1 - z2)
    wgtfacq_c[:, 0] = 1.0 - (wgtfacq_c[:, 1] + wgtfacq_c[:, 2])

    wgtfacq_c_dsl[:, nlev - 1] = wgtfacq_c[:, 0]
    wgtfacq_c_dsl[:, nlev - 2] = wgtfacq_c[:, 1]
    wgtfacq_c_dsl[:, nlev - 3] = wgtfacq_c[:, 2]

    return wgtfacq_c_dsl


def compute_wgtfacq_e_dsl(
    e2c,
    z_ifc: np.array,
    z_aux_c: np.array,
    c_lin_e: np.array,
    n_edges: int,
    nlev: int,
):
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        e2c: Edge to Cell offset
        z_ifc: geometric height at the vertical interface of cells.
        z_aux_c: interpolation of weighting coefficients to edges
        c_lin_e: interpolation field
        n_edges: number of edges
        nlev: int, last k level
    Returns:
    Field[EdgeDim, KDim] (full levels)
    """
    wgtfacq_e_dsl = np.zeros(shape=(n_edges, nlev + 1))
    z1, z2, z3 = compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)
    wgtfacq_c_dsl = compute_wgtfacq_c_dsl(z_ifc, nlev)
    z_aux_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 1] = (z1 - wgtfacq_c_dsl[:, nlev - 3] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 0] = 1.0 - (wgtfacq_c_dsl[:, nlev - 2] + wgtfacq_c_dsl[:, nlev - 3])

    z1, z2, z3 = compute_z1_z2_z3(z_ifc, 0, 1, 2, 3)
    z_aux_c[:, 5] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 4] = (z1 - z_aux_c[:, 5] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 3] = 1.0 - (z_aux_c[:, 4] + z_aux_c[:, 5])

    c_lin_e = c_lin_e[:, :, np.newaxis]
    z_aux_e = np.sum(c_lin_e * z_aux_c[e2c], axis=1)

    wgtfacq_e_dsl[:, nlev] = z_aux_e[:, 0]
    wgtfacq_e_dsl[:, nlev - 1] = z_aux_e[:, 1]
    wgtfacq_e_dsl[:, nlev - 2] = z_aux_e[:, 2]

    return wgtfacq_e_dsl
