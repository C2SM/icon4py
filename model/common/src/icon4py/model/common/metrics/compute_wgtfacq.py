# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common.settings import xp


def _compute_z1_z2_z3(
    z_ifc: xp.ndarray, i1: int, i2: int, i3: int, i4: int
) -> tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    z1 = 0.5 * (z_ifc[:, i2] - z_ifc[:, i1])
    z2 = 0.5 * (z_ifc[:, i2] + z_ifc[:, i3]) - z_ifc[:, i1]
    z3 = 0.5 * (z_ifc[:, i3] + z_ifc[:, i4]) - z_ifc[:, i1]
    return z1, z2, z3


def compute_wgtfacq_c_dsl(
    z_ifc: xp.ndarray,
    nlev: int,
) -> xp.ndarray:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        z_ifc: Field[CellDim, KDim] (half levels), geometric height at the vertical interface of cells.
        nlev: int, last k level
    Returns:
    Field[CellDim, KDim] (full levels)
    """
    wgtfacq_c = xp.zeros((z_ifc.shape[0], nlev + 1))
    wgtfacq_c_dsl = xp.zeros((z_ifc.shape[0], nlev))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)

    wgtfacq_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    wgtfacq_c[:, 1] = (z1 - wgtfacq_c[:, 2] * (z1 - z3)) / (z1 - z2)
    wgtfacq_c[:, 0] = 1.0 - (wgtfacq_c[:, 1] + wgtfacq_c[:, 2])

    wgtfacq_c_dsl[:, nlev - 1] = wgtfacq_c[:, 0]
    wgtfacq_c_dsl[:, nlev - 2] = wgtfacq_c[:, 1]
    wgtfacq_c_dsl[:, nlev - 3] = wgtfacq_c[:, 2]

    return wgtfacq_c_dsl


def compute_wgtfacq_e_dsl(
    e2c: xp.ndarray,
    z_ifc: xp.ndarray,
    c_lin_e: xp.ndarray,
    wgtfacq_c_dsl: xp.ndarray,
    n_edges: int,
    nlev: int,
):
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        e2c: Edge to Cell offset
        z_ifc: geometric height at the vertical interface of cells.
        wgtfacq_c_dsl: weighting factor for quadratic interpolation to surface
        c_lin_e: interpolation field
        n_edges: number of edges
        nlev: int, last k level
    Returns:
    Field[EdgeDim, KDim] (full levels)
    """
    wgtfacq_e_dsl = xp.zeros(shape=(n_edges, nlev + 1))
    z_aux_c = xp.zeros((z_ifc.shape[0], 6))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)
    z_aux_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 1] = (z1 - wgtfacq_c_dsl[:, nlev - 3] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 0] = 1.0 - (wgtfacq_c_dsl[:, nlev - 2] + wgtfacq_c_dsl[:, nlev - 3])

    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, 0, 1, 2, 3)
    z_aux_c[:, 5] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 4] = (z1 - z_aux_c[:, 5] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 3] = 1.0 - (z_aux_c[:, 4] + z_aux_c[:, 5])

    c_lin_e = c_lin_e[:, :, xp.newaxis]
    z_aux_e = xp.sum(c_lin_e * z_aux_c[e2c], axis=1)

    wgtfacq_e_dsl[:, nlev] = z_aux_e[:, 0]
    wgtfacq_e_dsl[:, nlev - 1] = z_aux_e[:, 1]
    wgtfacq_e_dsl[:, nlev - 2] = z_aux_e[:, 2]

    return wgtfacq_e_dsl
