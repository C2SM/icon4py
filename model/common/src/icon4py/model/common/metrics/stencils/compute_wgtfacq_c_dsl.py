# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
