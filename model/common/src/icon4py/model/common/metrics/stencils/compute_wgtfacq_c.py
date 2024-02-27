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


def compute_wgtfacq_c(
    z_ifc: np.array,
    nlevp1: int,
) -> np.array:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        z_ifc: Field[CellDim, KDim] (half levels), geometric height at the vertical interface of cells.
        nlev: int, last level
    Returns:
    Field[CellDim, KDim] (full levels)
    """
    nlev = nlevp1 - 1
    wgtfacq_c = np.zeros((z_ifc.shape[0], nlevp1))
    z1 = 0.5 * (z_ifc[:, nlev] - z_ifc[:, nlevp1])
    z2 = 0.5 * (z_ifc[:, nlev] + z_ifc[:, nlev - 1]) - z_ifc[:, nlevp1]
    z3 = 0.5 * (z_ifc[:, nlev - 1] + z_ifc[:, nlev - 2]) - z_ifc[:, nlevp1]

    wgtfacq_c[:, nlev - 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    wgtfacq_c[:, nlev - 1] = (z1 - wgtfacq_c[:, nlev - 2] * (z1 - z3)) / (z1 - z2)
    wgtfacq_c[:, nlev] = 1.0 - (wgtfacq_c[:, nlev - 1] + wgtfacq_c[:, nlev - 2])

    return wgtfacq_c
