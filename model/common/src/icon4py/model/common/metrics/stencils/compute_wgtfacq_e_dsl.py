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

from icon4py.model.common.metrics.stencils.compute_wgtfacq_c_dsl import (
    compute_wgtfacq_c_dsl,
    compute_z1_z2_z3,
)


def compute_wgtfacq_e_dsl(
    e2c,
    z_ifc: np.array,
    z_aux_c: np.array,
    c_lin_e: np.array,
    wgtfacq_e_dsl: np.array,
    nlev: int,
):
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        e2c: Edge to Cell offset
        z_ifc: geometric height at the vertical interface of cells.
        z_aux_c: interpolation of weighting coefficients to edges
        c_lin_e: interpolation field
        wgtfacq_e_dsl: output
        nlev: int, last k level
    Returns:
    Field[EdgeDim, KDim] (full levels)
    """
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
