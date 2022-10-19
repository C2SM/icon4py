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

from functional.ffront.decorator import program, scan_operator
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import CellDim, KDim


@scan_operator(
    axis=KDim,
    forward=True,
    init=(0.0, 0.0),
)
def _graupel(
    carry: tuple[float, float, float],
    qc_in: float,
    qr_in: float,
):

    a = 0.1
    b = 0.2

    # unpack carry
    qc_kMinus1, qr_kMinus1, _ = carry

    # Autoconversion: Cloud Drops -> Rain Drops
    qc = qc_in + qc_kMinus1
    qr = qr_in + qr_kMinus1

    return qr, qc


@program
def graupel(
    qc: Field[[CellDim, KDim], float],
    qr: Field[[CellDim, KDim], float],
):
    # Writing to several output fields currently breaks due to gt4py bugs
    _graupel(qc, qr, out=(qr, qc))
