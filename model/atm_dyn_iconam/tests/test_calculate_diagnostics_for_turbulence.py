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

from icon4py.model.atm_dyn_iconam.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def calculate_diagnostics_for_turbulence_numpy(
    div: np.array,
    k_hc: np.array,
    wgtfac_c: np.array,
) -> tuple[np.array, np.array]:
    kc_offset_1 = np.roll(k_hc, shift=1, axis=1)
    div_offset_1 = np.roll(div, shift=1, axis=1)
    div_ic = wgtfac_c * div + (1.0 - wgtfac_c) * div_offset_1
    hdef_ic = (wgtfac_c * k_hc + (1.0 - wgtfac_c) * kc_offset_1) ** 2
    return div_ic, hdef_ic


def test_calculate_diagnostics_for_turbulence():
    mesh = SimpleMesh()

    wgtfac_c = random_field(mesh, CellDim, KDim)
    div = random_field(mesh, CellDim, KDim)
    kh_c = random_field(mesh, CellDim, KDim)

    div_ic = zero_field(mesh, CellDim, KDim)
    hdef_ic = zero_field(mesh, CellDim, KDim)

    div_ref, kh_c_ref = calculate_diagnostics_for_turbulence_numpy(
        np.asarray(div),
        np.asarray(kh_c),
        np.asarray(wgtfac_c),
    )

    calculate_diagnostics_for_turbulence(
        div,
        kh_c,
        wgtfac_c,
        div_ic,
        hdef_ic,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(hdef_ic[:, 1:], kh_c_ref[:, 1:])
    assert np.allclose(div_ic[:, 1:], div_ref[:, 1:])
