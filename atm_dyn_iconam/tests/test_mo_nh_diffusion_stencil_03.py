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

from icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_03 import (
    mo_nh_diffusion_stencil_03,
)
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


# if "Koff[-1]" is removed from `mo_nh_diffusion_stencil_03` the lower two versions
# verify -> problem is indeed with offset
def mo_nh_diffusion_stencil_03_div_ic_numpy(
    wgtfac_c: np.array,
    div: np.array,
) -> np.array:
    div_offset = np.delete(div, len(div[:][0]) - 1, axis=1)
    div = np.delete(div, 0, axis=1)
    wgtfac_c = np.delete(wgtfac_c, 0, axis=1)
    div_ic = wgtfac_c * div + (1.0 - wgtfac_c) * div_offset
    return div_ic


# if "Koff[-1]" is removed from `mo_nh_diffusion_stencil_03` the lower two versions
# verify -> problem is indeed with offset
def mo_nh_diffusion_stencil_03_hdef_ic_numpy(
    wgtfac_c: np.array,
    k_hc: np.array,
) -> np.array:
    k_hc_offset = np.delete(k_hc, len(k_hc[:][0]) - 1, axis=1)
    k_hc = np.delete(k_hc, 0, axis=1)
    wgtfac_c = np.delete(wgtfac_c, 0, axis=1)
    hdef_ic = wgtfac_c * k_hc + (1.0 - wgtfac_c) * k_hc_offset**2
    return hdef_ic


def mo_nh_diffusion_stencil_03_numpy(
    wgtfac_c: np.array,
    div: np.array,
    k_hc: np.array,
):
    div_ic = mo_nh_diffusion_stencil_03_div_ic_numpy(wgtfac_c, div)
    hdef_ic = mo_nh_diffusion_stencil_03_hdef_ic_numpy(wgtfac_c, k_hc)
    return div_ic, hdef_ic


def test_mo_nh_diffusion_stencil_03():
    mesh = SimpleMesh()

    wgtfac_c = random_field(mesh, CellDim, KDim)
    div = random_field(mesh, CellDim, KDim)
    k_hc = random_field(mesh, CellDim, KDim)

    div_ic = zero_field(mesh, CellDim, KDim)
    hdef_ic = zero_field(mesh, CellDim, KDim)

    div_ref, hdef_ic_ref = mo_nh_diffusion_stencil_03_numpy(
        np.asarray(wgtfac_c),
        np.asarray(div),
        np.asarray(k_hc),
    )

    mo_nh_diffusion_stencil_03(
        wgtfac_c,
        k_hc,
        div,
        div_ic,
        hdef_ic,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(hdef_ic[:, 1:], hdef_ic_ref)
    assert np.allclose(div_ic[:, 1:], div_ref)
