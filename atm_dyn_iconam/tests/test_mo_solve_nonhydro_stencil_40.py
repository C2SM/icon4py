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

from icon4py.atm_dyn_iconam.mo_solve_nonhydro_stencil_40 import (
    mo_solve_nonhydro_stencil_40,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_solve_nonhydro_stencil_40_numpy(
    e_bln_c_s: np.array,
    z_w_concorr_me: np.array,
    wgtfacq_c: np.array,
    c2e: np.array,
) -> np.array:
    e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
    z_w_concorr_me_offset_1 = np.roll(z_w_concorr_me, shift=1, axis=1)
    z_w_concorr_me_offset_2 = np.roll(z_w_concorr_me, shift=2, axis=1)
    z_w_concorr_me_offset_3 = np.roll(z_w_concorr_me, shift=3, axis=1)
    z_w_concorr_mc_m1 = np.sum(e_bln_c_s * z_w_concorr_me_offset_1[c2e], axis=1)
    z_w_concorr_mc_m2 = np.sum(e_bln_c_s * z_w_concorr_me_offset_2[c2e], axis=1)
    z_w_concorr_mc_m3 = np.sum(e_bln_c_s * z_w_concorr_me_offset_3[c2e], axis=1)
    w_concorr_c = (
        np.roll(wgtfacq_c, shift=1, axis=1) * z_w_concorr_mc_m1
        + np.roll(wgtfacq_c, shift=2, axis=1) * z_w_concorr_mc_m2
        + np.roll(wgtfacq_c, shift=3, axis=1) * z_w_concorr_mc_m3
    )
    return w_concorr_c


def test_mo_solve_nonhydro_stencil_40():
    mesh = SimpleMesh()

    e_bln_c_s = random_field(mesh, CellDim, C2EDim)
    z_w_concorr_me = random_field(mesh, EdgeDim, KDim)
    wgtfacq_c = random_field(mesh, CellDim, KDim)

    w_concorr_c = zero_field(mesh, CellDim, KDim)

    w_concorr_c_ref = mo_solve_nonhydro_stencil_40_numpy(
        np.asarray(e_bln_c_s),
        np.asarray(z_w_concorr_me),
        np.asarray(wgtfacq_c),
        mesh.c2e,
    )

    mo_solve_nonhydro_stencil_40(
        as_1D_sparse_field(e_bln_c_s, CEDim),
        z_w_concorr_me,
        wgtfacq_c,
        w_concorr_c,
        offset_provider={
            "Koff": KDim,
            "C2E": mesh.get_c2e_offset_provider(),
            "C2CE": mesh.get_c2ce_offset_provider(),
        },
    )

    assert np.allclose(w_concorr_c[:, 3:], w_concorr_c_ref[:, 3:])
