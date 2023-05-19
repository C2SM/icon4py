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
from gt4py.next.iterator.embedded import StridedNeighborOffsetProvider
from simple_mesh import SimpleMesh
from utils import as_1D_sparse_field, random_field, zero_field

from icon4py.atm_dyn_iconam.mo_velocity_advection_stencil_19 import (
    mo_velocity_advection_stencil_19,
)
from icon4py.common.dimension import (
    CellDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    VertexDim,
)


def mo_velocity_advection_stencil_19_numpy(
    e2v: np.array,
    e2c: np.array,
    z_kin_hor_e: np.array,
    coeff_gradekin: np.array,
    z_ekinh: np.array,
    zeta: np.array,
    vt: np.array,
    f_e: np.array,
    c_lin_e: np.array,
    z_w_con_c_full: np.array,
    vn_ie: np.array,
    ddqz_z_full_e: np.array,
) -> np.array:
    z_ekinh_e2c = z_ekinh[e2c]
    coeff_gradekin = np.expand_dims(coeff_gradekin, axis=-1)
    f_e = np.expand_dims(f_e, axis=-1)
    c_lin_e = np.expand_dims(c_lin_e, axis=-1)

    ddt_vn_adv = -(
        (coeff_gradekin[:, 0] - coeff_gradekin[:, 1]) * z_kin_hor_e
        + (
            -coeff_gradekin[:, 0] * z_ekinh_e2c[:, 0]
            + coeff_gradekin[:, 1] * z_ekinh_e2c[:, 1]
        )
        + vt * (f_e + 0.5 * np.sum(zeta[e2v], axis=1))
        + np.sum(z_w_con_c_full[e2c] * c_lin_e, axis=1)
        * (vn_ie[:, :-1] - vn_ie[:, 1:])
        / ddqz_z_full_e
    )
    return ddt_vn_adv


def test_mo_velocity_advection_stencil_19():
    mesh = SimpleMesh()

    z_kin_hor_e = random_field(mesh, EdgeDim, KDim)
    coeff_gradekin = random_field(mesh, EdgeDim, E2CDim)
    coeff_gradekin_new = as_1D_sparse_field(coeff_gradekin, ECDim)
    z_ekinh = random_field(mesh, CellDim, KDim)
    zeta = random_field(mesh, VertexDim, KDim)
    vt = random_field(mesh, EdgeDim, KDim)
    f_e = random_field(mesh, EdgeDim)
    c_lin_e = random_field(mesh, EdgeDim, E2CDim)
    z_w_con_c_full = random_field(mesh, CellDim, KDim)
    vn_ie = random_field(mesh, EdgeDim, KDim, extend={KDim: 1})
    ddqz_z_full_e = random_field(mesh, EdgeDim, KDim)
    ddt_vn_adv = zero_field(mesh, EdgeDim, KDim)

    ddt_vn_adv_ref = mo_velocity_advection_stencil_19_numpy(
        mesh.e2v,
        mesh.e2c,
        np.asarray(z_kin_hor_e),
        np.asarray(coeff_gradekin),
        np.asarray(z_ekinh),
        np.asarray(zeta),
        np.asarray(vt),
        np.asarray(f_e),
        np.asarray(c_lin_e),
        np.asarray(z_w_con_c_full),
        np.asarray(vn_ie),
        np.asarray(ddqz_z_full_e),
    )

    mo_velocity_advection_stencil_19(
        z_kin_hor_e,
        coeff_gradekin_new,
        z_ekinh,
        zeta,
        vt,
        f_e,
        c_lin_e,
        z_w_con_c_full,
        vn_ie,
        ddqz_z_full_e,
        ddt_vn_adv,
        offset_provider={
            "E2V": mesh.get_e2v_offset_provider(),
            "E2C": mesh.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
            "Koff": KDim,
        },
    )

    assert np.allclose(ddt_vn_adv, ddt_vn_adv_ref)
