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

from icon4py.atm_dyn_iconam.mo_icon_interpolation_fields_initalization_stencil import (
    mo_icon_interpolation_fields_initalization_stencil,
)
from icon4py.common.dimension import (
    CellDim,
    E2CDim,
    ECDim,
    EdgeDim,
    KDim,
    VertexDim,
)

from .test_utils.helpers import as_1D_sparse_field, random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_icon_interpolation_fields_initalization_stencil_numpy(
    edge_cell_length: np.array,
    dual_edge_length: np.array,
    c_lin_e: np.array,
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


def test_mo_icon_interpolation_fields_initalization_stencil():
    mesh = SimpleMesh()

    edge_cell_length = random_field(mesh, EdgeDim, KDim)
    dual_edge_length = random_field(mesh, EdgeDim, E2CDim)
    c_lin_e = random_field(mesh, EdgeDim, E2CDim)

    ddt_vn_adv_ref = mo_icon_interpolation_fields_initalization_stencil_numpy(
        mesh.e2v,
        mesh.e2c,
        np.asarray(edge_cell_length),
        np.asarray(dual_edge_length),
        np.asarray(c_lin_e),
    )

    mo_icon_interpolation_fields_initalization_stencil_19(
        edge_cell_length,
        dual_edge_length,
        c_lin_e,
        offset_provider={
            "E2V": mesh.get_e2v_offset_provider(),
            "E2C": mesh.get_e2c_offset_provider(),
            "E2EC": StridedNeighborOffsetProvider(EdgeDim, ECDim, mesh.n_e2c),
            "Koff": KDim,
        },
    )

    assert np.allclose(c_lin_e, c_lin_e_ref)
