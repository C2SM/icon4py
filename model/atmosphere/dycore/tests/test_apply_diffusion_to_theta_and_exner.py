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
import pytest

from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.apply_diffusion_to_theta_and_exner import apply_diffusion_to_theta_and_exner
from icon4py.model.common.dimension import CellDim, EdgeDim, CEDim, CECDim, C2E2CDim, KDim

from icon4py.model.common.test_utils.helpers import flatten_first_two_dims, random_field, random_mask, zero_field, StencilTest


class TestApplyDiffusionToThetaAndExner(StencilTest):
    PROGRAM = apply_diffusion_to_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:
        theta_v = 0.
        exner = 0.
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, mesh):
        kh_smag_e = random_field(mesh, EdgeDim, KDim)
        inv_dual_edge_length = random_field(mesh, EdgeDim)
        theta_v_in = random_field(mesh, CellDim, KDim)
        geofac_div = random_field(mesh, CEDim)
        mask = random_mask(mesh, CellDim, KDim)
        zd_vertoffset = zero_field(mesh, CellDim, C2E2CDim, KDim, dtype=int32)
        rng = np.random.default_rng()
        for k in range(mesh.k_level):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            zd_vertoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=mesh.k_level - k - 1,
                size=(zd_vertoffset.shape[0], zd_vertoffset.shape[1]),
            )
        zd_diffcoef = random_field(mesh, CellDim, KDim)
        geofac_n2s_c = random_field(mesh, CellDim)
        geofac_n2s_nbh = random_field(mesh, CellDim, C2E2CDim)
        vcoef = random_field(mesh, CellDim, C2E2CDim, KDim)
        area = random_field(mesh, CellDim)
        theta_v = random_field(mesh, CellDim, KDim)
        exner = random_field(mesh, CellDim, KDim)
        rd_o_cvd = 5.0

        vcoef_new = flatten_first_two_dims(CECDim, KDim, field=vcoef)
        zd_vertoffset_new = flatten_first_two_dims(CECDim, KDim, field=zd_vertoffset)
        geofac_n2s_nbh_new = flatten_first_two_dims(CECDim, field=geofac_n2s_nbh)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v_in=theta_v_in,
            geofac_div=geofac_div,
            mask=mask,
            zd_vertoffset=zd_vertoffset_new,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh_new,
            vcoef=vcoef_new,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
        )
