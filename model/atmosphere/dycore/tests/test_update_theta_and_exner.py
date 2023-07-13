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

from icon4py.model.atmosphere.dycore.update_theta_and_exner import (
    update_theta_and_exner,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestUpdateThetaAndExner(StencilTest):
    PROGRAM = update_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        mesh,
        z_temp: np.array,
        area: np.array,
        theta_v: np.array,
        exner: np.array,
        rd_o_cvd,
        **kwargs,
    ) -> tuple[np.array]:
        area = np.expand_dims(area, axis=0)
        z_theta = theta_v
        theta_v = theta_v + (np.expand_dims(area, axis=-1) * z_temp)
        exner = exner * (1.0 + rd_o_cvd * (theta_v / z_theta - 1.0))
        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, mesh):
        z_temp = random_field(mesh, CellDim, KDim)
        area = random_field(mesh, CellDim)
        theta_v = random_field(mesh, CellDim, KDim)
        exner = random_field(mesh, CellDim, KDim)
        rd_o_cvd = 5.0

        return dict(
            z_temp=z_temp,
            area=area,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
        )
