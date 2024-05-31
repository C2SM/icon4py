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

from dataclasses import dataclass

from gt4py.next import as_field
from gt4py.next.common import Field

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C2EDim, CellDim


@dataclass
class DiagnosticState:
    """Class that contains the diagnostic state which is not used in dycore but may be used in physics.
    These variables are also stored for output purpose.

    Corresponds to ICON t_nh_diag
    """

    pressure: fa.CKfloatField
    # pressure at half levels
    pressure_ifc: fa.CKfloatField
    temperature: fa.CKfloatField
    # zonal wind speed
    u: fa.CKfloatField
    # meridional wind speed
    v: fa.CKfloatField

    @property
    def pressure_sfc(self) -> fa.CfloatField:
        return as_field((CellDim,), self.pressure_ifc.ndarray[:, -1])


@dataclass
class DiagnosticMetricState:
    """Class that contains the diagnostic metric state for computing the diagnostic state."""

    ddqz_z_full: fa.CKfloatField
    rbf_vec_coeff_c1: Field[[CellDim, C2E2C2EDim], float]
    rbf_vec_coeff_c2: Field[[CellDim, C2E2C2EDim], float]
