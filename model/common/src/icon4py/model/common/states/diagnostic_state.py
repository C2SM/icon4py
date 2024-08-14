# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

    pressure: fa.CellKField[float]
    # pressure at half levels
    pressure_ifc: fa.CellKField[float]
    temperature: fa.CellKField[float]
    # zonal wind speed
    u: fa.CellKField[float]
    # meridional wind speed
    v: fa.CellKField[float]

    @property
    def pressure_sfc(self) -> fa.CellField[float]:
        return as_field((CellDim,), self.pressure_ifc.ndarray[:, -1])


@dataclass
class DiagnosticMetricState:
    """Class that contains the diagnostic metric state for computing the diagnostic state."""

    ddqz_z_full: fa.CellKField[float]
    rbf_vec_coeff_c1: Field[[CellDim, C2E2C2EDim], float]
    rbf_vec_coeff_c2: Field[[CellDim, C2E2C2EDim], float]
