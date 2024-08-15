# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

import dataclasses

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


@dataclasses.dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog
    """

    #: specific humidity [kg/kg] at cell center
    qv: fa.CellKField[ta.wpfloat]
    #: specific cloud water content [kg/kg] at cell center
    qc: fa.CellKField[ta.wpfloat]
    #: specific rain content [kg/kg] at cell center
    qr: fa.CellKField[ta.wpfloat]
    #: specific cloud ice content [kg/kg] at cell center
    qi: fa.CellKField[ta.wpfloat]
    #: specific snow content [kg/kg] at cell center
    qs: fa.CellKField[ta.wpfloat]
    #: specific graupel content [kg/kg] at cell center
    qg: fa.CellKField[ta.wpfloat]
