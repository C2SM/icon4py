# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common import constants as phy_const

@dataclasses.dataclass
class ExternalParameters:
    """Dataclass containing external parameters."""

    topo_c: fa.CellField[float]
    topo_smt_c: fa.CellField[float]

