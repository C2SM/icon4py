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
"""Prognostic one-moment bulk microphysical parameterization.

"""
import sys
from typing import Final
import dataclasses

from gt4py.next.embedded.context import offset_provider
from gt4py.next.program_processors.runners.double_roundtrip import backend
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn_cached,
    run_gtfn_gpu_cached,
)
import numpy as np
import gt4py.next as gtx
from gt4py.eve.utils import FrozenNamespace
from gt4py.next.ffront.fbuiltins import log, exp, maximum, minimum, sqrt
from icon4py.model.common import constants as global_const
from gt4py.next.ffront.decorator import program, field_operator, scan_operator
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.atmosphere.physics.microphysics.single_moment_six_class_gscp_graupel import icon_graupel_params
from icon4py.model.atmosphere.physics.microphysics import single_moment_six_class_gscp_graupel as graupel
from icon4py.model.atmosphere.physics.microphysics import saturation_adjustment
from icon4py.model.common.type_alias import wpfloat, vpfloat



my_a: Final[wpfloat] = wpfloat("1.0")

for k ,v in list(globals().items()):
    print(k,v)

