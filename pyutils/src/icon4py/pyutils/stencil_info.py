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

from functional.ffront import program_ast as past
from functional.ffront.decorator import Program


@dataclasses.dataclass(frozen=True)
class StencilInfo:
    fvprog: Program
    connectivity_chains: list[str]
    stencil_header: str


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    field: past.DataSymbol
    inp: bool
    out: bool
