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
from typing import Optional


@dataclass(frozen=True)
class BoundsData:
    hlower: str | int
    hupper: str | int
    vlower: str | int
    vupper: str | int


@dataclass(frozen=False)  # not frozen as tolerances are updated after object creation
class FieldAssociationData:
    variable_name: str
    variable_association: str
    inp: bool
    out: bool
    abs_tol: Optional[str] = None
    rel_tol: Optional[str] = None


@dataclass(frozen=True)
class StencilData:
    name: str
    fields: list[FieldAssociationData]
    bounds: BoundsData
    startln: int
    endln: int


@dataclass(frozen=True)
class DeclareData:
    startln: int
    endln: int
    declarations: dict[str, str]


@dataclass(frozen=True)
class CreateData:
    startln: int
    endln: int
    variables: list[str]
