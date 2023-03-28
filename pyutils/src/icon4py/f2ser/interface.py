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

from icon4py.liskov.codegen.types import CodeGenInput


@dataclass
class Metadata:
    key: str
    value: str


@dataclass
class FieldSerialisationData:
    variable: str
    association: str


@dataclass
class InitData(CodeGenInput):
    directory_path: str


@dataclass
class SavepointData(CodeGenInput):
    name: str
    fields: list[FieldSerialisationData]
    metadata: Optional[list[Metadata]]


@dataclass
class SerialisationInterface:
    init: InitData
    savepoint: list[SavepointData]
