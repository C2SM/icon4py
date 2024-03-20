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

from icon4pytools.liskov.codegen.shared.types import CodeGenInput


@dataclass
class InitData(CodeGenInput):
    directory: str
    prefix: str


@dataclass
class Metadata:
    key: str
    value: str


@dataclass
class FieldSerialisationData:
    variable: str
    association: str
    decomposed: bool = False
    device: Optional[str] = "cpu"
    dimension: Optional[list[str]] = None
    typespec: Optional[str] = None
    typename: Optional[str] = None
    ptr_var: Optional[str] = None


@dataclass
class SavepointData(CodeGenInput):
    subroutine: str
    intent: str
    fields: list[FieldSerialisationData]
    metadata: Optional[list[Metadata]]


class ImportData(CodeGenInput):
    ...


@dataclass
class SerialisationCodeInterface:
    Import: ImportData
    Init: InitData
    Savepoint: list[SavepointData]
