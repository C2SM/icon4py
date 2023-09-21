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
import abc
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Union

from gt4py.next.type_system import type_specifications as ts

from icon4pytools.icon4pygen.bindings.locations import (
    BasicLocation,
    ChainedLocation,
    CompoundLocation,
)


@dataclass(frozen=True)
class FieldIntent:
    inp: bool
    out: bool


class FieldEntity(ABC):
    name: str
    field_type: ts.ScalarKind
    intent: FieldIntent
    has_vertical_dimension: bool
    includes_center: bool
    location: Optional[ChainedLocation | CompoundLocation | BasicLocation]

    @abc.abstractmethod
    def is_sparse(self) -> bool:
        ...

    @abc.abstractmethod
    def is_dense(self) -> bool:
        ...

    @abc.abstractmethod
    def is_compound(self) -> bool:
        ...

    @abc.abstractmethod
    def rank(self) -> int:
        ...

    @abc.abstractmethod
    def get_num_neighbors(self) -> int:
        ...


class OffsetEntity(ABC):
    includes_center: bool
    source: Union[BasicLocation, CompoundLocation]
    target: tuple[BasicLocation, ChainedLocation]

    @abc.abstractmethod
    def is_compound_location(self) -> bool:
        ...

    @abc.abstractmethod
    def get_num_neighbors(self) -> int:
        ...
