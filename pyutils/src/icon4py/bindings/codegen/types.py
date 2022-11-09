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

from abc import ABC

from functional.ffront.common_types import ScalarKind

from icon4py.bindings.locations import (
    BasicLocation,
    ChainedLocation,
    CompoundLocation,
)


class FieldEntity(ABC):
    location: ChainedLocation | CompoundLocation | BasicLocation | None
    field_type: ScalarKind
    name: str
    has_vertical_dimension: bool

    def rank(self) -> int:
        ...

    def is_sparse(self) -> bool:
        ...

    def is_dense(self) -> bool:
        ...

    def is_compound(self) -> bool:
        ...


class OffsetEntity(ABC):
    includes_center: bool
    target: tuple[BasicLocation, ChainedLocation]

    def is_compound_location(self) -> bool:
        ...

    def is_compound(self) -> bool:
        ...

    def has_vertical_dimension(self) -> bool:
        ...

    def name(self) -> str:
        ...

    def location(self) -> str:
        ...

    def field_type(self) -> ScalarKind:
        ...
