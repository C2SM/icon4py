# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
