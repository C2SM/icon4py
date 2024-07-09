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

from abc import ABCMeta, abstractmethod
from typing import Iterator

from gt4py.next.ffront.fbuiltins import Dimension
from icon4py.model.common.dimension import global_dimensions

from icon4pytools.icon4pygen.bindings.codegen.render.location import LocationRenderer


class BasicLocation:
    renderer = LocationRenderer

    @classmethod
    def render_location_type(cls) -> str:
        return cls.renderer.location_type(cls.__name__)


class Cell(BasicLocation):
    def __str__(self) -> str:
        return "C"


class Edge(BasicLocation):
    def __str__(self) -> str:
        return "E"


class Vertex(BasicLocation):
    def __str__(self) -> str:
        return "V"


BASIC_LOCATIONS = {location.__name__: location for location in [Cell, Edge, Vertex]}


def is_valid(nbh_list: list[BasicLocation]) -> bool:
    for i in range(0, len(nbh_list) - 1):  # This doesn't look very pythonic
        if isinstance(type(nbh_list[i]), type(nbh_list[i + 1])):
            return False
    return True


class MultiLocation(metaclass=ABCMeta):
    chain: list[BasicLocation]

    @abstractmethod
    def __str__(self) -> str:
        ...

    def __init__(self, chain: list[BasicLocation]) -> None:
        if is_valid(chain):
            self.chain = chain
        else:
            raise Exception(f"chain {chain} contains two of the same elements in succession")

    def __iter__(self) -> Iterator[BasicLocation]:
        return iter(self.chain)

    def __getitem__(self, item: int) -> BasicLocation:
        return self.chain[item]

    def to_dim_list(self) -> list[Dimension]:
        dims_initials = [key[0] for key in global_dimensions.keys()]
        map_to_dim = {d: list(global_dimensions.values())[d_i] for d_i, d in enumerate(dims_initials)}
        return [map_to_dim[str(c)] for c in self.chain]


class CompoundLocation(MultiLocation):
    def __str__(self) -> str:
        return "".join([str(loc) for loc in self.chain])


class ChainedLocation(MultiLocation):
    def __str__(self) -> str:
        return "2".join([str(loc) for loc in self.chain])
