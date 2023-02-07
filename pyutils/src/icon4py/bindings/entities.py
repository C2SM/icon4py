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

from typing import Union

from gt4py.eve import Node
from gt4py.next.ffront import program_ast as past
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings.codegen.render.field import FieldRenderer
from icon4py.bindings.codegen.render.offset import OffsetRenderer
from icon4py.bindings.codegen.types import FieldEntity, FieldIntent, OffsetEntity
from icon4py.bindings.exceptions import BindingsTypeConsistencyException
from icon4py.bindings.locations import (
    BASIC_LOCATIONS,
    BasicLocation,
    Cell,
    ChainedLocation,
    CompoundLocation,
    Edge,
    Vertex,
)
from icon4py.bindings.utils import calc_num_neighbors
from icon4py.pyutils.metadata import FieldInfo


def chain_from_str(chain: list[str] | str) -> list[BasicLocation]:
    chain_ctor_dispatcher = {"E": Edge, "C": Cell, "V": Vertex}
    if not all(c in chain_ctor_dispatcher for c in chain):
        raise BindingsTypeConsistencyException(
            f"invalid chain {chain} passed to chain_from_str, chain should only contain letters E,C,V"
        )
    return [chain_ctor_dispatcher[c]() for c in chain]


class Offset(Node, OffsetEntity):
    def __init__(self, chain: str) -> None:
        self.includes_center = self._includes_center(chain)
        self.source = self._handle_source(chain)
        self.target = self._make_target(chain, self.source)
        self.renderer = OffsetRenderer(self)

    def is_compound_location(self) -> bool:
        return isinstance(self.source, CompoundLocation)

    def get_num_neighbors(self) -> int:
        return calc_num_neighbors(self.target[1].to_dim_list(), self.includes_center)

    @staticmethod
    def _includes_center(chain: str) -> bool:
        if chain.endswith("O"):
            return True
        return False

    @staticmethod
    def _handle_source(chain: str) -> Union[BasicLocation, CompoundLocation]:
        if chain.endswith("O"):
            chain = chain[:-1]

        source = chain.split("2")[-1]

        if source in [str(loc()) for loc in BASIC_LOCATIONS.values()]:
            return chain_from_str(source)[0]
        elif all(
            char in [str(loc()) for loc in BASIC_LOCATIONS.values()] for char in source
        ):
            return CompoundLocation(chain_from_str(source))
        else:
            raise BindingsTypeConsistencyException(f"Invalid source {source}")

    @staticmethod
    def _make_target(
        chain: str, source: Union[BasicLocation, CompoundLocation]
    ) -> tuple[BasicLocation, ChainedLocation]:
        if chain.endswith("O"):
            chain = chain[:-1]

        target_0 = chain_from_str(chain.split("2")[0])[0]
        if isinstance(source, CompoundLocation):
            target_1 = ChainedLocation(chain_from_str(str(source)))
        else:
            target_1 = ChainedLocation(chain_from_str("".join(chain).split("2")))

        return target_0, target_1


class Field(Node, FieldEntity):
    def __init__(self, name: str, field_info: FieldInfo) -> None:
        self.name = str(name)
        self.field_type = self._extract_field_type(field_info.field)
        self.intent = FieldIntent(inp=field_info.inp, out=field_info.out)
        self.has_vertical_dimension = self._has_vertical_dimension(field_info.field)
        self.includes_center = False
        self._update_horizontal_location(field_info.field)
        self.renderer = FieldRenderer(self)

    def is_sparse(self) -> bool:
        return self.location is not None and isinstance(self.location, ChainedLocation)

    def is_dense(self) -> bool:
        return self.location is not None and isinstance(self.location, BasicLocation)

    def is_compound(self) -> bool:
        return self.location is not None and isinstance(self.location, CompoundLocation)

    def is_integral(self) -> bool:
        scalar_types = [
            ts.ScalarKind.INT,
            ts.ScalarKind.INT32,
            ts.ScalarKind.INT64,
            ts.ScalarKind.BOOL,
        ]
        return self.field_type in scalar_types

    def rank(self) -> int:
        rank = int(self.has_vertical_dimension) + int(self.location is not None)
        if self.location is not None:
            rank += int(isinstance(self.location, ChainedLocation)) + int(
                isinstance(self.location, CompoundLocation)
            )
        return rank

    def get_num_neighbors(self) -> int:
        if not self.is_sparse():
            raise BindingsTypeConsistencyException(
                "num nbh only defined for sparse fields"
            )
        return calc_num_neighbors(self.location.to_dim_list(), self.includes_center)  # type: ignore

    @staticmethod
    def _extract_field_type(field: past.DataSymbol) -> ts.ScalarKind:
        """Handle extraction of field types for different fields e.g. Scalar."""
        if not isinstance(field.type, ts.FieldType):
            return field.type.kind  # type: ignore
        return field.type.dtype.kind

    @staticmethod
    def _has_vertical_dimension(field: past.DataSymbol) -> bool:
        if not isinstance(field.type, ts.FieldType):
            return False
        return any(dim.value == "K" for dim in field.type.dims)

    def _update_horizontal_location(self, field: past.DataSymbol) -> None:
        self.location = None

        # early abort if field is in fact scalar
        if not isinstance(field.type, ts.FieldType):
            return

        maybe_horizontal_dimension = list(
            filter(lambda dim: dim.value != "K", field.type.dims)
        )

        # early abort if field is vertical
        if not len(maybe_horizontal_dimension):
            return

        # for sparse fields, throw out dense "root" since it's redundant
        horizontal_dimension = (
            maybe_horizontal_dimension[1].value
            if len(maybe_horizontal_dimension) > 1
            else maybe_horizontal_dimension[0].value
        )

        original_horizontal_dimension = horizontal_dimension

        # consume indicator that signals inclusion of center
        if horizontal_dimension.endswith("O"):
            self.includes_center = True
            horizontal_dimension = horizontal_dimension[:-1]

        # actual case distinction of field types
        if horizontal_dimension in [loc for loc in BASIC_LOCATIONS.keys()]:
            self.location = BASIC_LOCATIONS[horizontal_dimension]()
        elif all(len(token) == 1 for token in horizontal_dimension.split("2")):
            self.location = ChainedLocation(
                chain_from_str("".join(horizontal_dimension.split("2")))
            )
        elif all(
            char in [str(loc()) for loc in BASIC_LOCATIONS.values()]
            for char in horizontal_dimension
        ):
            self.location = CompoundLocation(chain_from_str(horizontal_dimension))
        else:
            raise BindingsTypeConsistencyException(
                f"Invalid chain: {original_horizontal_dimension}"
            )
