# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Sequence

from icon4pytools.icon4pygen.bindings.codegen.types import OffsetEntity
from icon4pytools.icon4pygen.bindings.locations import BasicLocation, ChainedLocation


@dataclass(frozen=True)
class OffsetRenderer:
    """A class to render Offset attributes for their the respective c++ or f90 bindings."""

    entity: OffsetEntity

    def render_lowercase_shorthand(self) -> str:
        return self.lowercase_shorthand(
            self.entity.is_compound_location(),
            self.entity.includes_center,
            self.entity.target,
        )

    def render_uppercase_shorthand(self) -> str:
        return self.uppercase_shorthand(
            self.entity.is_compound_location(),
            self.entity.includes_center,
            self.entity.target,
        )

    @staticmethod
    def lowercase_shorthand(
        is_compound_location: bool,
        includes_center: bool,
        target: tuple[BasicLocation, ChainedLocation],
    ) -> str:
        if is_compound_location:
            lhs = str(target[0]).lower()
            rhs = "".join([char for char in str(target[1]) if char != "2"]).lower()
            return f"{lhs}2{rhs}" + ("o" if includes_center else "")
        else:
            return "".join([char for char in str(target[1]) if char != "2"]).lower() + (
                "o" if includes_center else ""
            )

    @staticmethod
    def uppercase_shorthand(
        is_compound_location: bool,
        includes_center: bool,
        target: tuple[BasicLocation, ChainedLocation],
    ) -> str:
        if is_compound_location:
            return OffsetRenderer.lowercase_shorthand(
                is_compound_location, includes_center, target
            ).upper()
        else:
            return str(target[1]) + ("O" if includes_center else "")


class GpuTriMeshOffsetRenderer:
    """A helper class to render a GpuTriMeshOffset for the c++ bindings."""

    def __init__(self, offsets: Sequence[OffsetEntity]):
        self.offsets = offsets
        self.has_offsets = True if len(offsets) > 0 else False

    def make_table_vars(self) -> list[str]:
        if not self.has_offsets:
            return []
        unique_offsets = sorted({self._make_table_var(offset) for offset in self.offsets})
        return list(unique_offsets)

    def make_neighbor_tables(self) -> list[str]:
        if not self.has_offsets:
            return []

        unique_locations = sorted(
            {
                (
                    f"{self._make_table_var(offset)}Table = mesh->NeighborTables.at(std::tuple<std::vector<LocationType>, bool>{{"
                    f"{{{', '.join(self._make_location_type(offset))}}}, {1 if offset.includes_center else 0}}});"
                )
                for offset in self.offsets
            }
        )
        return list(unique_locations)

    @staticmethod
    def _make_table_var(offset: OffsetEntity) -> str:
        return f"{offset.target[1].__str__().replace('2', '').lower()}{'o' if offset.includes_center else ''}"

    @staticmethod
    def _make_location_type(offset: OffsetEntity) -> list[str]:
        return [f"LocationType::{loc.render_location_type()}" for loc in offset.target[1].chain]
