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

from collections import namedtuple
from typing import List, Optional, Tuple, Union

from eve import Node
from functional.ffront import program_ast as past
from functional.ffront.common_types import FieldType, ScalarKind
from functional.ffront.fbuiltins import Dimension

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.pyutils.icochainsize import IcoChainSize
from icon4py.pyutils.metadata import FieldInfo, StencilInfo, get_field_infos


Intent = namedtuple("Intent", ["inp", "out"])

BUILTIN_TO_ISO_C_TYPE = {
    ScalarKind.FLOAT64: "real(c_double)",
    ScalarKind.FLOAT32: "real(c_float)",
    ScalarKind.BOOL: "logical(c_int)",
    ScalarKind.INT32: "c_int",
    ScalarKind.INT64: "c_long",
}

BUILTIN_TO_CPP_TYPE = {
    ScalarKind.FLOAT64: "double",
    ScalarKind.FLOAT32: "float",
    ScalarKind.BOOL: "int",
    ScalarKind.INT32: "int",
    ScalarKind.INT64: "long",
}


class BasicLocation:
    ...


class Cell(BasicLocation):
    def __str__(self) -> str:
        return "C"

    @staticmethod
    def location_type() -> str:
        return "Cells"


class Edge(BasicLocation):
    def __str__(self) -> str:
        return "E"

    @staticmethod
    def location_type() -> str:
        return "Edges"


class Vertex(BasicLocation):
    def __str__(self) -> str:
        return "V"

    @staticmethod
    def location_type() -> str:
        return "Vertices"


__BASIC_LOCATIONS__ = {location.__name__: location for location in [Cell, Edge, Vertex]}


def chain_from_str(chain: str) -> list[BasicLocation]:
    _chain_ctor_dispatcher_ = {"E": Edge, "C": Cell, "V": Vertex}
    return [_chain_ctor_dispatcher_[c]() for c in chain]


def is_valid(nbh_list: List[BasicLocation]) -> bool:
    for i in range(0, len(nbh_list) - 1):  # This doesn't look very pythonic
        if isinstance(type(nbh_list[i]), type(nbh_list[i + 1])):
            return False
    return True


class CompoundLocation:
    compound: List[BasicLocation]

    def __str__(self) -> str:
        return "".join([str(loc) for loc in self.compound])

    def __init__(self, compound: List[BasicLocation]) -> None:
        if is_valid(compound):
            self.compound = compound
        else:
            raise Exception()


class ChainedLocation:
    chain: List[BasicLocation]

    def __str__(self) -> str:
        return "2".join([str(loc) for loc in self.chain])

    def __init__(self, chain: List[BasicLocation]) -> None:
        if is_valid(chain):
            self.chain = chain
        else:
            raise Exception()

    def __iter__(self):
        return iter(self.chain)

    def __getitem__(self, item: int) -> BasicLocation:
        return self.chain[item]

    def to_dim_list(self) -> list[Dimension]:
        map_to_dim = {Cell: CellDim, Edge: EdgeDim, Vertex: VertexDim}
        return [map_to_dim[c.__class__] for c in self.chain]


class Offset(Node):
    source: Union[BasicLocation, CompoundLocation]
    target: Tuple[BasicLocation, ChainedLocation]
    includes_center: bool = False

    def emit_strided_connectivity(self) -> bool:
        return isinstance(self.source, CompoundLocation)

    # todo: these shorthands should be improved after regression passes
    def render_lc_shorthand(self) -> str:
        if self.emit_strided_connectivity():
            lhs = str(self.target[0]).lower()
            rhs = "".join([char for char in str(self.target[1]) if char != "2"]).lower()
            return f"{lhs}2{rhs}"
        else:
            return "".join(
                [char for char in str(self.target[1]) if char != "2"]
            ).lower()

    def render_uc_shorthand(self) -> str:
        if self.emit_strided_connectivity():
            return self.render_lc_shorthand().upper()
        else:
            return str(self.target[1])

    def num_nbh(self) -> int:
        return IcoChainSize.get(self.target[1].to_dim_list()) + self.includes_center

    def __init__(self, chain: str) -> None:
        self.includes_center = False
        if chain.endswith("O"):
            self.includes_center = True
            chain = chain[:-1]

        source = chain.split("2")[-1]

        if source in [str(loc()) for loc in __BASIC_LOCATIONS__.values()]:
            self.source = chain_from_str(source)[0]
        elif all(
            char in [str(loc()) for loc in __BASIC_LOCATIONS__.values()]
            for char in source
        ):
            self.source = CompoundLocation(chain_from_str(source))
        else:
            raise Exception("Invalid Source")

        target_0 = chain_from_str(chain.split("2")[0])[0]
        if isinstance(self.source, CompoundLocation):
            target_1 = ChainedLocation(chain_from_str(str(source)))
        else:
            target_1 = ChainedLocation(chain_from_str("".join(chain).split("2")))

        self.target = (target_0, target_1)


class Field(Node):
    location: Optional[Union[BasicLocation, CompoundLocation, ChainedLocation]]
    has_vertical_dimension: bool
    includes_center: bool
    name: str
    intent: Intent
    field_type: ScalarKind

    def is_sparse(self) -> bool:
        return isinstance(self.location, ChainedLocation)

    def is_dense(self) -> bool:
        return isinstance(self.location, BasicLocation)

    def is_compound(self) -> bool:
        return isinstance(self.location, CompoundLocation)

    def ctype(self, binding_type: str) -> str:
        match binding_type:
            case "f90":
                return BUILTIN_TO_ISO_C_TYPE[self.field_type]
            case "c++":
                return BUILTIN_TO_CPP_TYPE[self.field_type]

    def rank(self) -> int:
        rank = int(self.has_vertical_dimension) + int(self.location is not None)
        if self.location is not None:
            rank += int(isinstance(self.location, ChainedLocation)) + int(
                isinstance(self.location, CompoundLocation)
            )
        return rank

    def render_pointer(self) -> str:
        return "" if self.rank() == 0 else "*"

    def ranked_dim_string(self) -> str:
        return (
            "dimension(" + ",".join([":"] * self.rank()) + ")"
            if self.rank() != 0
            else "value"
        )

    def dim_string(self) -> str:
        return "dimension(*)" if self.rank() != 0 else "value"

    def stride_type(self) -> str:
        _strides = {"E": "EdgeStride", "C": "CellStride", "V": "VertexStride"}
        if self.is_dense():
            return _strides[str(self.location)]
        elif self.is_sparse():
            return _strides[str(self.location[0])]
        else:
            raise Exception("stride type called on compound location or scalar")

    def serialise_func(self) -> str:
        _serializers = {
            "E": "serialize_dense_edges",
            "C": "serialize_dense_cells",
            "V": "serialize_dense_verts",
        }
        return _serializers[str(self.location)]

    def mesh_type(self) -> str:
        _mesh_types = {"E": "EdgeStride", "C": "CellStride", "V": "VertexStride"}
        return _mesh_types[str(self.location)]

    def render_sid(self) -> str:
        if self.is_sparse():
            raise Exception("can not render sid of sparse field")

        if self.rank() == 0:
            raise Exception("can not render sid of a scalar")

        values_str = (
            "1"
            if self.rank() == 1 or self.is_compound()
            else f"1, mesh_.{self.stride_type()}"
        )
        return f"get_sid({self.name}, gridtools::hymap::keys<unstructured::dim::vertical>::make_values({values_str}))"

    def num_nbh(self) -> int:
        if not self.is_sparse():
            raise Exception("num nbh only defined for sparse fields")

        return IcoChainSize.get(self.location.to_dim_list()) + int(self.includes_center)

    def __init__(self, name: str, field_info: FieldInfo) -> None:
        self.name = str(name)  # why isn't this a str in the first place?
        self.field_type = self._extract_field_type(field_info.field)
        self.intent = Intent(inp=field_info.inp, out=field_info.out)
        self.has_vertical_dimension = self._has_vertical_dimension(field_info.field)
        self.includes_center = False
        self.location = None
        self._get_horizontal_dimension_and_update_location = (
            self._get_horizontal_dimension_and_update_location(field_info.field)
        )

    @staticmethod
    def _extract_field_type(field: past.DataSymbol) -> ScalarKind:
        """Handle extraction of field types for different fields e.g. Scalar."""
        if not isinstance(field.type, FieldType):
            return field.type.kind
        return field.type.dtype.kind

    @staticmethod
    def _has_vertical_dimension(field: past.DataSymbol) -> bool:
        if not isinstance(field.type, FieldType):
            return False
        return any(dim.value == "K" for dim in field.type.dims)

    def _get_horizontal_dimension_and_update_location(self, field: past.DataSymbol):
        if not isinstance(field.type, FieldType):
            return None

        maybe_horizontal_dimension = list(
            filter(lambda dim: dim.value != "K", field.type.dims)
        )

        # vertical field or scalar
        if not len(maybe_horizontal_dimension):
            return None

        # for sparse fields, throw out dense "root" since it's redundant
        horizontal_dimension = (
            maybe_horizontal_dimension[1].value
            if len(maybe_horizontal_dimension) > 1
            else maybe_horizontal_dimension[0].value
        )

        # consume indicator that signals inclusion of center
        if horizontal_dimension.endswith("O"):
            self.includes_center = True
            horizontal_dimension = horizontal_dimension[:-1]

        # actual case distinction of field types
        if horizontal_dimension in [loc for loc in __BASIC_LOCATIONS__.keys()]:
            self.location = __BASIC_LOCATIONS__[horizontal_dimension]()
        elif all(len(token) == 1 for token in horizontal_dimension.split("2")):
            self.location = ChainedLocation(
                chain_from_str("".join(horizontal_dimension.split("2")))
            )
        elif all(
            char in [str(loc()) for loc in __BASIC_LOCATIONS__.values()]
            for char in horizontal_dimension
        ):
            self.location = CompoundLocation(chain_from_str(horizontal_dimension))
        else:
            raise Exception("invalid chain")
        return horizontal_dimension


def stencil_info_to_binding_type(
    stencil_info: StencilInfo,
) -> tuple[list[Field], List[Offset]]:
    chains = stencil_info.connectivity_chains
    fields = get_field_infos(stencil_info.fvprog)
    binding_fields = [Field(name, info) for name, info in fields.items()]
    binding_offsets = [Offset(chain) for chain in chains]
    return binding_fields, binding_offsets
