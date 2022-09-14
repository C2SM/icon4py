import tarfile
from xml.etree.ElementInclude import include
from importlib.metadata import metadata
from typing import List, Union, Tuple, Optional
from webbrowser import get
from icon4py.pyutils.stencil_info import StencilInfo, FieldInfo
from icon4py.pyutils.metadata import get_field_infos
from functional.ffront.common_types import ScalarKind
from collections import namedtuple
from eve import Node

Intent = namedtuple("Intent", ["inp", "out"])


class BasicLocation:
    ...


class Cell(BasicLocation):
    def __str__(self):
        return "C"


class Edge(BasicLocation):
    def __str__(self):
        return "E"


class Vertex(BasicLocation):
    def __str__(self):
        return "V"


__BASIC_LOCATIONS__ = {location.__name__: location for location in [Cell, Edge, Vertex]}


def chain_from_str(chain: str):
    _chain_ctor_dispatcher_ = {"E": Edge, "C": Cell, "V": Vertex}
    return [_chain_ctor_dispatcher_[c]() for c in chain]


def is_valid(nbh_list: List[BasicLocation]):
    for i in range(0, len(nbh_list) - 1):  # This doesn't look very pythonic
        if type(nbh_list[i]) == type(nbh_list[i + 1]):
            return False
    return True


class CompoundLocation:
    compound: List[BasicLocation]

    def __str__(self):
        return "".join([str(loc) for loc in self.compound])

    def __init__(self, compound: List[BasicLocation]):
        if is_valid(compound):
            self.compound = compound
        else:
            raise Exception()


class ChainedLocation:
    chain: List[BasicLocation]

    def __str__(self):
        return "2".join([str(loc) for loc in self.chain])

    def __init__(self, chain: List[BasicLocation]):
        if is_valid(chain):
            self.chain = chain
        else:
            raise Exception()


class Offset:
    source: Union[BasicLocation, CompoundLocation]
    target: Tuple[BasicLocation, ChainedLocation]
    includes_center: bool

    def emit_strided_connectivity(self):
        return isinstance(self.source, CompoundLocation)

    def __init__(self, chain: str):
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
    type: ScalarKind

    def is_sparse(self):
        return isinstance(self.location, ChainedLocation)

    def is_dense(self):
        return isinstance(self.location, BasicLocation)

    def __init__(self, name: str, field_info: FieldInfo):
        self.name = str(name)  # why isn't this a str in the first place?
        self.type = field_info.field.type.dtype.kind
        self.intent = Intent(inp=field_info.inp, out=field_info.out)
        self.has_vertical_dimension = any(
            dim.value == "K" for dim in field_info.field.type.dims
        )
        maybe_horizontal_dimension = list(
            filter(lambda dim: dim.value != "K", field_info.field.type.dims)
        )
        self.includes_center = False
        if len(maybe_horizontal_dimension):
            horizontal_dimension = maybe_horizontal_dimension[0].value
            if horizontal_dimension.endswith("O"):
                self.includes_center = True
                horizontal_dimension = horizontal_dimension[:-1]
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


def stencil_info_to_binding_type(stencil_info: StencilInfo):
    chains = stencil_info.connectivity_chains
    fields = get_field_infos(stencil_info.fvprog)

    binding_fields = [Field(name, info) for name, info in fields.items()]
    binding_offsets = [Offset(chain) for chain in chains]

    return (binding_fields, binding_offsets)
