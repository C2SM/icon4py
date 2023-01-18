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
from __future__ import annotations

import importlib
import types
from dataclasses import dataclass
from typing import Any, TypeGuard

import eve
from functional.common import Dimension, DimensionKind
from functional.ffront import program_ast as past
from functional.ffront import type_specifications as ts
from functional.ffront.decorator import Program, program
from functional.iterator import ir as itir

from icon4py.common.dimension import CellDim, EdgeDim, Koff, VertexDim
from icon4py.pyutils.exceptions import (
    InvalidConnectivityException,
    MultipleFieldOperatorException,
)
from icon4py.pyutils.icochainsize import IcoChainSize


@dataclass(frozen=True)
class StencilInfo:
    fvprog: Program
    connectivity_chains: list[eve.concepts.SymbolRef]
    offset_provider: dict


@dataclass(frozen=True)
class FieldInfo:
    field: past.DataSymbol
    inp: bool
    out: bool


@dataclass
class DummyConnectivity:
    """Provides static information to the code generator (`max_neighbors`, `has_skip_values`)."""

    max_neighbors: int
    has_skip_values: int
    origin_axis: Dimension

    def mapped_index(_, __) -> int:
        raise AssertionError("Unreachable")
        return 0


def is_list_of_names(obj: Any) -> TypeGuard[list[past.Name]]:
    return isinstance(obj, list) and all(isinstance(i, past.Name) for i in obj)


def _ignore_subscript(node: past.Name | past.Subscript) -> past.Name:
    return node if isinstance(node, past.Name) else node.value


def get_field_infos(fvprog: Program) -> dict[str, FieldInfo]:
    """Extract and format the in/out fields from a Program."""
    assert is_list_of_names(
        fvprog.past_node.body[0].args
    ), "Found unsupported expression in input arguments."
    input_arg_ids = set(arg.id for arg in fvprog.past_node.body[0].args)

    out_arg = fvprog.past_node.body[0].kwargs["out"]
    output_fields = (
        [_ignore_subscript(f) for f in out_arg.elts]
        if isinstance(out_arg, past.TupleExpr)
        else [_ignore_subscript(out_arg)]
    )
    assert all(isinstance(f, past.Name) for f in output_fields)
    output_arg_ids = set(arg.id for arg in output_fields)  # type: ignore

    domain_arg_ids = _get_domain_arg_ids(fvprog)

    fields: dict[str, FieldInfo] = {
        field_node.id: FieldInfo(
            field=field_node,
            inp=(field_node.id in input_arg_ids),
            out=(field_node.id in output_arg_ids),
        )
        for field_node in fvprog.past_node.params
        if field_node.id not in domain_arg_ids
    }

    return fields


def _get_domain_arg_ids(fvprog: Program) -> list | set[str]:
    """Collect all argument names that are used within the 'domain' keyword argument."""
    domain_arg_ids = []
    if "domain" in fvprog.past_node.body[0].kwargs.keys():
        domain_arg = fvprog.past_node.body[0].kwargs["domain"]
        assert isinstance(domain_arg, past.Dict)
        for arg in domain_arg.values_:
            for arg_elt in arg.elts:
                if isinstance(arg_elt, past.Name):
                    domain_arg_ids.append(arg_elt.id)
        domain_arg_ids = set(domain_arg_ids)
    return domain_arg_ids


class StencilImporter:
    """Class which imports a fencil from a fencil import path..

    Args:
        fencil_import_path: Import path to the fencil member in the following format,
            icon4py.<package>.<module>:<member_name>, such for example
            icon4py.atm_dyn_iconam.mo_nh_diffusion_stencil_06:mo_nh_diffusion_stencil_06
    """

    def __init__(self, fencil_import_path: str) -> None:
        self.fencil = self._import_definition(fencil_import_path)
        self.fvprog = self._get_fvprog()

    @staticmethod
    def _import_definition(
        fencil_import_path: str,
    ) -> Program | types.FunctionType:
        """Import a stencil from a given module.

        Note:
            The stencil program and module are assumed to have the same name.
        """
        module_name, member_name = fencil_import_path.split(":")
        fencil = getattr(importlib.import_module(module_name), member_name)
        return fencil

    def _get_fvprog(self) -> Program:
        match self.fencil:
            case Program():
                fvprog = self.fencil
            case _:
                fvprog = program(self.fencil)

        if len(fvprog.past_node.body) > 1:
            raise MultipleFieldOperatorException()

        return fvprog


def get_stencil_info(fvprog: Program) -> StencilInfo:
    """Generate StencilInfo dataclass from a fencil definition."""
    offsets = _scan_for_offsets(fvprog)
    offset_provider = {}
    for offset in offsets:
        offset_provider[offset] = provide_offset(offset)
    connectivity_chains = [offset for offset in offsets if offset != Koff.value]
    return StencilInfo(fvprog, connectivity_chains, offset_provider)


def _scan_for_offsets(fvprog: Program) -> list[eve.concepts.SymbolRef]:
    """Scan PAST node for offsets and return a set of all offsets."""
    all_types = (
        fvprog.past_node.pre_walk_values().if_isinstance(past.Symbol).getattr("type")
    )
    all_field_types = [
        symbol_type
        for symbol_type in all_types
        if isinstance(symbol_type, ts.FieldType)
    ]
    all_dims = set(i for j in all_field_types for i in j.dims)
    all_offset_labels = (
        fvprog.itir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .to_list()
    )
    all_dim_labels = [dim.value for dim in all_dims if dim.kind == DimensionKind.LOCAL]

    # we want to preserve order in the offsets for code generation reproducibility
    sorted_dims = sorted(set(all_offset_labels + all_dim_labels))
    return sorted_dims


def provide_offset(offset: str) -> DummyConnectivity | Dimension:
    if offset == Koff.value:
        assert len(Koff.target) == 1
        assert Koff.source == Koff.target[0]
        return Koff.source
    else:
        return _provide_neighbor_table(offset)


def _provide_neighbor_table(chain: str) -> DummyConnectivity:
    """Build an offset provider based on connectivity chain string.

    Connectivity strings must contain one of the following connectivity type identifiers:
    C (cell), E (Edge), V (Vertex) and be separated by a '2' e.g. 'E2V'. If the origin is to
    be included, the string should terminate with O (uppercase o), e.g. 'C2E2CO`.

    Handling of "new" sparse dimensions

    A new sparse dimension may look like C2CE or V2CVEC. In this case, we need to strip the 2
    and pass the tokens after to the algorithm below
    """
    # note: this seems really brittle. maybe agree on a keyword to indicate new sparse fields?
    new_sparse_field = any(
        len(token) > 1 for token in chain.split("2")
    ) and not chain.endswith("O")
    if new_sparse_field:
        chain = chain.split("2")[1]

    location_chain = []
    include_center = False
    for letter in chain:
        if letter == "C":
            location_chain.append(CellDim)
        elif letter == "E":
            location_chain.append(EdgeDim)
        elif letter == "V":
            location_chain.append(VertexDim)
        elif letter == "O":
            include_center = True
        elif letter == "2":
            pass
        else:
            raise InvalidConnectivityException(location_chain)
    return DummyConnectivity(
        max_neighbors=IcoChainSize.get(location_chain) + include_center,
        has_skip_values=False,
        origin_axis=location_chain[0],
    )
