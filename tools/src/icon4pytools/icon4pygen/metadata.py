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
from typing import Any, Optional, TypeGuard

import icon4py.model.common.dimension
from gt4py import eve
from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.ffront import program_ast as past
from gt4py.next.ffront.decorator import FieldOperator, Program, program
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.runtime import FendefDispatcher
from gt4py.next.type_system import type_specifications as ts
from icon4py.model.common import dimension as dims

from icon4pytools.icon4pygen.bindings.utils import calc_num_neighbors


H_START = "horizontal_start"
H_END = "horizontal_end"
V_START = "vertical_start"
V_END = "vertical_end"

SPECIAL_DOMAIN_MARKERS = [H_START, H_END, V_START, V_END]


@dataclass(frozen=True)
class StencilInfo:
    fendef: itir.FencilDefinition
    fields: dict[str, FieldInfo]
    connectivity_chains: list[eve.concepts.SymbolRef]
    offset_provider: dict


@dataclass(frozen=True)
class FieldInfo:
    field: past.DataSymbol
    inp: bool
    out: bool


@dataclass
class DummyConnectivity(Connectivity):
    """Provides static information to the code generator (`max_neighbors`, `has_skip_values`)."""

    max_neighbors: int
    has_skip_values: bool
    origin_axis: Dimension
    neighbor_axis: Dimension = Dimension("unused")
    index_type: type[int] = int

    def mapped_index(self, cur_index, neigh_index) -> int:
        raise AssertionError("Unreachable")
        return 0


def is_list_of_names(obj: Any) -> TypeGuard[list[past.Name]]:
    return isinstance(obj, list) and all(isinstance(i, past.Name) for i in obj)


def is_name(node: past.Expr) -> TypeGuard[past.Name]:
    return isinstance(node, past.Name)


def is_subscript(node: past.Expr) -> TypeGuard[past.Subscript]:
    return isinstance(node, past.Subscript)


def _ignore_subscript(node: past.Expr) -> past.Name:
    if is_name(node):
        return node
    elif is_subscript(node):
        return node.value
    else:
        raise Exception("Can only have past.Name in output kwargs.")


def _get_field_infos(fvprog: Program) -> dict[str, FieldInfo]:
    """Extract and format the in/out fields from a Program."""
    assert all(
        is_list_of_names(body.args) for body in fvprog.past_stage.past_node.body
    ), "Found unsupported expression in input arguments."
    input_arg_ids = set(arg.id for body in fvprog.past_stage.past_node.body for arg in body.args)  # type: ignore[attr-defined] # Checked in the assert

    out_args = (body.kwargs["out"] for body in fvprog.past_stage.past_node.body)
    output_fields = []
    for out_arg in out_args:
        if isinstance(out_arg, past.TupleExpr):
            output_fields.extend([_ignore_subscript(f) for f in out_arg.elts])
        else:
            output_fields.extend([_ignore_subscript(out_arg)])
    assert all(isinstance(f, past.Name) for f in output_fields)
    output_arg_ids = set(arg.id for arg in output_fields)

    fields: dict[str, FieldInfo] = {
        field_node.id: FieldInfo(
            field=field_node,
            inp=(field_node.id in input_arg_ids),
            out=(field_node.id in output_arg_ids),
        )
        for field_node in fvprog.past_stage.past_node.params
        if field_node.id not in SPECIAL_DOMAIN_MARKERS
    }

    return fields


def _get_domain_arg_ids(fvprog: Program) -> set[Optional[eve.concepts.SymbolRef]]:
    """Collect all argument names that are used within the 'domain' keyword argument."""
    domain_arg_ids = []
    if "domain" in fvprog.past_stage.past_node.body[0].kwargs.keys():
        domain_arg = fvprog.past_stage.past_node.body[0].kwargs["domain"]
        assert isinstance(domain_arg, past.Dict)
        for arg in domain_arg.values_:
            for arg_elt in arg.elts:
                if isinstance(arg_elt, past.Name):
                    domain_arg_ids.append(arg_elt.id)
    return set(domain_arg_ids)


def import_definition(name: str) -> Program | FieldOperator | types.FunctionType:
    """Import a stencil from a given module.

    Note:
        The stencil program and module are assumed to have the same name.
    """
    module_name, member_name = name.split(":")
    program = getattr(importlib.import_module(module_name), member_name)
    return program


def get_fvprog(fencil_def: Program | Any) -> Program:
    match fencil_def:
        case Program():
            fvprog = fencil_def
        case _:
            fvprog = program(fencil_def)

    return fvprog


def provide_offset(offset: str, is_global: bool = False) -> DummyConnectivity | Dimension:
    if offset == dims.Koff.value:
        assert len(dims.Koff.target) == 1
        assert dims.Koff.source == dims.Koff.target[0]
        return dims.Koff.source
    else:
        return provide_neighbor_table(offset, is_global)


def provide_neighbor_table(chain: str, is_global: bool) -> DummyConnectivity:
    """Build an offset provider based on connectivity chain string.

    Connectivity strings must contain one of the following connectivity type identifiers:
    C (cell), E (Edge), V (Vertex) and be separated by a '2' e.g. 'E2V'. If the origin is to
    be included, the string should terminate with O (uppercase o), e.g. 'C2E2CO`.

    Handling of "new" sparse dimensions

    A new sparse dimension may look like C2CE or V2CVEC. In this case, we need to strip the 2
    and pass the tokens after to the algorithm below
    """
    offset = getattr(icon4py.model.common.dimension, chain)
    assert isinstance(offset, FieldOffset)

    # note: this seems really brittle. maybe agree on a keyword to indicate new sparse fields?
    new_sparse_field = any(len(token) > 1 for token in chain.split("2")) and not chain.endswith("O")
    chain = chain.split("2")[1] if new_sparse_field else chain

    skip_values = False
    if is_global and "V" in chain and (chain.count("V") > 1 or not chain.endswith("V")):
        skip_values = True

    include_center = True if chain.count("O") > 0 else False
    dims_initials = [key[0] for key in dims.global_dimensions.keys()]
    map_to_dim = {
        d: list(dims.global_dimensions.values())[d_i] for d_i, d in enumerate(dims_initials)
    }
    location_chain: list[Dimension] = [map_to_dim.get(c) for c in chain if c not in ("2", "O")]  # type: ignore[misc] # type specified

    return DummyConnectivity(
        max_neighbors=calc_num_neighbors(location_chain, include_center),
        has_skip_values=skip_values,
        origin_axis=offset.target[0],
        neighbor_axis=offset.source,
    )


def scan_for_offsets(fvprog: Program) -> list[eve.concepts.SymbolRef]:
    """Scan PAST node for offsets and return a set of all offsets."""
    all_fields_types = (
        fvprog.past_stage.past_node.pre_walk_values()
        .if_isinstance(past.Symbol)
        .getattr("type")
        .if_isinstance(ts.FieldType)
        .to_list()
    )

    all_dims = list(
        set(i.value for j in all_fields_types for i in j.dims if i.kind == DimensionKind.LOCAL)
    )

    all_offset_labels = (
        fvprog.itir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_list()
    )

    # we want to preserve order in the offsets for code generation reproducibility
    sorted_dims = sorted(set(all_offset_labels + all_dims))
    return sorted_dims


def get_stencil_info(
    fencil_def: Program | FieldOperator | types.FunctionType | FendefDispatcher,
    is_global: bool = False,
) -> StencilInfo:
    """Generate StencilInfo dataclass from a fencil definition."""
    fvprog = get_fvprog(fencil_def)
    offsets = scan_for_offsets(fvprog)
    fendef = fvprog.itir

    fields = _get_field_infos(fvprog)

    offset_provider = {}
    connectivity_chains = []
    for offset in offsets:
        offset_provider[offset] = provide_offset(offset, is_global)
        if offset != dims.Koff.value:
            connectivity_chains.append(offset)
    return StencilInfo(fendef, fields, connectivity_chains, offset_provider)
