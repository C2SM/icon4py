# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import types
import typing

import rich.tree

from icon4py.model.common.config import options as config_options


type HINT = type | typing.TypeAliasType | types.UnionType
type RESOLVED = type | types.UnionType


@dataclasses.dataclass(kw_only=True, frozen=True)
class TraversalContext:
    name_path: tuple[str, ...]
    current_field: dataclasses.Field | None
    current_typehint: HINT

    def append_path(
        self: typing.Self,
        name: str,
        field: dataclasses.Field | None,
        type_hint: HINT,
    ) -> typing.Self:
        return self.__class__(
            name_path=(*self.name_path, name), current_field=field, current_typehint=type_hint
        )

    @property
    def default(self: typing.Self) -> typing.Any:
        if not self.current_field:
            return MissingDefault()
        result = (
            self.current_field.default
            if not isinstance(self.current_field.default, dataclasses._MISSING_TYPE)
            else MissingDefault()
        )
        if not isinstance(self.current_field.default_factory, dataclasses._MISSING_TYPE):
            result = self.current_field.default_factory()
        return result


@dataclasses.dataclass(kw_only=True, frozen=True)
class MissingDefault:
    def __str__(self: typing.Self) -> str:
        return "No default value."

    def __repr__(self: typing.Self) -> str:
        return str(self)


def resolve_type(type_hint: HINT) -> RESOLVED:
    actual_type = type_hint
    if hasattr(type_hint, "__metadata__") and hasattr(type_hint, "__origin__"):
        actual_type = resolve_type(typing.cast(RESOLVED, type_hint.__origin__))
    if isinstance(actual_type, typing.TypeAliasType):
        actual_type = resolve_type(actual_type.__value__)
    return actual_type


@dataclasses.dataclass(kw_only=True, frozen=True)
class Record:
    qualified_name: tuple[str, ...]
    option_meta: config_options.ConfigOption
    default: typing.Any = MissingDefault()
    allowed_type: RESOLVED

    @property
    def optional(self: typing.Self) -> bool:
        return not isinstance(self.default, MissingDefault)

    @classmethod
    def from_type(
        cls: type[typing.Self], type_hint: RESOLVED, ctx: TraversalContext
    ) -> typing.Self:
        actual_type = resolve_type(type_hint)
        option_meta = (
            config_options.ConfigOption.from_type_hint(type_hint)
            if hasattr(type_hint, "__metadata__")
            else config_options.ConfigOption(description="??")
        )
        return cls(
            qualified_name=ctx.name_path,
            allowed_type=actual_type,
            option_meta=option_meta,
            default=ctx.default,
        )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConfigClassContainer:
    record: Record
    children: tuple[Record | ConfigClassContainer | UnionContainer, ...]

    @classmethod
    def children_from_dataclass(
        cls: type[typing.Self], data_class: HINT, ctx: TraversalContext
    ) -> typing.Iterator[typing.Self | Record | UnionContainer]:
        assert dataclasses.is_dataclass(data_class)
        for field, field_type in children_with_type(data_class):
            resolved = resolve_type(field_type)
            new_ctx = ctx.append_path(field.name, field=field, type_hint=field_type)
            if dataclasses.is_dataclass(field_type):
                yield cls.from_type(resolved, ctx=new_ctx)
            elif isinstance(resolved, types.UnionType):
                yield UnionContainer.from_type(resolved, ctx=new_ctx)
            else:
                yield Record.from_type(resolved, ctx=new_ctx)

    @classmethod
    def from_type(
        cls: type[typing.Self], data_class: RESOLVED, ctx: TraversalContext
    ) -> typing.Self:
        return cls(
            record=Record.from_type(data_class, ctx),
            children=tuple(cls.children_from_dataclass(data_class, ctx)),
        )


@dataclasses.dataclass(kw_only=True, frozen=True)
class UnionContainer:
    record: Record
    alternatives: tuple[Record | ConfigClassContainer | UnionContainer, ...]

    @classmethod
    def from_type(
        cls: type[typing.Self], type_hint: types.UnionType, ctx: TraversalContext
    ) -> typing.Self:
        actual_type = resolve_type(type_hint)
        assert isinstance(actual_type, types.UnionType)
        alts: list[Record | ConfigClassContainer | UnionContainer] = []
        for t in actual_type.__args__:
            if dataclasses.is_dataclass(typing.cast(type, t)):
                alts.append(ConfigClassContainer.from_type(t, ctx=ctx))
            else:
                alts.append(Record.from_type(t, ctx))
        return cls(record=Record.from_type(type_hint, ctx), alternatives=tuple(alts))


def selfdoc(config_class: type) -> rich.tree.Tree:
    ir = node_from_type(
        config_class,
        ctx=TraversalContext(
            name_path=(config_class.__name__,), current_field=None, current_typehint=type
        ),
    )
    return tree_from_node(ir)


def children_with_type(data_class: HINT) -> typing.Iterator[tuple[dataclasses.Field, type]]:
    assert dataclasses.is_dataclass(data_class)
    annotations = typing.get_type_hints(data_class, include_extras=True)
    for field in dataclasses.fields(data_class):
        yield field, annotations[field.name]


def node_from_type(
    type_hint: HINT, ctx: TraversalContext
) -> ConfigClassContainer | Record | UnionContainer:
    resolved = resolve_type(type_hint)
    if dataclasses.is_dataclass(resolved):
        return node_from_dataclass(resolved, ctx)
    elif isinstance(resolved, types.UnionType):
        return node_from_union(resolved, ctx)
    else:
        return node_from_other_type(resolved, ctx)


def node_from_dataclass(type_hint: RESOLVED, ctx: TraversalContext) -> ConfigClassContainer:
    return ConfigClassContainer(
        record=Record.from_type(type_hint, ctx=ctx),
        children=tuple(
            node_from_type(resolve_type(t), ctx=ctx.append_path(name=f.name, field=f, type_hint=t))
            for f, t in children_with_type(type_hint)
        ),
    )


def node_from_union(type_hint: types.UnionType, ctx: TraversalContext) -> UnionContainer:
    return UnionContainer(
        record=Record.from_type(type_hint, ctx=ctx),
        alternatives=tuple(
            node_from_type(
                resolve_type(t),
                ctx=ctx.append_path(
                    name=str(resolve_type(t)), field=ctx.current_field, type_hint=t
                ),
            )
            for t in type_hint.__args__
        ),
    )


def node_from_other_type(type_hint: HINT, ctx: TraversalContext) -> Record:
    option_meta = (
        config_options.ConfigOption.from_type_hint(type_hint)
        if hasattr(type_hint, "__metadata__")
        else config_options.ConfigOption(description="??")
    )
    return Record(
        qualified_name=ctx.name_path,
        allowed_type=resolve_type(type_hint),
        option_meta=option_meta,
        default=ctx.default,
    )


@functools.singledispatch
def tree_from_node(node: Record | UnionContainer | ConfigClassContainer) -> rich.tree.Tree:
    _ = node
    return rich.tree.Tree("ERROR")


@tree_from_node.register
def tree_from_record(node: Record) -> rich.tree.Tree:
    result = rich.tree.Tree(node.qualified_name[-1])
    info = result.add("INFO")
    info.add(f"description: {node.option_meta.description}")
    info.add(f"required: {not node.optional}")
    info.add(f"default: {node.default}")
    info.add("units: NOT IMPLEMENTED YET")
    info.add(f"type: {node.allowed_type}")
    info.add("syntax: NOT IMPLEMENTED YET")
    return result


@tree_from_node.register
def tree_from_conf_container(node: ConfigClassContainer) -> rich.tree.Tree:
    result = tree_from_node(node.record)
    for child in node.children:
        result.add(tree_from_node(child))
    return result


@tree_from_node.register
def tree_from_union_container(node: UnionContainer) -> rich.tree.Tree:
    result = tree_from_node(node.record)
    info = next(n for n in result.children if n.label == "INFO")
    types_node = info.add("allowed types:")
    for alternative in node.alternatives:
        types_node.add(tree_from_node(alternative))
    return result
