# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import functools
import types
import typing
from collections.abc import Callable, Iterator

import textual.app
import textual.containers
import textual.reactive
import textual.widget
import textual.widgets.tree
from typing_extensions import _AnnotatedAlias

from icon4py.model.common import time, type_alias as ta
from icon4py.model.common.config import config_io, options as config_options


type HINT = (
    type
    | typing.TypeAliasType
    | types.UnionType
    | types.GenericAlias
    | enum.EnumType
    | _AnnotatedAlias
    | None
)
type RESOLVED = type | types.UnionType | types.GenericAlias | enum.EnumType | None


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


@typing.overload
def resolve_type(type_hint: enum.EnumType) -> enum.EnumType: ...


@typing.overload
def resolve_type(type_hint: type) -> type: ...


@typing.overload
def resolve_type(type_hint: types.UnionType) -> types.UnionType: ...


@typing.overload
def resolve_type(type_hint: types.GenericAlias) -> types.GenericAlias: ...


@typing.overload
def resolve_type(type_hint: None) -> None: ...


@typing.overload
def resolve_type(type_hint: HINT) -> RESOLVED: ...


def resolve_type(type_hint: HINT) -> RESOLVED:
    if type_hint is None:
        return None
    actual_type = type_hint
    if hasattr(type_hint, "__metadata__") and hasattr(type_hint, "__origin__"):
        actual_type = resolve_type(typing.cast(RESOLVED, type_hint.__origin__))
    if isinstance(actual_type, typing.TypeAliasType):
        actual_type = resolve_type(actual_type.__value__)
    if actual_type is None:
        return actual_type
    if (
        hasattr(actual_type, "__args__")
        and hasattr(actual_type, "__name__")
        and actual_type.__name__ == "Final"
    ):
        actual_type = resolve_type(next(iter(actual_type.__args__)))
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
            config_options.ConfigOption.from_type_hint(ctx.current_typehint)
            if hasattr(ctx.current_typehint, "__metadata__")
            else config_options.ConfigOption(description=description_from_type(actual_type))
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


@dataclasses.dataclass(kw_only=True, frozen=True)
class UnionContainer:
    record: Record
    alternatives: tuple[Record | ConfigClassContainer | UnionContainer, ...]


def children_with_type(data_class: RESOLVED) -> typing.Iterator[tuple[dataclasses.Field, type]]:
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
        record=(rec := Record.from_type(type_hint, ctx=ctx)),
        alternatives=tuple(
            node_from_type(
                resolve_type(t),
                ctx=ctx.append_path(
                    name=str(resolve_type(t)),
                    field=ctx.current_field,
                    type_hint=typing.Annotated[t, rec.option_meta],
                ),
            )
            for t in type_hint.__args__
        ),
    )


def description_from_type(type_hint: RESOLVED) -> str:
    if dataclasses.is_dataclass(type_hint) or isinstance(type_hint, enum.EnumType):
        desc = type_hint.__doc__ or "??"
        if desc.startswith(f"{type_hint.__name__}("):
            desc = "??"
    else:
        desc = "??"
    return desc


def node_from_other_type(type_hint: HINT, ctx: TraversalContext) -> Record:
    option_meta = (
        config_options.ConfigOption.from_type_hint(ctx.current_typehint)
        if hasattr(ctx.current_typehint, "__metadata__")
        else config_options.ConfigOption(description=description_from_type(resolve_type(type_hint)))
    )
    return Record(
        qualified_name=ctx.name_path,
        allowed_type=resolve_type(type_hint),
        option_meta=option_meta,
        default=ctx.default,
    )


@functools.singledispatch
def add_node_to_ttree(
    node: Record | ConfigClassContainer | UnionContainer, ttree: textual.widgets.tree.TreeNode
) -> None: ...


@add_node_to_ttree.register
def add_record_to_ttree(node: Record, ttree: textual.widgets.tree.TreeNode) -> None:
    ttree.add_leaf(node.qualified_name[-1], data=node)


@add_node_to_ttree.register
def add_conf_container(node: ConfigClassContainer, ttree: textual.widgets.tree.TreeNode) -> None:
    container = ttree.add(node.record.qualified_name[-1], data=node)
    for child in node.children:
        add_node_to_ttree(child, container)


@add_node_to_ttree.register
def add_union_container(node: UnionContainer, ttree: textual.widgets.tree.TreeNode) -> None:
    union = ttree.add(
        f"{node.record.qualified_name[-1]}: one of the following", data=node, expand=True
    )
    for alt in node.alternatives:
        add_node_to_ttree(alt, union)


def record_from_node(node: Record | ConfigClassContainer | UnionContainer) -> Record:
    match node:
        case Record():
            return node
        case _:
            return node.record


@functools.singledispatch
def examples_for(some_type: HINT) -> typing.Iterator[tuple[object, RESOLVED]]:
    _ = some_type
    yield "No example found.", str


@examples_for.register
def examples_for_type(
    some_type: type, *, try_call: bool = True
) -> typing.Iterator[tuple[object, RESOLVED]]:
    resolved = resolve_type(some_type)
    match resolved:
        case _ if hasattr(resolved, "__examples__") and isinstance(
            resolved.__examples__, types.MethodType
        ):
            yield from resolved.__examples__()
        case Callable() if try_call:
            try:
                yield resolved(), resolved
            except TypeError:
                yield from examples_for_type(some_type, try_call=False)
        case _ if dataclasses.is_dataclass(resolved):
            field_types = typing.get_type_hints(resolved)
            yield (
                resolved(
                    **{
                        f.name: next(examples_for(resolve_type(field_types[f.name])))[0]
                        for f in dataclasses.fields(resolved)
                    }
                ),
                resolved,
            )
        case time.AbsoluteTime:
            yield from examples_for_abstime(resolved)
        case ta.wpfloat:
            yield from examples_for_wpfloat(resolved)
        case _:
            yield "No example found.", str


@examples_for.register
def examples_for_enum(
    some_type: enum.EnumType,
) -> typing.Iterator[tuple[object, enum.EnumType]]:
    enum_values: Iterator[object] = iter(some_type)
    yield from ((e, some_type) for e in enum_values)


@examples_for.register
def examples_for_none(some_type: None) -> typing.Iterator[tuple[None, None]]:
    yield None, some_type


@examples_for.register
def examples_for_union(
    some_type: types.UnionType,
) -> typing.Iterator[tuple[object, types.UnionType]]:
    for allowed_type in some_type.__args__:
        yield from ((ex, some_type) for ex, _ in examples_for(allowed_type))


@examples_for.register
def examples_for_seq(
    some_type: types.GenericAlias,
) -> typing.Iterator[tuple[list, types.GenericAlias]]:
    if some_type.__name__ == "list":
        yield [], some_type
        for arg in some_type.__args__:
            e, _ = next(examples_for(arg))
            yield [e], some_type


def examples_for_abstime(
    some_type: type[time.AbsoluteTime],
) -> typing.Iterator[tuple[time.AbsoluteTime, type[time.AbsoluteTime]]]:
    yield time.AbsoluteTime(year=2026, month=1, day=1, hour=0, minute=0, second=0), some_type


def examples_for_wpfloat(
    some_type: type[ta.wpfloat],
) -> typing.Iterator[tuple[float, type[ta.wpfloat]]]:
    yield 0.0, some_type


@dataclasses.dataclass
class NoRoot:
    """
    You are seeing this because something went wrong and the configuration browser
    was not initialized properly!
    """


class ConfigDocWidget(textual.widget.Widget):
    """Broswe config options."""

    root_class: textual.reactive.var[type | None] = textual.reactive.var(None, init=True)

    def compose(self: typing.Self) -> textual.app.ComposeResult:
        with textual.containers.Horizontal():
            with textual.containers.VerticalScroll(id="left"):
                yield textual.widgets.Tree[Record | ConfigClassContainer | UnionContainer](
                    "Icon4Py Config File", id="tree"
                )
            with textual.containers.VerticalScroll(id="right"):
                yield textual.widgets.Markdown(id="description")

    def on_mount(self: typing.Self) -> None:
        self.root_class = self.root_class or NoRoot
        self.query_one("#left").styles.width = "1fr"
        self.query_one("#right").styles.width = "2fr"
        tree: textual.widgets.Tree[Record | ConfigClassContainer | UnionContainer] = self.query_one(
            "#tree", expect_type=textual.widgets.Tree
        )
        tree.root.expand()
        tree.auto_expand = False
        tree.root.data = Record(
            qualified_name=("Config File",),
            option_meta=config_options.ConfigOption(description=str(self.root_class.__doc__)),
            allowed_type=self.root_class,
        )
        tree.root.label = tree.root.data.qualified_name[-1]
        doctree = node_from_dataclass(
            self.root_class,
            ctx=TraversalContext(
                name_path=(), current_field=None, current_typehint=self.root_class
            ),
        )
        for child in doctree.children:
            add_node_to_ttree(child, tree.root)

        tree.select_node(tree.root)
        tree.focus()

    def on_tree_node_selected(
        self: typing.Self, message: textual.widgets.Tree.NodeSelected
    ) -> None:
        node = message.node
        node.expand()
        info = self.query_one("#description", expect_type=textual.widgets.Markdown)
        record = record_from_node(
            typing.cast(Record | ConfigClassContainer | UnionContainer, node.data)
        )
        example_name = (
            record.qualified_name[-1]
            if not record.qualified_name[-1].startswith("<class")
            else record.qualified_name[-2]
        )
        examples = "\n--\n\n".join(
            config_io.write_yaml_str(
                {example_name: config_io.CONV.unstructure(example, unstructure_as=unstructure_type)}
            )
            for example, unstructure_type in examples_for(record.allowed_type)
        )
        desc = record.option_meta.description
        markdown = "\n".join(
            (
                "## Description",
                "",
                f"{desc}",
                "",
                "## Info",
                "",
                f"- type: {record.allowed_type}",
                f"- default: {record.default}",
                f"- can{' ' if record.optional else ' not '}be omitted",
                "",
                "## Examples",
                "",
                "```yaml",
                f"{examples}",
                "```",
            )
        )
        info.update(markdown)


class ConfigDocApp(textual.app.App):
    root_class: textual.reactive.var[type | None] = textual.reactive.var(None, init=True)

    def __init__(self, root_class: type | None = None, **kwargs: typing.Any):
        super().__init__(**kwargs)
        self.root_class = root_class

    def compose(self) -> textual.app.ComposeResult:
        config_widget = ConfigDocWidget()
        config_widget.root_class = self.root_class
        yield config_widget
