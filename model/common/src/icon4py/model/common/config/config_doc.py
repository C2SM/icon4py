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
import textwrap
import types
import typing

import rich.tree
import textual.app
import textual.containers
import textual.reactive
import textual.widget
import textual.widgets.tree
import yaml

from icon4py.model.common import time, type_alias as ta
from icon4py.model.common.config import config_io, options as config_options
from icon4py.model.driver.config import ExperimentConfig


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


def description_from_type(type_hint: RESOLVED) -> str:
    if dataclasses.is_dataclass(type_hint) or isinstance(type_hint, enum.EnumType):
        return type_hint.__doc__ or "??"
    else:
        return "??"


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


@functools.singledispatch
def add_node_to_ttree(node: Record, ttree: textual.widgets.tree.TreeNode) -> None:
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


T = typing.TypeVar("T")


@functools.singledispatch
def examples_for[T](some_type: type[T]) -> typing.Iterator[tuple[T | dict | str, type]]:
    some_type = resolve_type(some_type)
    if some_type is config_io.SharedOptionSet:
        yield (
            config_io.SharedOptionSet(
                options={"<option 1>": "<value 1>", "<option 2>": "<value 2>"},
                consumers=["<section A>", "<section B>"],
            ),
            some_type,
        )
    else:
        try:
            yield some_type(), some_type
        except TypeError:
            if dataclasses.is_dataclass(some_type):
                field_types = typing.get_type_hints(some_type)
                yield (
                    some_type(
                        **{
                            f.name: next(examples_for(resolve_type(field_types[f.name])))[0]
                            for f in dataclasses.fields(some_type)
                        }
                    ),
                    some_type,
                )
            elif some_type is time.AbsoluteTime:
                yield from examples_for_abstime(some_type)
            elif some_type is ta.wpfloat:
                yield from examples_for_wpfloat(some_type)
            else:
                yield "No example found.", str


@examples_for.register
def examples_for_enum(
    some_type: enum.EnumType,
) -> typing.Iterator[tuple[typing.Any, enum.EnumType]]:
    yield from ((e, some_type) for e in some_type)


@examples_for.register
def examples_for_none(some_type: types.NoneType) -> typing.Iterator[tuple[None, types.NoneType]]:
    yield None, some_type


@examples_for.register
def examples_for_union(some_type: types.UnionType) -> typing.Iterator[tuple[None, types.UnionType]]:
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
    yield 0.0, ta.wpfloat


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
        desc = self.query_one("#description", expect_type=textual.widgets.Markdown)
        record = record_from_node(
            typing.cast(Record | ConfigClassContainer | UnionContainer, node.data)
        )
        example_name = (
            record.qualified_name[-1]
            if not record.qualified_name[-1].startswith("<class")
            else record.qualified_name[-2]
        )
        examples = "\n--\n\n".join(
            yaml.dump(  # TODO(ricoh): [c38] make this part of API
                {
                    example_name: config_io.CONV.unstructure(
                        example, unstructure_as=unstructure_type
                    )
                },
                sort_keys=False,
                Dumper=config_io.IndentSequencesDumper,
            )
            for example, unstructure_type in examples_for(record.allowed_type)
        )
        markdown = textwrap.dedent(
            f"""
            ## Description

            {record.option_meta.description}

            ## Info

            - type: {record.allowed_type}
            - default: {record.default}

            ## Examples

            ```yaml
            {{examples}}
            ```
            """
        ).format(examples=examples)
        desc.update(markdown)


class ConfigDocApp(textual.app.App):
    root_class: textual.reactive.var[type | None] = textual.reactive.var(None, init=True)

    def __init__(self, root_class: type | None = None, **kwargs):
        super().__init__(**kwargs)
        self.root_class = root_class if root_class else ExperimentConfig

    def compose(self) -> textual.app.ComposeResult:
        config_widget = ConfigDocWidget()
        config_widget.root_class = self.root_class
        yield config_widget
