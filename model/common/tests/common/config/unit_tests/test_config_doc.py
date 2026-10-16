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
import typing
from collections.abc import Callable, Iterator

import pytest
import textual
import textual.widgets

from icon4py.model.common import time
from icon4py.model.common.config import config_doc, config_io, options as config_options


@config_io.register_enum
class ExampleEnum(enum.Enum):
    """Represents a choice between modes, algorithms, etc."""

    FOO = enum.auto()
    BAR = enum.auto()


@dataclasses.dataclass
class ExampleMetaNoMeta:
    """Config class to test options with and without description metadata."""

    no_meta: int
    with_meta: typing.Annotated[
        int,
        config_options.ConfigOption(
            description="This is a long-ish description, long enough to wrap in the snapshot."
        ),
    ]


@dataclasses.dataclass
class NoDocString:
    foo: int


@dataclasses.dataclass
class ExampleNested:
    """Config class with another one nested under it."""

    nested: typing.Annotated[
        ExampleMetaNoMeta, config_options.ConfigOption(description="Nested Config")
    ]


@dataclasses.dataclass
class ExampleNestedNoMeta:
    """Config class with another one nested under it."""

    nested: ExampleMetaNoMeta


@dataclasses.dataclass
class ExampleEnumOption:
    """Config class with another one nested under it."""

    choice: ExampleEnum


@dataclasses.dataclass
class ExampleEndTime:
    """Config class with an time.EndOfSimulation option"""

    sim_end: typing.Annotated[
        time.EndOfSimulation,
        config_options.ConfigOption(description="When the simulation should end"),
    ]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExampleWithShared(config_io.ConfigWithShared):
    foo: ExampleNested
    bar: ExampleNestedNoMeta


@dataclasses.dataclass
class ExampleNoDoc:
    nodoc: NoDocString


@dataclasses.dataclass
class ExampleOptional:
    """Config class with a nested optional config class and no metadata."""

    nested: ExampleEnumOption | None = None


@pytest.mark.parametrize(
    ("typehint", "check"),
    (
        (int, lambda x: next(x)[0] == 0),
        (float, lambda x: next(x)[0] == 0.0),
        (None, lambda x: next(x)[0] is None),
        (ExampleEnum, lambda x: tuple(i[0] for i in x) == (ExampleEnum.FOO, ExampleEnum.BAR)),
        (time.AbsoluteTime, lambda x: isinstance(next(x)[0], time.AbsoluteTime)),
        (time.RelativeTime, lambda x: isinstance(next(x)[0], time.RelativeTime)),
        (int | float, lambda x: tuple(i[0] for i in x) == (0, 0.0)),
    ),
)
def test_examples_for(
    typehint: type, check: Callable[[Iterator[tuple[object, config_doc.RESOLVED]]], bool]
) -> None:
    """
    Check the resulting examples for a given tpye with a checking function also given in the params.
    """
    examples = config_doc.examples_for(typehint)
    assert check(examples)


@pytest.mark.parametrize(
    ("config_class", "key_presses"),
    (
        (ExampleMetaNoMeta, ("down", "enter")),
        (ExampleMetaNoMeta, ("down", "down", "enter")),
        (ExampleNested, ()),
        (ExampleNested, ("down", "enter")),
        (ExampleNestedNoMeta, ("down", "enter")),
        (ExampleEnumOption, ("down", "enter")),
        (ExampleEndTime, ("down", "enter", "tab", "pagedown")),
        (ExampleEndTime, ("down", "enter", "down", "enter", "tab", "pagedown")),
        (ExampleWithShared, ("down", "enter", "tab", "pagedown")),
        (ExampleNoDoc, ("down", "enter")),
        (ExampleOptional, ("down", "enter", "down", "enter")),
    ),
    ids=lambda val: val if not isinstance(val, tuple) else "".join(k[0] for k in val),
)
def test_snapshots(snap_compare: Callable, config_class: type, key_presses: tuple[str]) -> None:
    """
    Compare svg 'screenshots' of a specific TUI app state.

    The state is reached by starting the browser from the given class and
    executing the given key presses.
    """
    app = config_doc.ConfigDocApp(config_class)
    assert snap_compare(app, press=key_presses, terminal_size=(160, 40))
