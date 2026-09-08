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

import pytest
import textual
import textual.widgets

from icon4py.model.common import time
from icon4py.model.common.config import config_doc, config_io, options as config_options


@config_io.register_enum
class TestEnum(enum.Enum):
    """Represents a choice between modes, algorithms, etc."""

    FOO = enum.auto()
    BAR = enum.auto()


@dataclasses.dataclass
class TestMetaNoMeta:
    """Config class to test options with and without description metadata."""

    no_meta: int
    with_meta: typing.Annotated[
        int,
        config_options.ConfigOption(
            description="This is a long-ish description, long enough to wrap in the snapshot."
        ),
    ]


@dataclasses.dataclass
class TestNested:
    """Config class with another one nested under it."""

    nested: typing.Annotated[
        TestMetaNoMeta, config_options.ConfigOption(description="Nested Config")
    ]


@dataclasses.dataclass
class TestNestedNoMeta:
    """Config class with another one nested under it."""

    nested: TestMetaNoMeta


@dataclasses.dataclass
class TestEnumOption:
    """Config class with another one nested under it."""

    choice: TestEnum


@dataclasses.dataclass
class TestEndTime:
    """Config class with an time.EndOfSimulation option"""

    sim_end: time.EndOfSimulation


@dataclasses.dataclass(frozen=True, kw_only=True)
class TestWithShared(config_io.ConfigWithShared):
    foo: TestNested
    bar: TestNestedNoMeta


@pytest.mark.parametrize(
    ("typehint", "check"),
    (
        (int, lambda x: next(x)[0] == 0),
        (float, lambda x: next(x)[0] == 0.0),
        (None, lambda x: next(x)[0] is None),
        (TestEnum, lambda x: tuple(i[0] for i in x) == (TestEnum.FOO, TestEnum.BAR)),
        (time.AbsoluteTime, lambda x: isinstance(next(x)[0], time.AbsoluteTime)),
        (time.RelativeTime, lambda x: isinstance(next(x)[0], time.RelativeTime)),
        (int | float, lambda x: tuple(i[0] for i in x) == (0, 0.0)),
    ),
)
def test_examples_for(typehint, check):
    examples = config_doc.examples_for(typehint)
    assert check(examples)


@pytest.mark.parametrize(
    ("config_class", "key_presses"),
    (
        (TestMetaNoMeta, ("down", "enter")),
        (TestMetaNoMeta, ("down", "down", "enter")),
        (TestNested, ()),
        (TestNested, ("down", "enter")),
        (TestNestedNoMeta, ("down", "enter")),
        (TestEnumOption, ("down", "enter")),
        (TestEndTime, ("down", "enter", "tab", "pagedown")),
        (TestWithShared, ("down", "enter", "tab", "pagedown")),
    ),
    ids=lambda val: val if not isinstance(val, tuple) else "".join(k[0] for k in val),
)
def test_snapshots(snap_compare, config_class, key_presses):
    app = config_doc.ConfigDocApp(config_class)
    assert snap_compare(app, press=key_presses)
