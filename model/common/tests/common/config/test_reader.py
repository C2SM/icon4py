# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import pathlib
from typing import Optional

import omegaconf as oc
import pytest

from icon4py.model.common.config import reader as config_reader


@dataclasses.dataclass
class OptionalFoo:
    a: int = oc.MISSING
    b: str | None = None
    c: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Foo:
    a: int
    b: str
    c: list[int]


class Meridiem(enum.Enum):
    AM = 1
    PM = 2


@dataclasses.dataclass
class Time:
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    meridiem: Meridiem = Meridiem.AM


def test_config_reader_validate_default_config() -> None:
    reader = config_reader.ConfigReader(Foo(1, "b", [1, 2, 3]))
    foo = reader.as_type()
    assert foo.a == 1
    assert foo.b == "b"
    assert foo.c == [1, 2, 3]


def test_config_reader_raises_missing_value() -> None:
    reader = config_reader.ConfigReader(Foo)
    with pytest.raises(oc.MissingMandatoryValue):
        reader.as_type()


def test_config_reader_raises_for_missing() -> None:
    reader = config_reader.ConfigReader(OptionalFoo)
    with pytest.raises(oc.MissingMandatoryValue) as e:
        reader.as_type()


def test_config_reader_type_validates() -> None:
    reader = config_reader.ConfigReader(Foo)
    wrong_foo = Foo("a", "b", [1, 2, 3])
    with pytest.raises(oc.ValidationError):
        reader.update(wrong_foo)


def test_config_reader_supports_optional() -> None:
    reader = config_reader.ConfigReader(OptionalFoo(a=3))
    config = reader.as_type()
    assert len(config.c) == 0
    assert config.a == 3
    assert config.b is None


def test_config_reader_config_equals_default_without_update() -> None:
    reader = config_reader.ConfigReader(Foo(1, "b", [1, 2, 3]))
    foo = reader.as_type()
    default = reader.default
    assert reader.default == foo


def test_config_reader_update_from_dataclass() -> None:
    reader = config_reader.ConfigReader(Foo(1, "b", [1, 2]))
    original_config = reader.as_type()
    assert original_config.a == 1
    assert original_config.b == "b"
    assert original_config.c == [1, 2]
    reader.update(Foo(2, "b*", [8, 7]))
    update = reader.as_type()
    assert update.a == 2
    assert update.b == "b*"
    # TODO (@halungge): should Sequences replace or append?, same question for general dicts
    assert update.c == [8, 7]


def test_config_reader_update_from_file() -> None:
    reader = config_reader.ConfigReader(Foo(1, "b", [1, 2]))
    original_config = reader.as_type()
    file = pathlib.Path(__file__).parent.joinpath("foo_update.yaml")
    reader.update(file)
    config = reader.as_type()
    assert config.a == 42
    assert config.b == "this is the update"
    assert config.c == original_config.c


def test_config_enum_parsing_from_value_and_name() -> None:
    reader = config_reader.ConfigReader(Time)
    assert reader.as_type().meridiem == Meridiem.AM
    reader.update({"meridiem": 2})
    assert reader.as_type().meridiem == Meridiem.PM
    reader.update({"meridiem": "AM"})
    assert reader.as_type().meridiem == Meridiem.AM


def test_config_enum_creation() -> None:
    reader = config_reader.ConfigReader(Time)
    file = pathlib.Path(__file__).parent.joinpath("time.yaml")
    reader.update(file)
    config = reader.as_type()
    assert config.hours == 12
    assert config.seconds == 0
    assert config.minutes == 33
    assert config.meridiem == Meridiem.PM
    assert reader.default.meridiem == Meridiem.AM
