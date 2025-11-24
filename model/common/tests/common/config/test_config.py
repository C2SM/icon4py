# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import enum
import pathlib

import omegaconf as oc
import pytest
from gt4py.eve import utils as eve_utils

from icon4py.model.common import exceptions, utils
from icon4py.model.common.config import config as common_config


@dataclasses.dataclass
class OptionalFoo:
    a: int = common_config.MISSING
    b: str | None = None
    c: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Foo:
    a: int
    b: str
    c: list[int]


class Meridiem(utils.NamespaceMixin, enum.Enum):
    AM = 1
    PM = 2


def test_namespace():
    namespace = Meridiem.namespace()
    assert namespace.AM == 1
    assert namespace.PM == 2


@dataclasses.dataclass
class Time:
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    meridiem: Meridiem = Meridiem.AM


def test_frozen_namespace_mixin():
    namespace = Meridiem.namespace()
    assert isinstance(namespace, eve_utils.FrozenNamespace)


def test_config_reader_validate_default_config() -> None:
    reader = common_config.ConfigurationHandler(Foo(1, "b", [1, 2, 3]))
    foo = reader.get()
    assert foo.a == 1
    assert foo.b == "b"
    assert foo.c == [1, 2, 3]


def test_config_reader_raises_missing_value() -> None:
    reader = common_config.ConfigurationHandler(Foo)
    with pytest.raises(oc.MissingMandatoryValue):
        reader.get()


def test_config_reader_raises_for_missing() -> None:
    reader = common_config.ConfigurationHandler(OptionalFoo)
    with pytest.raises(oc.MissingMandatoryValue):
        reader.get()


def test_config_reader_type_validates() -> None:
    reader = common_config.ConfigurationHandler(Foo)
    wrong_foo = Foo("a", "b", [1, 2, 3])
    with pytest.raises(exceptions.InvalidConfigError):
        reader.update(wrong_foo)


def test_config_reader_supports_optional() -> None:
    reader = common_config.ConfigurationHandler(OptionalFoo(a=3))
    config = reader.get()
    assert len(config.c) == 0
    assert config.a == 3
    assert config.b is None


def test_config_reader_config_equals_default_without_update() -> None:
    reader = common_config.ConfigurationHandler(Foo(1, "b", [1, 2, 3]))
    foo = reader.get()
    default = reader.get(is_default=True)
    assert foo == default


def test_config_reader_update_from_dataclass() -> None:
    reader = common_config.ConfigurationHandler(Foo(1, "b", [1, 2]))
    original_config = reader.get()
    assert original_config.a == 1
    assert original_config.b == "b"
    assert original_config.c == [1, 2]
    reader.update(Foo(2, "b*", [8, 7]))
    update = reader.get()
    assert update.a == 2
    assert update.b == "b*"
    # TODO (@halungge): should Sequences replace or append?, same question for general dicts
    assert update.c == [8, 7]


def test_config_reader_update_from_file() -> None:
    reader = common_config.ConfigurationHandler(Foo(1, "b", [1, 2]))
    original_config = reader.get()
    file = pathlib.Path(__file__).parent.joinpath("foo_update.yaml")
    reader.update(file)
    config = reader.get()
    assert config.a == 42
    assert config.b == "this is the update"
    assert config.c == original_config.c


def test_configuration_update_from_dict() -> None:
    reader = common_config.ConfigurationHandler(Time(13, 12, 0))
    assert reader.get().minutes == 12
    assert reader.get().seconds == 0
    reader.update(dict(seconds=23, minutes=10))
    assert reader.get().minutes == 10
    assert reader.get().seconds == 23


def test_configuration_get_returns_read_only_dict() -> None:
    reader = common_config.ConfigurationHandler(Foo(a=1, b="foo", c=[1, 2]))
    with pytest.raises(oc.ReadonlyConfigError):
        reader._get(
            format_=common_config.Format.DICT, type_=common_config.ConfigType.CUSTOM, read_only=True
        ).b = "bar"


def test_configuration_get_returns_writable_type() -> None:
    reader = common_config.ConfigurationHandler(Foo(a=1, b="foo", c=[1, 2]))
    cfg = reader.get()
    cfg.b = "bar"
    assert cfg.b == "bar"
    assert reader.get().b == "foo"


def test_config_enum_parsing_from_value_and_name() -> None:
    reader = common_config.ConfigurationHandler(Time)
    assert reader.get().meridiem == Meridiem.AM
    reader.update({"meridiem": 2})
    assert reader.get().meridiem == Meridiem.PM
    reader.update({"meridiem": "AM"})
    assert reader.get().meridiem == Meridiem.AM


def test_config_enum_creation() -> None:
    reader = common_config.ConfigurationHandler(Time())
    file = pathlib.Path(__file__).parent.joinpath("time.yaml")
    reader.update(file)
    config = reader.get()
    assert config.hours == 12
    assert config.seconds == 0
    assert config.minutes == 33
    assert config.meridiem == Meridiem.PM
    assert (
        reader._get(
            type_=common_config.ConfigType.DEFAULT,
            format_=common_config.Format.CLASS,
            read_only=True,
        ).meridiem
        == Meridiem.AM
    )


def test_resolve_or_default():
    value = common_config.resolve_or_else("foo", 42)
    assert oc.II("oc.select:foo, 42") == value


def test_configuration_to_yaml(tmpdir):
    reader = common_config.ConfigurationHandler(
        Time(hours=2, minutes=11, seconds=23, meridiem=Meridiem.PM)
    )
    reader.update(dict(seconds=1))
    fname = tmpdir.join("time_config.yaml")
    reader.to_yaml(fname)
    expected = """hours: 2
meridiem: PM
minutes: 11
seconds: 1
"""
    assert fname.exists()
    assert fname.read_text(encoding="utf-8") == expected


def test_configuration_default_to_yaml(tmpdir):
    reader = common_config.ConfigurationHandler(Time())
    reader.update(dict(seconds=1))
    fname = tmpdir.join("time_default.yaml")
    reader.to_yaml(fname, common_config.ConfigType.DEFAULT)
    expected = """hours: 0
meridiem: AM
minutes: 0
seconds: 0
"""
    assert fname.exists()
    assert fname.read_text(encoding="utf-8") == expected


def test_dtime_resolver():
    handler = common_config.ConfigurationHandler(dict(a=12, b="${dtime:10}"))
    config = handler.get()
    assert config["a"] == 12
    assert config["b"] == datetime.timedelta(seconds=10)


def test_datetime_resolver():
    handler = common_config.ConfigurationHandler(
        dict(a=12, time="${datetime:2021-06-21T12:00:10.000}")
    )
    config = handler._get(
        format_=common_config.Format.DICT, type_=common_config.ConfigType.CUSTOM, read_only=True
    )
    assert config.a == 12
    assert config.time == datetime.datetime(
        year=2021, month=6, day=21, hour=12, minute=0, second=10, microsecond=0
    )


def test_dtime_resolver_in_structured_config():
    @dataclasses.dataclass
    class TimeDiff:
        seconds: datetime.timedelta = dataclasses.field(default=common_config.to_timedelta(10))

    handler = common_config.ConfigurationHandler(TimeDiff())
    config = handler.get()
    assert config.seconds == datetime.timedelta(seconds=10)


def test_datetime_resolver_in_structured_config():
    @dataclasses.dataclass
    class Time:
        time: datetime.datetime = dataclasses.field(
            default=common_config.to_datetime("2021-06-21T12:00:10.000"),
        )

    handler = common_config.ConfigurationHandler(Time())
    config = handler.get()
    assert config.time == datetime.datetime(
        year=2021, month=6, day=21, hour=12, minute=0, second=10, microsecond=0
    )
