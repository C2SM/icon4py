# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import textwrap
import typing

import cattrs
import pytest

from icon4py.model.common import time
from icon4py.model.common.config import config_io


@dataclasses.dataclass
class ExampleConfig:
    required_flag: bool
    optional_int: int = 5


@dataclasses.dataclass
class AlternativeConfig:
    unrelated_int: int = 4


@config_io.register_enum
class ExampleEnum(int, enum.Enum):
    FOO = enum.auto()
    BAR = enum.auto()


type CONFIG_UNION = ExampleConfig | AlternativeConfig
config_io.register_config_union(CONFIG_UNION.__value__, {"example": ExampleConfig, "alt": AlternativeConfig})


@dataclasses.dataclass
class UnionConfig:
    union: CONFIG_UNION


def test_read_empty_fails() -> None:
    with pytest.raises(TypeError):
        _ = config_io.read("", ExampleConfig)


def test_read_required_only() -> None:
    config = config_io.read(
        textwrap.dedent(
            """
            required_flag: false
            """
        ),
        ExampleConfig,
    )
    assert config.required_flag is False
    assert config.optional_int == 5


def test_read_all() -> None:
    config = config_io.read(
        textwrap.dedent(
            """
            required_flag: true
            optional_int: 42
            """
        ),
        ExampleConfig,
    )
    assert config.required_flag is True
    assert config.optional_int == 42


def test_read_extra_keys_fails() -> None:
    with pytest.raises(cattrs.ClassValidationError):
        _ = config_io.read(
            textwrap.dedent(
                """
                required_flag: true
                extra_key: "this should trigger an error"
                """
            ),
            ExampleConfig,
        )


def test_read_write_roundtrip() -> None:
    reference = textwrap.dedent(
        """\
        required_flag: false
        optional_int: 3
        """
    )
    assert config_io.write(config_io.read(reference, ExampleConfig)) == reference


def test_write_read_roundtrip() -> None:
    reference = ExampleConfig(True, 6)
    assert config_io.read(config_io.write(reference), ExampleConfig) == reference


def test_roundtrip_abstime() -> None:
    abstime_str = "'2026-07-30T14:41:25'\n"
    abstime = config_io.read(abstime_str, time.AbsoluteTime)
    assert abstime.year == 2026
    assert abstime.second == 25
    assert config_io.write(abstime) == abstime_str


def test_roundtrip_reltime() -> None:
    reltime_str = "300\n...\n"
    reltime = config_io.read(reltime_str, time.RelativeTime)
    assert reltime.seconds == 300
    assert config_io.write(reltime) == reltime_str


@pytest.mark.parametrize(
    ("endtime_str", "check"),
    [
        ("endtime:\n  type: absolute\n  value: '2026-07-30T14:46:00'\n", lambda v: v.minute == 46),
        ("endtime:\n  type: relative\n  value: 50\n", lambda v: v.seconds == 50),
        ("endtime:\n  type: numsteps\n  value: 42\n", lambda v: v == 42),
    ],
)
def test_roundtrip_endtime_abs(
    endtime_str: str, check: typing.Callable[[typing.Any], bool]
) -> None:
    @dataclasses.dataclass
    class EndtimeConfig:
        endtime: time.EndOfSimulation

    config = config_io.read(endtime_str, EndtimeConfig)
    assert check(config.endtime)
    assert config_io.write(config) == endtime_str


def test_roundtrip_enum() -> None:
    enum_str = "foo\n...\n"
    foo = config_io.read(enum_str, ExampleEnum)
    assert foo is ExampleEnum.FOO
    assert config_io.write(foo) == enum_str


@pytest.mark.parametrize(
    ("cfg_str", "reference"),
    (
        (
            textwrap.dedent(
                """\
                union:
                  type: example
                  required_flag: true
                  optional_int: 42
                """
            ),
            ExampleConfig(True, 42)
        ),
        (
            textwrap.dedent(
                """\
                union:
                  type: alt
                  unrelated_int: 7
                """
            ),
            AlternativeConfig(7)
        )

    )
)
def test_roundtrip_config_union(cfg_str: str, reference: CONFIG_UNION) -> None:
    config = config_io.read(cfg_str, UnionConfig)
    assert config == UnionConfig(reference)
    assert config_io.write(config) == cfg_str
