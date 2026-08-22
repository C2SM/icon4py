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
config_io.register_config_union(
    CONFIG_UNION.__value__, {"example": ExampleConfig, "alt": AlternativeConfig}
)


@dataclasses.dataclass
class UnionConfig:
    union: CONFIG_UNION


@dataclasses.dataclass
class EndtimeConfig:
    endtime: time.EndOfSimulation


def test_read_yaml_str_empty_fails() -> None:
    with pytest.raises(TypeError):
        _ = config_io.read_yaml_str("", ExampleConfig)


def test_read_yaml_str_required_only() -> None:
    config = config_io.read_yaml_str(
        textwrap.dedent(
            """
            required_flag: false
            """
        ),
        ExampleConfig,
    )
    assert config.required_flag is False
    assert config.optional_int == 5


def test_read_yaml_str_all() -> None:
    config = config_io.read_yaml_str(
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


def test_read_yaml_str_extra_keys_fails() -> None:
    with pytest.raises(cattrs.ClassValidationError):
        _ = config_io.read_yaml_str(
            textwrap.dedent(
                """
                required_flag: true
                extra_key: "this should trigger an error"
                """
            ),
            ExampleConfig,
        )


def test_read_yaml_str_write_yaml_str_roundtrip() -> None:
    reference = textwrap.dedent(
        """\
        required_flag: false
        optional_int: 3
        """
    )
    assert config_io.write_yaml_str(config_io.read_yaml_str(reference, ExampleConfig)) == reference


def test_write_yaml_str_read_yaml_str_roundtrip() -> None:
    reference = ExampleConfig(True, 6)
    assert config_io.read_yaml_str(config_io.write_yaml_str(reference), ExampleConfig) == reference


@pytest.mark.parametrize(
    ("input_str", "config_type", "reference"),
    (
        (
            "'2026-07-30T14:41:25'\n",
            time.AbsoluteTime,
            time.AbsoluteTime(year=2026, month=7, day=30, hour=14, minute=41, second=25),
        ),
        ("300.0\n...\n", time.RelativeTime, time.RelativeTime(seconds=300)),
        (
            "endtime:\n  type: absolute\n  value: '2026-07-30T14:41:46'\n",
            EndtimeConfig,
            EndtimeConfig(
                time.AbsoluteTime(year=2026, month=7, day=30, hour=14, minute=41, second=46)
            ),
        ),
        (
            "endtime:\n  type: relative\n  value: 50.0\n",
            EndtimeConfig,
            EndtimeConfig(time.RelativeTime(seconds=50)),
        ),
        ("endtime:\n  type: numsteps\n  value: 42\n", EndtimeConfig, EndtimeConfig(42)),
        ("foo\n...\n", ExampleEnum, ExampleEnum.FOO),
        (
            textwrap.dedent(
                """\
                union:
                  type: example
                  required_flag: true
                  optional_int: 42
                """
            ),
            UnionConfig,
            UnionConfig(ExampleConfig(True, 42)),
        ),
        (
            textwrap.dedent(
                """\
                union:
                  type: alt
                  unrelated_int: 7
                """
            ),
            UnionConfig,
            UnionConfig(AlternativeConfig(7)),
        ),
    ),
)
def test_roundtrip_customized_type(
    input_str: str,
    config_type: type[time.AbsoluteTime | time.RelativeTime | time.EndOfSimulation],
    reference: time.AbsoluteTime | time.RelativeTime,
) -> None:
    read_value = config_io.read_yaml_str(input_str, config_type)
    assert read_value == reference
    assert config_io.write_yaml_str(read_value) == input_str
