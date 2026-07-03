# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing

import pytest

from icon4py.model.common.config import options


@dataclasses.dataclass
class ConfigClass:
    """A configuration class for testing."""

    choice: typing.Annotated[
        int,
        options.ConfigOption(
            description="A choice of methods.",
            icon_equivalent=options.IconOption(name="isomchce", path=("nested_1", "nested_2")),
        ),
    ]
    flag: typing.Annotated[
        bool,
        options.ConfigOption(
            description="A configuration flag.",
            icon_equivalent=options.IconOption(name="lsomflg", path=(), list_to_value=True),
        ),
    ]
    other: typing.Annotated[int, options.ConfigOption(description="Non-ICON config choice.")] = 5


def test_config_option_from_annotated_type_hint() -> None:
    class TesteeConfig:
        testee: typing.Annotated[int, options.ConfigOption(description="Just for testing.")]

    result = options.ConfigOption.from_type_hint(
        typing.get_type_hints(TesteeConfig, include_extras=True)["testee"]
    )
    assert result


def test_config_option_from_unannotated_type_hint_fails() -> None:
    class TesteeConfig:
        testee: int

    with pytest.raises(options.MissingConfigOptionAnnotationError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["testee"]
        )


def test_config_option_from_wrongly_annotated_type_hint_fails() -> None:
    class TesteeConfig:
        no_option: typing.Annotated[int, "Just for testing."]
        more_than_one_option: typing.Annotated[
            bool, options.ConfigOption(description="foo"), options.ConfigOption(description="bar")
        ]

    with pytest.raises(options.MissingConfigOptionAnnotationError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["no_option"]
        )

    with pytest.raises(options.MultipleOptionAnnotationsError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["more_than_one_option"]
        )


def test_iter_config_options_from_config_class() -> None:
    @dataclasses.dataclass
    class TesteeConfig:
        testee_choice: typing.Annotated[int, options.ConfigOption(description="Just for testing.")]
        testee_flag: typing.Annotated[bool, options.ConfigOption(description="Just for testing.")]

    result = dict(options.ConfigOption.iter_from_config_class(TesteeConfig))

    assert "testee_choice" in result
    assert "testee_flag" in result


def test_iter_pairs_from_icon() -> None:
    result = dict(
        options.iter_pairs_from_icon(
            config_cls=ConfigClass,
            icon_config={
                "nested_1": {"nested_2": {"isomchce": 42}},
                "lsomflg": [False, False, False],
            },
        )
    )

    assert result["choice"] == 42
    assert not result["flag"]


def test_construct_config_from_icon() -> None:
    result = options.construct_config_from_icon(
        config_cls=ConfigClass,
        icon_config={"nested_1": {"nested_2": {"isomchce": 42}}, "lsomflg": [False, False, False]},
        other=3,
    )
    assert result.choice == 42
    assert result.flag is False
    assert result.other == 3


@dataclasses.dataclass
class ConfigClassWithDefaults:
    """A configuration class with defaults for all ICON options, for testing."""

    choice: typing.Annotated[
        int,
        options.ConfigOption(
            description="A choice of methods.",
            icon_equivalent=options.IconOption(name="isomchce", path=("nested",)),
        ),
    ] = 1
    flag: typing.Annotated[
        bool,
        options.ConfigOption(
            description="A configuration flag.",
            icon_equivalent=options.IconOption(name="lsomflg", path=()),
        ),
    ] = True


def test_iter_pairs_from_icon_missing_option_fails() -> None:
    with pytest.raises(KeyError):
        dict(
            options.iter_pairs_from_icon(
                config_cls=ConfigClassWithDefaults, icon_config={"nested": {"isomchce": 42}}
            )
        )


def test_construct_config_from_icon_allow_missing() -> None:
    result = options.construct_config_from_icon(
        config_cls=ConfigClassWithDefaults,
        icon_config={"nested": {"isomchce": 42}},
        allow_missing=True,
    )
    assert result.choice == 42
    assert result.flag is True  # not in the config dict: dataclass default applies
