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
            bool,
            options.ConfigOption(description="foo"),
            options.ConfigOption(description="bar"),
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
