# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import typing
from typing import Self


class NotDataclassError(Exception): ...


class MissingConfigOptionAnnotationError(Exception): ...


class MultipleOptionAnnotationsError(Exception): ...


@dataclasses.dataclass(frozen=True)
class ConfigOption:
    description: str

    @classmethod
    def from_type_hint(cls: type[Self], annotated: typing.Any) -> Self:
        """
        Read a ConfigOption from a (pre-evaluated) type annotation.

        Example:

        >>> @dataclasses.dataclass
        >>> class SomeConfig:
        >>>     an_option: typing.Annotated[int, ConfigOption(description="test")] = 0
        >>>
        >>> ConfigOption.from_type_hint(
        ...     typing.get_type_hints(SomeConfig, include_extras=True)["an_option"]
        ... )
        ConfigOption(description='test')
        """
        if not hasattr(annotated, "__metadata__"):
            raise MissingConfigOptionAnnotationError(
                f"'{annotated}' is not an annotated type hint."
            )
        opts = [a for a in annotated.__metadata__ if isinstance(a, cls)]
        if not opts:
            raise MissingConfigOptionAnnotationError(
                f"'{annotated}' is missing a ConfigOption annotation."
            )
        if len(opts) > 1:
            raise MultipleOptionAnnotationsError(
                f"'{annotated}' is annotated with multiple ConfigOptions."
            )
        (opt,) = opts
        return opt

    @classmethod
    def iter_from_config_class(
        cls: type[Self], config_cls: type
    ) -> typing.Iterator[tuple[str, Self]]:
        """
        Iterate over ConfigOption annotations of a given configuration dataclass.

        Example:

        >>> @dataclasses.dataclass
        >>> class SomeConfig:
        >>>     an_option: typing.Annotated[int, ConfigOption(description="first")] = 0
        >>>     other_option: typing.Annotated[int, ConfigOption(description="second")] = 0
        >>>
        >>> dict(ConfigOption.iter_from_config_class(SomeConfig))
        {'an_option': ConfigOption(description='first'), 'other_option': ConfigOption(description='second')}
        """
        if not dataclasses.is_dataclass(config_cls):
            raise NotDataclassError(str(config_cls))
        annotations = typing.get_type_hints(config_cls, include_extras=True)
        for field in dataclasses.fields(config_cls):
            yield field.name, cls.from_type_hint(annotations[field.name])
