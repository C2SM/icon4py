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

from icon4py.model.common.utils import fortran_config


class NotDataclassError(Exception): ...


class MissingConfigOptionAnnotationError(Exception): ...


class MultipleOptionAnnotationsError(Exception): ...


class NotRead: ...


@dataclasses.dataclass
class IconOption:
    """
    Describe how to find the equivalent option in ICON namelists.
    """

    #: option name
    name: str
    #: full path through nested namelist sections
    path: tuple[str, ...]
    list_to_value: bool = False
    read_from_icon: bool = True
    converter: typing.Callable[[typing.Any], typing.Any] | None = None

    def convert(
        self: Self,
        icon_config: dict,
        fallback_converter: typing.Callable[[typing.Any], typing.Any] | None,
    ) -> typing.Any | type[NotRead]:
        """
        Convert from ICON namelist value.

        The `fallback_converter` parameter is usually the type of the dataclass field annotated with this instance.

        Examples:

            >>> icon_opt = IconOption(name="a", path=())
            >>> icon_opt.convert({"a": "1"}, fallback_converter=int)
            1

            >>> icon_opt = IconOption(name="b", path=(), converter=lambda v: v + 1)
            >>> icon_opt.convert({"b": 1}, fallback_converter=int)
            2

            >>> IconOption(name="c", path=(), read_from_icon=False).convert(
            ...     {"c": 3}, fallback_converter=int
            ... )
            NotRead
        """
        if self.read_from_icon:
            data = icon_config
            for subsection in self.path:
                data = data[subsection]
            raw_value = data[self.name]
            de_listified = (
                fortran_config.list_to_value(raw_value) if self.list_to_value else raw_value
            )
            converter = self.converter or fallback_converter
            return converter(de_listified) if converter else de_listified
        else:
            return NotRead


@dataclasses.dataclass
class IconMultiOption:
    """
    Merge multiple ICON config options into one ICON4Py option.

    Individual options apply their own converter first if they have one.
    They then pass the result into the multi-option's '.converter' as kwargs,
    with keys corresponding to their '.name' attribute.
    """

    options: list[IconOption]
    converter: typing.Callable[..., typing.Any]

    def convert(self: Self, icon_config: dict, fallback_converter: type) -> typing.Any:
        """
        Merge and convert from multiple ICON namelist values

        Examples:

            >>> simple = IconMultiOption(
            ...     options=[IconOption(name="a", path=()), IconOption(name="b", path=())],
            ...     converter=lambda a, b: b or a,
            ... )
            >>> simple.convert({"a": True, "b": False})
            True

            >>> advanced = IconMultiOption(
            ...     options=[
            ...         IconOption(name="a", path=(), converter=lambda a: a == "on"),
            ...         IconOption(name="b", path=(), converter=lambda b: b % 2 > 1),
            ...     ],
            ...     converter=lambda a, b: b or a,
            ... )
            >>> advanced.convert({"a": "on", "b": 6})
            True
        """
        _ = fallback_converter
        return self.converter(
            **{
                icon_opt.name: icon_opt.convert(icon_config, fallback_converter=None)
                for icon_opt in self.options
                if icon_opt.read_from_icon
            }
        )


@dataclasses.dataclass
class ConfigOption:
    description: str
    icon_equivalent: IconOption | IconMultiOption | None = None

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
        ConfigOption(description='test', icon_equivalent=None)
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
        {'an_option': ConfigOption(description='first', icon_equivalent=None), 'other_option': ConfigOption(description='second', icon_equivalent=None)}
        """
        if not dataclasses.is_dataclass(config_cls):
            raise NotDataclassError(str(config_cls))
        annotations = typing.get_type_hints(config_cls, include_extras=True)
        for field in dataclasses.fields(config_cls):
            yield field.name, cls.from_type_hint(annotations[field.name])


def iter_pairs_from_icon(
    config_cls: type, icon_config: dict[str, typing.Any]
) -> typing.Iterator[tuple[str, typing.Any]]:
    """
    Iter name-value pairs for config options, reading from a fortran ICON config dict.

    Example:

    >>> @dataclasses.dataclass
    >>> class ConfigClass:
    >>>     choice: typing.Annotated[
    >>>         int,
    >>>         options.ConfigOption(
    >>>             description="A choice of methods.",
    >>>             icon_equivalent=options.IconOption(
    >>>                 name="isomchce",
    >>>                 path=("nested",)
    >>>             )
    >>>         )
    >>>     ]
    >>> next(iter_pairs_from_icon(ConfigClass, {"nested": {"isomche": 1}}))
    ("choice", 1)
    """
    annotations = typing.get_type_hints(config_cls, include_extras=True)
    if not dataclasses.is_dataclass(config_cls):
        raise NotDataclassError(str(config_cls))
    for name, opt in ConfigOption.iter_from_config_class(config_cls):
        if opt.icon_equivalent:
            converted = opt.icon_equivalent.convert(icon_config, annotations[name])
            if converted is not NotRead:
                yield name, converted


def construct_config_from_icon(
    config_cls: type, icon_config: dict[str, typing.Any], **overrides: typing.Any
) -> typing.Any:
    """
    Construct a configuration instance from a fortran ICON config dict.

    Example:

    >>> @dataclasses.dataclass
    >>> class ConfigClass:
    >>>     choice: typing.Annotated[
    >>>         int,
    >>>         options.ConfigOption(
    >>>             description="A choice of methods.",
    >>>             icon_equivalent=options.IconOption(
    >>>                 name="isomchce",
    >>>                 path=()
    >>>             )
    >>>         )
    >>>     ]
    >>> construct_config_from_icon(ConfigClass, {"isomche": 1})
    ConfigClass(choice=1)
    """
    return config_cls(
        **dict(iter_pairs_from_icon(config_cls, icon_config)),
        **overrides,
    )
