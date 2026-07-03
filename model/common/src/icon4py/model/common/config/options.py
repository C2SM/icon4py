# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import typing

from typing_extensions import Self

from icon4py.model.common.utils import fortran_config


T = typing.TypeVar("T")


class NotDataclassError(Exception): ...


class MissingConfigOptionAnnotationError(Exception): ...


class MultipleOptionAnnotationsError(Exception): ...


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


@dataclasses.dataclass
class ConfigOption:
    description: str
    icon_equivalent: IconOption | None = None

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
    config_cls: type, icon_config: dict[str, typing.Any], *, allow_missing: bool = False
) -> typing.Iterator[tuple[str, typing.Any]]:
    """
    Iter name-value pairs for config options, reading from a fortran ICON config dict.

    With ``allow_missing`` options not found in ``icon_config`` are skipped (so
    that the dataclass defaults apply). This is meant for sources that only
    contain the explicitly set options, e.g. the converted *input* namelists:
    derived-type namelists (such as ``aes_vdf_nml``) are echoed by ICON as an
    anonymous positional array, so for them the complete named values of the
    standard echoed-output source are not available.

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
        if opt.icon_equivalent and opt.icon_equivalent.read_from_icon:
            data = icon_config
            try:
                for subsection in opt.icon_equivalent.path:
                    data = data[subsection]
                raw_value = data[opt.icon_equivalent.name]
            except KeyError:
                if allow_missing:
                    continue
                raise
            de_listified = (
                fortran_config.list_to_value(raw_value)
                if opt.icon_equivalent.list_to_value
                else raw_value
            )
            yield name, annotations[name](de_listified)


def construct_config_from_icon(
    config_cls: type[T],
    icon_config: dict[str, typing.Any],
    *,
    allow_missing: bool = False,
    **overrides: typing.Any,
) -> T:
    """
    Construct a configuration instance from a fortran ICON config dict.

    ``allow_missing`` is forwarded to :func:`iter_pairs_from_icon`: options not
    found in ``icon_config`` fall back to the dataclass defaults.

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
        **dict(iter_pairs_from_icon(config_cls, icon_config, allow_missing=allow_missing)),
        **overrides,
    )
