# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import typing

from icon4py.model.common.utils import fortran_config


T = typing.TypeVar("T")


class NotDataclassError(Exception): ...


class MissingConfigOptionAnnotationError(Exception): ...


class MultipleOptionAnnotationsError(Exception): ...


@dataclasses.dataclass
class IconOption:
    name: str
    path: tuple[str, ...]
    list_to_value: bool = False
    read_from_icon: bool = True


@dataclasses.dataclass
class ConfigOption:
    description: str
    icon_equivalent: IconOption | None = None


def _opt_from_type_hint(annotated: typing.Any) -> ConfigOption:
    if not hasattr(annotated, "__metadata__"):
        raise MissingConfigOptionAnnotationError(f"'{annotated}' is not an annotated type hint.")
    opts = [a for a in annotated.__metadata__ if isinstance(a, ConfigOption)]
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


def opts_from_config_dataclass(cls: type) -> dict[str, ConfigOption]:
    if not dataclasses.is_dataclass(cls):
        raise NotDataclassError(str(cls))
    annotations = typing.get_type_hints(cls, include_extras=True)
    return {
        field.name: _opt_from_type_hint(annotations[field.name])
        for field in dataclasses.fields(cls)
    }


def iter_translate_from_icon_config(
    cls: type, icon_config: dict[str, typing.Any]
) -> typing.Iterator[tuple[str, typing.Any]]:
    annotations = typing.get_type_hints(cls, include_extras=True)
    if not dataclasses.is_dataclass(cls):
        raise NotDataclassError(str(cls))
    for name, opt in opts_from_config_dataclass(cls).items():
        if opt.icon_equivalent and opt.icon_equivalent.read_from_icon:
            data = icon_config
            for subsection in opt.icon_equivalent.path:
                data = data[subsection]
                raw_value = data[opt.icon_equivalent.name]
                de_listified = (
                    fortran_config.list_to_value(raw_value)
                    if opt.icon_equivalent.list_to_value
                    else raw_value
                )
                yield name, annotations[name](de_listified)


def construct_config_from_icon(
    cls: type[T], icon_config: dict[str, typing.Any], **overrides: typing.Any
) -> T:
    return cls(
        **{key: value for key, value in iter_translate_from_icon_config(cls, icon_config)},
        **overrides,
    )
