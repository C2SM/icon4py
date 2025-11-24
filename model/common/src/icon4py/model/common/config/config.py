# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import datetime
import enum
import functools
import logging
import pathlib
from typing import Any, Protocol, TypeAlias, TypeVar

import omegaconf as oc
import yaml

from icon4py.model.common import exceptions


log = logging.getLogger(__file__)

MISSING = oc.MISSING


def simple_str_representer(dumper: yaml.Dumper, data: Any) -> yaml.nodes.ScalarNode:
    """represent pathlib.Path as str"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_multi_representer(pathlib.PurePath, simple_str_representer)


def to_timedelta(secs: int | float) -> str:
    interpolation = f"dtime:{secs}"
    return oc.II(interpolation)


def to_datetime(time_str: str) -> str:
    interpolation = f"datetime:{time_str}"
    return oc.II(interpolation)


def _timedelta_resolver(secs: int | float) -> datetime.timedelta:
    return datetime.timedelta(seconds=secs)


def _datetime_resolver(time_str: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(time_str)


oc.OmegaConf.register_new_resolver("dtime", _timedelta_resolver)
oc.OmegaConf.register_new_resolver("datetime", _datetime_resolver)


class ConfigType(enum.Enum):
    DEFAULT = enum.auto()
    CUSTOM = enum.auto()


class Format(enum.Enum):
    DICT = enum.auto()
    CLASS = enum.auto()


T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")
"""
Type variable used in the Generic ConfigurationHandler: T is a data class which _needs to have all its property type-annotated_ with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
If a type annotation is missing the property will be missing from the resulting Configuration.
"""

_CT = TypeVar("_CT", int, str, float, enum.Enum, bool, bytes, dict, list, pathlib.Path)
""" TypeVar denoting possible value types in OmegaConf DictConfig"""
OCConfigType: TypeAlias = oc.DictConfig | oc.ListConfig


def as_dict(c: OCConfigType) -> oc.DictConfig:
    return c if isinstance(c, oc.DictConfig) else oc.DictConfig({i: v for i, v in enumerate(c)})


def resolve_or_else(key: str, value: _CT) -> _CT:
    """Convenience function to be used for value interpolation in configs. For example: if a given configuration
    data class has a property that is managed by another module it should be declared as:
    >>> @dataclass
    >>> class FooConfig:
    >>>   foreign_x: int = field(
    >>>    init=False, default=reader.resolve_or_else("x", 5)
    >>>    )

    If the config class is used on its own, the default value will be picked, if it is used in the context of a larger
    configuration that declares `x` it will interpolate to that value.
    """
    interpolation = f"oc.select:{key}, {value!r}"
    return oc.II(interpolation)


class Configuration(Protocol[T_co]):
    def to_yaml(self, file: str, is_default: bool = False) -> None: ...

    def _write_to_yaml(self, config: oc.DictConfig, file: str | pathlib.Path) -> None:
        if isinstance(file, str):
            file = pathlib.Path(file)

        stream = oc.OmegaConf.to_yaml(config, resolve=True, sort_keys=True)

        with file.open("w", encoding="utf-8") as f:
            f.write(stream)

    def get(
        self,
        *,
        is_default: bool = False,
    ) -> T_co: ...


class Updatable(Protocol[T_contra]):
    def update(self, patch: T_contra | oc.DictConfig | str | pathlib.Path | dict) -> None: ...


class ConfigurationHandler(Configuration[T], Updatable[T]):
    def __init__(self, schema: T):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config: oc.DictConfig = oc.OmegaConf.structured(
            schema, flags={"allow_objects": True}
        )
        self._config: oc.DictConfig = self._default_config.copy()
        oc.OmegaConf.set_readonly(self._default_config, True)

    @functools.cached_property
    def _name(self) -> str:
        return self._schema.__name__.lower()

    def _load_update(self, patch: T | oc.DictConfig | dict | str | pathlib.Path) -> oc.DictConfig:
        if isinstance(patch, (pathlib.Path, str)):
            return as_dict(oc.OmegaConf.load(patch))

        if isinstance(patch, dict):
            return oc.OmegaConf.create(patch)
        elif isinstance(patch, (self._schema, oc.DictConfig)):
            return oc.OmegaConf.structured(patch)
        else:
            raise ValueError(
                f"wrong type for config, expected {self._schema} or 'str' but got {type(patch)}"
            )

    def update(self, patch: T | oc.DictConfig | str | pathlib.Path | dict) -> None:
        try:
            update = self._load_update(patch)
            if self._name in update:
                update = update.get(self._name)
            if oc.OmegaConf.is_readonly(self._config):
                oc.OmegaConf.set_readonly(self._config, False)
            self._config = as_dict(oc.OmegaConf.merge(self._config, update))

        except (oc.ValidationError, ValueError) as e:
            log.error(f"patch {patch} does not validate against configuration {self._schema}")
            raise exceptions.InvalidConfigError(
                f"patch {patch} does not validate against configuration {self._schema}"
            ) from e

    def _as_type(self, config: oc.DictConfig, raise_on_missing: bool) -> T:
        mode = (
            oc.SCMode.INSTANTIATE
            if not isinstance(self._schema, (oc.DictConfig, dict))
            else oc.SCMode.DICT
        )
        return oc.OmegaConf.to_container(  # type: ignore  [return-value]
            config, resolve=True, throw_on_missing=raise_on_missing, structured_config_mode=mode
        )

    def to_yaml(self, file: str | pathlib.Path, is_default: bool = False) -> None:
        config = self._default_config if is_default else self._config
        self._write_to_yaml(config, file)

    def _get(
        self,
        *,
        type_: ConfigType,
        format_: Format,
        read_only: bool,
    ) -> oc.DictConfig | T:
        config = self._config if type_ == ConfigType.CUSTOM else self._default_config
        raise_on_missing = type_ == ConfigType.CUSTOM
        match format_:
            case Format.CLASS:
                return self._as_type(config, raise_on_missing)
            case Format.DICT:
                oc.OmegaConf.set_readonly(config, read_only)
                return config  # TODO (halungge): change to std dict?

    def get(self, *, is_default: bool = False) -> T:
        config_type = ConfigType.DEFAULT if is_default else ConfigType.CUSTOM
        return self._get(type_=config_type, format_=Format.CLASS, read_only=True)  # type: ignore  [return-value]


def init_config() -> ConfigurationHandler[dict]:
    return ConfigurationHandler(dict())
