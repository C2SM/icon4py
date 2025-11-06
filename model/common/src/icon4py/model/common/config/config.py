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
from typing import Generic, TypeVar, Protocol, overload, TypeAlias, TypeGuard
from typing_extensions import Self

import omegaconf as oc

from icon4py.model.common import exceptions


log = logging.getLogger(__file__)

MISSING = oc.MISSING


def to_timedelta(secs: int | float) -> str:
    interpolation = f"dtime:{secs}"
    return oc.II(interpolation)


def to_datetime(time_str: str) -> str:
    interpolation = f"datetime:{time_str}"
    return oc.II(interpolation)


def _timedelta_resolver(secs: int | float)->datetime.timedelta:
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

T = TypeVar("T", covariant=True)
"""
Type variable used in the Generic ConfigurationHandler: T is a data class which _needs to have all its property type-annotated_ with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
If a type annotation is missing the property will be missing from the resulting Configuration.
"""

_CT = TypeVar("_CT", int, str, float, enum.Enum, bool, bytes, dict, list, pathlib.Path)
""" TypeVar denoting possible value types in OmegaConf DictConfig"""
OCDictType:TypeAlias = oc.DictConfig|oc.ListConfig

def dict_config(c: OCDictType)-> TypeGuard[oc.DictConfig]:
    return  isinstance(c, oc.DictConfig)

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


# PROTOCOL should contain
# - update (atch: T | oc.DictConfig | dict | str | pathlib.Path)->None
# - to_yaml(self, file: str | pathlib.Path, config_type=ConfigType.USER) -> None
# - get(ConfigType, FORMAT)

class Configuration(Protocol[T]):
    def to_yaml(self, file: str, config_type:ConfigType) -> None: ...
    def get(self, type_:ConfigType, format_:Format)-> oc.DictConfig | T: ...

class Updatable(Protocol[T]):
    def update(self, patch: T | oc.DictConfig | str | pathlib.Path | dict, read_only: bool) -> Self: ...

class ConfigurationHandler(Configuration[T], Updatable[T]):
    def __init__(self, schema: T):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config: oc.DictConfig = oc.OmegaConf.create(schema)
        self._config: oc.DictConfig = self._default_config.copy()
        oc.OmegaConf.set_readonly(self._default_config, True)

    @functools.cached_property
    def _name(self)->str:
        return self._schema.__name__.lower()


    def _load_update(self, patch: T | oc.DictConfig | dict | str | pathlib.Path)->oc.DictConfig:
        if isinstance(patch, (pathlib.Path, str)):
            cfg = oc.OmegaConf.load(patch)
        if isinstance(patch, dict):
            cfg= oc.OmegaConf.create(patch)
        elif isinstance(patch, (self._schema, oc.DictConfig)):
            cfg= oc.OmegaConf.structured(patch)
        else:
            raise ValueError(
                f"wrong type for config, expected {self._schema} or 'str' but got {type(patch)}"
            )
        return cfg

    def update(
        self, patch: T | oc.DictConfig | str | pathlib.Path | dict, read_only:bool=False
    ) -> "ConfigurationHandler[T]":
        try:
            update = self._load_update(patch)
            if self._name in update:
                if dict_config(update):
                    update = update.get(self._name)
            if oc.OmegaConf.is_readonly(self._config):
                oc.OmegaConf.set_readonly(self._config, False)
            self._config = oc.OmegaConf.merge(self._config, update)
            oc.OmegaConf.set_readonly(self._config, read_only)
            return self
        except (oc.ValidationError, ValueError) as e:
            log.error(f"patch {patch} does not validate against configuration {self._schema}")
            raise exceptions.InvalidConfigError(
                f"patch {patch} does not validate against configuration {self._schema}"
            ) from e

    def _as_type(self, config:oc.DictConfig) -> T:
        mode = (
            oc.SCMode.INSTANTIATE if not isinstance(self._schema, oc.DictConfig) else oc.SCMode.DICT
        )
        return oc.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True, structured_config_mode=mode
        )

    def to_yaml(self, file: str | pathlib.Path, config_type:ConfigType=ConfigType.CUSTOM) -> None:
        if isinstance(file, str):
            file = pathlib.Path(file)

        config = self._config if config_type == ConfigType.CUSTOM else self._default_config
        stream = oc.OmegaConf.to_yaml(config, resolve=True, sort_keys=True)

        with file.open("w", encoding="utf-8") as f:
            f.write(stream)


    def get(self, type_: ConfigType=ConfigType.CUSTOM, format_: Format=Format.CLASS)->oc.DictConfig|T:
        config = self._config if type_ == ConfigType.CUSTOM else self._default_config
        match format_:
            case Format.CLASS:
                return self._as_type(config)
            case Format.DICT:
                oc.OmegaConf.set_readonly(config, True)
                return config # TODO (halungge): change to std dict




def init_config() -> ConfigurationHandler[dict]:
    return ConfigurationHandler(dict())
