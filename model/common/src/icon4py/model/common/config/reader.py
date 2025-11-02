# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import functools
import logging
import pathlib
from typing import Generic, TypeVar

import omegaconf as oc


# TODO (@halungge): address
#  - interpolation
#  - extract protocol
#  - error cases


log = logging.getLogger(__file__)


class ConfigType(enum.Enum):
    DEFAULT = enum.auto()
    USER = enum.auto()


T = TypeVar("T")
""" T is a data class to which needs to have all its members type anotated with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
"""

_CT = TypeVar("_CT", int, str, float, enum.Enum, bool, bytes, dict, list, pathlib.Path)
""" TypeVar denoting possible value types in OmegaConf DictConfig"""


def resolve_or_else(key: str, value: _CT) -> _CT:
    interpolation = f"oc.select:{key}, {value}"
    return oc.II(interpolation)


class Configuration(Generic[T]):
    def __init__(self, schema: T | type[T]):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config: oc.DictConfig = oc.OmegaConf.create(
            schema
        )  # TODO (halungge): should this use create?
        self._config: oc.DictConfig = self._default_config.copy()
        oc.OmegaConf.set_readonly(self._default_config, True)

    @functools.cached_property
    def _name(self):
        return self._schema.__name__.lower()

    def _load_update(self, patch: T | oc.DictConfig | dict | str | pathlib.Path):
        if isinstance(patch, (pathlib.Path, str)):
            return oc.OmegaConf.load(patch)
        if isinstance(patch, dict):
            return oc.OmegaConf.create(patch)
        elif isinstance(patch, (self._schema, oc.DictConfig)):
            return oc.OmegaConf.structured(patch)
        else:
            raise ValueError(
                f"wrong type for config, expected {self._schema} or 'str' but got {type(patch)}"
            )

    def update(self, patch: T | oc.DictConfig | str | pathlib.Path | dict, read_only=False):
        try:
            update = self._load_update(patch)
            if self._name in update:
                update = update.get(self._name)
            if oc.OmegaConf.is_readonly(self._config):
                oc.OmegaConf.set_readonly(self._config, False)
            self._config = oc.OmegaConf.merge(self._config, update)
            oc.OmegaConf.set_readonly(self._config, read_only)
        except (oc.ValidationError, ValueError) as e:
            log.error(f"patch {patch} does not validate against configuration {self._schema}")
            raise e

    def as_type(self) -> T:
        mode = (
            oc.SCMode.INSTANTIATE if not isinstance(self._schema, oc.DictConfig) else oc.SCMode.DICT
        )
        return oc.OmegaConf.to_container(
            self._config, resolve=True, throw_on_missing=True, structured_config_mode=mode
        )

    def to_yaml(self, file: str | pathlib.Path, config_type=ConfigType.USER) -> None:
        config = self._config if config_type == ConfigType.USER else self.default
        stream = oc.OmegaConf.to_yaml(config, resolve=True, sort_keys=True)
        with open(file, "w", encoding="utf-8") as f:
            f.write(stream)

    @property
    def config(self) -> oc.DictConfig:
        oc.OmegaConf.set_readonly(self._config, True)
        return self._config

    @property
    def config_as_type(self) -> T:
        return oc.DictConfig._to_object(self._config)

    @property
    def default(self) -> oc.DictConfig:
        return oc.OmegaConf.to_object(self._default_config)


def init_config() -> Configuration[dict]:
    return Configuration(dict())
