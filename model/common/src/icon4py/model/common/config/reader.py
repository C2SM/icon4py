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


# TODO (@halungge):
#  - interpolation
#  - extract protocol
#  - error cases

# error cases for update...

log = logging.getLogger(__file__)


class ConfigType(enum.Enum):
    DEFAULT = enum.auto()
    USER = enum.auto()


T = TypeVar("T")
DictType = TypeVar("DictType", int, str, float, enum.Enum, bool, bytes, dict, list, pathlib.Path)
""" T is a data class to which needs to have all its members type anotated with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
"""


class ConfigReader(Generic[T]):
    def __init__(self, schema: T | type[T]):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config: oc.DictConfig = oc.OmegaConf.structured(
            schema
        )  # TODO should this use create?
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
            if self._name in update.keys():
                update = update.get(self._name)
            self._config = oc.OmegaConf.merge(self._config, update)
            oc.OmegaConf.set_readonly(self._config, read_only)
        except oc.ValidationError or ValueError as e:
            log.error(f"patch {patch} does not validate against configuration {self._schema}")
            raise e

    def as_type(self) -> T:
        mode = (
            oc.SCMode.INSTANTIATE if not isinstance(self._schema, oc.DictConfig) else oc.SCMode.DICT
        )
        return oc.OmegaConf.to_container(
            self._config, resolve=True, throw_on_missing=True, structured_config_mode=mode
        )

    def to_yaml(self, type=ConfigType.USER): ...

    @property
    def config(self) -> oc.DictConfig:
        # TODO set readonly flag?
        return self._config

    @property
    def default(self) -> T:
        return oc.OmegaConf.to_object(self._default_config)


# TODO  replace parent class by protocol...
class DictConfigReader(ConfigReader[dict]):
    def __init__(self, values: dict | oc.DictConfig):
        self._config = oc.OmegaConf.create(values)
        oc.OmegaConf.set_readonly(self._config, True)

    def get_config(self) -> dict:
        return oc.OmegaConf.to_container(self._config, resolve=True)

    @property
    def default(self) -> dict:
        return self.get_config()


def interpolate_or_else(interpolation: str, default: DictType) -> DictType:
    return
