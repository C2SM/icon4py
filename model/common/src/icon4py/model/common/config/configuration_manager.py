# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import importlib
import logging
import pathlib
import sys
from types import ModuleType

import omegaconf as oc

from icon4py.model.common.config import config as common_config
from icon4py.model.common.grid import vertical as v_grid


MODEL_NODE = "model"
log = logging.getLogger(__file__)


# TODO (halungge): address

# [ ] dump configuration (should be done from recursively from ConfigReader as we want to be able
#      to do that standalone as well (additional change name if it writes as well)
# [ ] create configs that do not have their own setup or are mostly configured from "Other" components (for example metrics)?
# [ ] generally handle exceptions.. wrong keys, missing keys, wrong value types...
# [ ] what if a package does not have the `config.py`


@dataclasses.dataclass
class RunConfig:
    input_path: pathlib.Path = common_config.MISSING
    output_path: pathlib.Path = common_config.MISSING
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)
    start_date: datetime.datetime = common_config.MISSING
    end_date: datetime.datetime = common_config.MISSING


@dataclasses.dataclass
class ModelConfig:
    ndyn_substeps: int = 5
    vertical: v_grid.VerticalGridConfig = dataclasses.field(default=v_grid.VerticalGridConfig())
    grid: pathlib.Path = common_config.MISSING
    components: dict[str, str] = dataclasses.field(default_factory=dict)


def load_reader(module: ModuleType, update: oc.DictConfig) -> common_config.ConfigurationHandler:
    if hasattr(module, "init_config"):
        log.warning(f" module {module.__name__} has no `init_config` function")
        default_reader = module.init_config()
        default_reader.update(update)
    else:
        default_reader = common_config.init_config()
        default_reader.update(update)
    return default_reader


def init_reader(module: ModuleType) -> common_config.ConfigurationHandler:
    if hasattr(module, "init_config"):
        log.warning(f" module {module.__name__} has no `init_config` function")
        default_reader = module.init_config()
    else:
        default_reader = common_config.init_config()
    return default_reader


# TODO (halungge): change T type to custom type or std lib dict
class ConfigurationManager(common_config.Configuration[oc.DictConfig]):
    def __init__(self, model_config: pathlib.Path | str):
        self._handlers = {}
        if isinstance(model_config, str):
            model_config = pathlib.Path(model_config)
        if not model_config.exists():
            log.error(f"Configuration path {model_config} does not exist. Exiting")
            sys.exit()
        self._config_file = model_config.name if model_config.is_file() else None
        path = model_config.parent if model_config.is_file() else model_config
        try:
            self._config_path = path.resolve(strict=True)
        except OSError as e:
            log.error(f"Could not resolve path to {model_config} - {e}")
            sys.exit()

    def read_config(self):
        # lets assume the configuration is all in one file
        # this should do direct merging fo the config then the interpolation will work again....
        handler = common_config.init_config()
        handler.update(self._config_path.joinpath(self._config_file))
        user_config = handler._get(
            format_=common_config.Format.DICT,
            type_=common_config.ConfigType.CUSTOM,
            read_only=False,
        )

        model_config = common_config.ConfigurationHandler(ModelConfig())
        model_config.update(user_config.model)
        self._handlers["model"] = model_config
        run_config = common_config.ConfigurationHandler(RunConfig())
        run_config.update(user_config.run)
        self._handlers["run"] = run_config
        self._initialize_components(user_config)

    def get(
        self,
        is_default: bool = False,
    ) -> oc.DictConfig:
        config_type = (
            common_config.ConfigType.DEFAULT if is_default else common_config.ConfigType.CUSTOM
        )
        merged = oc.OmegaConf.create({})
        oc.OmegaConf.set_readonly(merged, False)
        for name, h in self._handlers.items():
            merged[name] = h._get(
                type_=config_type, format_=common_config.Format.DICT, read_only=False
            )

        oc.OmegaConf.resolve(merged)
        oc.OmegaConf.set_readonly(merged, True)
        return merged

    def _initialize_components(self, config: oc.DictConfig):
        model_components = config.model.components.items()
        for name, package in model_components:
            module_config = config.get(name)
            try:
                config_module = package + ".config"
                module = importlib.import_module(config_module)
                # merge the default of the package with the section configured in the file
                reader = load_reader(module, module_config)
            except ModuleNotFoundError:
                log.warning(
                    f"no config module in package {package}, instantiating general dict-config instead"
                )
                reader = common_config.init_config()
                if module_config is None:
                    log.warning(
                        f"No default configuration and no user-configuration for component {name} defined in file {self._config_file}"
                    )
                else:
                    reader.update(module_config)
            self._handlers[name] = reader

    def get_configured_modules(self) -> dict:
        return oc.OmegaConf.to_container(
            self._handlers["model"]
            ._get(
                format_=common_config.Format.DICT,
                type_=common_config.ConfigType.CUSTOM,
                read_only=True,
            )
            .components,
            structured_config_mode=oc.SCMode.DICT,
        )

    def to_yaml(self, file: str | pathlib.Path, is_default: bool = False):
        config = self.get(is_default)
        self._write_to_yaml(config, file)
