import dataclasses
import logging
import pathlib
import sys
import importlib
from types import ModuleType
from typing import Any
import omegaconf as oc
from icon4py.model.common.config import reader
MODEL_NODE = "model"
log = logging.getLogger(__file__)


# TODO interpolation of values
# TODO set all configs to readonly when updated and interpolations done
# TODO register resolvers for ENUMS (from string or enum-value...)
# TODO dump configuration (should be done from recursively from ConfigReader as we want to be able
#      to do that standalone as well (additional change name if it writes as well)
# TODO create configs that do not have their own setup or are mostly configured from "Other" components (for example metrics)?
# TODO generally handle exceptions.. wrong keys, missing keys, wrong value types...
# TODO what if a package does not have the `config.py`

@dataclasses.dataclass
class ModelConfig:
    components: dict[str, str] = oc.MISSING




def load_reader(module:ModuleType, update:oc.DictConfig)->reader.ConfigReader:
    if hasattr(module, "init_config"):
        default_reader = module.init_config()
        default_reader.update(update)
    else:
        default_reader = reader.DictConfigReader(update)
    return default_reader

class ConfigurationManager:
    def __init__(self, model_config: pathlib.Path|str):
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
        model_config_reader = reader.ConfigReader(ModelConfig())
        self._readers:dict[str, reader.ConfigReader] = {MODEL_NODE: model_config_reader}

    def read_config(self):
        # lets assume the configuration is all in one file
        config = oc.OmegaConf.load(self._config_path.joinpath(self._config_file))
        self._readers[MODEL_NODE].update(config.model)
        self.initialize_configs(config)

    def initialize_configs(self, config:oc.DictConfig):
        model_config = self._readers[MODEL_NODE].get_config()
        for name, package in model_config.components.items():
            config_module = package+".config"
            module = importlib.import_module(config_module)
            module_config = config[name]
            #TODO (halungge) what happens if there is no config block for this module??
            reader = load_reader(module, module_config)

            self._readers[name] = reader


    def get_configured_modules(self):
        return self._readers.keys()


    def get_config(self, module_name:str)->Any:
        module_config = self._readers.get(module_name)
        return module_config.get_config() if module_config else {}









