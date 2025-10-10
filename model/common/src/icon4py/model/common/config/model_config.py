import pathlib
from typing import Any

import himl
import logging
import enum
import omegaconf as oc

log = logging.Logger(__file__)

class ConfigType(enum.Enum):
    DEFAULT = enum.auto()
    FINAL = enum.auto()


class OmegaParser:
    def __init__(self):
        self._default_config = self._read_default_config()
        self._config = self._default_config

    def _read_default_config(self)->Any:
        default_path = pathlib.Path(__file__).parent.joinpath("default")
        configs = []
        for file in default_path.glob("*.yaml"):
            cfg = oc.OmegaConf.load(file)
            configs.append(cfg)
        return oc.OmegaConf.merge(*configs)

    def get(self, key:str, type_: ConfigType = ConfigType.FINAL) -> Any:
        if type_ == ConfigType.DEFAULT:
            return self._default_config.get(key)
        else:
            return self._config.get(key)

    def process(self, path:pathlib.Path)->dict|oc.DictConfig:
        cfg = oc.OmegaConf.load(path)
        merged = oc.OmegaConf.merge(self._default_config, cfg)
        self._config = merged
        return merged




class HimlParser:
    def __init__(self):
        self._processor = himl.ConfigProcessor()
        self._default_format = "yaml"
        self._config = {}
        self._read_default_config()


    def _read_default_config(self) -> None:
        default_path = pathlib.Path(__file__).parent.joinpath("default")
        log.debug(f"reading default config from {default_path}")
        self._default_config = self.process(path = default_path)


    def process(self, path:pathlib.Path)-> dict:
        filters = ()
        exclude_keys = ()
        default_format = "yaml"
        config = self._processor.process(path=str(path), filters=filters, exclude_keys=exclude_keys,
                                output_format=default_format, print_data=True)
        merged_config = self._config.copy()
        merged_config.update(config)
        self._config = merged_config
        return merged_config


    def get(self, key:str, type_: ConfigType = ConfigType.FINAL) -> Any:
        if type_ == ConfigType.DEFAULT:
            return self._default_config.get(key)
        else:
            return self._config.get(key)

