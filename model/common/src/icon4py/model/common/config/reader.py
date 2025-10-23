import logging
import pathlib
from typing import TypeVar, Generic, TypeAlias
import omegaconf as oc

#TODO (@halungge):
#  - interpolation
#  - extract protocol
#  - error cases

# error cases for update...

log = logging.getLogger(__file__)


T = TypeVar('T')
""" T is a data class to which needs to have all its members type anotated with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
"""
class ConfigReader(Generic[T]):


    def __init__(self, schema: T|type[T]):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config:oc.DictConfig = oc.OmegaConf.structured(schema)
        self._config:oc.DictConfig = self._default_config.copy()


    def _load_update(self, patch: T | oc.DictConfig | str | pathlib.Path):
        if isinstance(patch, (pathlib.Path, str)):
            return oc.OmegaConf.load(patch)
        elif isinstance(patch, (self._schema, oc.DictConfig)):
            return oc.OmegaConf.structured(patch)
        else:
            raise ValueError(f"wrong type for config, expected {self._schema} or 'str' but got {type(patch)}")

    def update(self, patch: T | oc.DictConfig | str | pathlib.Path):
        try:
            update = self._load_update(patch)
            self._config = oc.OmegaConf.merge(self._config, update)
        except oc.ValidationError or ValueError as e:
            log.error(f"patch {patch} does not validate against configuration {self._schema}")

    def get_config(self)->T:
        return oc.OmegaConf.to_object(self._config)

    @property
    def default(self)->T:
        return oc.OmegaConf.to_object(self._default_config)


#TODO  replace parent class by protocol...
class DictConfigReader(ConfigReader[dict]):
    def __init__(self, values : dict | oc.DictConfig):
        self._config = oc.OmegaConf.create(values)
        oc.OmegaConf.set_readonly(self._config, True)

    def get_config(self)->dict:
        return  oc.OmegaConf.to_container(self._config, resolve=True)

    def default(self)->dict:
        return self.get_config()





