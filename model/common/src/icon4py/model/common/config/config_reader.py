import logging
import pathlib
from typing import TypeVar, Generic, overload
import omegaconf as oc

#TODO (@halungge):
#  - interpolation

# update from json, dataclass
# error cases for update...

log = logging.getLogger(__file__)


T = TypeVar('T')
""" T is a data class to which needs to have all its members type annotated with types that
OmegaConf supports in structured configs: https://omegaconf.readthedocs.io/en/2.3_branch/structured_config.html
"""
class ConfigReader(Generic[T]):


    def __init__(self, schema: T|type[T]):
        self._schema: type[T] = schema if isinstance(schema, type) else type(schema)
        self._default_config:oc.DictConfig = oc.OmegaConf.structured(schema)
        self._config:oc.DictConfig = self._default_config.copy()

    @overload
    def update(self, patch:T):
        assert type(patch) == self._schema, f"wrong type for config, expected {self._schema} but got {type(patch)}"

        try:
            update = oc.OmegaConf.structured(patch)
            self._config = oc.OmegaConf.merge(self._config, update)
        except oc.ValidationError as e:
            log.error(f"Patch {patch} does not validate against configuration {self._schema}")

    @overload
    def update(self, file: pathlib.Path|str):
        try:
            config_from_file = oc.OmegaConf.load(file)
            merged = oc.OmegaConf.merge(self._config, config_from_file)
            self._config = merged
        except oc.ValidationError as e:
            log.error(f"file {file} does not validate against configuration {self._schema}")


    def update(self, update: T | str| pathlib.Path):
        ...

    def get_config(self)->T:
        return oc.OmegaConf.to_object(self._config)

    @property
    def default(self)->T:
        return oc.OmegaConf.to_object(self._default_config)








