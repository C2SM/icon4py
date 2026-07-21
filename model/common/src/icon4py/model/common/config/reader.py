from __future__ import annotations

import enum
import typing

import cattrs
import yaml


T = typing.TypeVar("T", bound=enum.Enum)


CONV = cattrs.Converter()


def read[T](yaml_str: str, config_cls: type[T]) -> T:
    return CONV.structure(yaml.safe_load(yaml_str), config_cls)


def structure_enum(val: str, enum_type: type[enum.Enum]) -> enum.Enum:
    return enum_type.__members__[val.upper()]


def unstructure_enum(val: enum.Enum) -> str:
    return val.name.lower()


def register_enum[T](enum_type: type[T]) -> type[T]:
    CONV.register_structure_hook(enum_type, structure_enum)
    CONV.register_unstructure_hook(enum_type, unstructure_enum)
    return enum_type
