# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import dataclasses
import enum
import typing

import cattrs.gen
import cattrs.preconf.pyyaml
import yaml

from icon4py.model.common import time, type_alias as ta


ET = typing.TypeVar("ET", bound=enum.Enum)
T = typing.TypeVar("T")
ST = typing.TypeVar("ST", bound="ConfigWithShared")


CONV = cattrs.preconf.pyyaml.PyyamlConverter(forbid_extra_keys=True)


@dataclasses.dataclass
class ConfigUnionStructurer[T]:
    union_type: type[T]
    type_map: typing.Mapping[str, type[T]]

    def __call__(self, spec: dict, _: typing.Any) -> T:
        config_type = spec.pop("type")
        if config_type not in self.type_map:
            raise TypeError(f"Unsupported type spec for {self.union_type}: {config_type}")
        return CONV.structure(spec, self.type_map[config_type])


@dataclasses.dataclass
class SharedOption:
    name: str
    value: typing.Any
    consumers: list[str]


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConfigWithShared:
    shared: list[SharedOption] = dataclasses.field(default_factory=list)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CONV.register_structure_hook(cls, structure_with_shared)
        CONV.register_unstructure_hook(cls, unstructure_with_shared)



@CONV.register_structure_hook
def structure_shared_option(data: dict, _: typing.Any) -> SharedOption:
    match tuple(data.items()):
        case ((name, value), ("consumers", [*consumers])):
            data = {"consumers": consumers, "name": name, "value": value}
        case (("consumers", [*consumers]), (name, value)):
            data = {"consumers": consumers, "name": name, "value": value}
    return cattrs.gen.make_dict_structure_fn(SharedOption, CONV)(data, SharedOption)


CONV.register_unstructure_hook(ta.wpfloat, lambda v: CONV.unstructure(float(v)))
yaml.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", ""))


def read_yaml_str[T](yaml_str: str, config_cls: type[T]) -> T:
    return CONV.structure(yaml.safe_load(yaml_str), config_cls)


def write_yaml_str[T](config_inst: T) -> str:
    return yaml.dump(CONV.unstructure(config_inst), sort_keys=False)


def structure_enum(val: str, enum_type: type[enum.Enum]) -> enum.Enum:
    return enum_type.__members__[val.upper()]


def unstructure_enum(val: enum.Enum) -> str:
    return val.name.lower()


def register_enum[ET](enum_type: type[ET]) -> type[ET]:
    CONV.register_structure_hook(enum_type, structure_enum)
    CONV.register_unstructure_hook(enum_type, unstructure_enum)
    return enum_type


def structure_with_shared[ST](spec: dict, config_cls: type[ST]) -> ST:
    for shared_option_spec in spec.get("shared", []):
        shared_option = CONV.structure(shared_option_spec, SharedOption)
        for consumer_name in shared_option.consumers:
            if spec[consumer_name] is None:
                spec[consumer_name] = {}
            if shared_option.name in spec[consumer_name]:
                raise ValueError(f"duplicate option for {consumer_name}: {shared_option.name} given in 'shared' as well as directly.")
            spec[consumer_name] |= {shared_option.name: shared_option.value}
    return cattrs.gen.make_dict_structure_fn(config_cls, CONV)(spec, config_cls)


def unstructure_with_shared[ST](config_obj: ST) -> dict:
    spec = cattrs.gen.make_dict_unstructure_fn(type(config_obj), CONV)(config_obj)
    config_copy = copy.deepcopy(config_obj)
    for shared_option in config_copy.shared:
        for consumer_name in shared_option.consumers:
            if spec[consumer_name][shared_option.name] == shared_option.value:
                spec[consumer_name].pop(shared_option.name)
                if spec[consumer_name] == {}:
                    spec[consumer_name] = None
            else:
                shared_option.consumers.remove(consumer_name)
        if not shared_option.consumers:
            config_copy.shared.remove(shared_option)

    if not spec["shared"]:
        spec.pop("shared")
    else:
        spec["shared"] = CONV.unstructure(config_copy.shared)
    return spec


@CONV.register_structure_hook
def structure_abstime(abstime_val: str, _: typing.Any) -> time.AbsoluteTime:
    if isinstance(abstime_val, time.AbsoluteTime):
        return abstime_val
    return time.AbsoluteTime.fromisoformat(abstime_val)


@CONV.register_unstructure_hook
def unstructure_abstime(abstime: time.AbsoluteTime) -> str:
    return abstime.isoformat()


@CONV.register_structure_hook
def structure_reltime(reltime_val: str, _: typing.Any) -> time.RelativeTime:
    if isinstance(reltime_val, time.RelativeTime):
        return reltime_val
    return time.RelativeTime(seconds=int(reltime_val))


@CONV.register_unstructure_hook
def unstructure_reltime(reltime: time.RelativeTime) -> int:
    return int(reltime.total_seconds())


@CONV.register_structure_hook
def structure_endtime(endtime_dict: dict, _: typing.Any) -> time.EndOfSimulation:
    timeclass: type | None = None
    match timetype := endtime_dict.pop("type"):
        case "absolute":
            timeclass = time.AbsoluteTime
        case "relative":
            timeclass = time.RelativeTime
        case "numsteps":
            timeclass = time.NumTimeSteps
    if not timeclass:
        raise TypeError(f"unsupported end of simulation time type: '{timetype}'")
    return CONV.structure(endtime_dict["value"], timeclass)


@CONV.register_unstructure_hook
def unstructure_endtime(endtime: time.EndOfSimulation) -> dict:
    timetype: str = ""
    match endtime:
        case time.AbsoluteTime():
            timetype = "absolute"
        case time.RelativeTime():
            timetype = "relative"
        case time.NumTimeSteps():
            timetype = "numsteps"
        case _:
            raise TypeError(f"Unsupported time type: '{type(endtime)}'.")
    return {"type": timetype, "value": CONV.unstructure(endtime)}


def register_config_union[T](union_type: type[T], mapping: typing.Mapping[str, type[T]]) -> None:

    inverse_mapping: dict[type[T], str] = {v: k for k, v in mapping.items()}

    def unstructure(instance: T) -> dict:
        return {"type": inverse_mapping[type(instance)], **CONV.unstructure(instance)}

    CONV.register_structure_hook(
        union_type, ConfigUnionStructurer(union_type=union_type, type_map=mapping)
    )
    CONV.register_unstructure_hook(union_type, unstructure)
