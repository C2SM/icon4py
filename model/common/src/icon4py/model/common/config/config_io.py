# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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


class IndentSequencesDumper(yaml.Dumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


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
class SharedOptionSet:
    options: dict[str, typing.Any]
    consumers: list[str]


@CONV.register_structure_hook
def structure_shared_set(data: dict, _: typing.Any) -> SharedOptionSet:
    print(data)
    return SharedOptionSet(
        consumers=data["consumers"], options={k: v for k, v in data.items() if k != "consumers"}
    )


@CONV.register_unstructure_hook
def unstructure_shared_set(shared_set: SharedOptionSet) -> dict:
    return shared_set.options | {"consumers": shared_set.consumers}


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConfigWithShared:
    shared: list[SharedOptionSet] = dataclasses.field(default_factory=list)

    def __init_subclass__(cls: type[typing.Self], **kwargs: typing.Any):
        super().__init_subclass__(**kwargs)
        CONV.register_structure_hook(cls, structure_with_shared)
        CONV.register_unstructure_hook(cls, unstructure_with_shared)


CONV.register_unstructure_hook(ta.wpfloat, lambda v: CONV.unstructure(float(v)))
yaml.add_representer(type(None), lambda d, _: d.represent_scalar("tag:yaml.org,2002:null", ""))


def read_yaml_str[T](yaml_str: str, config_cls: type[T]) -> T:
    return CONV.structure(yaml.safe_load(yaml_str), config_cls)


def write_yaml_str[T](config_inst: T) -> str:
    return yaml.dump(CONV.unstructure(config_inst), sort_keys=False, Dumper=IndentSequencesDumper)


def structure_enum(val: str, enum_type: type[enum.Enum]) -> enum.Enum:
    return enum_type.__members__[val.upper()]


def unstructure_enum(val: enum.Enum) -> str:
    return val.name.lower()


def register_enum[ET](enum_type: type[ET]) -> type[ET]:
    CONV.register_structure_hook(enum_type, structure_enum)
    CONV.register_unstructure_hook(enum_type, unstructure_enum)
    return enum_type


def structure_with_shared[ST](spec: dict, config_cls: type[ST]) -> ST:
    for option_set in [CONV.structure(i, SharedOptionSet) for i in spec.get("shared", [])]:
        for consumer in option_set.consumers:
            if spec[consumer] is None:
                spec[consumer] = {}
            if clashing := set(spec[consumer]).intersection(set(option_set.options)):
                raise ValueError(
                    f"multiple options given for {consumer}: {clashing} given in 'shared' as well as directly."
                )
            spec[consumer] |= option_set.options
    return cattrs.gen.make_dict_structure_fn(config_cls, CONV)(spec, config_cls)


def unstructure_with_shared(config_obj: ConfigWithShared) -> dict:
    spec = cattrs.gen.make_dict_unstructure_fn(type(config_obj), CONV)(config_obj)
    for shared_set in config_obj.shared:
        for consumer in shared_set.consumers:
            consistent = {k for k, v in shared_set.options.items() if spec[consumer][k] == v}
            spec[consumer] = {
                k: v for k, v in spec[consumer].items() if k not in consistent
            } or None
    if not spec["shared"]:
        spec.pop("shared")
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
