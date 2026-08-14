# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import typing

import cattrs.preconf.pyyaml
import yaml

from icon4py.model.common import time, type_alias as ta


T = typing.TypeVar("T", bound=enum.Enum)


CONV = cattrs.preconf.pyyaml.PyyamlConverter(forbid_extra_keys=True)


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


def register_enum[T](enum_type: type[T]) -> type[T]:
    CONV.register_structure_hook(enum_type, structure_enum)
    CONV.register_unstructure_hook(enum_type, unstructure_enum)
    return enum_type


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
    def structure(config_dict: dict, _: typing.Any) -> T:
        config_type = config_dict.pop("type")
        if config_type not in mapping:
            raise TypeError(f"Unsupported type spec for {union_type}: {config_type}")
        return CONV.structure(config_dict, mapping[config_type])

    inverse_mapping: dict[type[T], str] = {v: k for k, v in mapping.items()}

    def unstructure(instance: T) -> dict:
        return {"type": inverse_mapping[type(instance)], **CONV.unstructure(instance)}

    CONV.register_structure_hook(union_type, structure)
    CONV.register_unstructure_hook(union_type, unstructure)
