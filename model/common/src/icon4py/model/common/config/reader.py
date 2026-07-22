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

import cattrs
import yaml

from icon4py.model.common import time


T = typing.TypeVar("T", bound=enum.Enum)


CONV = cattrs.Converter(forbid_extra_keys=True)


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
def unstructure_reltime(reltime: time.RelativeTime) -> str:
    return str(int(reltime.total_seconds()))


@CONV.register_structure_hook
def structure_endtime(endtime_dict: dict, _: typing.Any) -> time.EndOfSimulation:
    timeclass: type | None = None
    match timetype := endtime_dict.pop("type"):
        case "absolute":
            timeclass = time.AbsoluteTime
        case "relative":
            timeclass = time.RelativeTime
        case "numstep":
            timeclass = time.NumTimeSteps
    if not timeclass:
        raise TypeError(f"unsupported end of simulation time type: '{timetype}'")
    return CONV.structure(endtime_dict["value"], timeclass)


@CONV.register_unstructure_hook
def unstructure_endtime(endtime: time.EndOfSimulation) -> dict:
    timetype = ""
    match endtime:
        case time.AbsoluteTime():
            timetype = "absolute"
        case time.RelativeTime():
            timetype = "relative"
        case time.NumTimeSteps:
            timetype = "numsteps"
        case _:
            raise TypeError(f"Unsupported time type: '{type(endtime)}'.")
    return {"type": timetype, "value": CONV.unstructure(endtime)}
