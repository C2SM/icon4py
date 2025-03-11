# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import os
from typing import TypeVar


def _env_flag_to_bool(name: str, default: bool) -> bool:
    """Recognize true or false signaling string values."""
    flag_value = None
    if name in os.environ:
        flag_value = os.environ[name].lower()
    match flag_value:
        case None:
            return default
        case "0" | "false" | "off":
            return False
        case "1" | "true" | "on":
            return True
        case _:
            raise ValueError(
                "Invalid GT4Py environment flag value: use '0 | false | off' or '1 | true | on'."
            )


class _StrEnum(str, enum.Enum):
    """:class:`enum.Enum` subclass whose members are considered as real strings."""

    def __str__(self) -> str:
        assert isinstance(self.value, str)
        return self.value


_T = TypeVar("_T", bound=_StrEnum)


def _env_to_strenum(name: str, enum_type: type[_T], default: _T) -> _T:
    """Recognize string values as members of an enumeration."""
    value = os.environ.get(name, default).upper()
    if value not in enum_type.__members__:
        allowed_values = ", ".join(f"'{m}'" for m in enum_type.__members__)
        raise ValueError(
            f"Invalid value '{value}' for '{name}', allowed values are {allowed_values}."
        )
    return enum_type(value)


PROFILING: bool = _env_flag_to_bool("PY2FGEN_PROFILING", False)


class PY2FGEN_LOG_LEVELS(_StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


LOG_LEVEL: str = _env_to_strenum("PY2FGEN_LOG_LEVEL", PY2FGEN_LOG_LEVELS, PY2FGEN_LOG_LEVELS.INFO)
