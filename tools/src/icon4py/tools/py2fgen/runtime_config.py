# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Callable
from typing import TypeVar

from gt4py import eve


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


_T = TypeVar("_T", bound=eve.StrEnum)


def _env_to_strenum(name: str, enum_type: type[_T], default: _T) -> _T:
    """Read an enum value from an environment variable (with checking)."""
    value = os.environ.get(name, default).upper()
    if value not in enum_type.__members__:
        allowed_values = ", ".join(f"'{m}'" for m in enum_type.__members__)
        raise ValueError(
            f"Invalid value '{value}' for '{name}', allowed values are {allowed_values}."
        )
    return enum_type(value)


PROFILING: bool = _env_flag_to_bool("PY2FGEN_PROFILING", False)
"""Enable profiling for the PY2FGEN generated bindings."""


class Py2fgenLogLevels(eve.StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


LOG_LEVEL: str = _env_to_strenum("PY2FGEN_LOG_LEVEL", Py2fgenLogLevels, Py2fgenLogLevels.INFO)
"""Set the log level for the PY2FGEN generated bindings."""


def _split_and_strip(s: str | None, sep: str = ",") -> list[str]:
    if not s:
        return []
    return [part.strip() for part in s.split(sep) if part.strip()]


EXTRA_MODULES: list[str] = _split_and_strip(os.environ.get("PY2FGEN_EXTRA_MODULES", ""))
"""
Extra Python modules to be loaded when the PY2FGEN bindings are initialized.

Modules should be specified as a comma-separated list of module paths ('my.path.to.module').
"""

# CUSTOMIZATION HOOKS

# TODO: str == function name or pass the function that we are wrapping as identifier?
HOOK_FUNCTION_ENTER: Callable[[str], None] = lambda func_name: None  # noqa: E731
HOOK_FUNCTION_EXIT: Callable[[str], None] = lambda func_name: None  # noqa: E731
