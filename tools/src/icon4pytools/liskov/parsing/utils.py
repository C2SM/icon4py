# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Sequence, Type

from icon4pytools.liskov.parsing import types as ts


def flatten_list_of_dicts(list_of_dicts: list[dict]) -> dict:
    """Flatten a list of dictionaries into a single dictionary."""
    if not isinstance(list_of_dicts, list):
        raise TypeError("Input must be a list")
    for d in list_of_dicts:
        if not isinstance(d, dict):
            raise TypeError("Input list must contain dictionaries only")

    return {k: v for d in list_of_dicts for k, v in d.items()}


def string_to_bool(string: str) -> bool:
    """Convert a string representation of a boolean to a bool."""
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert '{string}' to a boolean.")


def extract_directive(
    directives: Sequence[ts.ParsedDirective],
    required_type: Type[ts.ParsedDirective],
) -> Sequence[ts.ParsedDirective]:
    """Extract a directive type from a list of directives."""
    directives = [d for d in directives if type(d) is required_type]
    return directives


def print_parsed_directive(directive: ts.ParsedDirective) -> str:
    """Print a parsed directive, including its contents, and start and end line numbers."""
    return f"Directive: {directive.string}, start line: {directive.startln}, end line: {directive.endln}\n"


def remove_directive_types(
    directives: Sequence[ts.ParsedDirective],
    exclude_types: Sequence[Type[ts.ParsedDirective]],
) -> Sequence[ts.ParsedDirective]:
    """Remove specified directive types from a list of directives."""
    return [d for d in directives if type(d) not in exclude_types]
