# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re

from icon4py.liskov.directives import TypedDirective
from icon4py.liskov.exceptions import SyntaxExceptionHandler


class DirectiveSyntaxValidator:
    """Syntax validation method dispatcher for each directive type."""

    def __init__(self):
        self.parentheses_regex = r"\((\w*?)\)"
        self.exception_handler = SyntaxExceptionHandler

    def validate(self, directives: list[TypedDirective]) -> None:
        for d in directives:
            type_name = d.directive_type.__class__.__name__
            getattr(self, type_name)(d)

    def StartStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def EndStencil(self, directive: TypedDirective):
        regex = rf"{directive.directive_type.pattern}{self.parentheses_regex}"
        self._validate_syntax(directive, regex)

    def Create(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def Declare(self, directive: TypedDirective):
        regex = directive.directive_type.pattern
        self._validate_syntax(directive, regex)

    def _validate_syntax(self, directive, regex):
        matches = re.fullmatch(regex, directive.string)
        self.exception_handler.check_for_matches(directive, matches, regex)


class DirectiveSemanticsValidator:
    # todo: check not more than one declare directive (at least 1)
    # todo: check not more than one create directive (at least 1)
    # todo: number of StencilStart and StencilEnd must be the same
    # todo: stencil names for StencilStart must be unique
    # todo: stencil names for StencilEnd must be unique
    def validate(self, directives: list[TypedDirective]):
        pass
