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

from typing import Type

from icon4py.liskov.directives import DirectiveType, TypedDirective


def extract_directive(
    directives: list[TypedDirective],
    required_type: Type[DirectiveType],
) -> list[TypedDirective]:
    directives = [d for d in directives if type(d.directive_type) == required_type]
    return directives
