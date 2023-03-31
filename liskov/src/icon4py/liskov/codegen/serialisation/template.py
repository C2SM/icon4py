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

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator


class InitStatement(eve.Node):
    InitStatement = as_jinja("todo")


class InitStatementGenerator(TemplatedGenerator):
    InitStatementGenerator = as_jinja("todo")


class SavepointStatement(eve.Node):
    SavepointStatement = as_jinja("todo")


class SavepointStatementGenerator(TemplatedGenerator):
    SavepointStatementGenerator = as_jinja("todo")
