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

from icon4py.liskov.codegen.serialisation.interface import SavepointData


class InitStatement(eve.Node):
    directory: str


class InitStatementGenerator(TemplatedGenerator):
    InitStatement = as_jinja("!$ser init directory={{directory}}")


class SavepointStatement(eve.Node):
    savepoint: SavepointData


class SavepointStatementGenerator(TemplatedGenerator):
    SavepointStatement = as_jinja(
        """
        !$ser savepoint {{ _this_node.savepoint.name }} {% if _this_node.savepoint.metadata %} {%- for m in _this_node.savepoint.metadata -%} {{ m.key }}={{ m.value }} {%- endfor -%} {% endif %}

        {% for f in _this_node.savepoint.fields %}
        !$ser data {{ f.variable }}={{ f.association }}
        {% endfor %}
        """
    )
