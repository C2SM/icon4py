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
from dataclasses import asdict
from typing import Optional

import gt4py.eve as eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator

from icon4py.liskov.codegen.serialisation.interface import SavepointData


class InitStatement(eve.Node):
    directory: str
    prefix: str


class InitStatementGenerator(TemplatedGenerator):
    InitStatement = as_jinja(
        '!$ser init directory="{{directory}}" prefix="{{ prefix }}"'
    )


class Field(eve.Node):
    variable: str
    association: str
    decomposed: bool
    dimension: Optional[list[str]]
    typespec: Optional[str]
    typename: Optional[str]
    ptr_var: Optional[str]


class StandardFields(eve.Node):
    fields: list[Field]


class DecomposedFields(StandardFields):
    ...


class DecomposedFieldDeclarations(DecomposedFields):
    ...


class SavepointStatement(eve.Node):
    savepoint: SavepointData
    standard_fields: StandardFields = eve.datamodels.field(init=False)
    decomposed_fields: DecomposedFields = eve.datamodels.field(init=False)
    decomposed_field_declarations: DecomposedFieldDeclarations = eve.datamodels.field(
        init=False
    )

    def __post_init__(self):
        self.standard_fields = StandardFields(
            fields=[
                Field(**asdict(f)) for f in self.savepoint.fields if not f.decomposed
            ]
        )
        self.decomposed_fields = DecomposedFields(
            fields=[Field(**asdict(f)) for f in self.savepoint.fields if f.decomposed]
        )
        self.decomposed_field_declarations = DecomposedFieldDeclarations(
            fields=[Field(**asdict(f)) for f in self.savepoint.fields if f.decomposed]
        )


class SavepointStatementGenerator(TemplatedGenerator):
    SavepointStatement = as_jinja(
        """
        {{ decomposed_field_declarations }}

        !$ser savepoint {{ _this_node.savepoint.subroutine }}_{{ _this_node.savepoint.intent }} {% if _this_node.savepoint.metadata %} {%- for m in _this_node.savepoint.metadata -%} {{ m.key }}={{ m.value }} {%- endfor -%} {% endif %}

        {{ decomposed_fields }}

        {{ standard_fields }}
        """
    )

    StandardFields = as_jinja(
        """
    {% for f in _this_node.fields %}
    !$ser data {{ f.variable }}={{ f.association }}
    {% endfor %}
    """
    )

    DecomposedFieldDeclarations = as_jinja(
        """
        {% for f in _this_node.fields %}
        !$ser verbatim {{ f.typespec }}, dimension({{ ",".join(f.dimension) }}), allocatable :: {{ f.variable }}_{{ f.ptr_var}}
        {% endfor %}
        """
    )

    DecomposedFields = as_jinja(
        """
    {% for f in _this_node.fields %}
    !$ser verbatim allocate({{ f.variable }}_{{ f.ptr_var}}({{ f.alloc_dims }}))
    !$ser data {{ f.variable }}_{{ f.ptr_var}}={{ f.association }}
    !$ser verbatim deallocate({{ f.variable }}_{{ f.ptr_var}})
    {% endfor %}
    """
    )

    def visit_DecomposedFields(self, node: DecomposedFields):
        def generate_size_strings(colon_list, var_name):
            size_strings = []
            for i in range(len(colon_list)):
                size_strings.append(f"size({var_name}, {i + 1})")
            return size_strings

        for f in node.fields:
            f.variable = f.variable.replace(f"_{f.ptr_var}", "")
            f.alloc_dims = ", ".join(generate_size_strings(f.dimension, f.variable))

        return self.generic_visit(node)
