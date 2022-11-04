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

from pathlib import Path
from typing import Sequence, Union

import eve
from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.bindings.entities import Field, Offset
from icon4py.bindings.utils import format_fortran_code, write_string


class F90Generator(TemplatedGenerator):
    F90File = as_jinja(
        """
        #define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
        #define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
        module {{stencil_name}}
        use, intrinsic :: iso_c_binding
        implicit none
        interface
        {{run_fun}}
        {{run_and_verify_fun}}
        {{setup_fun}}
        subroutine &
        free_{{stencil_name}}( ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        end subroutine
        end interface
        contains

        {{wrap_run_fun}}
        {{wrap_setup_fun}}
        end module
    """
    )

    F90FieldNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.name}}, & {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90FieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.name}}_before, & {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90TypedFieldNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.renderer.render_ctype('f90')}}, {{ field.renderer.render_dim_string() }}, target :: {{field.name}} {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90TypedFieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.renderer.render_ctype('f90')}}, {{ field.renderer.render_dim_string() }}, target :: {{field.name}}_before {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90RankedFieldNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.renderer.render_ctype('f90')}}, {{field.renderer.render_ranked_dim_string()}}, target :: {{field.name}} {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90RankedFieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.renderer.render_ctype('f90')}}, {{field.renderer.render_ranked_dim_string()}}, target :: {{field.name}}_before {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90ToleranceArgs = as_jinja(
        """
        {%- for field in _this_node.fields -%}
        {{field.name}}_rel_tol, &
        {{field.name}}_abs_tol {% if not loop.last %}, &
        {% else %} & {% endif %}
        {% endfor -%}
      """
    )

    F90ErrToleranceArgs = as_jinja(
        """
        {%- for field in _this_node.fields -%}
        {{field.name}}_rel_err_tol, &
        {{field.name}}_abs_err_tol {% if not loop.last %}, &
        {% else %} & {% endif %}
        {%- endfor -%}
      """
    )

    F90TypedToleranceArgs = as_jinja(
        """
        {%- for field in _this_node.fields -%}
          real(c_double), value, target :: {{field.name}}_rel_tol
          real(c_double), value, target :: {{field.name}}_abs_tol
        {% endfor -%}
      """
    )

    F90TypedToleranceArgsOptional = as_jinja(
        """
        {%- for field in _this_node.fields -%}
          real(c_double), value, target, optional :: {{field.name}}_rel_tol
          real(c_double), value, target, optional :: {{field.name}}_abs_tol
        {% endfor -%}
      """
    )

    F90ErrToleranceDeclarations = as_jinja(
        """
        {%- for field in _this_node.fields -%}
        real(c_double) :: {{field.name}}_rel_err_tol
        real(c_double) :: {{field.name}}_abs_err_tol
        {% endfor -%}
        """
    )

    F90ToleranceConditionals = as_jinja(
        """{%- for field in _this_node.fields -%}
        if (present({{field.name}}_rel_tol)) then
            {{field.name}}_rel_err_tol = {{field.name}}_rel_tol
        else
            {{field.name}}_rel_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD
        endif

        if (present({{field.name}}_abs_tol)) then
            {{field.name}}_abs_err_tol = {{field.name}}_abs_tol
        else
            {{field.name}}_abs_err_tol = DEFAULT_ABSOLUTE_ERROR_THRESHOLD
        endif
        {% endfor %}
        """
    )

    F90OpenACCSection = as_jinja(
        """{%- for field in _this_node.all_fields -%}
        !$ACC {{field.name}}, &
        {% endfor -%}
        {%- for field in _this_node.out_fields -%}
        !$ACC {{field.name}}_before{% if not loop.last %}, &
        {% else %} & {% endif %}
        {%- endfor -%}
        """
    )

    F90KNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.name}}_kmax{% if not loop.last %}, &
            {% else %} &
            {% endif %}
        {% endfor -%}
        """
    )

    F90TypedKNamesOptional = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            integer(c_int), value, target, optional :: {{field.name}}_kmax
        {% endfor -%}
        """
    )

    F90TypedKNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            integer(c_int), value, target :: {{field.name}}_kmax
        {% endfor -%}
        """
    )

    F90VertDeclarations = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            integer(c_int) :: {{field.name}}_kvert_max
        {% endfor -%}
        """
    )

    F90VertNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.name}}_kvert_max{% if not loop.last %}, &
        {% else %} &
        {% endif %}
        {%- endfor -%}
        """
    )

    F90VertConditionals = as_jinja(
        """
        {%- for field in _this_node.fields -%}
        if (present({{field.name}}_kmax)) then
            {{field.name}}_kvert_max = {{field.name}}_kmax
        else
            {{field.name}}_kvert_max = k_size
        endif
        {% endfor -%}
        """
    )

    F90RunFun = as_jinja(
        """\
        subroutine &
        run_{{stencil_name}}( &
        {{params}}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {{binds}}        
        end subroutine
        """
    )

    F90RunAndVerifyFun = as_jinja(
        """
        subroutine &
        run_and_verify_{{stencil_name}}( &
        {{params}}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {{binds}}
        end subroutine
        """
    )

    F90SetupFun = as_jinja(
        """\
        subroutine &
        setup_{{stencil_name}}( &
        {{params}}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {{binds}}
        end subroutine
        """
    )

    F90WrapRunFun = as_jinja(
        """\
        subroutine &
        wrap_run_{{stencil_name}}( &
        {{params}}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        {{binds}}
        {{tol_decls}}
        integer(c_int) :: vertical_start
        integer(c_int) :: vertical_end
        integer(c_int) :: horizontal_start
        integer(c_int) :: horizontal_end
        vertical_start = vertical_lower-1
        vertical_end = vertical_upper
        horizontal_start = horizontal_lower-1
        horizontal_end = horizontal_upper
        {{conditionals}}
        !$ACC host_data use_device( &
        {{openacc}}
        !$ACC )
        #ifdef __DSL_VERIFY
            call run_and_verify_{{stencil_name}} &
            ( &
            {{run_ver_params}}
            )
        #else
            call run_{{stencil_name}} &
            ( &
            {{run_params}}
            )
        #endif
        !$ACC end host_data
        end subroutine
        """
    )

    F90WrapSetupFun = as_jinja(
        """\
        subroutine &
        wrap_setup_{{stencil_name}}( &
        {{params}}
        )
        use, intrinsic :: iso_c_binding
        use openacc        
        {{binds}}
        {{vert_decls}}
        {{vert_conditionals}}
        call setup_{{stencil_name}} &
        ( &
            {{setup_params}}
        )
        end subroutine
        """
    )

    F90FieldList = as_jinja(
        """
        {%- for field in fields -%}
        {{ field }}{% if not loop.last %}{{ line_end }}
        {% else %}{{ line_end_last }}{% endif %}
        {%- endfor -%}
        """
    )

    F90Field = as_jinja("{{ name }}{% if suffix %}_{{ suffix }}{% endif %}")

    F90OpenACCField = as_jinja(
        "!$ACC    {{ name }}{% if suffix %}_{{ suffix }}{% endif %}"
    )

    F90TypedField = as_jinja(
        "{{ type }}, {% if dims %}{{ dims }},{% endif %} target {% if _this_node.optional %} , optional {% endif %}:: {{ name }}{% if suffix %}_{{ suffix }}{% endif %} "
    )

    F90Conditional = as_jinja(
        """if ({{ predicate }}) then
         {{ if_branch }}
      else 
         {{ else_branch }}
      end if"""
    )


class F90FieldContainer(Node):
    fields: Sequence[Field]


class F90FieldNames(F90FieldContainer):
    ...


class F90FieldNamesBefore(F90FieldContainer):
    ...


class F90TypedFieldNames(F90FieldContainer):
    ...


class F90TypedFieldNamesBefore(F90FieldContainer):
    ...


class F90RankedFieldNames(F90FieldContainer):
    ...


class F90RankedFieldNamesBefore(F90FieldContainer):
    ...


class F90ToleranceArgs(F90FieldContainer):
    ...


class F90ErrToleranceArgs(F90FieldContainer):
    ...


class F90TypedToleranceArgs(F90FieldContainer):
    ...


class F90TypedToleranceArgsOptional(F90FieldContainer):
    ...


class F90ErrToleranceDeclarations(F90FieldContainer):
    ...


class F90ToleranceConditionals(F90FieldContainer):
    ...


class F90KNames(F90FieldContainer):
    ...


class F90TypedKNames(F90FieldContainer):
    ...


class F90TypedKNamesOptional(F90FieldContainer):
    ...


class F90VertDeclarations(F90FieldContainer):
    ...


class F90VertNames(F90FieldContainer):
    ...


class F90VertConditionals(F90FieldContainer):
    ...


class F90OpenACCSection(Node):
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]


class F90Field(eve.Node):
    name: str
    suffix: str = ""


class F90OpenACCField(F90Field):
    ...


class F90TypedField(F90Field):
    type: str
    dims: str = ""
    optional: bool = False


class F90Conditional(eve.Node):
    predicate: str
    if_branch: str
    else_branch: str


class F90FieldList(eve.Node):
    fields: Sequence[Union[F90Field, F90Conditional]]
    line_end: str = ""
    line_end_last: str = ""


class F90RunFun(eve.Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90FieldList = eve.datamodels.field(init=False)
    binds: F90FieldList = eve.datamodels.field(init=False)

    def __post_init__(self):
        param_decls = [F90Field(name=field.name) for field in self.all_fields] + [
            F90Field(name=name)
            for name in [
                "vertical_lower",
                "vertical_upper",
                "horizontal_lower",
                "horizontal_upper",
            ]
        ]
        bind_decls = [
            F90TypedField(
                name=field.name,
                type=field.renderer.render_ctype("f90"),
                dims=field.renderer.render_dim_string(),
            )
            for field in self.all_fields
        ] + [
            F90TypedField(name=name, type="integer(c_int)", dims="value")
            for name in [
                "vertical_lower",
                "vertical_upper",
                "horizontal_lower",
                "horizontal_upper",
            ]
        ]

        self.params = F90FieldList(
            fields=param_decls, line_end=", &", line_end_last=" &"
        )
        self.binds = F90FieldList(fields=bind_decls)


class F90RunAndVerifyFun(eve.Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90FieldList = eve.datamodels.field(init=False)
    binds: F90FieldList = eve.datamodels.field(init=False)

    def __post_init__(self):
        param_decls = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [
                F90Field(name=name)
                for name in [
                    "vertical_lower",
                    "vertical_upper",
                    "horizontal_lower",
                    "horizontal_upper",
                ]
            ]
        )
        bind_decls = (
            [
                F90TypedField(
                    name=field.name,
                    type=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_dim_string(),
                )
                for field in self.all_fields
            ]
            + [
                F90TypedField(
                    name=field.name,
                    suffix="before",
                    type=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_dim_string(),
                )
                for field in self.out_fields
            ]
            + [
                F90TypedField(name=name, type="integer(c_int)", dims="value")
                for name in [
                    "vertical_lower",
                    "vertical_upper",
                    "horizontal_lower",
                    "horizontal_upper",
                ]
            ]
        )

        for field in self.out_fields:
            param_decls += [
                F90Field(name=field.name, suffix=s) for s in ["rel_tol", "abs_tol"]
            ]
            bind_decls += [
                F90TypedField(
                    name=field.name, suffix=s, type="real(c_double)", dims="value"
                )
                for s in ["rel_tol", "abs_tol"]
            ]

        self.params = F90FieldList(
            fields=param_decls, line_end=", &", line_end_last=" &"
        )
        self.binds = F90FieldList(fields=bind_decls)


class F90SetupFun(Node):
    stencil_name: str
    out_fields: Sequence[Field]

    params: F90FieldList = eve.datamodels.field(init=False)
    binds: F90FieldList = eve.datamodels.field(init=False)

    def __post_init__(self):
        param_decls = [F90Field(name=name) for name in ["mesh", "k_size", "stream"]] + [
            F90Field(name=field.name, suffix="kmax") for field in self.out_fields
        ]
        bind_decls = [
            F90TypedField(name="mesh", type="type(c_ptr)", dims="value"),
            F90TypedField(name="k_size", type="integer(c_int)", dims="value"),
            F90TypedField(
                name="stream", type="integer(kind=acc_handle_kind)", dims="value"
            ),
        ] + [
            F90TypedField(
                name=field.name, type="integer(c_int)", dims="value", suffix="kmax"
            )
            for field in self.out_fields
        ]

        self.params = F90FieldList(
            fields=param_decls, line_end=", &", line_end_last=" &"
        )
        self.binds = F90FieldList(fields=bind_decls)


class F90WrapRunFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90FieldList = eve.datamodels.field(init=False)
    binds: F90FieldList = eve.datamodels.field(init=False)
    conditionals: F90FieldList = eve.datamodels.field(init=False)
    openacc: F90FieldList = eve.datamodels.field(init=False)
    tol_decls: F90FieldList = eve.datamodels.field(init=False)
    run_ver_params: F90FieldList = eve.datamodels.field(init=False)
    run_params: F90FieldList = eve.datamodels.field(init=False)

    def __post_init__(self):
        param_decls = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [
                F90Field(name=name)
                for name in [
                    "vertical_lower",
                    "vertical_upper",
                    "horizontal_lower",
                    "horizontal_upper",
                ]
            ]
        )
        bind_decls = (
            [
                F90TypedField(
                    name=field.name,
                    type=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_ranked_dim_string(),
                )
                for field in self.all_fields
            ]
            + [
                F90TypedField(
                    name=field.name,
                    suffix="before",
                    type=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_ranked_dim_string(),
                )
                for field in self.out_fields
            ]
            + [
                F90TypedField(name=name, type="integer(c_int)", dims="value")
                for name in [
                    "vertical_lower",
                    "vertical_upper",
                    "horizontal_lower",
                    "horizontal_upper",
                ]
            ]
        )
        tol = [
            F90TypedField(name=field.name, suffix=s, type="real(c_double)")
            for s in ["rel_err_tol", "abs_err_tol"]
            for field in self.out_fields
        ]
        cond_decls = [
            F90Conditional(
                predicate=f"present({field.name}_{s}_tol)",
                if_branch=f"{field.name}_{s}_err_tol = {field.name}_{s}_tol",
                else_branch=f"{field.name}_{s}_err_tol = DEFAULT_RELATIVE_ERROR_THRESHOLD",
            )
            for s in ["rel", "abs"]
            for field in self.out_fields
        ]
        open_acc_decls = [
            F90OpenACCField(name=field.name)
            for field in self.all_fields
            if field.rank() != 0
        ] + [
            F90OpenACCField(name=field.name)
            for field in self.out_fields
            if field.rank() != 0
        ]
        run_ver_param_decls = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [
                F90Field(name=name)
                for name in [
                    "vertical_start",
                    "vertical_end",
                    "horizontal_start",
                    "horizontal_end",
                ]
            ]
        )
        run_param_decls = [F90Field(name=field.name) for field in self.all_fields] + [
            F90Field(name=name)
            for name in [
                "vertical_start",
                "vertical_end",
                "horizontal_start",
                "horizontal_end",
            ]
        ]

        for field in self.out_fields:
            param_decls += [
                F90Field(name=field.name, suffix=s) for s in ["rel_tol", "abs_tol"]
            ]
            bind_decls += [
                F90TypedField(
                    name=field.name,
                    suffix=s,
                    type="real(c_double)",
                    dims="value",
                    optional=True,
                )
                for s in ["rel_tol", "abs_tol"]
            ]
            run_ver_param_decls += [
                F90Field(name=field.name, suffix=s)
                for s in ["rel_err_tol", "abs_err_tol"]
            ]

        self.params = F90FieldList(
            fields=param_decls, line_end=", &", line_end_last=" &"
        )
        self.binds = F90FieldList(fields=bind_decls)
        self.tol_decls = F90FieldList(fields=tol)
        self.conditionals = F90FieldList(fields=cond_decls)
        self.openacc = F90FieldList(
            fields=open_acc_decls, line_end=", &", line_end_last=" &"
        )
        self.run_ver_params = F90FieldList(
            fields=run_ver_param_decls, line_end=", &", line_end_last=" &"
        )
        self.run_params = F90FieldList(
            fields=run_param_decls, line_end=", &", line_end_last=" &"
        )


class F90WrapSetupFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90FieldList = eve.datamodels.field(init=False)
    binds: F90FieldList = eve.datamodels.field(init=False)
    vert_decls: F90FieldList = eve.datamodels.field(init=False)
    vert_conditionals: F90FieldList = eve.datamodels.field(init=False)
    setup_params: F90FieldList = eve.datamodels.field(init=False)

    def __post_init__(self):
        param_decls = [F90Field(name=name) for name in ["mesh", "k_size", "stream"]] + [
            F90Field(name=field.name, suffix="kmax") for field in self.out_fields
        ]
        bind_decls = [
            F90TypedField(name="mesh", type="type(c_ptr)", dims="value"),
            F90TypedField(name="k_size", type="integer(c_int)", dims="value"),
            F90TypedField(
                name="stream", type="integer(kind=acc_handle_kind)", dims="value"
            ),
        ] + [
            F90TypedField(
                name=field.name,
                type="integer(c_int)",
                dims="value",
                suffix="kmax",
                optional=True,
            )
            for field in self.out_fields
        ]
        vert = [
            F90TypedField(name=field.name, suffix="kvert_max", type="integer(c_int)")
            for field in self.out_fields
        ]
        vert_conditionals_decl = [
            F90Conditional(
                predicate=f"present({field.name}_kmax)",
                if_branch=f"{field.name}_kvert_max = {field.name}_kmax",
                else_branch=f"{field.name}_kvert_max = k_size",
            )
            for field in self.out_fields
        ]
        setup_params_decl = [F90Field(name=name) for name in ["mesh", "k_size", "stream"]] + [
            F90Field(name=field.name, suffix="kvert_max") for field in self.out_fields
        ]

        self.params = F90FieldList(
            fields=param_decls, line_end=", &", line_end_last=" &"
        )
        self.binds = F90FieldList(fields=bind_decls)
        self.vert_decls = F90FieldList(fields=vert)
        self.vert_conditionals = F90FieldList(fields=vert_conditionals_decl)
        self.setup_params = F90FieldList(fields=setup_params_decl, line_end=", &", line_end_last=" &")


class F90File(Node):
    stencil_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    run_fun: F90RunFun = eve.datamodels.field(init=False)
    run_and_verify_fun: F90RunAndVerifyFun = eve.datamodels.field(init=False)
    setup_fun: F90SetupFun = eve.datamodels.field(init=False)
    wrap_run_fun: F90WrapRunFun = eve.datamodels.field(init=False)
    wrap_setup_fun: F90WrapSetupFun = eve.datamodels.field(init=False)

    def __post_init__(self):
        all_fields = self.fields
        out_fields = [field for field in self.fields if field.intent.out]

        self.run_fun = F90RunFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )

        self.run_and_verify_fun = F90RunAndVerifyFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )

        self.setup_fun = F90SetupFun(
            stencil_name=self.stencil_name,
            out_fields=out_fields,
        )

        self.wrap_run_fun = F90WrapRunFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )

        self.wrap_setup_fun = F90WrapSetupFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )


def generate_f90_file(
    stencil_name: str, fields: Sequence[Field], offsets: Sequence[Offset], outpath: Path
) -> None:
    f90 = F90File(stencil_name=stencil_name, fields=fields, offsets=offsets)
    source = F90Generator.apply(f90)
    formatted_source = format_fortran_code(source)
    write_string(formatted_source, outpath, f"{stencil_name}.f90")
