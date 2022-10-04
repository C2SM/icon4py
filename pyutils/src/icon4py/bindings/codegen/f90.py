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

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.bindings.entities import Field, Offset
from icon4py.bindings.utils import write_string


class F90Generator(TemplatedGenerator):
    F90File = as_jinja(
        """
        #define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12
        #define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1
        module {{sten_name}}
        use, intrinsic :: iso_c_binding
        implicit none
        interface
        {{run_fun}}
        {{run_and_verify_fun}}
        {{setup_fun}}
        subroutine &
        free_{{sten_name}}( ) bind(c)
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
            {{field.render_ctype('f90')}}, {{ field.render_dim_string() }}, target :: {{field.name}} {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90TypedFieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.render_ctype('f90')}}, {{ field.render_dim_string() }}, target :: {{field.name}}_before {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90RankedFieldNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.render_ctype('f90')}}, {{field.render_ranked_dim_string()}}, target :: {{field.name}} {% if not loop.last %}
            {% else %}{% endif %}
           {%- endfor -%}
        """
    )

    F90RankedFieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.render_ctype('f90')}}, {{field.render_ranked_dim_string()}}, target :: {{field.name}}_before {% if not loop.last %}
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
        run_{{sten_name}}( &
        {{field_names}}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper &
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {{typed_field_names}}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        end subroutine
        """
    )

    F90RunAndVerifyFun = as_jinja(
        """
        subroutine &
        run_and_verify_{{sten_name}}( &
        {{field_names}}
        {{field_names_before}}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper, &
        {{tolerance_args}}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        {{typed_field_names}}
        {{typed_field_names_before}}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        {{typed_tolerance_args}}
        end subroutine
        """
    )

    F90SetupFun = as_jinja(
        """\
        subroutine &
        setup_{{sten_name}}( &
        mesh, &
        k_size, &
        stream, &
        {{vert_names}}
        ) bind(c)
        use, intrinsic :: iso_c_binding
        use openacc
        type(c_ptr), value, target :: mesh
        integer(c_int), value, target :: k_size
        integer(kind=acc_handle_kind), value, target :: stream
        {{typed_k_names}}
        end subroutine
        """
    )

    F90WrapRunFun = as_jinja(
        """\
        subroutine &
        wrap_run_{{sten_name}}( &
        {{field_names}}
        {{field_names_before}}
        vertical_lower, &
        vertical_upper, &
        horizontal_lower, &
        horizontal_upper, &
        {{tolerance_args}}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        {{ranked_field_names}}
        {{ranked_field_names_before}}
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        {{typed_tolerance_args_optional}}
        {{error_tolerance_declarations}}
        integer(c_int) :: vertical_start
        integer(c_int) :: vertical_end
        integer(c_int) :: horizontal_start
        integer(c_int) :: horizontal_end
        vertical_start = vertical_lower-1
        vertical_end = vertical_upper
        horizontal_start = horizontal_lower-1
        horizontal_end = horizontal_upper
        {{tolerance_conditionals}}
        !$ACC host_data use_device( &
        {{openacc_section}}
        !$ACC )
        #ifdef __DSL_VERIFY
            call run_and_verify_{{sten_name}} &
            ( &
            {{field_names}}
            {{field_names_before}}
            vertical_start, &
            vertical_end, &
            horizontal_start, &
            horizontal_end, &
            {{err_tolerance_args}}
            )
        #else
            call run_{{sten_name}} &
            ( &
            {{field_names}}
            vertical_start, &
            vertical_end, &
            horizontal_start, &
            horizontal_end &
            )
        #endif
        !$ACC end host_data
        end subroutine
        """
    )

    F90WrapSetupFun = as_jinja(
        """\
        subroutine &
        wrap_setup_{{sten_name}}( &
        mesh, &
        k_size, &
        stream, &
        {{k_names}}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        type(c_ptr), value, target :: mesh
        integer(c_int), value, target :: k_size
        integer(kind=acc_handle_kind), value, target :: stream
        {{typed_k_names_optional}}
        {{vert_declarations}}
        {{vert_conditionals}}
        call setup_{{sten_name}} &
        ( &
            mesh, &
            k_size, &
            stream, &
            {{vert_names}}
        )
        end subroutine
        """
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


class F90RunFun(Node):
    sten_name: str
    field_names: F90FieldNames
    typed_field_names: F90TypedFieldNames


class F90RunAndVerifyFun(Node):
    sten_name: str
    field_names: F90FieldNames
    field_names_before: F90FieldNamesBefore
    tolerance_args: F90ToleranceArgs
    typed_field_names: F90TypedFieldNames
    typed_field_names_before: F90TypedFieldNamesBefore
    typed_tolerance_args: F90TypedToleranceArgs


class F90SetupFun(Node):
    sten_name: str
    vert_names: F90KNames
    typed_k_names: F90TypedKNames


class F90WrapRunFun(Node):
    sten_name: str
    field_names: F90FieldNames
    field_names_before: F90FieldNamesBefore
    tolerance_args: F90ToleranceArgs
    ranked_field_names: F90RankedFieldNames
    ranked_field_names_before: F90RankedFieldNamesBefore
    typed_tolerance_args_optional: F90TypedToleranceArgsOptional
    error_tolerance_declarations: F90ErrToleranceDeclarations
    tolerance_conditionals: F90ToleranceConditionals
    openacc_section: F90OpenACCSection
    err_tolerance_args: F90ErrToleranceArgs


class F90WrapSetupFun(Node):
    sten_name: str
    k_names: F90KNames
    typed_k_names_optional: F90TypedKNamesOptional
    vert_declarations: F90VertDeclarations
    vert_conditionals: F90VertConditionals
    vert_names: F90VertNames


class F90File(Node):
    sten_name: str
    run_fun: F90RunFun
    run_and_verify_fun: F90RunAndVerifyFun
    setup_fun: F90SetupFun
    wrap_run_fun: F90WrapRunFun
    wrap_setup_fun: F90WrapSetupFun


@dataclass
class F90Iface:
    sten_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    def _format_code_subprocesss(self, source: str) -> str:
        args = ["fprettify"]
        p1 = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        return p1.communicate(source.encode("UTF-8"))[0].decode("UTF-8").rstrip()

    def write(self, outpath: Path) -> None:
        iface = self._generate_iface()
        source = F90Generator.apply(iface)
        formatted_source = self._format_code_subprocesss(source)
        write_string(formatted_source, outpath, f"{self.sten_name}.f90")

    def _generate_iface(self):
        all_fields = self.fields
        out_fields = [field for field in self.fields if field.intent.out]
        iface = F90File(
            sten_name=self.sten_name,
            run_fun=F90RunFun(
                sten_name=self.sten_name,
                field_names=F90FieldNames(fields=all_fields),
                typed_field_names=F90TypedFieldNames(fields=all_fields),
            ),
            run_and_verify_fun=F90RunAndVerifyFun(
                sten_name=self.sten_name,
                field_names=F90FieldNames(fields=all_fields),
                field_names_before=F90FieldNamesBefore(fields=out_fields),
                tolerance_args=F90ToleranceArgs(fields=out_fields),
                typed_field_names=F90TypedFieldNames(fields=all_fields),
                typed_field_names_before=F90TypedFieldNamesBefore(fields=out_fields),
                typed_tolerance_args=F90TypedToleranceArgs(fields=out_fields),
            ),
            setup_fun=F90SetupFun(
                sten_name=self.sten_name,
                vert_names=F90KNames(fields=out_fields),
                typed_k_names=F90TypedKNames(fields=out_fields),
            ),
            wrap_run_fun=F90WrapRunFun(
                sten_name=self.sten_name,
                field_names=F90FieldNames(fields=all_fields),
                field_names_before=F90FieldNamesBefore(fields=out_fields),
                tolerance_args=F90ToleranceArgs(fields=out_fields),
                ranked_field_names=F90RankedFieldNames(fields=all_fields),
                ranked_field_names_before=F90RankedFieldNamesBefore(fields=out_fields),
                typed_tolerance_args_optional=F90TypedToleranceArgsOptional(
                    fields=out_fields
                ),
                error_tolerance_declarations=F90ErrToleranceDeclarations(
                    fields=out_fields
                ),
                tolerance_conditionals=F90ToleranceConditionals(fields=out_fields),
                openacc_section=F90OpenACCSection(
                    all_fields=[field for field in all_fields if field.rank() != 0],
                    out_fields=[field for field in out_fields if field.rank() != 0],
                ),
                err_tolerance_args=F90ErrToleranceArgs(fields=out_fields),
            ),
            wrap_setup_fun=F90WrapSetupFun(
                sten_name=self.sten_name,
                k_names=F90KNames(fields=out_fields),
                typed_k_names_optional=F90TypedKNamesOptional(fields=out_fields),
                vert_declarations=F90VertDeclarations(fields=out_fields),
                vert_conditionals=F90VertConditionals(fields=out_fields),
                vert_names=F90VertNames(fields=out_fields),
            ),
        )
        return iface
