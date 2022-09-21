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

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eve import Node
from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import TemplatedGenerator

from icon4py.bindings.types import Field, Offset


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
            {{field.ctype('f90')}}, dimension(*), target :: {{field.name}} {% if not loop.last %}
            {% else %}{% endif %}            
           {%- endfor -%}
        """
    )

    F90TypedFieldNamesBefore = as_jinja(
        """
        {%- for field in _this_node.fields -%} 
            {{field.ctype('f90')}}, dimension(*), target :: {{field.name}}_before {% if not loop.last %}
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
        {%- endfor -%}
      """
    )

    F90TypedToleranceArgs = as_jinja(
        """
        {%- for field in _this_node.fields -%}
          real(c_double), value, target :: {{field.name}}_rel_tol
          real(c_double), value, target :: {{field.name}}_abs_tol
        {%- endfor -%}
      """
    )

    F90ErrToleranceDeclarations = as_jinja(
        """
        {%- for field in _this_node.fields -%}
        real(c_double) :: {{field.name}}_rel_err_tol
        real(c_double) :: {{field.name}}_abs_err_tol
        {%- endfor -%}
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
        {%- endfor %}
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

    F90VertNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            {{field.name}}_kmax, & {% if not loop.last %}
            {% else %}{% endif %}             
        {%- endfor -%}
        """
    )

    F90TypedVertNames = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            integer(c_int), value, target, optional :: {{field.name}}_kmax
        {%- endfor -%}
        """
    )

    F90VertDeclarations = as_jinja(
        """
        {%- for field in _this_node.fields -%}
            integer(c_int) :: {{field.name}}_kvert_max
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
        {%- endfor -%}
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
        {{typed_vert_names}}
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
        {{typed_field_names}}
        {{typed_field_names_before}}        
        integer(c_int), value, target :: vertical_lower
        integer(c_int), value, target :: vertical_upper
        integer(c_int), value, target :: horizontal_lower
        integer(c_int), value, target :: horizontal_upper
        {{typed_tolerance_args}}
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
            {{tolerance_args}}
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
        {{vert_names}}
        )
        use, intrinsic :: iso_c_binding
        use openacc
        type(c_ptr), value, target :: mesh
        integer(c_int), value, target :: k_size
        integer(kind=acc_handle_kind), value, target :: stream
        {{typed_vert_names}}
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


class F90FieldNames(Node):
    fields: Sequence[Field]


class F90FieldNamesBefore(Node):
    fields: Sequence[Field]


class F90TypedFieldNames(Node):
    fields: Sequence[Field]


class F90TypedFieldNamesBefore(Node):
    fields: Sequence[Field]


class F90ToleranceArgs(Node):
    fields: Sequence[Field]


class F90TypedToleranceArgs(Node):
    fields: Sequence[Field]


class F90ErrToleranceDeclarations(Node):
    fields: Sequence[Field]


class F90ToleranceConditionals(Node):
    fields: Sequence[Field]


class F90VertNames(Node):
    fields: Sequence[Field]


class F90TypedVertNames(Node):
    fields: Sequence[Field]


class F90VertDeclarations(Node):
    fields: Sequence[Field]


class F90VertConditionals(Node):
    fields: Sequence[Field]


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
    vert_names: F90VertNames
    typed_vert_names: F90TypedVertNames


class F90WrapRunFun(Node):
    sten_name: str
    field_names: F90FieldNames
    field_names_before: F90FieldNamesBefore
    tolerance_args: F90ToleranceArgs
    typed_field_names: F90TypedFieldNames
    typed_field_names_before: F90TypedFieldNamesBefore
    typed_tolerance_args: F90TypedToleranceArgs
    error_tolerance_declarations: F90ErrToleranceDeclarations
    tolerance_conditionals: F90ToleranceConditionals
    openacc_section: F90OpenACCSection


class F90WrapSetupFun(Node):
    sten_name: str
    vert_names: F90VertNames
    typed_vert_names: F90TypedVertNames
    vert_declarations: F90VertDeclarations
    vert_conditionals: F90VertConditionals


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

    def write(self, outpath: Path):
        iface = self._generate_iface()
        source = F90Generator.apply(iface)
        # todo: code formatting & writing code to file.
        print(source)

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
                vert_names=F90VertNames(fields=out_fields),
                typed_vert_names=F90TypedVertNames(fields=out_fields),
            ),
            wrap_run_fun=F90WrapRunFun(
                sten_name=self.sten_name,
                field_names=F90FieldNames(fields=all_fields),
                field_names_before=F90FieldNamesBefore(fields=out_fields),
                tolerance_args=F90ToleranceArgs(fields=out_fields),
                typed_field_names=F90TypedFieldNames(fields=all_fields),
                typed_field_names_before=F90TypedFieldNamesBefore(fields=out_fields),
                typed_tolerance_args=F90TypedToleranceArgs(fields=out_fields),
                error_tolerance_declarations=F90ErrToleranceDeclarations(
                    fields=out_fields
                ),
                tolerance_conditionals=F90ToleranceConditionals(fields=out_fields),
                openacc_section=F90OpenACCSection(
                    all_fields=all_fields,
                    out_fields=out_fields,
                ),
            ),
            wrap_setup_fun=F90WrapSetupFun(
                sten_name=self.sten_name,
                vert_names=F90VertNames(fields=out_fields),
                typed_vert_names=F90TypedVertNames(fields=out_fields),
                vert_declarations=F90VertDeclarations(fields=out_fields),
                vert_conditionals=F90VertConditionals(fields=out_fields),
            ),
        )
        return iface
