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
from typing import Any, Sequence, Union

from gt4py import eve
from gt4py.eve import Node
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator

from icon4pytools.icon4pygen.bindings.entities import Field, Offset
from icon4pytools.icon4pygen.bindings.utils import format_fortran_code, write_string


_DOMAIN_ARGS = [
    "vertical_lower",
    "vertical_upper",
    "horizontal_lower",
    "horizontal_upper",
]


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
        {{k_sizes}}
        vertical_start = vertical_lower-1
        vertical_end = vertical_upper
        horizontal_start = horizontal_lower-1
        horizontal_end = horizontal_upper
        {{k_sizes_assignments}}
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
        call setup_{{stencil_name}} &
        ( &
            {{setup_params}}
        )
        end subroutine
        """
    )

    F90EntityList = as_jinja(
        """
        {%- for field in fields -%}
        {{ field }}{% if not loop.last %}{{ line_end }}
        {% else %}{{ line_end_last }}{% endif %}
        {%- endfor -%}
        """
    )

    F90Field = as_jinja("{{ name }}{% if suffix %}_{{ suffix }}{% endif %}")

    F90OpenACCField = as_jinja("!$ACC    {{ name }}{% if suffix %}_{{ suffix }}{% endif %}")

    F90TypedField = as_jinja(
        "{{ dtype }}, {% if dims %}{{ dims }},{% endif %} target {% if _this_node.optional %} , optional {% endif %}:: {{ name }}{% if suffix %}_{{ suffix }}{% endif %} "
    )

    F90Conditional = as_jinja(
        """if ({{ predicate }}) then
         {{ if_branch }}
      else
         {{ else_branch }}
      end if"""
    )

    F90Assignment = as_jinja("{{ left_side }} = {{ right_side }}")


class F90Field(eve.Node):
    name: str
    suffix: str = ""


class F90OpenACCField(F90Field):
    ...


class F90TypedField(F90Field):
    dtype: str
    dims: str = ""
    optional: bool = False


class F90Conditional(eve.Node):
    predicate: str
    if_branch: str
    else_branch: str


class F90Assignment(eve.Node):
    left_side: str
    right_side: str


class F90EntityList(eve.Node):
    fields: Sequence[Union[F90Field, F90Conditional, F90Assignment]]
    line_end: str = ""
    line_end_last: str = ""


class F90RunFun(eve.Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90EntityList = eve.datamodels.field(init=False)
    binds: F90EntityList = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        param_fields = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="k_size") for field in self.out_fields]
            + [F90Field(name=name) for name in _DOMAIN_ARGS]
        )
        bind_fields = [
            F90TypedField(
                name=field.name,
                dtype=field.renderer.render_ctype("f90"),
                dims=field.renderer.render_dim_string(),
            )
            for field in self.all_fields
        ] + [
            F90TypedField(name=field.name, suffix="k_size", dtype="integer(c_int)", dims="value")
            for field in self.out_fields
        ] + [
            F90TypedField(name=name, dtype="integer(c_int)", dims="value") for name in _DOMAIN_ARGS
        ]

        self.params = F90EntityList(fields=param_fields, line_end=", &", line_end_last=" &")
        self.binds = F90EntityList(fields=bind_fields)


class F90RunAndVerifyFun(eve.Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]
    tol_fields: Sequence[Field]

    params: F90EntityList = eve.datamodels.field(init=False)
    binds: F90EntityList = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        param_fields = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [F90Field(name=field.name, suffix="k_size") for field in self.out_fields]
            + [F90Field(name=name) for name in _DOMAIN_ARGS]
        )
        bind_fields = (
            [
                F90TypedField(
                    name=field.name,
                    dtype=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_dim_string(),
                )
                for field in self.all_fields
            ]
            + [
                F90TypedField(
                    name=field.name,
                    suffix="before",
                    dtype=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_dim_string(),
                )
                for field in self.out_fields
            ]
            + [
                F90TypedField(name=field.name, suffix="k_size", dtype="integer(c_int)", dims="value")
                for field in self.out_fields
            ]
            + [
                F90TypedField(name=name, dtype="integer(c_int)", dims="value")
                for name in _DOMAIN_ARGS
            ]
        )

        for field in self.tol_fields:
            param_fields += [F90Field(name=field.name, suffix=s) for s in ["rel_tol", "abs_tol"]]
            bind_fields += [
                F90TypedField(
                    name=field.name,
                    suffix=s,
                    dtype="real(c_double)",
                    dims="value",
                )
                for s in ["rel_tol", "abs_tol"]
            ]

        self.params = F90EntityList(fields=param_fields, line_end=", &", line_end_last=" &")
        self.binds = F90EntityList(fields=bind_fields)


class F90SetupFun(Node):
    stencil_name: str
    out_fields: Sequence[Field]

    params: F90EntityList = eve.datamodels.field(init=False)
    binds: F90EntityList = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        param_fields = [
            F90Field(name=name)
            for name in [
                "mesh",
                "stream",
                "json_record",
                "mesh_info_vtk",
                "verify",
            ]
        ]
        bind_fields = [
            F90TypedField(name="mesh", dtype="type(c_ptr)", dims="value"),
            F90TypedField(
                name="stream",
                dtype="integer(kind=acc_handle_kind)",
                dims="value",
            ),
            F90TypedField(name="json_record", dtype="type(c_ptr)", dims="value"),
            F90TypedField(name="mesh_info_vtk", dtype="type(c_ptr)", dims="value"),
            F90TypedField(name="verify", dtype="type(c_ptr)", dims="value"),
        ]

        self.params = F90EntityList(fields=param_fields, line_end=", &", line_end_last=" &")
        self.binds = F90EntityList(fields=bind_fields)


class F90WrapRunFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]
    tol_fields: Sequence[Field]

    params: F90EntityList = eve.datamodels.field(init=False)
    binds: F90EntityList = eve.datamodels.field(init=False)
    conditionals: F90EntityList = eve.datamodels.field(init=False)
    k_sizes: F90EntityList = eve.datamodels.field(init=False)
    k_sizes_assignments: F90EntityList = eve.datamodels.field(init=False)
    openacc: F90EntityList = eve.datamodels.field(init=False)
    tol_decls: F90EntityList = eve.datamodels.field(init=False)
    run_ver_params: F90EntityList = eve.datamodels.field(init=False)
    run_params: F90EntityList = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        param_fields = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [F90Field(name=name) for name in _DOMAIN_ARGS]
        )
        bind_fields = (
            [
                F90TypedField(
                    name=field.name,
                    dtype=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_ranked_dim_string(),
                )
                for field in self.all_fields
            ]
            + [
                F90TypedField(
                    name=field.name,
                    suffix="before",
                    dtype=field.renderer.render_ctype("f90"),
                    dims=field.renderer.render_ranked_dim_string(),
                )
                for field in self.out_fields
            ]
            + [
                F90TypedField(name=name, dtype="integer(c_int)", dims="value")
                for name in _DOMAIN_ARGS
            ]
        )
        tol_fields = [
            F90TypedField(name=field.name, suffix=s, dtype="real(c_double)")
            for s in ["rel_err_tol", "abs_err_tol"]
            for field in self.tol_fields
        ]
        k_sizes_fields = [
            F90TypedField(name=field.name, suffix=s, dtype="integer")
            for s in ["k_size"]
            for field in self.out_fields
        ]
        k_sizes_assignment_fields = [
            F90Assignment(
                left_side=f"{field.name}_k_size",
                right_side=f"SIZE({field.name}, 2)",
            )
            for field in self.out_fields
        ]
        cond_fields = [
            F90Conditional(
                predicate=f"present({field.name}_{short}_tol)",
                if_branch=f"{field.name}_{short}_err_tol = {field.name}_{short}_tol",
                else_branch=f"{field.name}_{short}_err_tol = DEFAULT_{long}_ERROR_THRESHOLD",
            )
            for short, long in [("rel", "RELATIVE"), ("abs", "ABSOLUTE")]
            for field in self.tol_fields
        ]
        open_acc_fields = [
            F90OpenACCField(name=field.name) for field in self.all_fields if field.rank() != 0
        ] + [
            F90OpenACCField(name=field.name, suffix="before")
            for field in self.out_fields
            if field.rank() != 0
        ]
        run_ver_param_fields = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="before") for field in self.out_fields]
            + [F90Field(name=field.name, suffix="k_size") for field in self.out_fields]
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
        run_param_fields = (
            [F90Field(name=field.name) for field in self.all_fields]
            + [F90Field(name=field.name, suffix="k_size") for field in self.out_fields]
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

        for field in self.tol_fields:
            param_fields += [F90Field(name=field.name, suffix=s) for s in ["rel_tol", "abs_tol"]]
            bind_fields += [
                F90TypedField(
                    name=field.name,
                    suffix=s,
                    dtype="real(c_double)",
                    dims="value",
                    optional=True,
                )
                for s in ["rel_tol", "abs_tol"]
            ]
            run_ver_param_fields += [
                F90Field(name=field.name, suffix=s) for s in ["rel_err_tol", "abs_err_tol"]
            ]

        self.params = F90EntityList(fields=param_fields, line_end=", &", line_end_last=" &")
        self.binds = F90EntityList(fields=bind_fields)
        self.tol_decls = F90EntityList(fields=tol_fields)
        self.conditionals = F90EntityList(fields=cond_fields)
        self.k_sizes = F90EntityList(fields=k_sizes_fields)
        self.k_sizes_assignments = F90EntityList(fields=k_sizes_assignment_fields)
        self.openacc = F90EntityList(fields=open_acc_fields, line_end=", &", line_end_last=" &")
        self.run_ver_params = F90EntityList(
            fields=run_ver_param_fields, line_end=", &", line_end_last=" &"
        )
        self.run_params = F90EntityList(fields=run_param_fields, line_end=", &", line_end_last=" &")


class F90WrapSetupFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    out_fields: Sequence[Field]

    params: F90EntityList = eve.datamodels.field(init=False)
    binds: F90EntityList = eve.datamodels.field(init=False)
    setup_params: F90EntityList = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        param_fields = [
            F90Field(name=name)
            for name in [
                "mesh",
                "stream",
                "json_record",
                "mesh_info_vtk",
                "verify",
            ]
        ]
        bind_fields = [
            F90TypedField(name="mesh", dtype="type(c_ptr)", dims="value"),
            F90TypedField(
                name="stream",
                dtype="integer(kind=acc_handle_kind)",
                dims="value",
            ),
            F90TypedField(name="json_record", dtype="type(c_ptr)", dims="value"),
            F90TypedField(name="mesh_info_vtk", dtype="type(c_ptr)", dims="value"),
            F90TypedField(name="verify", dtype="type(c_ptr)", dims="value"),
        ]
        setup_params_fields = [
            F90Field(name=name)
            for name in [
                "mesh",
                "stream",
                "json_record",
                "mesh_info_vtk",
                "verify",
            ]
        ]

        self.params = F90EntityList(fields=param_fields, line_end=", &", line_end_last=" &")
        self.binds = F90EntityList(fields=bind_fields)
        self.setup_params = F90EntityList(
            fields=setup_params_fields, line_end=", &", line_end_last=" &"
        )


class F90File(Node):
    stencil_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]

    run_fun: F90RunFun = eve.datamodels.field(init=False)
    run_and_verify_fun: F90RunAndVerifyFun = eve.datamodels.field(init=False)
    setup_fun: F90SetupFun = eve.datamodels.field(init=False)
    wrap_run_fun: F90WrapRunFun = eve.datamodels.field(init=False)
    wrap_setup_fun: F90WrapSetupFun = eve.datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        all_fields = self.fields
        out_fields = [field for field in self.fields if field.intent.out]
        tol_fields = [field for field in out_fields if not field.is_integral()]

        self.run_fun = F90RunFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )

        self.run_and_verify_fun = F90RunAndVerifyFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
            tol_fields=tol_fields,
        )

        self.setup_fun = F90SetupFun(
            stencil_name=self.stencil_name,
            out_fields=out_fields,
        )

        self.wrap_run_fun = F90WrapRunFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
            tol_fields=tol_fields,
        )

        self.wrap_setup_fun = F90WrapSetupFun(
            stencil_name=self.stencil_name,
            all_fields=all_fields,
            out_fields=out_fields,
        )


def generate_f90_file(
    stencil_name: str,
    fields: Sequence[Field],
    offsets: Sequence[Offset],
    outpath: Path,
) -> None:
    f90 = F90File(stencil_name=stencil_name, fields=fields, offsets=offsets)
    source = F90Generator.apply(f90)
    formatted_source = format_fortran_code(source)
    write_string(formatted_source, outpath, f"{stencil_name}.f90")
