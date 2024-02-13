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
from typing import Any, Sequence

from gt4py.eve import Node, codegen, datamodels
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import TemplatedGenerator
from gt4py.next import Dimension
from gt4py.next.type_system.type_specifications import ScalarKind

from icon4pytools.icon4pygen.bindings.codegen.type_conversion import (
    BUILTIN_TO_CPP_TYPE,
    BUILTIN_TO_ISO_C_TYPE,
)
from icon4pytools.icon4pygen.bindings.utils import format_fortran_code, write_string
from icon4pytools.py2fgen.common import ARRAY_SIZE_ARGS


class DimensionType(Node):
    name: str
    length: int


class FuncParameter(Node):
    name: str
    d_type: ScalarKind
    dimensions: Sequence[Dimension]
    py_type_hint: str
    size_args: list[str] = datamodels.field(init=False)

    def __post_init__(self):
        self.size_args = dims_to_size_strings(self.dimensions)


class Func(Node):
    name: str
    args: Sequence[FuncParameter]
    size_args: Sequence[str] = datamodels.field(init=False)

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.args]
        )


class CffiPlugin(Node):
    name: str
    function: Func


FFI_EXTERN_DECORATOR = "@ffi.def_extern()"

CFFI_FUNCS = """\
def unpack(ptr, *sizes) -> np.ndarray:
    '''
    Unpacks an n-dimensional Fortran (column-major) array into a numpy array (row-major).

    :param ptr: c_pointer to the field
    :param sizes: variable number of arguments representing the dimensions of the array in Fortran order
    :return: a numpy array with shape specified by the reverse of sizes and dtype = ctype of the pointer
    '''
    shape = sizes[
        ::-1
    ]  # Reverse the sizes to convert from Fortran (column-major) to C/numpy (row-major) order
    length = np.prod(shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    arr = np.frombuffer(
        ffi.buffer(ptr, length * ffi.sizeof(c_type)),
        dtype=np.dtype(c_type),
        count=-1,
        offset=0,
    ).reshape(shape)
    return arr


def pack(ptr, arr: np.ndarray):
    '''
    memcopies a numpy array into a pointer.

    :param ptr: c pointer
    :param arr: numpy array
    :return:
    '''
    # for now only 2d
    length = np.prod(arr.shape)
    c_type = ffi.getctype(ffi.typeof(ptr).item)
    ffi.memmove(ptr, np.ravel(arr), length * ffi.sizeof(c_type))
"""


def to_c_type(scalar_type: ScalarKind) -> str:
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_f_type(scalar_type: ScalarKind) -> str:
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: FuncParameter) -> str:
    """
    If param is a scalar type (dimension=0) then return the F90 'value' keyword.

    Used for F90 generation only.
    """
    return "value," if len(param.dimensions) == 0 else ""


def get_intent(param: FuncParameter) -> str:
    # todo(samkellerhals): quick hack to set correct intent for domain bounds
    #   by default all other variables are assumed to be arrays for now
    #   and passed by reference (pointers) in the C interface and thus
    #   annotated with inout in the f90 interface. All params need to have
    #   corresponding in/out/inout types associated with them going forward
    if "_start" in param.name or "_end" in param.name:
        return "in"
    else:
        return "inout"


def render_c_pointer(param: FuncParameter) -> str:
    return "" if len(param.dimensions) == 0 else "*"


def render_fortran_array_dimensions(param: FuncParameter) -> str:
    size = len(param.dimensions)
    if size > 0:
        dims = ",".join(map(lambda x: ":", range(size)))
        return f"dimension({dims}),"
    return ""


def dims_to_size_strings(dimensions: Sequence[Dimension]) -> list[str]:
    # Map the dimension values to their corresponding Fortran array access strings
    access_strings = []
    for dim in dimensions:
        if dim.value in ARRAY_SIZE_ARGS:
            fortran_dim = ARRAY_SIZE_ARGS[dim.value]
            access_strings.append(fortran_dim)
    return sorted(access_strings)


def render_fortran_array_sizes(param: FuncParameter) -> str:
    size = len(param.dimensions)
    if size == 0:
        return ""

    size_strings = dims_to_size_strings(param.dimensions)
    return "(" + ", ".join(size_strings) + ")"


def flatten_and_get_unique_elts(list_of_lists: list[list[str]]):
    flattened = [item for sublist in list_of_lists for item in sublist]
    return sorted(list(set(flattened)))


# TODO(samkellerhals): Actually generate the imports (look at original module imports)
def render_py_imports(plugin_name: str) -> str:
    return f"""\
from {plugin_name} import ffi
import numpy as np
from gt4py.next.ffront.fbuiltins import Field, float64, int32
from icon4py.model.common.dimension import CellDim, KDim
from icon4pytools.py2fgen.wrappers.square import square
    """


class PythonWrapper(Node):
    plugin_name: str
    function: Func
    imports: str = datamodels.field(init=False)
    func_args: Sequence[FuncParameter] = datamodels.field(init=False)
    size_args: Sequence[str] = datamodels.field(init=False)
    ffi_decorator: str = FFI_EXTERN_DECORATOR
    cffi_funcs: str = CFFI_FUNCS

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        self.func_args = [arg for arg in self.function.args]
        self.size_args = flatten_and_get_unique_elts(
            [dims_to_size_strings(arg.dimensions) for arg in self.func_args]
        )
        self.imports = render_py_imports(self.plugin_name)


class PythonWrapperGenerator(TemplatedGenerator):
    PythonWrapper = as_jinja(
        """\
{{ imports }}

{{ cffi_funcs }}

{{ ffi_decorator }}
def {{ _this_node.function.name }}_wrapper(
{%- for arg in _this_node.func_args -%}
{{ arg.name }}: {{ arg.py_type_hint }}{% if not loop.last or _this_node.size_args %}, {% endif %}
{%- endfor %}
{%- for arg in _this_node.size_args -%}
{{ arg }}: int32{{ ", " if not loop.last else "" }}
{%- endfor -%}
):
    {% for arg in _this_node.func_args %}
    {{ arg.name }} = unpack({{ arg.name }}, {{ ", ".join(arg.size_args) }})
    {% endfor %}
    {{ _this_node.function.name }}(
    {%- for arg in _this_node.func_args -%}
    {{ arg.name }}{{ ", " if not loop.last else "" }}
    {%- endfor -%}
)
"""
    )


class CHeaderGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja("""extern void {{_this_node.function.name}}_wrapper({{function}});""")

    Func = as_jinja("{%- for arg in args -%}{{ arg }}{% if not loop.last or size_args|length > 0 %}, {% endif %}{% endfor -%}{%- for sarg in size_args -%} int {{ sarg }}{% if not loop.last %}, {% endif %}{% endfor -%}")

    def visit_FuncParameter(self, param: FuncParameter):
        return self.generic_visit(
            param, rendered_type=to_c_type(param.d_type), pointer=render_c_pointer(param)
        )

    FuncParameter = as_jinja("""{{rendered_type}}{{pointer}} {{name}}""")


class F90InterfaceGenerator(TemplatedGenerator):
    CffiPlugin = as_jinja(
        """\
    module {{name}}
    use, intrinsic:: iso_c_binding
    implicit none

    public
    interface
    {{ function }}
    end interface
    end module
    """
    )

    def visit_Func(self, func: Func, **kwargs):
        arg_names = ", &\n ".join(map(lambda x: x.name, func.args))
        if func.size_args:
            arg_names += ",&\n" + ", &\n".join(func.size_args)

        return self.generic_visit(func, param_names=arg_names)

    Func = as_jinja(
        """subroutine {{name}}_wrapper({{param_names}}) bind(c, name='{{name}}_wrapper')
       use, intrinsic :: iso_c_binding
       {% for size_arg in size_args -%}
       integer(c_int), value, target :: {{ size_arg }}
       {% endfor %}
       {% for arg in args: %}\
       {{arg}}\
       {% endfor %}\
    end subroutine {{name}}_wrapper
    """
    )

    def visit_FuncParameter(self, param: FuncParameter, **kwargs):
        return self.generic_visit(
            param,
            value=as_f90_value(param),
            intent=get_intent(param),
            rendered_type=to_f_type(param.d_type),
            dim=render_fortran_array_dimensions(param),
            explicit_size=render_fortran_array_sizes(param),
        )

    FuncParameter = as_jinja(
        """{{rendered_type}}, {{dim}} {{value}} target :: {{name}}{{ explicit_size }}
    """
    )


def generate_c_header(plugin: CffiPlugin) -> str:
    generated_code = CHeaderGenerator.apply(plugin)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def generate_python_wrapper(plugin: CffiPlugin) -> str:
    node = PythonWrapper(plugin_name=plugin.name, function=plugin.function)
    generated_code = PythonWrapperGenerator.apply(node)
    return codegen.format_source("python", generated_code)


def generate_and_write_f90_interface(build_path: Path, plugin: CffiPlugin):
    generated_code = F90InterfaceGenerator.apply(plugin)
    formatted_source = format_fortran_code(generated_code)
    write_string(formatted_source, build_path, f"{plugin.name}.f90")
