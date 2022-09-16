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

import ctypes
from types import MappingProxyType
from typing import Final, Type

import numpy


LANGUAGE_ID: Final = "cpp"

_TYPE_MAPPING: Final = MappingProxyType(
    {
        bool: "int",
        int: "long",
        float: "double",
        complex: "std::complex<double>",
        numpy.bool_: "int",
        numpy.byte: "signed char",
        numpy.ubyte: "unsigned char",
        numpy.short: "short",
        numpy.ushort: "unsigned short",
        numpy.intc: "int",
        numpy.uintc: "unsigned int",
        numpy.int_: "long",
        numpy.uint: "unsigned long",
        numpy.longlong: "long long",
        numpy.ulonglong: "unsigned long long",
        numpy.single: "float",
        numpy.double: "double",
        numpy.longdouble: "long double",
        numpy.csingle: "std::complex<float>",
        numpy.cdouble: "std::complex<double>",
        numpy.clongdouble: "std::complex<long double>",
        ctypes.c_bool: "int",
        ctypes.c_char: "char",
        ctypes.c_wchar: "wchar_t",
        ctypes.c_byte: "char",
        ctypes.c_ubyte: "unsigned char",
        ctypes.c_short: "short",
        ctypes.c_ushort: "unsigned short",
        ctypes.c_int: "int",
        ctypes.c_uint: "unsigned int",
        ctypes.c_long: "long",
        ctypes.c_ulong: "unsigned long",
        ctypes.c_longlong: "long long",
        ctypes.c_ulonglong: "unsigned long long",
        ctypes.c_size_t: "std::size_t",
        ctypes.c_ssize_t: "std::ptrdiff_t",
        ctypes.c_float: "float",
        ctypes.c_double: "double",
        ctypes.c_longdouble: "long double",
    }
)


def render_python_type(python_type: Type) -> str:
    return _TYPE_MAPPING[python_type]
