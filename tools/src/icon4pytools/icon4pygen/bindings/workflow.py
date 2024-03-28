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

from icon4pytools.icon4pygen.bindings.codegen.cpp import generate_cpp_definition
from icon4pytools.icon4pygen.bindings.codegen.dace import generate_cpp_definition as generate_dace_cpp_definition
from icon4pytools.icon4pygen.bindings.codegen.f90 import generate_f90_file
from icon4pytools.icon4pygen.bindings.codegen.header import generate_cpp_header
from icon4pytools.icon4pygen.bindings.entities import Field, Offset
from icon4pytools.icon4pygen.bindings.utils import check_dir_exists
from icon4pytools.icon4pygen.metadata import StencilInfo


class PyBindGen:
    """Class to handle the bindings generation workflow.

    The workflow consists of generating the following bindings for the ICON model.

    - A Fortran Interface (.f90), which declares a wrapper function which can be called from within the ICON source code.
    - A C interface (.cpp, .h) which enables passing Fortran pointers to the Gridtools stencil.

    Note:
        Within the C interface, we also carry out verification of the DSL output fields against the Fortran output fields.
        Furthermore, we also serialise data to .csv or .vtk files in case of verification failure.
    """

    def __init__(self, stencil_info: StencilInfo, levels_per_thread: int, block_size: int) -> None:
        self.stencil_name = stencil_info.itir.id
        self.fields, self.offsets = self._stencil_info_to_binding_type(stencil_info)
        self.levels_per_thread = levels_per_thread
        self.block_size = block_size

    @staticmethod
    def _stencil_info_to_binding_type(
        stencil_info: StencilInfo,
    ) -> tuple[list[Field], list[Offset]]:
        chains = stencil_info.connectivity_chains
        binding_fields = [
            Field(name, info) for name, info in stencil_info.fields.items()
        ]
        binding_offsets = [Offset(chain) for chain in chains]
        return binding_fields, binding_offsets

    def __call__(self, outpath: Path) -> None:
        check_dir_exists(outpath)
        generate_f90_file(self.stencil_name, self.fields, self.offsets, outpath)
        generate_cpp_header(self.stencil_name, self.fields, outpath)
        generate_cpp_definition(
            self.stencil_name,
            self.fields,
            self.offsets,
            self.levels_per_thread,
            self.block_size,
            outpath,
        )


class DacePyBindGen:
    """Class to handle the bindings generation workflow.

    The workflow consists of generating the following bindings for the ICON model.

    - A Fortran Interface (.f90), which declares a wrapper function which can be called from within the ICON source code.
    - A C interface (.cpp, .h) which enables passing Fortran pointers to the Gridtools stencil.

    Note:
        Within the C interface, we also carry out verification of the DSL output fields against the Fortran output fields.
        Furthermore, we also serialise data to .csv or .vtk files in case of verification failure.
    """

    def __init__(self, stencil_info: StencilInfo, on_gpu: bool) -> None:
        self.stencil_name = stencil_info.itir.id
        self.fields, self.offsets = self._stencil_info_to_binding_type(stencil_info)
        self.on_gpu = on_gpu

    @staticmethod
    def _stencil_info_to_binding_type(
        stencil_info: StencilInfo,
    ) -> tuple[list[Field], list[Offset]]:
        chains = stencil_info.connectivity_chains
        binding_fields = [
            Field(name, info) for name, info in stencil_info.fields.items()
        ]
        binding_offsets = [Offset(chain) for chain in chains]
        return binding_fields, binding_offsets

    def __call__(self, outpath: Path) -> None:
        check_dir_exists(outpath)
        generate_f90_file(self.stencil_name, self.fields, self.offsets, outpath)
        generate_cpp_header(self.stencil_name, self.fields, outpath)
        generate_dace_cpp_definition(
            self.stencil_name,
            self.fields,
            self.offsets,
            outpath,
        )
