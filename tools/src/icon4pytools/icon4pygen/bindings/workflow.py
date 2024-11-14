# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4pytools.common.metadata import StencilInfo
from icon4pytools.icon4pygen.bindings.codegen.cpp import generate_cpp_definition
from icon4pytools.icon4pygen.bindings.codegen.f90 import generate_f90_file
from icon4pytools.icon4pygen.bindings.codegen.header import generate_cpp_header
from icon4pytools.icon4pygen.bindings.entities import Field, Offset
from icon4pytools.icon4pygen.bindings.utils import check_dir_exists


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
        self.stencil_name = stencil_info.program.id
        self.fields, self.offsets = self._stencil_info_to_binding_type(stencil_info)
        self.levels_per_thread = levels_per_thread
        self.block_size = block_size

    @staticmethod
    def _stencil_info_to_binding_type(
        stencil_info: StencilInfo,
    ) -> tuple[list[Field], list[Offset]]:
        chains = stencil_info.connectivity_chains
        binding_fields = [Field(name, info) for name, info in stencil_info.fields.items()]
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
