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

from eve.codegen import format_source

from icon4py.bindings.build import CppbindgenBuilder
from icon4py.bindings.codegen.cpp import CppDef
from icon4py.bindings.codegen.f90 import F90Iface
from icon4py.bindings.codegen.header import CppHeader
from icon4py.bindings.types import stencil_info_to_binding_type
from icon4py.bindings.utils import check_dir_exists, run_subprocess
from icon4py.pyutils.metadata import StencilInfo, format_metadata


# todo: this can be deleted once PyBindGen is complete
@dataclass(frozen=True)
class CppBindGen:
    stencil_info: StencilInfo
    binary_name: str = "bindings_generator"
    build_folder: str = "build"
    gen_folder: str = "generated"

    def __call__(self, source_path: Path):
        build_path = source_path / self.build_folder
        CppbindgenBuilder(source_path).build(build_path)
        self._run_codegen(build_path)

    def _run_codegen(self, build_path: Path):
        gen_path = self._create_gen_folder(build_path)
        metadata_path = self._write_metadata(gen_path)
        self._execute_cppbindgen(metadata_path, build_path)
        self._format_source_code(gen_path)

    def _format_source_code(self, gen_path: Path):
        extensions = ["cpp", "h"]

        for ext in extensions:
            files = gen_path.glob(f"*.{ext}")
            for file in files:
                with open(file, "r+") as f:
                    formatted = format_source("cpp", f.read(), style="LLVM")
                    f.seek(0)
                    f.write(formatted)
                    f.truncate()

    def _execute_cppbindgen(self, metadata_path: Path, build_path: Path):
        binary_path = build_path / self.binary_name
        run_subprocess([binary_path, metadata_path.__str__()])

    def _create_gen_folder(self, build_path: Path):
        gen_path = build_path / self.gen_folder
        check_dir_exists(gen_path)
        return gen_path

    def _write_metadata(self, gen_path: Path):
        stencil_name = self.stencil_info.fvprog.past_node.id
        metadata_path = gen_path / f"{stencil_name}.dat"
        metadata_path.write_text(
            format_metadata(
                self.stencil_info.fvprog, self.stencil_info.connectivity_chains
            )
        )
        return metadata_path


@dataclass(frozen=True)
class PyBindGen:
    stencil_info: StencilInfo
    levels_per_thread: int
    block_size: int

    def __call__(self, outpath: Path):
        check_dir_exists(outpath)

        # from stencil_meta data to bindgen internal data structures
        stencil_name = self.stencil_info.fvprog.itir.id
        fields, offsets = stencil_info_to_binding_type(self.stencil_info)

        F90Iface(stencil_name, fields, offsets).write(outpath)
        CppHeader(stencil_name, fields).write(outpath)
        CppDef(
            stencil_name, fields, offsets, self.levels_per_thread, self.block_size
        ).write(outpath)

        # todo: implement code generation for .cpp file
