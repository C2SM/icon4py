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

from icon4py.bindings.build import CppbindgenBuilder
from icon4py.bindings.utils import check_dir_exists, run_subprocess
from icon4py.pyutils.metadata import format_metadata
from icon4py.pyutils.stencil_info import StencilInfo


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

    def __call__(self, outpath: Path):
        # todo: implement code generation for f90 interface, cpp and h files.
        pass
