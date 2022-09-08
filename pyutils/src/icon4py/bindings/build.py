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

import pathlib
from dataclasses import dataclass

from icon4py.bindings.utils import check_dir_exists, run_subprocess


# todo: use logger


@dataclass
class CppbindgenBuilder:
    source_folder: pathlib.Path
    build_folder: str = "build"

    def build(self, build_path: pathlib.Path) -> None:
        """Build and compiles C++ bindings generator."""
        check_dir_exists(build_path)
        self._cmake_build(build_path)
        self._compile(build_path)

    def _cmake_build(self, build_path: str) -> None:
        """Build the CMake project."""
        run_subprocess(["cmake", "-S", f"{self.source_folder}", "-B", f"{build_path}"])

    @staticmethod
    def _compile(build_path: str) -> None:
        """Compile the generated Makefile."""
        run_subprocess(["make"], cwd=build_path)
