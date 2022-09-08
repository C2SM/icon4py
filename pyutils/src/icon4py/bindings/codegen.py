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

from icon4py.pyutils.stencil_info import StencilInfo


@dataclass(frozen=True)
class CppBindGen:
    stencil_info: StencilInfo

    def __call__(self, bindgen_source_path: pathlib.Path):
        # todo: compile cppbindgen using cmake, make and run bindings generator.
        # todo: ensure all generated files are written to build folder.
        pass

    def _write_metadata(self):
        # todo: write metadata file for consumption by compiled binary.
        # todo: metadata_path = os.path.join(outpath, f"{fencil}.dat")
        # todo: metadata_path.write_text(format_metadata(fvprog, connectivity_chains))
        pass


@dataclass(frozen=True)
class PyBindGen:
    stencil_info: StencilInfo

    def __call__(self, outpath: pathlib.Path):
        # todo: implement code generation for f90 interface, cpp and h files.
        pass
