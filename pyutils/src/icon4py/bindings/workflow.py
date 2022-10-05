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

from icon4py.bindings.codegen.cpp import CppDef
from icon4py.bindings.codegen.f90 import F90Iface
from icon4py.bindings.codegen.header import CppHeader
from icon4py.bindings.entities import Field, Offset
from icon4py.bindings.utils import check_dir_exists
from icon4py.pyutils.metadata import StencilInfo, get_field_infos


@dataclass(frozen=True)
class PyBindGen:
    """Class to handle the bindings generation workflow."""

    stencil_info: StencilInfo
    levels_per_thread: int
    block_size: int

    @staticmethod
    def _stencil_info_to_binding_type(
        stencil_info: StencilInfo,
    ) -> tuple[list[Field], list[Offset]]:
        chains = stencil_info.connectivity_chains
        fields = get_field_infos(stencil_info.fvprog)
        binding_fields = [Field(name, info) for name, info in fields.items()]
        binding_offsets = [Offset(chain) for chain in chains]
        return binding_fields, binding_offsets

    def __call__(self, outpath: Path) -> None:
        check_dir_exists(outpath)
        stencil_name = self.stencil_info.fvprog.itir.id
        fields, offsets = self._stencil_info_to_binding_type(self.stencil_info)

        F90Iface(stencil_name, fields, offsets).write(outpath)
        CppHeader(stencil_name, fields).write(outpath)
        CppDef(
            stencil_name, fields, offsets, self.levels_per_thread, self.block_size
        ).write(outpath)
