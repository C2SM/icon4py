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
from typing import Any

from functional.iterator import ir as itir
from functional.program_processors.codegens.gtfn.gtfn_backend import generate

from icon4py.bindings.utils import write_string
from icon4py.common.dimension import Koff
from icon4py.pyutils.exceptions import MultipleFieldOperatorException
from icon4py.pyutils.metadata import StencilInfo


class GTHeader:
    """Class for generating Gridtools C++ header using the GTFN backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path) -> None:
        """Generate C++ code using the GTFN backend and write it to a file."""
        gtheader = self._generate_cpp_code(
            self._adapt_domain(self.stencil_info.fvprog.itir)
        )
        write_string(gtheader, outpath, f"{self.stencil_info.fvprog.itir.id}.hpp")

    def _generate_cpp_code(self, fencil: itir.FencilDefinition, **kwargs: Any) -> str:
        return generate(
            fencil,
            offset_provider=self.stencil_info.offset_provider,
            **kwargs,
        )

    def _adapt_domain(self, fencil: itir.FencilDefinition) -> itir.FencilDefinition:
        """Replace field view size parameters by horizontal and vertical range parameters."""
        if len(fencil.closures) > 1:
            raise MultipleFieldOperatorException()

        fencil.closures[0].domain = itir.FunCall(
            fun=itir.SymRef(id="unstructured_domain"),
            args=[
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value="horizontal"),
                        itir.SymRef(id="horizontal_start"),
                        itir.SymRef(id="horizontal_end"),
                    ],
                ),
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=Koff.source.value),
                        itir.SymRef(id="vertical_start"),
                        itir.SymRef(id="vertical_end"),
                    ],
                ),
            ],
        )
        return itir.FencilDefinition(
            id=fencil.id,
            function_definitions=fencil.function_definitions,
            params=[
                *(p for p in fencil.params if not self._is_size_param(p)),
                itir.Sym(id="horizontal_start"),
                itir.Sym(id="horizontal_end"),
                itir.Sym(id="vertical_start"),
                itir.Sym(id="vertical_end"),
            ],
            closures=fencil.closures,
        )

    @staticmethod
    def _is_size_param(param: itir.Sym) -> bool:
        """Check if parameter is a size parameter introduced by field view frontend."""
        return param.id.startswith("__") and "_size_" in param.id
