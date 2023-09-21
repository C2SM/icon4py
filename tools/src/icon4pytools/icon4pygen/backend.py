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
from typing import Any, Iterable, List

from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.codegens.gtfn.gtfn_backend import generate
from icon4py.model.common.dimension import Koff

from icon4pytools.icon4pygen.bindings.utils import write_string
from icon4pytools.icon4pygen.exceptions import MultipleFieldOperatorException
from icon4pytools.icon4pygen.metadata import StencilInfo


H_START = "horizontal_start"
H_END = "horizontal_end"
V_START = "vertical_start"
V_END = "vertical_end"

_DOMAIN_ARGS = [H_START, H_END, V_START, V_END]


class GTHeader:
    """Class for generating Gridtools C++ header using the GTFN backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, imperative: bool) -> None:
        """Generate C++ code using the GTFN backend and write it to a file."""
        gtheader = self._generate_cpp_code(
            self._adapt_domain(self.stencil_info.itir), imperative=imperative
        )
        write_string(gtheader, outpath, f"{self.stencil_info.itir.id}.hpp")

    def _generate_cpp_code(self, fencil: itir.FencilDefinition, **kwargs: Any) -> str:
        return generate(
            fencil,
            offset_provider=self.stencil_info.offset_provider,
            column_axis=self.stencil_info.column_axis,
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
                        itir.SymRef(id=H_START),
                        itir.SymRef(id=H_END),
                    ],
                ),
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=Koff.source.value),
                        itir.SymRef(id=V_START),
                        itir.SymRef(id=V_END),
                    ],
                ),
            ],
        )
        return itir.FencilDefinition(
            id=fencil.id,
            function_definitions=fencil.function_definitions,
            params=[
                *(p for p in fencil.params if not self._is_size_param(p)),
                *(p for p in self._missing_domain_params(fencil.params)),
            ],
            closures=fencil.closures,
        )

    @staticmethod
    def _is_size_param(param: itir.Sym) -> bool:
        """Check if parameter is a size parameter introduced by field view frontend."""
        return param.id.startswith("__") and "_size_" in param.id

    @staticmethod
    def _missing_domain_params(params: List[itir.Sym]) -> Iterable[itir.Sym]:
        """Get domain limit params that are not present in param list."""
        param_ids = [p.id for p in params]
        missing_args = [s for s in _DOMAIN_ARGS if s not in param_ids]
        return (itir.Sym(id=p) for p in missing_args)
