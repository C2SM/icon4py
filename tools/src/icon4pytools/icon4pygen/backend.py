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
from typing import Iterable, List, Optional

from gt4py.next import common
from gt4py.next.common import Connectivity
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.program_processors.runners import gtfn
from icon4py.model.common.dimension import Koff

from icon4pytools.icon4pygen.bindings.utils import write_string
from icon4pytools.icon4pygen.exceptions import MultipleFieldOperatorException
from icon4pytools.icon4pygen.metadata import StencilInfo


H_START = "horizontal_start"
H_END = "horizontal_end"
V_START = "vertical_start"
V_END = "vertical_end"

_DOMAIN_ARGS = [H_START, H_END, V_START, V_END]


def adapt_domain(fencil: itir.FencilDefinition) -> itir.FencilDefinition:
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

    fencil_params = [
        *(p for p in fencil.params if not is_size_param(p)),
        *(p for p in get_missing_domain_params(fencil.params)),
    ]

    ordered_params = order_grid_size_symbols(fencil_params)

    return itir.FencilDefinition(
        id=fencil.id,
        function_definitions=fencil.function_definitions,
        params=ordered_params,
        closures=fencil.closures,
    )


def order_grid_size_symbols(symbols):
    move_to_end = {"num_cells", "num_edges", "num_vertices"}

    front, back = [], []
    for symbol in symbols:
        if str(symbol.id) in move_to_end:
            back.append(symbol)
        else:
            front.append(symbol)
    return front + back


def is_size_param(param: itir.Sym) -> bool:
    """Check if parameter is a size parameter introduced by field view frontend."""
    return param.id.startswith("__") and "_size_" in param.id


def get_missing_domain_params(params: List[itir.Sym]) -> Iterable[itir.Sym]:
    """Get domain limit params that are not present in param list."""
    param_ids = [p.id for p in params]
    missing_args = [s for s in _DOMAIN_ARGS if s not in param_ids]
    return (itir.Sym(id=p) for p in missing_args)


def generate_gtheader(
    fencil: itir.FencilDefinition,
    offset_provider: dict[str, Connectivity],
    column_axis: Optional[common.Dimension],
    imperative: bool,
    temporaries: bool,
) -> str:
    """Generate a GridTools C++ header for a given stencil definition using specified configuration parameters."""
    gtfn_translation = gtfn.run_gtfn.executor.otf_workflow.translation  # type: ignore
    assert isinstance(gtfn_translation, gtfn.gtfn_module.GTFNTranslationStep)

    lift_mode = LiftMode.FORCE_TEMPORARIES if temporaries else LiftMode.FORCE_INLINE

    # configure backend
    gtfn_translation = gtfn_translation.replace(
        use_imperative_backend=imperative, lift_mode=lift_mode
    )

    if temporaries:
        gtfn_translation = gtfn_translation.replace(
            symbolic_domain_sizes={
                "Cell": "num_cells",
                "Edge": "num_edges",
                "Vertex": "num_vertices",
            }
        )

    return gtfn_translation.generate_stencil_source(
        fencil,
        offset_provider=offset_provider,  # type: ignore
        column_axis=column_axis,
    )


class GTHeader:
    """Class for generating Gridtools C++ header using the GTFN backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, imperative: bool, temporaries: bool) -> None:
        """Generate C++ code using the GTFN backend and write it to a file."""
        fencil_with_adapted_domain = adapt_domain(self.stencil_info.itir)
        gtheader = generate_gtheader(
            fencil=fencil_with_adapted_domain,
            offset_provider=self.stencil_info.offset_provider,
            column_axis=self.stencil_info.column_axis,
            imperative=imperative,
            temporaries=temporaries,
        )
        write_string(gtheader, outpath, f"{self.stencil_info.itir.id}.hpp")
