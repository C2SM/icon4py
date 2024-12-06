# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from pathlib import Path
from typing import Any, Iterable, List

from gt4py.next.common import Connectivity, Dimension, DimensionKind
from gt4py.next.iterator import ir as itir
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.type_system import type_specifications as ts
from icon4py.model.common import dimension as dims

from icon4pytools.common.metadata import StencilInfo
from icon4pytools.common.utils import write_string


H_START = "horizontal_start"
H_END = "horizontal_end"
V_START = "vertical_start"
V_END = "vertical_end"

DOMAIN_ARGS = [H_START, H_END, V_START, V_END]
GRID_SIZE_ARGS = ["num_cells", "num_edges", "num_vertices"]

_SIZE_TYPE = ts.ScalarType(kind=ts.ScalarKind.INT32)


def transform_and_configure_fencil(
    program: itir.Program,
) -> itir.Program:
    """Transform the domain representation and configure the FencilDefinition parameters."""
    grid_size_symbols = [itir.Sym(id=arg, type=_SIZE_TYPE) for arg in GRID_SIZE_ARGS]

    for stmt in program.body:
        assert isinstance(stmt, itir.SetAt)
        if not len(stmt.domain.args) == 2:
            raise TypeError(f"Output domain of '{program.id}' must be 2-dimensional.")
        assert isinstance(stmt.domain.args[0], itir.FunCall) and isinstance(
            stmt.domain.args[1], itir.FunCall
        )
        horizontal_axis = stmt.domain.args[0].args[0]
        vertical_axis = stmt.domain.args[1].args[0]
        assert isinstance(horizontal_axis, itir.AxisLiteral) and isinstance(
            vertical_axis, itir.AxisLiteral
        )
        assert horizontal_axis.value in ["Vertex", "Edge", "Cell"]
        assert vertical_axis.value == "K"

        stmt.domain = itir.FunCall(
            fun=itir.SymRef(id="unstructured_domain"),
            args=[
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        horizontal_axis,
                        itir.SymRef(id=H_START),
                        itir.SymRef(id=H_END),
                    ],
                ),
                itir.FunCall(
                    fun=itir.SymRef(id="named_range"),
                    args=[
                        itir.AxisLiteral(value=dims.Koff.source.value, kind=DimensionKind.VERTICAL),
                        itir.SymRef(id=V_START),
                        itir.SymRef(id=V_END),
                    ],
                ),
            ],
        )

    fencil_params = [
        *(p for p in program.params if not is_size_param(p) and p not in grid_size_symbols),
        *(p for p in get_missing_domain_params(program.params)),
        *grid_size_symbols,
    ]

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=fencil_params,
        declarations=program.declarations,
        body=program.body,
        implicit_domain=program.implicit_domain
    )


def is_size_param(param: itir.Sym) -> bool:
    """Check if parameter is a size parameter introduced by field view frontend."""
    return param.id.startswith("__") and "_size_" in param.id


def get_missing_domain_params(params: List[itir.Sym]) -> Iterable[itir.Sym]:
    """Get domain limit params that are not present in param list."""
    param_ids = [p.id for p in params]
    missing_args = [s for s in DOMAIN_ARGS if s not in param_ids]
    return (itir.Sym(id=p, type=_SIZE_TYPE) for p in missing_args)


def check_for_domain_bounds(program: itir.Program) -> None:
    """Check that fencil params contain domain boundaries, emit warning otherwise."""
    param_ids = {param.id for param in program.params}
    all_domain_params_present = all(
        param in param_ids for param in [H_START, H_END, V_START, V_END]
    )
    if not all_domain_params_present:
        warnings.warn(
            f"Domain boundaries are missing or have non-standard names for '{program.id}'. "
            "Adapting domain to use the standard names. This feature will be removed in the future.",
            DeprecationWarning,
            stacklevel=2,
        )


def generate_gtheader(
    program: itir.Program,
    offset_provider: dict[str, Connectivity | Dimension],
    imperative: bool,
    temporaries: bool,
    **kwargs: Any,
) -> str:
    """Generate a GridTools C++ header for a given stencil definition using specified configuration parameters."""
    check_for_domain_bounds(program)

    transformed_fencil = transform_and_configure_fencil(program)

    translation = gtfn_module.GTFNTranslationStep(
        enable_itir_transforms=True,
        use_imperative_backend=imperative,
    )

    if temporaries:
        translation = translation.replace(
            symbolic_domain_sizes={
                "Cell": "num_cells",
                "Edge": "num_edges",
                "Vertex": "num_vertices",
            },
        )

    return translation.generate_stencil_source(
        transformed_fencil,
        offset_provider=offset_provider,
        column_axis=dims.KDim,  # only used for ScanOperator
        **kwargs,
    )


class GTHeader:
    """Class for generating Gridtools C++ header using the GTFN backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, imperative: bool, temporaries: bool) -> None:
        """Generate C++ code using the GTFN backend and write it to a file."""
        gtheader = generate_gtheader(
            program=self.stencil_info.program,
            offset_provider=self.stencil_info.offset_provider,
            imperative=imperative,
            temporaries=temporaries,
        )
        write_string(gtheader, outpath, f"{self.stencil_info.program.id}.hpp")
