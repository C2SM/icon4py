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
import warnings
from pathlib import Path
from typing import Any, Iterable, List, Optional

from gt4py._core import definitions as core_defs
from gt4py.next.common import Connectivity, Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import LiftMode
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners.dace_iterator import (
    workflow as dace_workflow,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt
from icon4py.model.common.dimension import KDim, Koff

from icon4pytools.icon4pygen.bindings.utils import write_string
from icon4pytools.icon4pygen.metadata import StencilInfo


H_START = "horizontal_start"
H_END = "horizontal_end"
V_START = "vertical_start"
V_END = "vertical_end"

DOMAIN_ARGS = [H_START, H_END, V_START, V_END]
GRID_SIZE_ARGS = ["num_cells", "num_edges", "num_vertices"]


def transform_and_configure_fencil(
    fencil: itir.FencilDefinition,
) -> itir.FencilDefinition:
    """Transform the domain representation and configure the FencilDefinition parameters."""
    grid_size_symbols = [itir.Sym(id=arg) for arg in GRID_SIZE_ARGS]

    for closure in fencil.closures:
        if not len(closure.domain.args) == 2:
            raise TypeError(f"Output domain of '{fencil.id}' must be 2-dimensional.")
        assert isinstance(closure.domain.args[0], itir.FunCall) and isinstance(
            closure.domain.args[1], itir.FunCall
        )
        horizontal_axis = closure.domain.args[0].args[0]
        vertical_axis = closure.domain.args[1].args[0]
        assert isinstance(horizontal_axis, itir.AxisLiteral) and isinstance(
            vertical_axis, itir.AxisLiteral
        )
        assert horizontal_axis.value in ["Vertex", "Edge", "Cell"]
        assert vertical_axis.value == "K"

        closure.domain = itir.FunCall(
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
                        itir.AxisLiteral(value=Koff.source.value),
                        itir.SymRef(id=V_START),
                        itir.SymRef(id=V_END),
                    ],
                ),
            ],
        )

    fencil_params = [
        *(p for p in fencil.params if not is_size_param(p) and p not in grid_size_symbols),
        *(p for p in get_missing_domain_params(fencil.params)),
        *grid_size_symbols,
    ]

    return itir.FencilDefinition(
        id=fencil.id,
        function_definitions=fencil.function_definitions,
        params=fencil_params,
        closures=fencil.closures,
    )


def is_size_param(param: itir.Sym) -> bool:
    """Check if parameter is a size parameter introduced by field view frontend."""
    return param.id.startswith("__") and "_size_" in param.id


def get_missing_domain_params(params: List[itir.Sym]) -> Iterable[itir.Sym]:
    """Get domain limit params that are not present in param list."""
    param_ids = [p.id for p in params]
    missing_args = [s for s in DOMAIN_ARGS if s not in param_ids]
    return (itir.Sym(id=p) for p in missing_args)


def check_for_domain_bounds(fencil: itir.FencilDefinition) -> None:
    """Check that fencil params contain domain boundaries, emit warning otherwise."""
    param_ids = {param.id for param in fencil.params}
    all_domain_params_present = all(
        param in param_ids for param in [H_START, H_END, V_START, V_END]
    )
    if not all_domain_params_present:
        warnings.warn(
            f"Domain boundaries are missing or have non-standard names for '{fencil.id}'. "
            "Adapting domain to use the standard names. This feature will be removed in the future.",
            DeprecationWarning,
            stacklevel=2,
        )


def generate_gtheader(
    fencil: itir.FencilDefinition,
    offset_provider: dict[str, Connectivity | Dimension],
    imperative: bool,
    temporaries: bool,
    **kwargs: Any,
) -> str:
    """Generate a GridTools C++ header for a given stencil definition using specified configuration parameters."""
    check_for_domain_bounds(fencil)

    transformed_fencil = transform_and_configure_fencil(fencil)

    translation = gtfn_module.GTFNTranslationStep(
        enable_itir_transforms=True,
        use_imperative_backend=imperative,
    )

    if temporaries:
        translation = translation.replace(
            lift_mode=LiftMode.USE_TEMPORARIES,
            symbolic_domain_sizes={
                "Cell": "num_cells",
                "Edge": "num_edges",
                "Vertex": "num_vertices",
            },
        )

    return translation.generate_stencil_source(
        transformed_fencil,
        offset_provider=offset_provider,
        column_axis=KDim,  # only used for ScanOperator
        **kwargs,
    )


class GTHeader:
    """Class for generating Gridtools C++ header using the GTFN backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, imperative: bool, temporaries: bool) -> None:
        """Generate C++ code using the GTFN backend and write it to a file."""
        gtheader = generate_gtheader(
            fencil=self.stencil_info.itir,
            offset_provider=self.stencil_info.offset_provider,
            imperative=imperative,
            temporaries=temporaries,
        )
        write_string(gtheader, outpath, f"{self.stencil_info.itir.id}.hpp")


def generate_dace_code(
    stencil_info: StencilInfo,
    offset_provider: dict[str, Connectivity | Dimension],
    on_gpu: bool,
    temporaries: bool,
    **kwargs: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    import dace  # type: ignore[import-untyped]

    """Generate a GridTools C++ header for a given stencil definition using specified configuration parameters."""
    check_for_domain_bounds(stencil_info.itir)

    transformed_fencil = transform_and_configure_fencil(stencil_info.itir)

    translation = dace_workflow.DaCeTranslator(
        auto_optimize=True,
        device_type=core_defs.DeviceType.CUDA if on_gpu else core_defs.DeviceType.CPU,
    )

    if temporaries:
        translation = translation.replace(
            lift_mode=LiftMode.USE_TEMPORARIES,
            symbolic_domain_sizes={
                "Cell": "num_cells",
                "Edge": "num_edges",
                "Vertex": "num_vertices",
            },
        )

    params = [str(p.id) for p in stencil_info.itir.params]
    arg_types = [
        stencil_info.fields[pname].field.type
        if pname in stencil_info.fields
        else ts.ScalarType(kind=tt.get_scalar_kind(p.dtype))
        if p.dtype is not None
        else ts.ScalarType(kind=ts.ScalarKind.INT32)
        for p, pname in zip(stencil_info.itir.params, params, strict=False)
    ]

    sdfg = translation.generate_sdfg(
        transformed_fencil,
        arg_types,
        offset_provider=offset_provider,
        column_axis=KDim,  # only used for ScanOperator
        **kwargs,
    )

    code_objs = sdfg.generate_code()
    hdr_objs = [obj for obj in code_objs if obj.language == "h"]
    assert len(hdr_objs) == 1
    src_objs = [obj for obj in code_objs if obj.language == "cpp" and obj.linkable]
    assert len(src_objs) == 1
    # for gpu codegen, also return the cuda file
    if on_gpu:
        cuda_objs = [obj for obj in code_objs if obj.language == "cu" and obj.linkable]
        assert len(cuda_objs) == 1
        return None, None, cuda_objs[0].clean_code
    else:
        return hdr_objs[0].clean_code, src_objs[0].clean_code, None


class DaceCodegen:
    """Class for generating Gridtools C++ header using the DaCe backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, on_gpu: bool, temporaries: bool) -> None:
        """Generate C++ code using the DaCe backend and write it to a file."""
        dc_hdr, dc_src, dc_cuda = generate_dace_code(
            self.stencil_info,
            self.stencil_info.offset_provider,
            on_gpu,
            temporaries,
        )
        if on_gpu:
            write_string(dc_cuda, outpath, f"{self.stencil_info.itir.id}_dace.cu")
        else:
            write_string(dc_hdr, outpath, f"{self.stencil_info.itir.id}_dace.h")
            write_string(dc_src, outpath, f"{self.stencil_info.itir.id}_dace.cpp")
