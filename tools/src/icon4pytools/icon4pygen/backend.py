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
import re
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


def transform_and_configure_fencil(fencil: itir.FencilDefinition) -> itir.FencilDefinition:
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
            fencil=self.stencil_info.fendef,
            offset_provider=self.stencil_info.offset_provider,
            imperative=imperative,
            temporaries=temporaries,
        )
        write_string(gtheader, outpath, f"{self.stencil_info.fendef.id}.hpp")


def generate_dace_code(
    stencil_info: StencilInfo,
    offset_provider: dict[str, Connectivity | Dimension],
    on_gpu: bool,
    temporaries: bool,
    **kwargs: Any,
) -> tuple[str, str, Optional[str]]:
    import dace

    """Generate a GridTools C++ header for a given stencil definition using specified configuration parameters."""
    check_for_domain_bounds(stencil_info.fendef)

    transformed_fencil = transform_and_configure_fencil(stencil_info.fendef)

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

    params = [str(p.id) for p in stencil_info.fendef.params]
    arg_types = [
        stencil_info.fields[pname].field.type
        if pname in stencil_info.fields
        else ts.ScalarType(kind=tt.get_scalar_kind(p.dtype))
        if p.dtype is not None
        else ts.ScalarType(kind=ts.ScalarKind.INT32)
        for p, pname in zip(stencil_info.fendef.params, params, strict=False)
    ]

    sdfg = translation.generate_sdfg(
        transformed_fencil,
        arg_types,
        offset_provider=offset_provider,
        column_axis=KDim,  # only used for ScanOperator
        **kwargs,
    )

    # limit cuda code generation to only use one cuda stream and expose utility function
    # to override it
    if on_gpu:
        sdfg.append_global_code(
            f"""
        DACE_EXPORTED bool __dace_gpu_set_stream({sdfg.name}_state_t *__state, int streamid, gpuStream_t stream);
        DACE_EXPORTED void __set_stream_{sdfg.name}({sdfg.name}_state_t *__state, gpuStream_t stream) {{
            __dace_gpu_set_stream(__state, 0, stream);
        }}"""
        )

    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=1):
        code_objs = sdfg.generate_code()

    hdr_objs = [obj for obj in code_objs if obj.language == "h"]
    assert len(hdr_objs) == 1
    if on_gpu:
        # add utility function for cuda stream to header file
        line = f'extern "C" void __set_stream_{sdfg.name}({sdfg.name}Handle_t handle, cudaStream_t stream);\n'
        hdr_objs[0].code += line

    src_objs = [obj for obj in code_objs if obj.language == "cpp" and obj.linkable]
    assert len(src_objs) == 1
    # remove include of header file with hash string
    hash_include_line = "#include \"../../include/hash.h\""
    src_objs[0].code = src_objs[0].code.replace(hash_include_line, "")

    # for gpu codegen, also return the cuda file
    if on_gpu:
        cuda_objs = [obj for obj in code_objs if obj.language == "cu" and obj.linkable]
        assert len(cuda_objs) == 1



        init_func_name_before = f'__dace_init_cuda'
        init_func_name_after = f'__dace_init_cuda_{sdfg.name}'

        src_objs[0].code = src_objs[0].code.replace(init_func_name_before, init_func_name_after)
        cuda_objs[0].code = cuda_objs[0].code.replace(init_func_name_before, init_func_name_after)

        exit_func_name_before = f'__dace_exit_cuda'
        exit_func_name_after = f'__dace_exit_cuda_{sdfg.name}'

        src_objs[0].code = src_objs[0].code.replace(exit_func_name_before, exit_func_name_after)
        cuda_objs[0].code = cuda_objs[0].code.replace(exit_func_name_before, exit_func_name_after)

        set_stream_func_name_before = f'__dace_gpu_set_stream'
        set_stream_func_name_after = f'__dace_gpu_set_stream_{sdfg.name}'

        src_objs[0].code = src_objs[0].code.replace(set_stream_func_name_before, set_stream_func_name_after)
        cuda_objs[0].code = cuda_objs[0].code.replace(set_stream_func_name_before, set_stream_func_name_after)

        set_all_streams_func_name_before = f'__dace_gpu_set_all_streams'
        set_all_streams_func_name_after = f'__dace_gpu_set_all_streams_{sdfg.name}'

        src_objs[0].code = src_objs[0].code.replace(set_all_streams_func_name_before, set_all_streams_func_name_after)
        cuda_objs[0].code = cuda_objs[0].code.replace(set_all_streams_func_name_before, set_all_streams_func_name_after)

        include_line_before = f'#include <dace/dace.h>'
        include_line_after = f'#include <dace/dace.h>\n#include \"{sdfg.name}_dace.cpp\"'

        cuda_objs[0].code = cuda_objs[0].code.replace(include_line_before, include_line_after)

        context_line = f"""struct compute_avg_vn_state_t {{
    dace::cuda::Context *gpu_context;
}};"""
        cuda_objs[0].code = cuda_objs[0].code.replace(context_line, "")

        return hdr_objs[0].clean_code, src_objs[0].clean_code, cuda_objs[0].clean_code
    else:
        return hdr_objs[0].clean_code, src_objs[0].clean_code, None


class DaceCodegen:
    """Class for generating Gridtools CUDA code using the DaCe backend."""

    def __init__(self, stencil_info: StencilInfo) -> None:
        self.stencil_info = stencil_info

    def __call__(self, outpath: Path, temporaries: bool) -> None:
        """Generate CUDA code using the DaCe backend and write it to a file.

        Returns the name of the kernel to be invoked.
        """
        # only support cuda codegen
        on_gpu = True

        dc_hdr, dc_src, dc_cuda = generate_dace_code(
            self.stencil_info,
            self.stencil_info.offset_provider,
            on_gpu,
            temporaries,
        )

        write_string(dc_hdr, outpath, f"{self.stencil_info.fendef.id}_dace.h")
        write_string(dc_src, outpath, f"{self.stencil_info.fendef.id}_dace.cpp")

        assert dc_cuda is not None
        write_string(dc_cuda, outpath, f"{self.stencil_info.fendef.id}_dace.cu")
