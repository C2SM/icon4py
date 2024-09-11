# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

from __future__ import annotations

import copy
import site
import sys
from typing import Any, ClassVar, Optional, Sequence


try:
    import dace
except ImportError as e:
    raise ImportError("DaCe is required for this module") from e

from dace import dtypes
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg.state import SDFGState
from gt4py.next.ffront.fbuiltins import Dimension

import icon4py.model.common as common
from icon4py.model.common import dimension as dims


ghex_ptr_names = (
    "__context_ptr",
    "__comm_ptr",
    "__pattern_CellDim_ptr",
    "__pattern_VertexDim_ptr",
    "__pattern_EdgeDim_ptr",
    "__domain_descriptor_CellDim_ptr",
    "__domain_descriptor_VertexDim_ptr",
    "__domain_descriptor_EdgeDim_ptr",
)


def add_halo_tasklet(
    sdfg: dace.SDFG,
    state: SDFGState,
    global_buffers: dict[str, dace.data.Data],
    exchange: common.decomposition.mpi_decomposition.GHexMultiNodeExchange,
    dim: Dimension,
    unique_id: int,
    wait: bool,
    counter: int,
) -> None:
    """
    Add halo exchange tasklet to the SDFG state.

    Arg:
        sdfg: SDFG
        state: SDFG state to add the tasklet to
        global_buffers: dictionary of buffers to exchange -dict[buffer_name, data_descriptor]-
        exchange: GHexMultiNodeExchange object
        dim: dimension over which the exchange is performed
        unique_id:
        wait: Async exchange or not
        counter: counter to define the global variables only once
    """
    tasklet = dace.sdfg.nodes.Tasklet(
        "_halo_exchange_",
        inputs=None,
        outputs=None,
        code="",
        language=dace.dtypes.Language.CPP,
        side_effects=True,
    )
    state.add_node(tasklet)

    in_connectors = {}
    out_connectors = {}

    if counter == 0:  # define them only once
        for buffer_name in ghex_ptr_names:
            sdfg.add_scalar(buffer_name, dtype=dace.uintp)
            buffer = state.add_read(buffer_name)
            in_connectors["IN_" + buffer_name] = dace.uintp.dtype
            state.add_edge(
                buffer, None, tasklet, "IN_" + buffer_name, Memlet(buffer_name, subset="0")
            )

    for i, (buffer_name, data_descriptor) in enumerate(global_buffers.items()):
        sdfg.add_array(
            buffer_name,
            data_descriptor.shape,
            data_descriptor.dtype,
            storage=data_descriptor.storage,
            strides=data_descriptor.strides,
        )
        buffer = state.add_read(buffer_name)
        in_connectors["IN_" + f"field_{i}"] = dtypes.pointer(data_descriptor.dtype)
        state.add_edge(
            buffer,
            None,
            tasklet,
            "IN_" + f"field_{i}",
            Memlet.from_array(buffer_name, data_descriptor),
        )

        update = state.add_write(buffer_name)
        out_connectors["OUT_" + f"field_{i}"] = dtypes.pointer(data_descriptor.dtype)
        state.add_edge(
            tasklet,
            "OUT_" + f"field_{i}",
            update,
            None,
            Memlet.from_array(buffer_name, data_descriptor),
        )

    # Dummy return: preserve same interface with non-DaCe version
    sdfg.add_scalar(name="__return", dtype=dace.int32)
    ret = state.add_write("__return")
    state.add_edge(tasklet, "__out", ret, None, dace.Memlet(data="__return", subset="0"))
    out_connectors["__out"] = dace.int32

    tasklet.in_connectors = in_connectors
    tasklet.out_connectors = out_connectors
    tasklet.environments = [f"{DaceGHEX.__module__}.{DaceGHEX.__name__}"]

    pattern_type = exchange._patterns[dim].__cpp_type__
    domain_descriptor_type = exchange._domain_descriptors[dim].__cpp_type__
    communication_object_type = exchange._comm.__cpp_type__
    communication_handle_type = communication_object_type[
        communication_object_type.find("<") + 1 : communication_object_type.rfind(">")
    ]

    # Tasklet code generation part
    fields_desc_glob_vars = "\n"
    fields_desc = """
    bool levels_first;
    std::size_t outer_strides;
    std::size_t levels;

    """
    descr_unique_names = []
    for i, arg in enumerate(copy.deepcopy(list(global_buffers.values()))):
        # Checks below adapted from https://github.com/ghex-org/GHEX/blob/master/bindings/python/src/_pyghex/unstructured/field_descriptor.cpp
        # Keep in mind:
        # GHEX/NumPy strides: number of bytes to jump
        # DaCe strides: number of elements to jump
        fields_desc += f"""
        if ({len(arg.shape)} > 2u) throw std::runtime_error("field has too many dimensions");
        
        if ({arg.shape[0]} != {exchange._domain_descriptors[dim].size()}) throw std::runtime_error("field's first dimension must match the size of the domain");

        levels_first = true;
        outer_strides = 0u;
        if ({len(arg.shape)} == 2 && {arg.strides[1]} != 1)
        {{
            levels_first = false;
            if ({arg.strides[0]} != 1) throw std::runtime_error("field's strides are not compatible with GHEX");
            outer_strides = {arg.strides[1]};
        }}
        else if ({len(arg.shape)} == 2)
        {{
            if ({arg.strides[1]} != 1) throw std::runtime_error("field's strides are not compatible with GHEX");
            outer_strides = {arg.strides[0]};
        }}
        else
        {{
            if ({arg.strides[0]} != 1) throw std::runtime_error("field's strides are not compatible with GHEX");
        }}

        levels = ({len(arg.shape)} == 1) ? 1u : {arg.shape[1]};
        """

        descr_unique_name = f"field_desc_{i}_{counter}_{unique_id}"
        descr_unique_names.append(descr_unique_name)
        descr_type_ = f"ghex::unstructured::data_descriptor<ghex::{'cpu' if arg.storage.value <= 5 else 'gpu'}, int, int, {arg.dtype.ctype}>"
        if wait:
            # de-allocated descriptors once out-of-scope, no need for storing them in global vars
            fields_desc += f"{descr_type_} {descr_unique_name}{{*domain_descriptor, IN_field_{i}, levels, levels_first, outer_strides}};\n"
        else:
            # for async exchange, we need to keep the descriptors alive, until the wait
            fields_desc += f"{descr_unique_name} = std::make_unique<{descr_type_}>(*domain_descriptor, IN_field_{i}, levels, levels_first, outer_strides);\n"
            fields_desc_glob_vars += f"std::unique_ptr<{descr_type_}> {descr_unique_name};\n"

    code = ""
    if counter == 0:
        # The GHEX C++ ptrs are global variables
        # Here, they are initialized at the first encounter of the halo exchange node
        __pattern = ""
        __domain_descriptor = ""
        for dim_ in dims.global_dimensions.values():
            __pattern += f"__pattern_{dim_.value}Dim_ptr_{unique_id} = IN___pattern_{dim_.value}Dim_ptr;\n"  # IN_XXX comes from the DaCe connector
            __domain_descriptor += f"__domain_descriptor_{dim_.value}Dim_ptr_{unique_id} = IN___domain_descriptor_{dim_.value}Dim_ptr;\n"

        code = f"""
            __context_ptr_{unique_id} = IN___context_ptr;
            __comm_ptr_{unique_id} = IN___comm_ptr;
            {__pattern}
            {__domain_descriptor}
            """

    code += f"""
            ghex::context* m = reinterpret_cast<ghex::context*>(__context_ptr_{unique_id});
            
            {pattern_type}* pattern = reinterpret_cast<{pattern_type}*>(__pattern_{dim.value}Dim_ptr_{unique_id});
            {domain_descriptor_type}* domain_descriptor = reinterpret_cast<{domain_descriptor_type}*>(__domain_descriptor_{dim.value}Dim_ptr_{unique_id});
            {communication_object_type}* communication_object = reinterpret_cast<{communication_object_type}*>(__comm_ptr_{unique_id});

            {fields_desc}

            h_{unique_id} = communication_object->exchange({", ".join([f'(*pattern)({"" if wait else "*"}{descr_unique_names[i]})' for i in range(len(global_buffers))])});
            { 'h_'+str(unique_id)+'.wait();' if wait else ''}

            __out = 1234; // Dummy return;
            """

    tasklet.code = CodeBlock(code=code, language=dace.dtypes.Language.CPP)

    if counter == 0:
        # Set global variables
        __pattern = ""
        __domain_descriptor = ""
        for dim_ in dims.global_dimensions.values():
            __pattern += f"{dace.uintp.dtype} __pattern_{dim_.value}Dim_ptr_{unique_id};\n"
            __domain_descriptor += (
                f"{dace.uintp.dtype} __domain_descriptor_{dim_.value}Dim_ptr_{unique_id};\n"
            )

        code = f"""
                {dace.uintp.dtype} __context_ptr_{unique_id};
                {dace.uintp.dtype} __comm_ptr_{unique_id};
                {__pattern}
                {__domain_descriptor}
                {fields_desc_glob_vars}
                ghex::communication_handle<{communication_handle_type}> h_{unique_id};
                """
    else:
        code = fields_desc_glob_vars

    tasklet.code_global = CodeBlock(code=code, language=dace.dtypes.Language.CPP)


@dace.library.environment
class DaceGHEX:
    """Set GHEX environment for compilation in DaCe"""

    python_site_packages: ClassVar[str] = site.getsitepackages()[0]
    ghex_path: ClassVar[str] = (
        python_site_packages + "/ghex"
    )  # 'absolute_path_to/GHEX/build' [case of manual compilation]

    cmake_minimum_version: ClassVar = None
    cmake_packages: ClassVar = []
    cmake_variables: ClassVar = {}
    cmake_compile_flags: ClassVar = []
    cmake_link_flags: ClassVar[list[str]] = [
        f"-L{ghex_path}/lib -lghex -lhwmalloc -loomph_common -loomph_mpi"
    ]
    cmake_includes: ClassVar[list[str]] = [
        f"{sys.prefix}/include",
        f"{ghex_path}/include",
        f"{python_site_packages}/gridtools_cpp/data/include",
    ]
    cmake_files: ClassVar = []
    cmake_libraries: ClassVar = []

    headers: ClassVar[list[str]] = [
        "ghex/context.hpp",
        "ghex/unstructured/pattern.hpp",
        "ghex/unstructured/user_concepts.hpp",
        "ghex/communication_object.hpp",
        "stdexcept",
    ]
    state_fields: ClassVar = []
    init_code: ClassVar = ""
    finalize_code: ClassVar = ""
    dependencies: ClassVar = []


class DummyNestedSDFG:
    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        sdfg = dace.SDFG("DummyNestedSDFG")
        state = sdfg.add_state()

        sdfg.add_scalar(name="__return", dtype=dace.int32)

        tasklet = dace.sdfg.nodes.Tasklet(
            "DummyNestedSDFG",
            inputs=None,
            outputs=None,
            code="__out = 1;",
            language=dace.dtypes.Language.CPP,
            side_effects=False,
        )
        state.add_node(tasklet)

        state.add_edge(
            tasklet,
            "__out",
            state.add_write("__return"),
            None,
            dace.Memlet(data="__return", subset="0"),
        )
        tasklet.out_connectors = {"__out": dace.int32}

        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        return {}

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return ([], [])
