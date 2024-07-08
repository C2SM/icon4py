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
from icon4py.model.common.settings import backend

from collections.abc import Callable
from typing import Union
import warnings
from icon4py.model.common.decomposition.definitions import SingleNodeResult
from icon4py.model.common.decomposition.mpi_decomposition import MultiNodeResult

if "dace" in backend.executor.name:
    import sys
    from collections.abc import Sequence
    from dace.frontend.python.common import SDFGConvertible
    from typing import Any, Optional
    import site
    import inspect
    import copy

    import numpy as np
    import dace
    from dace.sdfg.utils import distributed_compile
    from dace import hooks, dtypes
    from dace.config import Config
    from dace.memlet import Memlet
    from dace.properties import CodeBlock
    from gt4py._core import definitions as core_defs
    from gt4py.next.program_processors.runners.dace_iterator.utility import connectivity_identifier
    from icon4py.model.common.decomposition.mpi_decomposition import GHexMultiNodeExchange
    from icon4py.model.common.decomposition.definitions import DecompositionInfo, SingleNodeExchange
    from icon4py.model.common.orchestration.dtypes import *
    from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
    from icon4py.model.common.grid.icon import IconGrid
    from ghex import expose_cpp_ptr
    import mpi4py


def orchestration(method=True):
    def decorator(fuse_func):
        compiled_sdfgs = {}
        def wrapper(*args, **kwargs):
            if "dace" in backend.executor.name:
                if method:
                    # self is used to retrieve the _exchange object -on the fly halo exchanges- and the grid object -offset providers-
                    self = args[0]
                    self_name = [param.name for param in inspect.signature(fuse_func).parameters.values()][0]
                else:
                    warnings.warn("The orchestration decorator is only for methods -at least for now-. Returning the original function.")
                    return fuse_func(*args, **kwargs)

                has_exchange = False
                has_grid = False
                for attr_name, attr_value in self.__dict__.items():
                    if isinstance(attr_value, GHexMultiNodeExchange) or isinstance(attr_value, SingleNodeExchange) and not has_exchange:
                        has_exchange = True
                        exchange_obj = getattr(self, attr_name)
                    if isinstance(attr_value, IconGrid) and not has_grid:
                        has_grid = True
                        grid = getattr(self, attr_name)
                
                if not has_exchange or not has_grid:
                    warnings.warn("No exchnage/grid object found. Returning the original function.")
                    return fuse_func(*args, **kwargs)

                dace_symbols_concretization(exchange_obj, grid, fuse_func, args, kwargs)

                # Parse and compile the fused SDFG -caches it-, i.e. every time the same fused function is called, the same SDFG is used
                parse_and_compile_sdfg(compiled_sdfgs, self, exchange_obj, fuse_func, kwargs)

                with dace.config.temporary_config():
                    configure_dace_temp_env()

                    args = mod_args_for_dace_structures(fuse_func, args)
                    sdfg_args = compiled_sdfgs[id(self)]['dace_program']._create_sdfg_args(compiled_sdfgs[id(self)]['sdfg'], args, kwargs)
                    if method:
                        del sdfg_args[self_name]
                    
                    return compiled_sdfgs[id(self)]['compiled_sdfg'](**sdfg_args)
            else:
                return fuse_func(*args, **kwargs)
        return wrapper
    return decorator


def dace_inhibitor(f: Callable):
    """Triggers callback generation wrapping `func` while doing DaCe parsing."""
    return f


@dace_inhibitor
def wait(comm_handle: Union[int, SingleNodeResult, MultiNodeResult]):
    if isinstance(comm_handle, int):
        pass
    else:
        comm_handle.wait()


if "dace" in backend.executor.name:
    def dace_symbols_concretization(exchange_obj: Union[SingleNodeExchange, GHexMultiNodeExchange], grid: IconGrid, fuse_func: Callable, args: Any, kwargs: Any):
        flattened_xargs_type_value = list(zip(list(fuse_func.__annotations__.values()), list(args) + list(kwargs.values())))
        kwargs.update({
            # DaCe symbols concretization
            **{"CellDim_sym": grid.offset_providers['C2E'].table.shape[0], "EdgeDim_sym": grid.offset_providers['E2C'].table.shape[0], "KDim_sym": grid.num_levels},
            **{f"DiffusionDiagnosticState_{member}_s{str(stride)}_sym": getattr(k_v[1], member).ndarray.strides[stride] // 8 for k_v in flattened_xargs_type_value for member in ["hdef_ic", "div_ic", "dwdx", "dwdy"] for stride in [0,1] if k_v[0] is DiffusionDiagnosticState_t},
            **{f"PrognosticState_{member}_s{str(stride)}_sym": getattr(k_v[1], member).ndarray.strides[stride] // 8 for k_v in flattened_xargs_type_value for member in ["rho", "w", "vn", "exner", "theta_v"] for stride in [0,1] if k_v[0] is PrognosticState_t},
            # [misc] GHEX C++ ptrs, connectivity tables, etc.
            **dace_specific_kwargs(exchange_obj, grid),
            })


    def dace_specific_kwargs(exchange_obj: Union[SingleNodeExchange, GHexMultiNodeExchange], grid: IconGrid) -> dict[str, Any]:
        return {
            # GHEX C++ ptrs
            "__context_ptr":expose_cpp_ptr(exchange_obj._context) if isinstance(exchange_obj, GHexMultiNodeExchange) else 0,
            "__comm_ptr":expose_cpp_ptr(exchange_obj._comm) if isinstance(exchange_obj, GHexMultiNodeExchange) else 0,
            **{f"__pattern_{dim.value}Dim_ptr":expose_cpp_ptr(exchange_obj._patterns[dim]) if isinstance(exchange_obj, GHexMultiNodeExchange) else 0 for dim in (CellDim, VertexDim, EdgeDim)},
            **{f"__domain_descriptor_{dim.value}Dim_ptr":expose_cpp_ptr(exchange_obj._domain_descriptors[dim].__wrapped__) if isinstance(exchange_obj, GHexMultiNodeExchange) else 0 for dim in (CellDim, VertexDim, EdgeDim)},
            # offset providers
            **{f"{connectivity_identifier(k)}":v.table for k,v in grid.offset_providers.items() if hasattr(v, "table")},
            # TODO(kotsaloscv): Possibly needed for future feature, i.e., build GHEX patterns on the fly
            **{f"__gids_{ind.name}_{dim.value}":exchange_obj._decomposition_info.global_index(dim, ind) if isinstance(exchange_obj, GHexMultiNodeExchange) else np.empty(1, dtype=np.int64) for ind in (DecompositionInfo.EntryType.ALL, DecompositionInfo.EntryType.OWNED, DecompositionInfo.EntryType.HALO) for dim in (CellDim, VertexDim, EdgeDim)},
        }


    def parse_and_compile_sdfg(compiled_sdfgs, self, exchange_obj, fuse_func, kwargs):
        """Function that parses and compiles the fused SDFG along with adding the halo exchanges."""
        if id(self) not in compiled_sdfgs:
            compiled_sdfgs[id(self)] = {}

            tmp_exchange_and_wait = exchange_obj.exchange_and_wait
            tmp_exchange = exchange_obj.exchange

            # Replace the manually placed halo exchanges with dummy sdfgs
            exchange_obj.exchange_and_wait = DummyNestedSDFG()
            exchange_obj.exchange = DummyNestedSDFG()

            with dace.config.temporary_config():
                device_type = configure_dace_temp_env()

                compiled_sdfgs[id(self)]['dace_program'] = dace.program(auto_optimize=False,
                                                                        device=dev_type_from_gt4py_to_dace(device_type),
                                                                        distributed_compilation=False)(fuse_func)

                compiled_sdfgs[id(self)]['sdfg'] = compiled_sdfgs[id(self)]['dace_program'].to_sdfg(self, simplify=False, validate=True)

                add_halo_exchanges(compiled_sdfgs[id(self)]['sdfg'], exchange_obj, self.grid.offset_providers, **kwargs)
                
                compiled_sdfgs[id(self)]['sdfg'].simplify(validate=True)

                with hooks.invoke_sdfg_call_hooks(compiled_sdfgs[id(self)]['sdfg']) as sdfg:
                    # TODO(kotsaloscv): Re-think the distributed compilation -all args need to be type annotated and not dace.compiletime-
                    compiled_sdfgs[id(self)]['compiled_sdfg'] = sdfg.compile(validate=compiled_sdfgs[id(self)]['dace_program'].validate)

            # Restore the original exchange methods
            exchange_obj.exchange_and_wait = tmp_exchange_and_wait
            exchange_obj.exchange = tmp_exchange


    def mod_args_for_dace_structures(fuse_func: Callable, args: Any) -> tuple:
        """Modify the args to support DaCe Structures, i.e., teach DaCe how to extract the data from the corresponding GT4Py structures"""
        new_args = [DiffusionDiagnosticState_t.dtype._typeclass.as_ctypes()(hdef_ic=k_v[1].hdef_ic.data_ptr(), div_ic=k_v[1].div_ic.data_ptr(), dwdx=k_v[1].dwdx.data_ptr(), dwdy=k_v[1].dwdy.data_ptr()) if k_v[0] is DiffusionDiagnosticState_t else k_v[1] for k_v in list(zip(list(fuse_func.__annotations__.values()), list(args)))]
        new_args = [PrognosticState_t.dtype._typeclass.as_ctypes()(rho=k_v[1].rho.data_ptr(), w=k_v[1].w.data_ptr(), vn=k_v[1].vn.data_ptr(), exner=k_v[1].exner.data_ptr(), theta_v=k_v[1].theta_v.data_ptr()) if k_v[0] is PrognosticState_t else k_v[1] for k_v in list(zip(list(fuse_func.__annotations__.values()), list(new_args)))]
        for new_arg_ in new_args:
            if isinstance(new_arg_, DiffusionDiagnosticState_t.dtype._typeclass.as_ctypes()):
                new_arg_.descriptor = DiffusionDiagnosticState_t
            if isinstance(new_arg_, PrognosticState_t.dtype._typeclass.as_ctypes()):
                new_arg_.descriptor = PrognosticState_t
        return tuple(new_args)


    def configure_dace_temp_env():
        dace.config.Config.set("cache", value="unique") # no caching or clashes can happen between different processes (MPI)
        dace.config.Config.set("compiler", "allow_view_arguments", value=True) # Allow numpy views as arguments: If true, allows users to call DaCe programs with NumPy views (for example, “A[:,1]” or “w.T”)
        dace.config.Config.set("optimizer", "automatic_simplification", value=False) # simplifications & optimizations after placing halo exchanges -need a sequential structure of nested sdfgs-
        dace.config.Config.set("optimizer", "autooptimize", value=False)
        device_type = backend.executor.otf_workflow.step.translation.device_type
        if device_type == core_defs.DeviceType.CPU:
            device = "cpu"
            compiler_args = dace.config.Config.get("compiler", "cpu", "args")
            compiler_args = compiler_args.replace("-std=c++14", "-std=c++17") # needed for GHEX
            # disable finite-math-only in order to support isfinite/isinf/isnan builtins
            if "-ffast-math" in compiler_args:
                compiler_args += " -fno-finite-math-only"
            if "-ffinite-math-only" in compiler_args:
                compiler_args = compiler_args.replace("-ffinite-math-only", "")
        elif device_type == core_defs.DeviceType.CUDA:
            device = "cuda"
            compiler_args = dace.config.Config.get("compiler", "cuda", "args")
            compiler_args = compiler_args.replace("-Xcompiler", "-Xcompiler -std=c++17")
            compiler_args += " -std=c++17"
        else:
            raise ValueError("The device type is not supported.")
        dace.config.Config.set("compiler", device, "args", value=compiler_args)

        return device_type


    def dev_type_from_gt4py_to_dace(device_type: core_defs.DeviceType) -> dace.dtypes.DeviceType:
        if device_type == core_defs.DeviceType.CPU:
            return dace.dtypes.DeviceType.CPU
        elif device_type == core_defs.DeviceType.CUDA:
            return dace.dtypes.DeviceType.GPU
        else:
            raise ValueError("The device type is not supported.")


    class DummyNestedSDFG(SDFGConvertible):
        """Dummy replacement of the manually placed halo exchanges"""
        def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
            sdfg = dace.SDFG('DummyNestedSDFG')
            state = sdfg.add_state()

            sdfg.add_scalar(name='__return', dtype=dace.int32)

            tasklet = dace.sdfg.nodes.Tasklet('DummyNestedSDFG',
                                              inputs=None,
                                              outputs=None,
                                              code="__out = 1;",
                                              language=dace.dtypes.Language.CPP,
                                              side_effects=False,)
            state.add_node(tasklet)

            state.add_edge(tasklet, '__out', state.add_write('__return'), None, dace.Memlet(data='__return', subset='0'))
            tasklet.out_connectors = {'__out':dace.int32}

            return sdfg

        def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
            return {}

        def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            return ([],[])


    def add_halo_exchanges(sdfg, exchange, offset_providers, **kwargs):
        '''Add halo exchange nodes to the SDFG only where needed.'''
        if not isinstance(exchange, GHexMultiNodeExchange):
            return

        # TODO(kotsaloscv): Work on asynchronous communication
        wait = True
        
        counter = 0
        for nested_sdfg in sdfg.all_sdfgs_recursive():
            if not hasattr(nested_sdfg, "gt4py_program_output_fields"):
                continue

            if len(nested_sdfg.gt4py_program_output_fields) == 0:
                continue

            field_dims = set()
            for dim in nested_sdfg.gt4py_program_output_fields.values():
                field_dims.add(dim)

            if len(field_dims) > 1:
                raise ValueError("The output fields to be communicated are not defined on the same dimension.")

            # dimension of the fields to be exchanged
            dim = {'Cell':CellDim, 'Edge':EdgeDim, 'Vertex':VertexDim}[list(field_dims)[0].value] if len(field_dims) == 1 else None
            if not dim:
                continue

            if nested_sdfg.gt4py_program_kwargs.get('horizontal_start', None) == None or nested_sdfg.gt4py_program_kwargs.get('horizontal_end', None) == None:
                continue

            halos_inds = exchange._decomposition_info.local_index(dim, DecompositionInfo.EntryType.HALO)
            # Consider overcomputing, i.e. computational update of the halo elements and not through communication.
            updated_halo_inds = np.intersect1d(halos_inds, np.arange(nested_sdfg.gt4py_program_kwargs['horizontal_start'], nested_sdfg.gt4py_program_kwargs['horizontal_end']))

            for nested_sdfg_state in sdfg.states():
                if nested_sdfg_state.label == nested_sdfg.parent.label:
                    break

            global_buffers = {} # the elements of this dictionary are going to be exchanged
            for buffer_name in nested_sdfg.gt4py_program_output_fields:
                global_buffer_name = None
                for edge in nested_sdfg_state.all_edges_recursive():
                    # local buffer_name [src] --> global_buffer_name [dst]
                    if hasattr(edge[0], "src_conn") and (edge[0].src_conn == buffer_name):
                        global_buffer_name = edge[0].dst.label
                        break
                if not global_buffer_name:
                    raise ValueError("Could not link the local buffer_name to the global one (coming from the closure).")
                
                # Visit all stencils/sdfgs below the current one (nested_sdfg), and see if halo update is needed for the specific (buffer_name) output field.
                halo_update = False
                cont = True
                for ones_after_nsdfg in sdfg.all_sdfgs_recursive():
                    if halo_update:
                        break
                    
                    if not hasattr(ones_after_nsdfg, "gt4py_program_input_fields"):
                        continue

                    if ones_after_nsdfg is nested_sdfg:
                        cont = False
                        continue

                    if cont:
                        continue

                    if ones_after_nsdfg.gt4py_program_kwargs.get('horizontal_start', None) == None or ones_after_nsdfg.gt4py_program_kwargs.get('horizontal_end', None) == None:
                        continue

                    for ones_after_nsdfg_state in sdfg.states():
                        if ones_after_nsdfg_state.label == ones_after_nsdfg.parent.label:
                            break

                    for buffer_name_ in ones_after_nsdfg.gt4py_program_input_fields:
                        if halo_update:
                            break
                        
                        global_buffer_name_ = None
                        for edge in ones_after_nsdfg_state.all_edges_recursive():
                            # global_buffer_name_ [src] --> local buffer_name_ [dst]
                            if hasattr(edge[0], "dst_conn") and (edge[0].dst_conn == buffer_name_):
                                global_buffer_name_ = edge[0].src.label
                                break
                        if not global_buffer_name_:
                            raise ValueError("Could not link the local buffer_name_ to the global one.")
                        
                        if global_buffer_name_ != global_buffer_name:
                            continue

                        for op_ in ones_after_nsdfg.offset_providers_per_input_field[buffer_name_]:
                            if len(op_) == 0:
                                continue
                            offset_literal_value = op_[0].value
                            if not hasattr(offset_providers[offset_literal_value], 'table'):
                                continue
                            op = offset_providers[offset_literal_value].table

                            source_dim = {'C':CellDim, 'E':EdgeDim, 'V':VertexDim}[offset_literal_value[0]]
                            if source_dim != list(ones_after_nsdfg.gt4py_program_output_fields.values())[0]:
                                continue
                            
                            accessed_inds = op[ones_after_nsdfg.gt4py_program_kwargs['horizontal_start']]
                            for ind in range(ones_after_nsdfg.gt4py_program_kwargs['horizontal_start']+1, ones_after_nsdfg.gt4py_program_kwargs['horizontal_end']):
                                accessed_inds = np.concatenate((accessed_inds, op[ind]))

                            accessed_halo_inds = np.intersect1d(halos_inds, accessed_inds)
                            if not np.all(np.isin(accessed_halo_inds, updated_halo_inds)):
                                # TODO(kotsaloscv): Communicate only the non-accesed halo elements
                                halo_update = True
                                break
                
                # sync MPI ranks, otherwise deadlock
                all_bools = mpi4py.MPI.COMM_WORLD.allgather(halo_update)
                if len(set(all_bools)) == 2:
                    halo_update = True
                else:
                    halo_update = all_bools[0]

                if halo_update:
                    global_buffers[global_buffer_name] = sdfg.arrays[global_buffer_name]
            
            if len(global_buffers) == 0:
                # There is no field to exchange
                continue

            # Start buidling the halo exchange node
            state = sdfg.add_state_after(nested_sdfg_state, label='_halo_exchange_')

            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_',
                                            inputs=None,
                                            outputs=None,
                                            code='',
                                            language=dace.dtypes.Language.CPP,
                                            side_effects=True,)
            state.add_node(tasklet)

            in_connectors = {}
            out_connectors = {}

            if counter == 0:
                for buffer_name in kwargs:
                    if '_ptr' in buffer_name:
                        # Add GHEX C++ obj ptrs (just once)
                        sdfg.add_scalar(buffer_name, dtype=dace.uintp)

            for i, (buffer_name, data_descriptor) in enumerate(global_buffers.items()):
                buffer = state.add_read(buffer_name)
                in_connectors['IN_' + f'field_{i}'] = dtypes.pointer(data_descriptor.dtype)
                state.add_edge(buffer, None, tasklet, 'IN_' + f'field_{i}', Memlet.from_array(buffer_name, data_descriptor))

                update = state.add_write(buffer_name)
                out_connectors['OUT_' + f'field_{i}'] = dtypes.pointer(data_descriptor.dtype)
                state.add_edge(tasklet, 'OUT_' + f'field_{i}', update, None, Memlet.from_array(buffer_name, data_descriptor))

            if counter == 0:
                for buffer_name in kwargs:
                    if '_ptr' in buffer_name:
                        buffer = state.add_read(buffer_name)
                        data_descriptor = dace.uintp
                        in_connectors['IN_' + buffer_name] = data_descriptor.dtype
                        memlet_ =  Memlet(buffer_name, subset='0')
                        state.add_edge(buffer, None, tasklet, 'IN_' + buffer_name, memlet_)

            tasklet.in_connectors = in_connectors
            tasklet.out_connectors = out_connectors
            tasklet.environments = ['icon4py.model.common.orchestration.decorator.DaceGHEX']

            pattern_type = exchange._patterns[dim].__cpp_type__
            domain_descriptor_type = exchange._domain_descriptors[dim].__cpp_type__
            communication_object_type = exchange._comm.__cpp_type__
            communication_handle_type = communication_object_type[communication_object_type.find('<')+1:communication_object_type.rfind('>')]

            fields_desc_glob_vars = '\n'
            fields_desc = '\n'
            descr_unique_names = []
            for i, arg in enumerate(copy.deepcopy(list(global_buffers.values()))):
                if isinstance(arg.shape[0], dace.symbolic.symbol):
                    arg.shape   = (kwargs[str(arg.shape[0])]  , kwargs[str(arg.shape[1])])
                    arg.strides = (kwargs[str(arg.strides[0])], kwargs[str(arg.strides[1])])
                # https://github.com/ghex-org/GHEX/blob/master/bindings/python/src/_pyghex/unstructured/field_descriptor.cpp
                if len(arg.shape) > 2:
                    raise ValueError("field has too many dimensions")
                if arg.shape[0] != exchange._domain_descriptors[dim].size():
                    raise ValueError("field's first dimension must match the size of the domain")
                
                levels_first = True
                outer_strides = 0
                # DaCe strides: number of elements to jump
                # GHEX/NumPy strides: number of bytes to jump
                if len(arg.shape) == 2 and arg.strides[1] != 1:
                    levels_first = False
                    if arg.strides[0] != 1:
                        raise ValueError("field's strides are not compatible with GHEX")
                    outer_strides = arg.strides[1]
                elif len(arg.shape) == 2:
                    if arg.strides[1] != 1:
                        raise ValueError("field's strides are not compatible with GHEX")
                    outer_strides = arg.strides[0]
                else:
                    if arg.strides[0] != 1:
                        raise ValueError("field's strides are not compatible with GHEX")

                levels = 1 if len(arg.shape) == 1 else arg.shape[1]

                device = 'cpu' if arg.storage.value <= 5 else 'gpu'
                field_dtype = arg.dtype.ctype
                
                descr_unique_name = f'field_desc_{i}_{counter}_{id(exchange)}'
                descr_unique_names.append(descr_unique_name)
                descr_type_ = f"ghex::unstructured::data_descriptor<ghex::{device}, int, int, {field_dtype}>"
                if wait:
                    # de-allocated descriptors once out-of-scope, no need for storing them in global vars
                    fields_desc += f"{descr_type_} {descr_unique_name}{{*domain_descriptor, IN_field_{i}, {levels}, {'true' if levels_first else 'false'}, {outer_strides}}};\n"
                else:
                    # for async exchange, we need to keep the descriptors alive, until the wait
                    fields_desc += f"{descr_unique_name} = std::make_unique<{descr_type_}>(*domain_descriptor, IN_field_{i}, {levels}, {'true' if levels_first else 'false'}, {outer_strides});\n"
                    fields_desc_glob_vars += f"std::unique_ptr<{descr_type_}> {descr_unique_name};\n"

            code = ''
            if counter == 0:
                __pattern = ''
                __domain_descriptor = ''
                for dim_ in (CellDim, VertexDim, EdgeDim):
                    __pattern += f"__pattern_{dim_.value}Dim_ptr_{id(exchange)} = IN___pattern_{dim_.value}Dim_ptr;\n"
                    __domain_descriptor += f"__domain_descriptor_{dim_.value}Dim_ptr_{id(exchange)} = IN___domain_descriptor_{dim_.value}Dim_ptr;\n"
                    
                code = f'''
                    __context_ptr_{id(exchange)} = IN___context_ptr;
                    __comm_ptr_{id(exchange)} = IN___comm_ptr;
                    {__pattern}
                    {__domain_descriptor}
                    '''

            code += f'''
                    ghex::context* m = reinterpret_cast<ghex::context*>(__context_ptr_{id(exchange)});
                    
                    {pattern_type}* pattern = reinterpret_cast<{pattern_type}*>(__pattern_{dim.value}Dim_ptr_{id(exchange)});
                    {domain_descriptor_type}* domain_descriptor = reinterpret_cast<{domain_descriptor_type}*>(__domain_descriptor_{dim.value}Dim_ptr_{id(exchange)});
                    {communication_object_type}* communication_object = reinterpret_cast<{communication_object_type}*>(__comm_ptr_{id(exchange)});

                    {fields_desc}

                    h_{id(exchange)} = communication_object->exchange({", ".join([f'(*pattern)({"" if wait else "*"}{descr_unique_names[i]})' for i in range(len(global_buffers))])});
                    { 'h_'+str(id(exchange))+'.wait();' if wait else ''}
                    '''

            tasklet.code = CodeBlock(code=code, language=dace.dtypes.Language.CPP)
            if counter == 0:
                __pattern = ''
                __domain_descriptor = ''
                for dim_ in (CellDim, VertexDim, EdgeDim):
                    __pattern += f"{dace.uintp.dtype} __pattern_{dim_.value}Dim_ptr_{id(exchange)};\n"
                    __domain_descriptor += f"{dace.uintp.dtype} __domain_descriptor_{dim_.value}Dim_ptr_{id(exchange)};\n"
                
                code = f'''
                        {dace.uintp.dtype} __context_ptr_{id(exchange)};
                        {dace.uintp.dtype} __comm_ptr_{id(exchange)};
                        {__pattern}
                        {__domain_descriptor}
                        {fields_desc_glob_vars}
                        ghex::communication_handle<{communication_handle_type}> h_{id(exchange)};
                        '''
            else:
                code = fields_desc_glob_vars
            tasklet.code_global = CodeBlock(code=code, language=dace.dtypes.Language.CPP)

            counter += 1


    @dace.library.environment
    class DaceGHEX:
        python_site_packages = site.getsitepackages()[0]
        ghex_path = python_site_packages + '/ghex' # 'absolute_path_to/GHEX/build' [case of manual compilation]

        cmake_minimum_version = None
        cmake_packages = []
        cmake_variables = {}
        cmake_compile_flags = []
        cmake_link_flags = [f"-L{ghex_path}/lib -lghex -lhwmalloc -loomph_common -loomph_mpi"]
        cmake_includes = [f'{sys.prefix}/include', f'{ghex_path}/include', f'{python_site_packages}/gridtools_cpp/data/include']
        cmake_files = []
        cmake_libraries = []

        headers = ["ghex/context.hpp", "ghex/unstructured/pattern.hpp", "ghex/unstructured/user_concepts.hpp", "ghex/communication_object.hpp"]
        state_fields = []
        init_code = ""
        finalize_code = ""
        dependencies = []
