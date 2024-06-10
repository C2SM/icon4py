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

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union, Any, Dict, Optional, Sequence, Tuple

from gt4py.next import Dimension, Field

from icon4py.model.common.decomposition.definitions import SingleNodeExchange


try:
    import ghex
    import mpi4py
    
    from ghex import expose_cpp_ptr
    from ghex.context import make_context
    from ghex.unstructured import (
        DomainDescriptor,
        HaloGenerator,
        make_communication_object,
        make_field_descriptor,
        make_pattern,
    )

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = True

except ImportError:
    mpi4py = None
    ghex = None
    unstructured = None

from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import CellDim, DimensionKind, EdgeDim, VertexDim
from icon4py.model.common.decomposition.definitions import DecompositionInfo as di

import sys
import site
import dace
import dace.library
from dace.frontend.python.common import SDFGConvertible
from dace.memlet import Memlet
from dace import dtypes
from dace.properties import CodeBlock
import numpy as np


if TYPE_CHECKING:
    import mpi4py.MPI


CommId = Union[int, "mpi4py.MPI.Comm", None]
log = logging.getLogger(__name__)


def as_dace_type(np_dtype):
    if np_dtype == np.int32:
        return dace.int32
    elif np_dtype == np.int64:
        return dace.int64
    raise ValueError(f"'{np_dtype}' not supported.")


def init_mpi():
    from mpi4py import MPI

    if not MPI.Is_initialized():
        log.info("initializing MPI")
        MPI.Init()


def finalize_mpi():
    from mpi4py import MPI

    if not MPI.Is_finalized():
        log.info("finalizing MPI")
        MPI.Finalize()


def _get_processor_properties(with_mpi=False, comm_id: CommId = None):
    def _get_current_comm_or_comm_world(comm_id: CommId) -> mpi4py.MPI.Comm:
        if isinstance(comm_id, int):
            comm = mpi4py.MPI.Comm.f2py(comm_id)
        elif isinstance(comm_id, mpi4py.MPI.Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    if with_mpi:
        init_mpi()
        current_comm = _get_current_comm_or_comm_world(comm_id)
        return MPICommProcessProperties(current_comm)


class ParallelLogger(logging.Filter):
    def __init__(self, process_properties: definitions.ProcessProperties = None):
        super().__init__()
        self._rank_info = ""
        if process_properties and process_properties.comm_size > 1:
            self._rank_info = f"rank={process_properties.rank}/{process_properties.comm_size} [{process_properties.comm_name}] "

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank_info
        return True


@definitions.get_processor_properties.register(definitions.MultiNodeRun)
def get_multinode_properties(s: definitions.MultiNodeRun) -> definitions.ProcessProperties:
    return _get_processor_properties(with_mpi=True)


@dataclass(frozen=True)
class MPICommProcessProperties(definitions.ProcessProperties):
    comm: mpi4py.MPI.Comm = None

    @functools.cached_property
    def rank(self):
        return self.comm.Get_rank()

    @functools.cached_property
    def comm_name(self):
        return self.comm.Get_name()

    @functools.cached_property
    def comm_size(self):
        return self.comm.Get_size()


class GHexMultiNodeExchange(SDFGConvertible):
    def __init__(
        self,
        props: definitions.ProcessProperties,
        domain_decomposition: definitions.DecompositionInfo,
    ):
        self._context = make_context(props.comm, False)
        self._domain_id_gen = definitions.DomainDescriptorIdGenerator(props)
        self._decomposition_info = domain_decomposition
        self._domain_descriptors = {
            CellDim: self._create_domain_descriptor(
                CellDim,
            ),
            VertexDim: self._create_domain_descriptor(
                VertexDim,
            ),
            EdgeDim: self._create_domain_descriptor(EdgeDim),
        }
        log.info(f"domain descriptors for dimensions {self._domain_descriptors.keys()} initialized")

        self._patterns = {
            CellDim: self._create_pattern(CellDim),
            VertexDim: self._create_pattern(VertexDim),
            EdgeDim: self._create_pattern(EdgeDim),
        }
        log.info(f"patterns for dimensions {self._patterns.keys()} initialized ")
        self._comm = make_communication_object(self._context)

        self.max_num_of_fields_to_communicate = 10 # maximum number of fields to perform halo exchange on (DaCe-related)
        self.return_sdfg = False # DaCe-related
        self.counter = 0 # Some SDFG variables need to be defined only once (DaCe-related)

        log.info("communication object initialized")

    def _domain_descriptor_info(self, descr):
        return f" domain_descriptor=[id='{descr.domain_id()}', size='{descr.size()}', inner_size='{descr.inner_size()}' (halo size='{descr.size() - descr.inner_size()}')"

    def get_size(self):
        return self._context.size()

    def my_rank(self):
        return self._context.rank()

    def _create_domain_descriptor(self, dim: Dimension):
        all_global = self._decomposition_info.global_index(
            dim, definitions.DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, definitions.DecompositionInfo.EntryType.HALO
        )
        # first arg is the domain ID which builds up an MPI Tag.
        # if those ids are not different for all domain descriptors the system might deadlock
        # if two parallel exchanges with the same domain id are done
        domain_desc = DomainDescriptor(
            self._domain_id_gen(), all_global.tolist(), local_halo.tolist()
        )
        log.debug(
            f"domain descriptor for dim='{dim.value}' with properties {self._domain_descriptor_info(domain_desc)} created"
        )
        return domain_desc

    def _create_pattern(self, horizontal_dim: Dimension):
        assert horizontal_dim.kind == DimensionKind.HORIZONTAL

        global_halo_idx = self._decomposition_info.global_index(
            horizontal_dim, definitions.DecompositionInfo.EntryType.HALO
        )
        halo_generator = HaloGenerator.from_gids(global_halo_idx)
        log.debug(f"halo generator for dim='{horizontal_dim.value}' created")
        pattern = make_pattern(
            self._context,
            halo_generator,
            [self._domain_descriptors[horizontal_dim]],
        )
        log.debug(
            f"pattern for dim='{horizontal_dim.value}' and {self._domain_descriptor_info(self._domain_descriptors[horizontal_dim])} created"
        )
        return pattern

    def exchange(self, dim: definitions.Dimension, *fields: Sequence[Field]):
        assert dim in [CellDim, EdgeDim, VertexDim]
        pattern = self._patterns[dim]
        assert pattern is not None, f"pattern for {dim.value} not found"
        domain_descriptor = self._domain_descriptors[dim]
        assert domain_descriptor is not None, f"domain descriptor for {dim.value} not found"
        applied_patterns = [
            pattern(make_field_descriptor(domain_descriptor, f.asnumpy())) for f in fields
        ]
        handle = self._comm.exchange(applied_patterns)
        log.info(f"exchange for {len(fields)} fields of dimension ='{dim.value}' initiated.")
        return MultiNodeResult(handle, applied_patterns)

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        res = self.exchange(dim, *fields)
        res.wait()
    
    def __call__(self, *args, **kwargs) -> Optional[dace.SDFG]:
        dim = kwargs.get('dim', None)
        if not dim:
            raise ValueError("Need to define a dimension.")
        wait = kwargs.get('wait', True)
        
        if self.return_sdfg:
            sdfg = dace.SDFG('_halo_exchange_')
            state = sdfg.add_state()

            if self.counter == 0:
                for buffer_name in self.__sdfg_closure__():
                    if '__gids_' in buffer_name or '__lids_' in buffer_name:
                        data_descriptor = dace.data.Array(dtype=as_dace_type(self.__sdfg_closure__()[buffer_name].dtype), 
                                                          shape=self.__sdfg_closure__()[buffer_name].shape)
                        sdfg.add_array(buffer_name,
                                       data_descriptor.shape,
                                       data_descriptor.dtype)
                    else:
                        sdfg.add_scalar(buffer_name, dtype=dace.uintp)

            for arg in zip(self.__sdfg_signature__()[0], args):
                buffer_name = arg[0]
                data_descriptor = arg[1]
                sdfg.add_array(buffer_name,
                               data_descriptor.shape,
                               data_descriptor.dtype,
                               storage=data_descriptor.storage,
                               strides=data_descriptor.strides)
            
            # Dummy return: preserve same interface with non-DaCe version
            sdfg.add_scalar(name='__return', dtype=dace.int32)

            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_',
                                              inputs=None,
                                              outputs=None,
                                              code='',
                                              language=dace.dtypes.Language.CPP,
                                              side_effects=True,)
            state.add_node(tasklet)

            in_connectors = {}
            out_connectors = {}

            for i, arg in enumerate(zip(self.__sdfg_signature__()[0], args)):
                buffer_name = arg[0]
                data_descriptor = arg[1]

                buffer = state.add_read(buffer_name)
                in_connectors['IN_' + buffer_name] = dtypes.pointer(data_descriptor.dtype)
                state.add_edge(buffer, None, tasklet, 'IN_' + buffer_name, Memlet.from_array(buffer_name, data_descriptor))

                update = state.add_write(buffer_name)
                out_connectors['OUT_' + buffer_name] = dtypes.pointer(data_descriptor.dtype)
                state.add_edge(tasklet, 'OUT_' + buffer_name, update, None, Memlet.from_array(buffer_name, data_descriptor))

            if self.counter == 0:
                for i, buffer_name in enumerate(self.__sdfg_closure__()):
                    buffer = state.add_read(buffer_name)
                    if '__gids_' in buffer_name or '__lids_' in buffer_name:
                        data_descriptor = dace.data.Array(dtype=as_dace_type(self.__sdfg_closure__()[buffer_name].dtype), 
                                                          shape=self.__sdfg_closure__()[buffer_name].shape)
                        in_connectors['IN_' + buffer_name] = dtypes.pointer(data_descriptor.dtype)
                        memlet_ = Memlet.from_array(buffer_name, data_descriptor)
                    else:
                        data_descriptor = dace.uintp
                        in_connectors['IN_' + buffer_name] = data_descriptor.dtype
                        memlet_ =  Memlet(buffer_name, subset='0')
                    state.add_edge(buffer, None, tasklet, 'IN_' + buffer_name, memlet_)

            ret = state.add_write('__return')
            state.add_edge(tasklet, '__out', ret, None, dace.Memlet(data='__return', subset='0'))
            out_connectors['__out'] = dace.int32

            tasklet.in_connectors = in_connectors
            tasklet.out_connectors = out_connectors
            tasklet.environments = ['icon4py.model.common.decomposition.mpi_decomposition.DaceGHEX']
            
            pattern_type = self._patterns[dim].__cpp_type__
            domain_descriptor_type = self._domain_descriptors[dim].__cpp_type__
            communication_object_type = self._comm.__cpp_type__
            communication_handle_type = communication_object_type[communication_object_type.find('<')+1:communication_object_type.rfind('>')]
            
            fields_desc_glob_vars = '\n'
            fields_desc = '\n'
            descr_unique_names = []
            for i, arg in enumerate(args):
                # https://github.com/ghex-org/GHEX/blob/master/bindings/python/src/_pyghex/unstructured/field_descriptor.cpp
                if len(arg.shape) > 2:
                    raise ValueError("field has too many dimensions")
                if arg.shape[0] != self._domain_descriptors[dim].size():
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
                
                descr_unique_name = f'field_desc_{i}_{self.counter}_{id(self)}'
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
            if self.counter == 0:
                __pattern = ''
                __domain_descriptor = ''
                __gids = ''
                for dim_ in (CellDim, VertexDim, EdgeDim):
                    __pattern += f"__pattern_{dim_.value}Dim_ptr_{id(self)} = IN___pattern_{dim_.value}Dim_ptr;\n"
                    __domain_descriptor += f"__domain_descriptor_{dim_.value}Dim_ptr_{id(self)} = IN___domain_descriptor_{dim_.value}Dim_ptr;\n"

                    for ind in (di.EntryType.ALL, di.EntryType.OWNED, di.EntryType.HALO):
                        __gids += f"__gids_{ind.name}_{dim_.value}_{id(self)} = IN___gids_{ind.name}_{dim_.value};\n"
                    
                code = f'''
                       __context_ptr_{id(self)} = IN___context_ptr;
                       __comm_ptr_{id(self)} = IN___comm_ptr;
                       {__pattern}
                       {__domain_descriptor}
                       {__gids}
                       '''

            code += f'''
                    ghex::context* m = reinterpret_cast<ghex::context*>(__context_ptr_{id(self)});
                    
                    {pattern_type}* pattern = reinterpret_cast<{pattern_type}*>(__pattern_{dim.value}Dim_ptr_{id(self)});
                    {domain_descriptor_type}* domain_descriptor = reinterpret_cast<{domain_descriptor_type}*>(__domain_descriptor_{dim.value}Dim_ptr_{id(self)});
                    {communication_object_type}* communication_object = reinterpret_cast<{communication_object_type}*>(__comm_ptr_{id(self)});

                    {fields_desc}

                    h_{id(self)} = communication_object->exchange({", ".join([f'(*pattern)({"" if wait else "*"}{descr_unique_names[i]})' for i in range(len(args))])});
                    { 'h_'+str(id(self))+'.wait();' if wait else ''}

                    __out = 1234; // Dummy return;
                    '''

            # # Debugging
            # field_desc_out = '\n'
            # for i, arg in enumerate(args):
            #     field_desc_out += f'outFile << {descr_unique_names[i]}.device_id() << ", " << {descr_unique_names[i]}.domain_id() << ", " << {descr_unique_names[i]}.domain_size() << ", " << {descr_unique_names[i]}.num_components() << std::endl;\n'
            # code += f'''
            #         std::stringstream filenameStream;
            #         filenameStream << "RANK_" << m->rank() << ".txt";
            #         std::string filename = filenameStream.str();
            #         std::ofstream outFile(filename);
            #         if (outFile.is_open()) {{
            #             outFile << m->rank() << ", " << m->size() << std::endl;
            #             outFile << pattern->size() << ", " << pattern->max_tag() << std::endl;
            #             outFile << domain_descriptor->domain_id() << ", " << domain_descriptor->inner_size() << ", " << domain_descriptor->size() << std::endl;
            #             {field_desc_out}
            #             outFile << __out << std::endl;
            #             outFile << IN___context_ptr << ", " << IN___comm_ptr << std::endl;
            #             outFile << IN___pattern_CellDim_ptr << ", " << IN___pattern_VertexDim_ptr << ", " << IN___pattern_EdgeDim_ptr << std::endl;
            #             outFile << IN___domain_descriptor_CellDim_ptr << ", " << IN___domain_descriptor_VertexDim_ptr << ", " << IN___domain_descriptor_EdgeDim_ptr << std::endl;
            #             outFile.close();
            #         }} else {{
            #             ;
            #         }}
            #         '''

            tasklet.code = CodeBlock(code=code, language=dace.dtypes.Language.CPP)
            if self.counter == 0:
                __pattern = ''
                __domain_descriptor = ''
                __gids = ''
                for dim_ in (CellDim, VertexDim, EdgeDim):
                    __pattern += f"{dace.uintp.dtype} __pattern_{dim_.value}Dim_ptr_{id(self)};\n"
                    __domain_descriptor += f"{dace.uintp.dtype} __domain_descriptor_{dim_.value}Dim_ptr_{id(self)};\n"

                    for ind in (di.EntryType.ALL, di.EntryType.OWNED, di.EntryType.HALO):
                        __gids += f"int* __gids_{ind.name}_{dim_.value}_{id(self)};\n"
                
                code = f'''
                        {dace.uintp.dtype} __context_ptr_{id(self)};
                        {dace.uintp.dtype} __comm_ptr_{id(self)};
                        {__pattern}
                        {__domain_descriptor}
                        {__gids}
                        {fields_desc_glob_vars}
                        ghex::communication_handle<{communication_handle_type}> h_{id(self)};
                        '''
            else:
                code = fields_desc_glob_vars
            tasklet.code_global = CodeBlock(code=code, language=dace.dtypes.Language.CPP)
            

            self.return_sdfg = False # reset
            return sdfg
        else:
            res = self.exchange(dim, *args)
            if wait:
                res.wait()
            else:
                return res

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        self.return_sdfg = True

        sdfg = self.__call__(*args, **kwargs)
        
        sdfg.arg_names.extend(self.__sdfg_signature__()[0])
        sdfg.arg_names.extend(list(self.__sdfg_closure__().keys()))

        self.counter += 1
        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {'__context_ptr':expose_cpp_ptr(self._context),
                '__comm_ptr':expose_cpp_ptr(self._comm),
                **{f"__pattern_{dim.value}Dim_ptr":expose_cpp_ptr(self._patterns[dim]) for dim in (CellDim, VertexDim, EdgeDim)},
                **{f"__domain_descriptor_{dim.value}Dim_ptr":expose_cpp_ptr(self._domain_descriptors[dim].__wrapped__) for dim in (CellDim, VertexDim, EdgeDim)},
                #
                **{f"__gids_{ind.name}_{dim.value}":self._decomposition_info.global_index(dim, ind) for ind in (di.EntryType.ALL, di.EntryType.OWNED, di.EntryType.HALO) for dim in (CellDim, VertexDim, EdgeDim)},
                }
    
    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        args = []
        for i in range(self.max_num_of_fields_to_communicate):
            args.append(f'field_{i}')
        return (args,[])


@dataclass
class WaitOnComm(SDFGConvertible):
    exchange_object: ...
    return_sdfg : bool = False

    def __call__(self, *args, **kwargs) -> Optional[dace.SDFG]:
        if self.return_sdfg:
            sdfg = dace.SDFG('_halo_exchange_wait_')
            state = sdfg.add_state()

            # Dummy return, otherwise dead-dataflow-elimination kicks in. Return something to generate code.
            sdfg.add_scalar(name='__return', dtype=dace.int32)

            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_wait_',
                                              inputs=None,
                                              outputs=None,
                                              code=f'h_{id(self.exchange_object)}.wait();\n__out = 1234;',
                                              language=dace.dtypes.Language.CPP,
                                              side_effects=False,)
            state.add_node(tasklet)

            ret = state.add_write('__return')
            state.add_edge(tasklet, '__out', ret, None, dace.Memlet(data='__return', subset='0'))
            tasklet.out_connectors = {'__out':dace.int32}

            self.return_sdfg = False # reset
            return sdfg
        else:
            communication_handle = args[0]
            communication_handle.wait()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        self.return_sdfg = True
        sdfg = self.__call__(*args, **kwargs)
        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}
    
    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        return ([],[])


@dataclass
class MultiNodeResult:
    handle: ...
    pattern_refs: ...

    def wait(self):
        self.handle.wait()
        del self.pattern_refs

    def is_ready(self) -> bool:
        return self.handle.is_ready()


@definitions.create_exchange.register(MPICommProcessProperties)
def create_multinode_node_exchange(
    props: MPICommProcessProperties, decomp_info: definitions.DecompositionInfo
) -> definitions.ExchangeRuntime:
    if props.comm_size > 1:
        return GHexMultiNodeExchange(props, decomp_info)
    else:
        return SingleNodeExchange()

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
