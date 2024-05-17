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
from typing import TYPE_CHECKING, Sequence, Union

from gt4py.next import Dimension, Field

from icon4py.model.common.decomposition.definitions import SingleNodeExchange
from icon4py.model.common.settings import device


#try:
import ghex
import mpi4py
from ghex.context import make_context
from ghex.util import Architecture
from ghex.unstructured import (
DomainDescriptor,
HaloGenerator,
make_communication_object,
make_field_descriptor,
make_pattern,
)

mpi4py.rc.initialize = False
mpi4py.rc.finalize = True

#except ImportError:
#    mpi4py = None
#    ghex = None
#    unstructured = None

from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import CellDim, DimensionKind, EdgeDim, VertexDim


if TYPE_CHECKING:
    import mpi4py.MPI


if device.name == "GPU":
    ghex_arch=Architecture.GPU
else:
    ghex_arch=Architecture.CPU

CommId = Union[int, "mpi4py.MPI.Comm", None]
log = logging.getLogger(__name__)


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
def get_multinode_properties(s: definitions.MultiNodeRun, comm_id: CommId = None) -> definitions.ProcessProperties:
    return _get_processor_properties(with_mpi=True, comm_id=comm_id)


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


class GHexMultiNodeExchange:
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
            pattern(make_field_descriptor(domain_descriptor, f, arch=ghex_arch)) for f in fields
        ]
        handle = self._comm.exchange(applied_patterns)
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' initiated.")
        return MultiNodeResult(handle, applied_patterns)

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        res = self.exchange(dim, *fields)
        res.wait()
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' done.")


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


#@definitions.create_exchange.register(MPICommProcessProperties)
#def create_multinode_node_exchange(
#    props: MPICommProcessProperties
#) -> definitions.ExchangeRuntime:
#    if props.comm_size > 1:
#        return GHexMultiNodeExchange(props)
#    else:
#        return SingleNodeExchange()
