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
import logging
from enum import Enum
from typing import Union

import mpi4py
import numpy as np
import numpy.ma as ma
from ghex import unstructured as ghex
from gt4py.next.common import Dimension, DimensionKind
from mpi4py.MPI import Comm

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.diffusion.diffusion_utils import builder


mpi4py.rc.initialize = False
# mpi4py.rc.finalize = False

CommId = Union[int, Comm, None]
log = logging.getLogger(__name__)


def get_processor_properties(comm_id: CommId = None):
    init_mpi()

    def _get_current_comm_or_comm_world(comm_id: CommId) -> Comm:
        if isinstance(comm_id, int):
            comm = Comm.f2py(comm_id)
        elif isinstance(comm_id, Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    current_comm = _get_current_comm_or_comm_world(comm_id)
    return ProcessProperties.from_mpi_comm(current_comm)


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

class DomainDescriptorIdGenerator():
    _counter = 0
    _roundtrips = 0

    def __init__(self, context):
        self._comm_size = context.size()
        self._roundtrips = context.rank()
        self._base = self._roundtrips * self._comm_size

    def __call__(self):
        id = self._base + self._counter
        if self._counter + 1 >= self._comm_size:
            self._roundtrips = self._roundtrips + self._comm_size
            self._base = self._roundtrips * self._comm_size
            self._counter = 0
        else:
            self._counter = self._counter + 1
        return id

class DecompositionInfo:
    class EntryType(int, Enum):
        ALL = (0,)
        OWNED = (1,)
        HALO = 2

    @builder
    def with_dimension(
        self, dim: Dimension, global_index: np.ndarray, owner_mask: np.ndarray
    ):
        masked_global_index = ma.array(global_index, mask=owner_mask)
        self._global_index[dim] = masked_global_index

    def __init__(self, klevels: int):
        self._global_index = {}
        self._klevels = klevels

    @property
    def klevels(self):
        return self._klevels

    def local_index(self, dim: Dimension, entry_type: EntryType = EntryType.ALL):
        match (entry_type):
            case DecompositionInfo.EntryType.ALL:
                return self._to_local_index(dim)
            case DecompositionInfo.EntryType.HALO:
                index = self._to_local_index(dim)
                mask = self._global_index[dim].mask
                return index[~mask]
            case DecompositionInfo.EntryType.OWNED:
                index = self._to_local_index(dim)
                mask = self._global_index[dim].mask
                return index[mask]

    def _to_local_index(self, dim):
        data = ma.getdata(self._global_index[dim], subok=False)
        assert data.ndim == 1
        return np.arange(data.shape[0])

    def owner_mask(self, dim: Dimension) -> np.ndarray:
        return self._global_index[dim].mask

    def global_index(self, dim: Dimension, entry_type: EntryType = EntryType.ALL):
        match (entry_type):
            case DecompositionInfo.EntryType.ALL:
                return ma.getdata(self._global_index[dim], subok=False)
            case DecompositionInfo.EntryType.OWNED:
                global_index = self._global_index[dim]
                return ma.getdata(global_index[global_index.mask])
            case DecompositionInfo.EntryType.HALO:
                global_index = self._global_index[dim]
                return ma.getdata(global_index[~global_index.mask])
            case _:
                raise NotImplementedError()


class Exchange:
    def __init__(self, context, domain_decomposition: DecompositionInfo):
        self._context = context
        self._domain_id_gen = DomainDescriptorIdGenerator(context)
        self._decomposition_info = domain_decomposition
        self._log_id = f"rank={self._context.rank()}/{self._context.size()}>>>"
        self._domain_descriptors = {
            CellDim: self._create_domain_descriptor(
                CellDim,
            ),
            VertexDim: self._create_domain_descriptor(
                VertexDim,
            ),
            EdgeDim: self._create_domain_descriptor(EdgeDim),
        }
        print(f"{self._log_id}: domain descriptors initialized")

        self._patterns = {
            CellDim: self._create_pattern(CellDim),
            VertexDim: self._create_pattern(VertexDim),
            EdgeDim: self._create_pattern(EdgeDim),
        }
        print(f"{self._log_id}: patterns initialized ")
        self._comm  = ghex.make_co(context)
        print(f"{self._log_id}: exchange initialized")

    def _domain_descriptor_info(self, descr):
        return f" id={descr.domain_id()}, size={descr.size()}, inner_size={descr.inner_size()} (halo size = {descr.size() - descr.inner_size()})"

    def get_size(self):
        return self._context.size()


    def my_rank(self):
        return self._context.rank()

    def _create_domain_descriptor(self, dim: Dimension):
        all_global = self._decomposition_info.global_index(
            dim, DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, DecompositionInfo.EntryType.HALO
        )

        # print(
        #     f"rank={self._context.rank()}/{self._context.size()}:  all global idx(dim={dim.value}) (shape = {all_global.shape}) {all_global}")
        # print(
        #     f"rank={self._context.rank()}/{self._context.size()}:  local halo idx(dim={dim.value}) (shape = {local_halo.shape}) {local_halo}")

        # first arg is the domain ID which builds up an MPI Tag.
        # if those ids are not different for all domain descriptors the system might deadlock
        # if two parallel exchanges with the same domain id are done
        domain_desc = ghex.domain_descriptor(
            self._domain_id_gen(), all_global.tolist(), local_halo.tolist()
        )
        print(
            f"{self._log_id}: domain descriptor for dim {dim} with properties {self._domain_descriptor_info(domain_desc)}"
        )
        return domain_desc

    def _create_pattern(self, horizontal_dim: Dimension):
        assert horizontal_dim.kind == DimensionKind.HORIZONTAL

        global_halo_idx = self._decomposition_info.global_index(horizontal_dim,
                                                      DecompositionInfo.EntryType.HALO)
        #print(f"rank={self._context.rank()}/{self._context.size()}:  global halo idx(dim={horizontal_dim.value}) (shape = {global_halo_idx.shape}) {global_halo_idx}")
        halo_generator = ghex.halo_generator_with_gids(
            global_halo_idx
        )
        print(
            f"{self._log_id}: halo generator for dim={horizontal_dim} created"
        )
        pattern = ghex.make_pattern(
            self._context, halo_generator, [self._domain_descriptors[horizontal_dim]]
        )
        print(
            f"{self._log_id}: pattern for dim={horizontal_dim} and {self._domain_descriptors[horizontal_dim]} created"
        )
        return pattern

    def prepare_field(self, dim:Dimension, field):
        assert dim in [CellDim, EdgeDim, VertexDim]
        pattern = self._patterns[dim]
        assert pattern is not None
        domain_descriptor = self._domain_descriptors[dim]
        assert domain_descriptor is not None
        descriptor = ghex.field_descriptor(domain_descriptor, np.asarray(field))
        return field, pattern(descriptor)





    def exchange(self, dim: Dimension, *fields):
        assert dim in [CellDim, EdgeDim, VertexDim]
        pattern = self._patterns[dim]
        assert pattern is not None
        fields = [np.asarray(f) for f in fields]
        for f in fields:
            print(
                f"{self._log_id}: communicating field of dim = {dim} : shape = {f.shape}"
            )
        domain_descriptor = self._domain_descriptors[dim]
        print(f"{self._log_id}:  applying pattern to field_descriptor of field f={f.shape} with domain descriptor {self._domain_descriptor_info(domain_descriptor)} ")

        patterns_of_field = [
            pattern(ghex.field_descriptor(domain_descriptor, f))
            for f in fields
        ]

        return self._comm.exchange(patterns_of_field)
