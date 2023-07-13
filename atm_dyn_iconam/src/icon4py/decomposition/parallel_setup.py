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

import ghex.unstructured as ghex
import mpi4py
import numpy as np
import numpy.ma as ma
from gt4py.next.common import Dimension
from mpi4py.MPI import Comm

from icon4py.common.dimension import CellDim, EdgeDim, KDim, KHalfDim, VertexDim
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
        print(
            f"rank={self._context.rank()}/{self._context.size()} :domain descriptors initialized"
        )
        self._field_size = {
            CellDim: self._decomposition_info.global_index(
                CellDim, DecompositionInfo.EntryType.ALL
            ).shape[0],
            EdgeDim: self._decomposition_info.global_index(
                EdgeDim, DecompositionInfo.EntryType.ALL
            ).shape[0],
            VertexDim: self._decomposition_info.global_index(
                VertexDim, DecompositionInfo.EntryType.ALL
            ).shape[0],
            KDim: domain_decomposition.klevels,
            KHalfDim: domain_decomposition.klevels + 1,
        }
        print(
            f"rank={self._context.rank()}/{self._context.size()} : field sizes = {self._field_size}"
        )

        self._patterns = {
            CellDim: self._create_pattern(CellDim),
            VertexDim: self._create_pattern(VertexDim),
            EdgeDim: self._create_pattern(EdgeDim),
        }
        self._comms = {k: ghex.make_co(context) for k, v in self._patterns.items()}
        print(
            f"rank={self._context.rank()}/{self._context.size()} : patterns and communicators initialized "
        )
        print(
            f"rank={self._context.rank()}/{self._context.size()} : exchange initialized"
        )

    def _domain_descriptor_info(self, descr):
        return f" id={descr.domain_id()}, size={descr.size()}, inner_size={descr.inner_size()}"

    def get_size(self):
        return self._context.size()

    # TODO [magdalena] is the tolist() necessary?
    def _create_domain_descriptor(self, dim: Dimension):
        all_global = self._decomposition_info.global_index(
            dim, DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, DecompositionInfo.EntryType.HALO
        )
        # TOOD [magdalena]  first arg is the domain ID which builds up an MPI Tag, doesn't need to be the MPI rank.
        # on the contrary it is safer if those are different for all domain descriptors (otherwise the system deadlocks if 2 parallel exchanges are done
        # with the same domain-id
        domain_desc = ghex.domain_descriptor(
            self._context.rank(), all_global.tolist(), local_halo.tolist()
        )
        print(
            f"rank={self._context.rank()}/{self._context.size()}: domain descriptor for dim {dim} with properties {self._domain_descriptor_info(domain_desc)}"
        )

        return domain_desc

    def _create_pattern(self, horizontal_dim: Dimension):
        halo_generator = ghex.halo_generator_with_gids(
            self._decomposition_info.global_index(
                horizontal_dim, DecompositionInfo.EntryType.HALO
            )
        )
        print(
            f"rank={self._context.rank()}/{self._context.size()}: halo generator for dim={horizontal_dim} created"
        )
        pattern = ghex.make_pattern(
            self._context, halo_generator, [self._domain_descriptors[horizontal_dim]]
        )
        print(
            f"rank={self._context.rank()}/{self._context.size()}: pattern for dim={horizontal_dim} and {self._domain_descriptors[horizontal_dim]} created"
        )
        return pattern

    def exchange(self, dim: Dimension, *fields):
        assert dim in [CellDim, EdgeDim, VertexDim]
        horizontal_size = self._field_size[dim]
        pattern = self._patterns[dim]
        assert pattern is not None
        fields = [np.asarray(f)[:horizontal_size, :] for f in fields]
        shapes = list(map(lambda f: f.shape, fields))
        print(
            f"rank = {self._context.rank()}/{self._context.size()}: communicating fields of dim = {dim} : shapes = {shapes}"
        )

        patterns_of_field = [
            pattern(ghex.field_descriptor(self._domain_descriptors[dim], f))
            for f in fields
        ]
        return self._comms[dim].exchange(patterns_of_field)
