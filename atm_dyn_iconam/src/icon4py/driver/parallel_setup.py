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

from icon4py.common.dimension import CellDim, VertexDim, EdgeDim
from icon4py.decomposition.decomposed import ProcessProperties
from icon4py.diffusion.utils import builder


mpi4py.rc.initialize = False

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
        MPI.Init()


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
            CellDim: self._create_domain_descriptor(CellDim),
            VertexDim: self._create_domain_descriptor(VertexDim),
            EdgeDim: self._create_domain_descriptor(EdgeDim)
        }
        log.info(f"exchange patterns initialized {self._domain_descriptors}")

        self._patterns = {
            CellDim: self._create_pattern(CellDim),
            VertexDim: self._create_pattern(VertexDim),
            EdgeDim: self._create_pattern(EdgeDim)
        }
        log.info(f"exchange patterns initialized {self._patterns}")

    def get_size(self):
        return self._context.size()
    def _create_domain_descriptor(self, dim: Dimension):
        all_global = self._decomposition_info.global_index(
            dim, DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, DecompositionInfo.EntryType.HALO
        )
        domain_descr = ghex.domain_descriptor(
            self._context.rank(),
            all_global.tolist(),
            local_halo.tolist(),
            self._decomposition_info.klevels,
        )
        return domain_descr

    def _create_pattern(self, dim):
        halo_generator = ghex.halo_generator_with_gids(
            self._decomposition_info.global_index(dim, DecompositionInfo.EntryType.HALO)
        )
        pattern = ghex.make_pattern(
            self._context, halo_generator, [self._domain_descriptors[dim]]
        )
        return pattern

    def exchange(self, dim: Dimension, *fields):
        assert dim in [CellDim, EdgeDim, VertexDim]
        log.info(f"exchanging fields for dim={dim}:")
        for f in fields:
            log.info(f"{f.shape}")
        domain_descriptor = self._domain_descriptors[dim]
        pattern = self._patterns[dim]
        communicator = ghex.make_co(self._context, pattern)
        pattern_of_fields = [
            pattern(ghex.field_descriptor(domain_descriptor, np.asarray(f))) for f in fields
        ]
        communicator.exchange(pattern_of_fields)
