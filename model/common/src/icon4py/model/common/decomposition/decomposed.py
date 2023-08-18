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
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import ghex
import ghex.unstructured as unstructured
import numpy as np
import numpy.ma as ma
from gt4py.next import Dimension

from icon4py.model.common.dimension import CellDim, DimensionKind, EdgeDim, VertexDim
from icon4py.model.common.decomposition.parallel_setup import ProcessProperties
from icon4py.diffusion.diffusion_utils import builder


log = logging.getLogger(__name__)


class DomainDescriptorIdGenerator:
    _counter = 0
    _roundtrips = 0

    def __init__(self, parallel_props: ProcessProperties):
        self._comm_size = parallel_props.comm_size
        self._roundtrips = parallel_props.rank
        self._base = self._roundtrips * self._comm_size

    def __call__(self):
        next_id = self._base + self._counter
        if self._counter + 1 >= self._comm_size:
            self._roundtrips = self._roundtrips + self._comm_size
            self._base = self._roundtrips * self._comm_size
            self._counter = 0
        else:
            self._counter = self._counter + 1
        return next_id


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


class ExchangeResult(Protocol):
    def wait(self):
        ...

    def is_ready(self) -> bool:
        ...


class ExchangeRuntime(Protocol):
    def exchange(self, dim: Dimension, *fields: tuple) -> ExchangeResult:
        ...

    def get_size(self):
        ...

    def my_rank(self):
        ...

    def wait(self):
        pass

    def is_ready(self) -> bool:
        return True


def create_exchange(
    props: ProcessProperties, decomp_info: DecompositionInfo
) -> ExchangeRuntime:
    """
    Create an Exchange depending on the runtime size.

    Depending on the number of processor a SingleNode version is returned or a GHEX context created and a Multinode returned.
    """
    if props.comm_size > 1:
        return GHexMultiNode(props, decomp_info)
    else:
        return SingleNode()


@dataclass
class SingleNode:
    def exchange(self, dim: Dimension, *fields: tuple) -> ExchangeResult:
        return SingleNodeResult()

    def my_rank(self):
        return 0

    def get_size(self):
        return 1


class SingleNodeResult:
    def wait(self):
        pass

    def is_ready(self) -> bool:
        return True


class GHexMultiNode:
    def __init__(
        self, props: ProcessProperties, domain_decomposition: DecompositionInfo
    ):
        self._context = ghex.context(ghex.mpi_comm(props.comm), True)
        self._domain_id_gen = DomainDescriptorIdGenerator(props)
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
        log.info(
            f"domain descriptors for dimensions {self._domain_descriptors.keys()} initialized"
        )

        self._patterns = {
            CellDim: self._create_pattern(CellDim),
            VertexDim: self._create_pattern(VertexDim),
            EdgeDim: self._create_pattern(EdgeDim),
        }
        log.info(f"patterns for dimensions {self._patterns.keys()} initialized ")
        self._comm = unstructured.make_co(self._context)
        log.info("communication object initialized")

    def _domain_descriptor_info(self, descr):
        return f" domain_descriptor=[id='{descr.domain_id()}', size='{descr.size()}', inner_size='{descr.inner_size()}' (halo size='{descr.size() - descr.inner_size()}')"

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
        # first arg is the domain ID which builds up an MPI Tag.
        # if those ids are not different for all domain descriptors the system might deadlock
        # if two parallel exchanges with the same domain id are done
        domain_desc = unstructured.domain_descriptor(
            self._domain_id_gen(), all_global.tolist(), local_halo.tolist()
        )
        log.debug(
            f"domain descriptor for dim='{dim.value}' with properties {self._domain_descriptor_info(domain_desc)} created"
        )
        return domain_desc

    def _create_pattern(self, horizontal_dim: Dimension):
        assert horizontal_dim.kind == DimensionKind.HORIZONTAL

        global_halo_idx = self._decomposition_info.global_index(
            horizontal_dim, DecompositionInfo.EntryType.HALO
        )
        halo_generator = unstructured.halo_generator_with_gids(global_halo_idx)
        log.debug(f"halo generator for dim='{horizontal_dim.value}' created")
        pattern = unstructured.make_pattern(
            self._context, halo_generator, [self._domain_descriptors[horizontal_dim]]
        )
        log.debug(
            f"pattern for dim='{horizontal_dim.value}' and {self._domain_descriptor_info(self._domain_descriptors[horizontal_dim])} created"
        )
        return pattern

    def exchange(self, dim: Dimension, *fields: tuple):
        assert dim in [CellDim, EdgeDim, VertexDim]
        pattern = self._patterns[dim]
        assert pattern is not None, f"pattern for {dim.value} not found"
        domain_descriptor = self._domain_descriptors[dim]
        assert (
            domain_descriptor is not None
        ), f"domain descriptor for {dim.value} not found"
        applied_patterns = [
            pattern(unstructured.field_descriptor(domain_descriptor, np.asarray(f)))
            for f in fields
        ]
        handle = self._comm.exchange(applied_patterns)
        log.info(
            f"exchange for {len(fields)} fields of dimension ='{dim.value}' initiated."
        )
        return MultiNodeResult(handle, applied_patterns)


@dataclass
class MultiNodeResult:
    handle: ...
    pattern_refs: ...

    def wait(self):
        self.handle.wait()
        del self.pattern_refs

    def is_ready(self) -> bool:
        return self.handle.is_ready()
