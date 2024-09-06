# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import functools
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol

import numpy as np
import numpy.ma as ma
from gt4py.next import Dimension

from icon4py.model.common.settings import xp
from icon4py.model.common.utils import builder


log = logging.getLogger(__name__)


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int

    def single_node(self) -> bool:
        return self.comm_size == 1


@dataclass(frozen=True, init=False)
class SingleNodeProcessProperties(ProcessProperties):
    def __init__(self):
        object.__setattr__(self, "comm", None)
        object.__setattr__(self, "rank", 0)
        object.__setattr__(self, "comm_name", "")
        object.__setattr__(self, "comm_size", 1)


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
    class EntryType(IntEnum):
        ALL = 0
        OWNED = 1
        HALO = 2

    @builder.builder
    def with_dimension(
        self,
        dim: Dimension,
        global_index: np.ndarray,
        owner_mask: np.ndarray,
        halo_levels: np.ndarray,
    ):
        masked_global_index = ma.array(global_index, mask=owner_mask)
        self._global_index[dim] = masked_global_index
        self._halo_levels[dim] = halo_levels

    def __init__(self, klevels: int):
        self._global_index = {}
        self._halo_levels = {}
        self._klevels = klevels

    @property
    def klevels(self):
        return self._klevels

    def local_index(self, dim: Dimension, entry_type: EntryType = EntryType.ALL):
        match entry_type:
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
        match entry_type:
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

    def halo_levels(self, dim: Dimension):
        return self._halo_levels[dim]

    def halo_level_mask(self, dim: Dimension, level: DecompositionFlag):
        return xp.where(self._halo_levels[dim] == level, True, False)


class ExchangeResult(Protocol):
    def wait(self):
        ...

    def is_ready(self) -> bool:
        ...


class ExchangeRuntime(Protocol):
    def exchange(self, dim: Dimension, *fields: tuple) -> ExchangeResult:
        ...

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        ...

    def get_size(self):
        ...

    def my_rank(self):
        ...

    def wait(self):
        pass

    def is_ready(self) -> bool:
        return True


@dataclass
class SingleNodeExchange:
    def exchange(self, dim: Dimension, *fields: tuple) -> ExchangeResult:
        return SingleNodeResult()

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        return

    def my_rank(self):
        return 0

    def get_size(self):
        return 1


class SingleNodeResult:
    def wait(self):
        pass

    def is_ready(self) -> bool:
        return True


class RunType:
    """Base type for marker types used to initialize the parallel or single node properites."""

    pass


class MultiNodeRun(RunType):
    """
    Mark multinode run.

    Dummy marker type used to initialize a multinode run and initialize
    construction multinode ProcessProperties.
    """

    pass


class SingleNodeRun(RunType):
    """
    Mark single node run.

    Dummy marker type used to initialize a single node run and initialize
    construction SingleNodeProcessProperties.
    """

    pass


def get_runtype(with_mpi: bool = False) -> RunType:
    if with_mpi:
        return MultiNodeRun()
    else:
        return SingleNodeRun()


@functools.singledispatch
def get_processor_properties(runtime) -> ProcessProperties:
    raise TypeError(f"Cannot define ProcessProperties for ({type(runtime)})")


@get_processor_properties.register(SingleNodeRun)
def get_single_node_properties(s: SingleNodeRun) -> ProcessProperties:
    return SingleNodeProcessProperties()


@functools.singledispatch
def create_exchange(props: ProcessProperties, decomp_info: DecompositionInfo) -> ExchangeRuntime:
    """
    Create an Exchange depending on the runtime size.

    Depending on the number of processor a SingleNode version is returned or a GHEX context created and a Multinode returned.
    """
    raise NotImplementedError(f"Unknown ProcessorProperties type ({type(props)})")


@create_exchange.register(SingleNodeProcessProperties)
def create_single_node_exchange(
    props: SingleNodeProcessProperties, decomp_info: DecompositionInfo
) -> ExchangeRuntime:
    return SingleNodeExchange()


class DecompositionFlag(enum.IntEnum):
    UNDEFINED = -1
    OWNED = 0
    """used for locally owned cells, vertices, edges"""

    FIRST_HALO_LINE = 1
    """
    used for:
    - cells that share 1 edge with an OWNED cell
    - vertices that are on OWNED cell 
    - edges that are on OWNED cell
    """

    SECOND_HALO_LINE = 2
    """
    used for:
    - cells that share a vertex with an OWNED cell
    - vertices that are on a cell(FIRST_HALO_LINE) but not on an owned cell
    - edges that have _exactly_ one vertex shared with and OWNED Cell
    """

    THIRD_HALO_LINE = 3
    """
    This type does not exist in ICON. It denotes the "closing/far" edges of the SECOND_HALO_LINE cells
    used for:
    - cells (NOT USED)
    - vertices (NOT USED)
    - edges that are only on the cell(SECOND_HALO_LINE)
    """
