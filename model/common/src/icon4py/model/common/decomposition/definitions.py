# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dataclasses
import enum
import functools
import logging
from typing import Any, Protocol

import gt4py.next as gtx

from icon4py.model.common.settings import xp
from icon4py.model.common.utils import builder


log = logging.getLogger(__name__)


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int


@dataclasses.dataclass(frozen=True, init=False)
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

@dataclasses.dataclass(frozen = True)
class MaskedArray:
    data: xp.ndarray
    mask: xp.ndarray
    
    def __post_init__(self, ):
        assert self.mask.shape == self.data.shape, "mask and value must have the same shape"
        assert self.mask.dtype == bool, "maks should be a boolean array"
    
    def get_masked(self):
        return self.data[self.mask]
    
    def get_unmasked(self):
        return self.data[~self.mask]

class DecompositionInfo:
    def __init__(self, klevels: int):
        self._global_index:dict[gtx.Dimension, MaskedArray] = {}
        self._klevels = klevels
        
    class EntryType(enum.Enum):
        ALL = 0
        OWNED = 1
        HALO = 2

    @builder.builder
    def with_dimension(self, dim: gtx.Dimension, global_index: xp.ndarray, owner_mask: xp.ndarray):
        masked_global_index = MaskedArray(global_index, mask=owner_mask)
        self._global_index[dim] = masked_global_index

    def __init__(self, klevels: int):
        self._global_index = {}
        self._klevels = klevels

    @property
    def klevels(self):
        return self._klevels

    def local_index(self, dim: gtx.Dimension, entry_type: EntryType = EntryType.ALL):
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
        data = self._global_index[dim].data
        assert data.ndim == 1
        return xp.arange(data.shape[0])

    def owner_mask(self, dim: gtx.Dimension) -> xp.ndarray:
        return self._global_index[dim].mask

    def global_index(self, dim: gtx.Dimension, entry_type: EntryType = EntryType.ALL):
        match entry_type:
            case DecompositionInfo.EntryType.ALL:
                return self._global_index[dim].data
            case DecompositionInfo.EntryType.OWNED:
                return self._global_index[dim].get_masked()
            case DecompositionInfo.EntryType.HALO:
                return self._global_index[dim].get_unmasked()
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

    def exchange_and_wait(self, dim: gtx.Dimension, *fields: tuple):
        ...

    def get_size(self):
        ...

    def my_rank(self):
        ...

    def wait(self):
        pass

    def is_ready(self) -> bool:
        return True


@dataclasses.dataclass
class SingleNodeExchange:
    def exchange(self, dim: gtx.Dimension, *fields: tuple) -> ExchangeResult:
        return SingleNodeResult()

    def exchange_and_wait(self, dim: gtx.Dimension, *fields: tuple):
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
