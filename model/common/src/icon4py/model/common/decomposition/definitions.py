# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import numpy.ma as ma
from gt4py.next import Dimension

from icon4py.model.common.utils import builder


try:
    import dace

    from icon4py.model.common.orchestration.halo_exchange import DummyNestedSDFG
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


log = logging.getLogger(__name__)


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int


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
    def with_dimension(self, dim: Dimension, global_index: np.ndarray, owner_mask: np.ndarray):
        masked_global_index = ma.array(global_index, mask=owner_mask)
        self._global_index[dim] = masked_global_index

    def __init__(self, klevels: int):
        self._global_index = {}
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

    def __call__(self, *args, **kwargs):
        dim = kwargs.get("dim", None)
        wait = kwargs.get("wait", True)

        res = self.exchange(dim, *args)
        if wait:
            res.wait()
        else:
            return res

    if dace:
        # Implementation of DaCe SDFGConvertible interface
        def dace__sdfg__(self, *args, **kwargs):
            sdfg = DummyNestedSDFG().__sdfg__()
            sdfg.name = "_halo_exchange_"
            return sdfg

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            return DummyNestedSDFG().__sdfg_closure__()

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            return DummyNestedSDFG().__sdfg_signature__()
    else:

        def dace__sdfg__(self, *args, **kwargs):
            raise NotImplementedError(
                "__sdfg__ is only supported when the 'dace' module is available."
            )

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            raise NotImplementedError(
                "__sdfg_closure__ is only supported when the 'dace' module is available."
            )

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            raise NotImplementedError(
                "__sdfg_signature__ is only supported when the 'dace' module is available."
            )

    __sdfg__ = dace__sdfg__
    __sdfg_closure__ = dace__sdfg_closure__
    __sdfg_signature__ = dace__sdfg_signature__


class HaloExchangeWaitRuntime(Protocol):
    def __call__(self, communication_handle: ExchangeResult):
        ...

    def __sdfg__(self, *args, **kwargs):
        ...

    def __sdfg_closure__(self, reevaluate: Optional[dict[str, str]] = None) -> dict[str, Any]:
        ...

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        ...


@dataclass
class HaloExchangeWait:
    exchange_object: SingleNodeExchange  # maintain the same interface with the MPI counterpart

    def __call__(self, communication_handle: SingleNodeResult):
        communication_handle.wait()

    if dace:
        # Implementation of DaCe SDFGConvertible interface
        def dace__sdfg__(self, *args, **kwargs):
            sdfg = DummyNestedSDFG().__sdfg__()
            sdfg.name = "_halo_exchange_wait_"
            return sdfg

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            return DummyNestedSDFG().__sdfg_closure__()

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            return DummyNestedSDFG().__sdfg_signature__()
    else:

        def dace__sdfg__(self, *args, **kwargs):
            raise NotImplementedError(
                "__sdfg__ is only supported when the 'dace' module is available."
            )

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            raise NotImplementedError(
                "__sdfg_closure__ is only supported when the 'dace' module is available."
            )

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            raise NotImplementedError(
                "__sdfg_signature__ is only supported when the 'dace' module is available."
            )

    __sdfg__ = dace__sdfg__
    __sdfg_closure__ = dace__sdfg_closure__
    __sdfg_signature__ = dace__sdfg_signature__


@functools.singledispatch
def create_halo_exchange_wait(runtime: ExchangeRuntime) -> HaloExchangeWaitRuntime:
    raise TypeError(f"Unknown ExchangeRuntime type ({type(runtime)})")


@create_halo_exchange_wait.register(SingleNodeExchange)
def create_single_node_halo_exchange_wait(runtime: SingleNodeExchange) -> HaloExchangeWait:
    return HaloExchangeWait(runtime)


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
