# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Sequence
from enum import Enum
from types import ModuleType
from typing import Any, Literal, Protocol, TypeAlias, overload, runtime_checkable

import dace  # type: ignore[import-untyped]
import gt4py.next as gtx
import numpy as np

from icon4py.model.common import utils
from icon4py.model.common.orchestration.halo_exchange import DummyNestedSDFG
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)

# TODO(all): Currently this is the only place that uses/needs stream. Move them to an
#   appropriate location once the need arises.


class CupyLikeStream(Protocol):
    """The type follows the CuPy convention of a stream.

    This means they have an attribute `ptr` that returns the address of the
    underlying GPU stream.
    See: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html#cupy-cuda-stream

    Todo:
        Drop once we fully translated to CuPy 13.
    """

    @property
    def ptr(self) -> int: ...


class CudaStreamProtocolLike(Protocol):
    """The type follows the CUDA stream protocol.

    This means it provides a method called `__cuda_stream__()` returning a pair of
    integers. The first is the protocol version and the second value is the
    address of the stream.
    See: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
    """

    def __cuda_stream__(self) -> tuple[int, int]: ...


StreamLike: TypeAlias = CupyLikeStream | CudaStreamProtocolLike


@dataclasses.dataclass(frozen=True)
class Stream:
    """Stream object used in ICON4Py

    Args:
        ptr: The address of the underlying stream.
    """

    ptr: int

    def __cuda_stream__(self) -> tuple[int, int]:
        return 1, self.ptr


DEFAULT_STREAM = Stream(0)
"""Default stream of the device.

Its availability is not tied to a particular device, it is thus also present in a
purely CPU setting, where it is save to use and usually represents fully blocking
semantic.
"""


class Block:
    pass


BLOCK = Block()
"""
Constant used by `ExchangeResult.finish()` to indicate that blocking semantic should
be used, i.e. wait until the exchange has fully finished not until it has merely been
scheduled on the device, should be used.
"""


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int


@dataclasses.dataclass(frozen=True, init=False)
class SingleNodeProcessProperties(ProcessProperties):
    comm: None
    rank: int
    comm_name: str
    comm_size: int

    def __init__(self) -> None:
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

    def __call__(self) -> int:
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
        ALL = 0
        OWNED = 1
        HALO = 2

    @utils.chainable
    def with_dimension(
        self, dim: gtx.Dimension, global_index: data_alloc.NDArray, owner_mask: data_alloc.NDArray
    ) -> None:
        self._global_index[dim] = global_index
        self._owner_mask[dim] = owner_mask

    def __init__(
        self,
        num_cells: int | None = None,
        num_edges: int | None = None,
        num_vertices: int | None = None,
    ):
        self._global_index: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._owner_mask: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._num_vertices = num_vertices
        self._num_cells = num_cells
        self._num_edges = num_edges

    @property
    def num_cells(self) -> int | None:
        return self._num_cells

    @property
    def num_edges(self) -> int | None:
        return self._num_edges

    @property
    def num_vertices(self) -> int | None:
        return self._num_vertices

    def local_index(
        self, dim: gtx.Dimension, entry_type: EntryType = EntryType.ALL
    ) -> data_alloc.NDArray:
        match entry_type:
            case DecompositionInfo.EntryType.ALL:
                return self._to_local_index(dim)
            case DecompositionInfo.EntryType.HALO:
                index = self._to_local_index(dim)
                mask = self._owner_mask[dim]
                return index[~mask]
            case DecompositionInfo.EntryType.OWNED:
                index = self._to_local_index(dim)
                mask = self._owner_mask[dim]
                return index[mask]

    def _to_local_index(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        data = self._global_index[dim]
        assert data.ndim == 1
        if isinstance(data, np.ndarray):
            import numpy as xp
        else:
            import cupy as xp  # type: ignore[import-not-found, no-redef]

            xp.arange(data.shape[0])
        return xp.arange(data.shape[0])

    def owner_mask(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        return self._owner_mask[dim]

    def global_index(
        self, dim: gtx.Dimension, entry_type: EntryType = EntryType.ALL
    ) -> data_alloc.NDArray:
        match entry_type:
            case DecompositionInfo.EntryType.ALL:
                return self._global_index[dim]
            case DecompositionInfo.EntryType.OWNED:
                return self._global_index[dim][self._owner_mask[dim]]
            case DecompositionInfo.EntryType.HALO:
                return self._global_index[dim][~self._owner_mask[dim]]
            case _:
                raise NotImplementedError()


class ExchangeResult(Protocol):
    def finish(
        self,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None:
        """Wait on the halo exchange.

        Finishes the halo exchange represented by this `ExchangeResult`, see
        `ExchangeRuntime.start()` for more.
        When the function returns the exchange has not necessarily completed yet,
        but has been scheduled on the device using stream `stream`. This means
        that all further work submitted to `stream` will wait until the exchange
        has completed. By default `DEFAULT_STREAM` is used.

        In case `stream` is `BLOCK` the function will only return after the exchange
        has been completed.

        For fields located on the host the only supported behaviour is `stream=BLOCK`.
        """
        ...

    def is_ready(self) -> bool:
        """Check if communication has been finished.

        For an exchange involving device memory, calling this function is equivalent
        to the call `self.finish(stream=BLOCK)`.
        """
        ...


@runtime_checkable
class ExchangeRuntime(Protocol):
    @overload
    def start(
        self,
        dim: gtx.Dimension,
        *buffers: data_alloc.NDArray,
        stream: StreamLike,
    ) -> ExchangeResult: ...

    @overload
    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult:
        """Initiate a halo exchanges.

        The exchange will synchronize with `stream`, i.e. not start before all work
        previously submitted to `stream` has finished. `stream` defaults to
        `DEFAULT_STREAM`. To complete the exchange `finish()` must be called on the
        returned `ExchangeResult`. There is also the `exchange()` function which
        combines these two steps into one.

        Once the function returns it is safe to reuse the memory of `fields` (is this
        still true for NCCL).

        Note:
            For fields on the host the exchange will begin immediately, regardless
            which stream has been passed.
        """
        ...

    @overload
    def exchange(
        self,
        dim: gtx.Dimension,
        *buffers: data_alloc.NDArray,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None: ...

    @overload
    def exchange(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None: ...

    def exchange(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None:
        """Perform a full hallo exchange.

        The exchange will synchronize with `stream`, i.e. not start before tasks
        previously submitted to it are done. The function returns before the exchange
        has been completed, but it will synchronize with `stream`, i.e. work submitted
        to `stream` will not start before the exchange has finished.

        It is possible to split the exchange into a send part, for which
        `exchange_request = self.start()` can be used and a receive part for which
        `exchange_request.finish()` can be used.

        In case `stream` is `BLOCK` then the function will only return once the exchange
        has been completed entirely. In this case the send part of will be performed
        as `DEFAULT_STREAM` was passed.

        Note:
            The protocol supplies a default implementation.
        """
        ex_req = self.start(
            dim,
            *fields,
            stream=(DEFAULT_STREAM if stream is BLOCK else stream),  # type: ignore[arg-type]
        )
        ex_req.finish(stream)

    @overload
    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: Literal[True],
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None: ...

    @overload
    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: Literal[False],
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult: ...

    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: bool = True,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None | ExchangeResult:
        """Performs either a full exchange or a partial exchange.

        If `full_exchange` is `True` then this function is equivalent to
        `self.exchange()` otherwise it behaves as `self.start()` and the exchange
        result object is returned.

        Note:
            - This function is deprecated and should no longer be used.
            - The order of `*fields` and `dim` is reversed compared to `exchange()`.
            - The protocol supplies a default implementation.
        """
        ex_req = self.start(
            dim,
            *fields,
            stream=(DEFAULT_STREAM if stream is BLOCK else stream),  # type: ignore[arg-type]
        )
        if not full_exchange:
            return ex_req
        ex_req.finish(stream=stream)
        return None

    def get_size(self) -> int: ...

    def my_rank(self) -> int: ...

    def __str__(self) -> str:
        return f"{self.__class__} (rank = {self.my_rank()} / {self.get_size()})"


@dataclasses.dataclass
class SingleNodeExchange(ExchangeRuntime):
    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult:
        return SingleNodeResult()

    def my_rank(self) -> int:
        return 0

    def get_size(self) -> int:
        return 1

    # Implementation of DaCe SDFGConvertible interface
    # For more see [dace repo]/dace/frontend/python/common.py#[class SDFGConvertible]
    # NOTE: Stream are not supported here.
    def dace__sdfg__(
        self, *args: Any, dim: gtx.Dimension, full_exchange: bool = True
    ) -> dace.sdfg.sdfg.SDFG:
        sdfg = DummyNestedSDFG().__sdfg__()
        sdfg.name = "_halo_exchange_"
        return sdfg

    def dace__sdfg_closure__(self, reevaluate: dict[str, str] | None = None) -> dict[str, Any]:
        return DummyNestedSDFG().__sdfg_closure__()

    def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return DummyNestedSDFG().__sdfg_signature__()

    __sdfg__ = dace__sdfg__
    __sdfg_closure__ = dace__sdfg_closure__
    __sdfg_signature__ = dace__sdfg_signature__


class HaloExchangeWaitRuntime(Protocol):
    """Protocol for halo exchange wait."""

    def __call__(
        self,
        communication_handle: ExchangeResult,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> None:
        """Calls `wait()` on the provided communication handle.

        Args:
            stream: The stream forwarded to the `wait()` call, defaults to `DEFAULT_STREAM`.

        Note:
            - The protocol provides a default implementation.
        """
        communication_handle.finish(stream=stream)

    def __sdfg__(self, *args: Any, **kwargs: dict[str, Any]) -> dace.sdfg.sdfg.SDFG:
        """DaCe related: SDFGConvertible interface."""
        ...

    def __sdfg_closure__(self, reevaluate: dict[str, str] | None = None) -> dict[str, Any]:
        """DaCe related: SDFGConvertible interface."""
        ...

    def __sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        """DaCe related: SDFGConvertible interface."""
        ...


@dataclasses.dataclass
class HaloExchangeWait(HaloExchangeWaitRuntime):
    exchange_object: SingleNodeExchange  # maintain the same interface with the MPI counterpart

    # Implementation of DaCe SDFGConvertible interface
    def dace__sdfg__(
        self,
        *args: Any,
        dim: gtx.Dimension,
        full_exchange: bool = True,
        stream: StreamLike | Block = DEFAULT_STREAM,
    ) -> dace.sdfg.sdfg.SDFG:
        sdfg = DummyNestedSDFG().__sdfg__()
        sdfg.name = "_halo_exchange_wait_"
        return sdfg

    def dace__sdfg_closure__(self, reevaluate: dict[str, str] | None = None) -> dict[str, Any]:
        return DummyNestedSDFG().__sdfg_closure__()

    def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return DummyNestedSDFG().__sdfg_signature__()

    __sdfg__ = dace__sdfg__  # type: ignore[assignment]
    __sdfg_closure__ = dace__sdfg_closure__
    __sdfg_signature__ = dace__sdfg_signature__


@functools.singledispatch
def create_halo_exchange_wait(runtime: ExchangeRuntime) -> HaloExchangeWaitRuntime:
    raise TypeError(f"Unknown ExchangeRuntime type ({type(runtime)})")


@create_halo_exchange_wait.register(SingleNodeExchange)
def create_single_node_halo_exchange_wait(runtime: SingleNodeExchange) -> HaloExchangeWait:
    return HaloExchangeWait(runtime)


class SingleNodeResult(ExchangeResult):
    def finish(self, stream: StreamLike | Block = DEFAULT_STREAM) -> None:
        pass

    def is_ready(self) -> bool:
        return True


class RunType:
    """Base type for marker types used to initialize the parallel or single node properties."""

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


class Reductions(Protocol):
    def min(
        self, buffer: data_alloc.NDArray, array_ns: ModuleType = np
    ) -> state_utils.ScalarType: ...

    def max(
        self, buffer: data_alloc.NDArray, array_ns: ModuleType = np
    ) -> state_utils.ScalarType: ...

    def sum(
        self, buffer: data_alloc.NDArray, array_ns: ModuleType = np
    ) -> state_utils.ScalarType: ...

    def mean(
        self, buffer: data_alloc.NDArray, array_ns: ModuleType = np
    ) -> state_utils.ScalarType: ...


class SingleNodeReductions(Reductions):
    def min(self, buffer: data_alloc.NDArray, array_ns: ModuleType = np) -> state_utils.ScalarType:
        return array_ns.min(buffer).item()

    def max(self, buffer: data_alloc.NDArray, array_ns: ModuleType = np) -> state_utils.ScalarType:
        return array_ns.max(buffer).item()

    def sum(self, buffer: data_alloc.NDArray, array_ns: ModuleType = np) -> state_utils.ScalarType:
        return array_ns.sum(buffer).item()

    def mean(self, buffer: data_alloc.NDArray, array_ns: ModuleType = np) -> state_utils.ScalarType:
        return array_ns.sum(buffer).item() / buffer.size


@overload
def get_runtype(with_mpi: Literal[True]) -> MultiNodeRun: ...


@overload
def get_runtype(with_mpi: Literal[False]) -> SingleNodeRun: ...


def get_runtype(with_mpi: bool = False) -> RunType:
    if with_mpi:
        return MultiNodeRun()
    else:
        return SingleNodeRun()


@functools.singledispatch
def get_processor_properties(runtime: RunType, comm_id: int | None = None) -> ProcessProperties:
    raise TypeError(f"Cannot define ProcessProperties for ({type(runtime)})")


@get_processor_properties.register(SingleNodeRun)
def get_single_node_properties(s: SingleNodeRun, comm_id: int | None = None) -> ProcessProperties:
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


@functools.singledispatch
def create_reduction(props: ProcessProperties) -> Reductions:
    """
    Create a Global Reduction depending on the runtime size.

    Depending on the number of processor a SingleNode version is returned or a GHEX context created and a Multinode returned.
    """
    raise NotImplementedError(f"Unknown ProcessorProperties type ({type(props)})")


@create_reduction.register(SingleNodeProcessProperties)
def create_single_reduction_exchange(props: SingleNodeProcessProperties) -> Reductions:
    return SingleNodeReductions()


single_node_default = SingleNodeExchange()
single_node_reductions = SingleNodeReductions()
