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
from typing import Any, Literal, Protocol, overload, runtime_checkable

import dace  # type: ignore[import-untyped]
import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims, utils
from icon4py.model.common.grid import base, gridfile
from icon4py.model.common.orchestration.halo_exchange import DummyNestedSDFG
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int

    def single_node(self) -> bool:
        return self.comm_size == 1


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
    def __init__(
        self,
    ) -> None:
        self._global_index: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._halo_levels: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._owner_mask: dict[gtx.Dimension, data_alloc.NDArray] = {}

    class EntryType(int, Enum):
        ALL = 0
        OWNED = 1
        HALO = 2

    @utils.chainable
    def set_dimension(
        self,
        dim: gtx.Dimension,
        global_index: data_alloc.NDArray,
        owner_mask: data_alloc.NDArray,
        halo_levels: data_alloc.NDArray,
    ) -> None:
        self._global_index[dim] = global_index
        self._owner_mask[dim] = owner_mask
        self._halo_levels[dim] = halo_levels

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

    def global_to_local(
        self, dim: gtx.Dimension, indices_to_translate: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        global_indices = self.global_index(dim)
        sorter = np.argsort(global_indices)

        mask = np.isin(indices_to_translate, global_indices)
        positions = np.searchsorted(global_indices, indices_to_translate, sorter=sorter)
        local_neighbors = np.full_like(indices_to_translate, gridfile.GridFile.INVALID_INDEX)
        local_neighbors[mask] = sorter[positions[mask]]
        return local_neighbors

        # TODO (halungge): use for test reference? in test_definitions.py

    def global_to_local_ref(
        self, dim: gtx.Dimension, indices_to_translate: data_alloc.NDArray
    ) -> data_alloc.NDArray:
        global_indices = self.global_index(dim)
        local_neighbors = np.full_like(indices_to_translate, gridfile.GridFile.INVALID_INDEX)
        for i in range(indices_to_translate.shape[0]):
            for j in range(indices_to_translate.shape[1]):
                if np.isin(indices_to_translate[i, j], global_indices):
                    pos = np.where(indices_to_translate[i, j] == global_indices)[0]
                    local_neighbors[i, j] = pos
        return local_neighbors

    def owner_mask(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        return self._owner_mask[dim]

    def global_index(
        self,
        dim: gtx.Dimension,
        entry_type: DecompositionInfo.EntryType = EntryType.ALL,
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

    def get_horizontal_size(self) -> base.HorizontalGridSize:
        return base.HorizontalGridSize(
            num_cells=self.global_index(dims.CellDim, self.EntryType.ALL).shape[0],
            num_edges=self.global_index(dims.EdgeDim, self.EntryType.ALL).shape[0],
            num_vertices=self.global_index(dims.VertexDim, self.EntryType.ALL).shape[0],
        )

    def get_halo_size(self, dim: gtx.Dimension, flag: DecompositionFlag) -> int:
        return np.count_nonzero(self.halo_level_mask(dim, flag))

    def halo_levels(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        return self._halo_levels[dim]

    def halo_level_mask(self, dim: gtx.Dimension, level: DecompositionFlag) -> data_alloc.NDArray:
        return np.where(self._halo_levels[dim] == level, True, False)


class ExchangeResult(Protocol):
    def wait(self) -> None: ...

    def is_ready(self) -> bool: ...


@runtime_checkable
class ExchangeRuntime(Protocol):
    @overload
    def exchange(self, dim: gtx.Dimension, *fields: gtx.Field) -> ExchangeResult: ...

    @overload
    def exchange(self, dim: gtx.Dimension, *buffers: data_alloc.NDArray) -> ExchangeResult: ...

    @overload
    def exchange_and_wait(self, dim: gtx.Dimension, *fields: gtx.Field) -> None: ...

    @overload
    def exchange_and_wait(self, dim: gtx.Dimension, *buffers: data_alloc.NDArray) -> None: ...

    def get_size(self) -> int: ...

    def my_rank(self) -> int: ...

    def __str__(self) -> str:
        return f"{self.__class__} (rank = {self.my_rank()} / {self.get_size()})"


@dataclasses.dataclass
class SingleNodeExchange:
    def exchange(
        self, dim: gtx.Dimension, *fields: gtx.Field | data_alloc.NDArray
    ) -> ExchangeResult:
        return SingleNodeResult()

    def exchange_and_wait(
        self, dim: gtx.Dimension, *fields: gtx.Field | data_alloc.NDArray
    ) -> None:
        return None

    def my_rank(self) -> int:
        return 0

    def get_size(self) -> int:
        return 1

    def __call__(self, *args: Any, dim: gtx.Dimension, wait: bool = True) -> ExchangeResult | None:  # type: ignore[return] # return statment in else condition
        """Perform a halo exchange operation.

        Args:
            args: The fields to be exchanged.

        Keyword Args:
            dim: The dimension along which the exchange is performed.
            wait: If True, the operation will block until the exchange is completed (default: True).
        """

        res = self.exchange(dim, *args)
        if wait:
            res.wait()
        else:
            return res

    # Implementation of DaCe SDFGConvertible interface
    # For more see [dace repo]/dace/frontend/python/common.py#[class SDFGConvertible]
    def dace__sdfg__(
        self, *args: Any, dim: gtx.Dimension, wait: bool = True
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

    def __call__(self, communication_handle: ExchangeResult) -> None:
        """Wait on the communication handle."""
        ...

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
class HaloExchangeWait:
    exchange_object: SingleNodeExchange  # maintain the same interface with the MPI counterpart

    def __call__(self, communication_handle: SingleNodeResult) -> None:
        communication_handle.wait()

    # Implementation of DaCe SDFGConvertible interface
    def dace__sdfg__(
        self, *args: Any, dim: gtx.Dimension, wait: bool = True
    ) -> dace.sdfg.sdfg.SDFG:
        sdfg = DummyNestedSDFG().__sdfg__()
        sdfg.name = "_halo_exchange_wait_"
        return sdfg

    def dace__sdfg_closure__(self, reevaluate: dict[str, str] | None = None) -> dict[str, Any]:
        return DummyNestedSDFG().__sdfg_closure__()

    def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
        return DummyNestedSDFG().__sdfg_signature__()

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
    def wait(self) -> None:
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


class DecompositionFlag(int, Enum):
    UNDEFINED = -1
    OWNED = 0
    """used for locally owned cells, vertices, edges"""

    FIRST_HALO_LEVEL = 1
    """
    used for:
    - cells that share 1 edge with an OWNED cell
    - vertices that are on OWNED cell, but not owned
    - edges that are on OWNED cell, but not owned
    """

    SECOND_HALO_LEVEL = 2
    """
    used for:
    - cells that share one vertex with an OWNED cell
    - vertices that are on a cell(FIRST_HALO_LINE) but not on an owned cell
    - edges that have _exactly_ one vertex shared with and OWNED Cell
    """

    THIRD_HALO_LEVEL = 3
    """
    This type does not exist in ICON. It denotes the "closing/far" edges of the SECOND_HALO_LINE cells
    used for:
    - cells (NOT USED)
    - vertices (NOT USED)
    - edges that are only on the cell(SECOND_HALO_LINE)
    """


single_node_default = SingleNodeExchange()
