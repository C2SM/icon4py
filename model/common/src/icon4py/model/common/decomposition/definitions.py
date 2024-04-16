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
from enum import IntEnum
from typing import Any, Protocol, Any, Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.ma as ma
from gt4py.next import Dimension

from icon4py.model.common.utils import builder

import dace
from dace.frontend.python.common import SDFGConvertible


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

    @builder
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
class SingleNodeExchange(SDFGConvertible):
    return_sdfg = False

    def exchange(self, dim: Dimension, *fields: tuple) -> ExchangeResult:
        return SingleNodeResult()

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        return

    def my_rank(self):
        return 0

    def get_size(self):
        return 1

    def __call__(self, *args, **kwargs) -> Optional[dace.SDFG]:
        dim = kwargs.get('dim', None)
        wait = kwargs.get('wait', True)

        if self.return_sdfg:
            sdfg = dace.SDFG('_halo_exchange_')
            state = sdfg.add_state()

            # Dummy return: preserve same interface with non-DaCe version
            sdfg.add_scalar(name='__return', dtype=dace.int32)

            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_',
                                              inputs=None,
                                              outputs=None,
                                              code='__out = 1234;',
                                              language=dace.dtypes.Language.CPP,
                                              side_effects=False,)
            state.add_node(tasklet)

            ret = state.add_write('__return')
            state.add_edge(tasklet, '__out', ret, None, dace.Memlet(data='__return', subset='0'))
            tasklet.out_connectors = {'__out':dace.int32}

            self.return_sdfg = False # reset
            return sdfg
        else:
            res = self.exchange(dim, *args)
            if wait:
                res.wait()
            else:
                return res

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        self.return_sdfg = True
        sdfg = self.__call__(*args, **kwargs)
        return sdfg
    
    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        return ([],[])


@dataclass
class WaitOnComm(SDFGConvertible):
    return_sdfg : bool = False

    def __call__(self, *args, **kwargs) -> Optional[dace.SDFG]:
        if self.return_sdfg:
            sdfg = dace.SDFG('_halo_exchange_wait_')
            state = sdfg.add_state()

            # Dummy return, otherwise dead-dataflow-elimination kicks in. Return something to generate code.
            sdfg.add_scalar(name='__return', dtype=dace.int32)

            tasklet = dace.sdfg.nodes.Tasklet('_halo_exchange_wait_',
                                              inputs=None,
                                              outputs=None,
                                              code='__out = 1234;',
                                              language=dace.dtypes.Language.CPP,
                                              side_effects=False,)
            state.add_node(tasklet)

            ret = state.add_write('__return')
            state.add_edge(tasklet, '__out', ret, None, dace.Memlet(data='__return', subset='0'))
            tasklet.out_connectors = {'__out':dace.int32}

            self.return_sdfg = False # reset
            return sdfg
        else:
            args[0].wait()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        self.return_sdfg = True
        sdfg = self.__call__(*args, **kwargs)
        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}
    
    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        return ([],[])


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
