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
from typing import TYPE_CHECKING, Any, ClassVar, Final, Optional, Sequence, Union

from gt4py.next import Dimension, Field

from icon4py.model.common import dimension as dims
from icon4py.model.common.config import Device
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.decomposition.definitions import SingleNodeExchange
from icon4py.model.common.settings import device, xp


try:
    import ghex
    import mpi4py
    from ghex.context import make_context
    from ghex.unstructured import (
        DomainDescriptor,
        HaloGenerator,
        make_communication_object,
        make_field_descriptor,
        make_pattern,
    )
    from ghex.util import Architecture

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = True

except ImportError:
    mpi4py = None
    ghex = None
    unstructured = None

try:
    import dace

    from icon4py.model.common.orchestration import halo_exchange
except ImportError:
    from types import ModuleType

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]

if TYPE_CHECKING:
    import mpi4py.MPI

CommId = Union[int, "mpi4py.MPI.Comm", None]
log = logging.getLogger(__name__)


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


def _get_processor_properties(with_mpi=False, comm_id: CommId = None):
    def _get_current_comm_or_comm_world(comm_id: CommId) -> mpi4py.MPI.Comm:
        if isinstance(comm_id, int):
            comm = mpi4py.MPI.Comm.f2py(comm_id)
        elif isinstance(comm_id, mpi4py.MPI.Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    if with_mpi:
        init_mpi()
        current_comm = _get_current_comm_or_comm_world(comm_id)
        return MPICommProcessProperties(current_comm)


class ParallelLogger(logging.Filter):
    def __init__(self, process_properties: definitions.ProcessProperties = None):
        super().__init__()
        self._rank_info = ""
        if process_properties and process_properties.comm_size > 1:
            self._rank_info = f"rank={process_properties.rank}/{process_properties.comm_size} [{process_properties.comm_name}] "

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self._rank_info
        return True


@definitions.get_processor_properties.register(definitions.MultiNodeRun)
def get_multinode_properties(
    s: definitions.MultiNodeRun, comm_id: CommId = None
) -> definitions.ProcessProperties:
    return _get_processor_properties(with_mpi=True, comm_id=comm_id)


@dataclass(frozen=True)
class MPICommProcessProperties(definitions.ProcessProperties):
    comm: mpi4py.MPI.Comm = None

    @functools.cached_property
    def rank(self):
        return self.comm.Get_rank()

    @functools.cached_property
    def comm_name(self):
        return self.comm.Get_name()

    @functools.cached_property
    def comm_size(self):
        return self.comm.Get_size()


class GHexMultiNodeExchange:
    max_num_of_fields_to_communicate_dace: Final[
        int
    ] = 10  # maximum number of fields to perform halo exchange on (DaCe-related)

    def __init__(
        self,
        props: definitions.ProcessProperties,
        domain_decomposition: definitions.DecompositionInfo,
    ):
        self._context = make_context(props.comm, False)
        self._domain_id_gen = definitions.DomainDescriptorIdGenerator(props)
        self._decomposition_info = domain_decomposition
        self._domain_descriptors = {
            dim: self._create_domain_descriptor(dim) for dim in dims.global_dimensions.values()
        }
        log.info(f"domain descriptors for dimensions {self._domain_descriptors.keys()} initialized")
        self._patterns = {dim: self._create_pattern(dim) for dim in dims.global_dimensions.values()}
        log.info(f"patterns for dimensions {self._patterns.keys()} initialized ")
        self._comm = make_communication_object(self._context)

        # DaCe SDFGConvertible interface
        self.num_of_halo_tasklets = (
            0  # Some SDFG variables need to be defined only once (per fused SDFG)
        )

        log.info("communication object initialized")

    def _domain_descriptor_info(self, descr):
        return f" domain_descriptor=[id='{descr.domain_id()}', size='{descr.size()}', inner_size='{descr.inner_size()}' (halo size='{descr.size() - descr.inner_size()}')"

    def get_size(self):
        return self._context.size()

    def my_rank(self):
        return self._context.rank()

    def _create_domain_descriptor(self, dim: Dimension):
        all_global = self._decomposition_info.global_index(
            dim, definitions.DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, definitions.DecompositionInfo.EntryType.HALO
        )
        # first arg is the domain ID which builds up an MPI Tag.
        # if those ids are not different for all domain descriptors the system might deadlock
        # if two parallel exchanges with the same domain id are done
        domain_desc = DomainDescriptor(
            self._domain_id_gen(), all_global.tolist(), local_halo.tolist()
        )
        log.debug(
            f"domain descriptor for dim='{dim.value}' with properties {self._domain_descriptor_info(domain_desc)} created"
        )
        return domain_desc

    def _create_pattern(self, horizontal_dim: Dimension):
        assert horizontal_dim.kind == dims.DimensionKind.HORIZONTAL

        global_halo_idx = self._decomposition_info.global_index(
            horizontal_dim, definitions.DecompositionInfo.EntryType.HALO
        )
        halo_generator = HaloGenerator.from_gids(global_halo_idx)
        log.debug(f"halo generator for dim='{horizontal_dim.value}' created")
        pattern = make_pattern(
            self._context,
            halo_generator,
            [self._domain_descriptors[horizontal_dim]],
        )
        log.debug(
            f"pattern for dim='{horizontal_dim.value}' and {self._domain_descriptor_info(self._domain_descriptors[horizontal_dim])} created"
        )
        return pattern

    def _slice_field_based_on_dim(self, field: Field, dim: definitions.Dimension) -> xp.ndarray:
        """
        Slices the field based on the dimension passed in.
        """
        if dim == dims.VertexDim:
            return field.ndarray[: self._decomposition_info.num_vertices, :]
        elif dim == dims.EdgeDim:
            return field.ndarray[: self._decomposition_info.num_edges, :]
        elif dim == dims.CellDim:
            return field.ndarray[: self._decomposition_info.num_cells, :]
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def exchange(self, dim: definitions.Dimension, *fields: Sequence[Field]):
        """
        Exchange method that slices the fields based on the dimension and then performs halo exchange.

            This ensures that the exchanged fields always have the correct size, thereby also working in
            the granule context where fields otherwise have length nproma.
        """
        assert dim in dims.global_dimensions.values()
        pattern = self._patterns[dim]
        assert pattern is not None, f"pattern for {dim.value} not found"
        domain_descriptor = self._domain_descriptors[dim]
        assert domain_descriptor is not None, f"domain descriptor for {dim.value} not found"

        # Slice the fields based on the dimension
        sliced_fields = [self._slice_field_based_on_dim(f, dim) for f in fields]

        # Create field descriptors and perform the exchange
        arch = Architecture.CPU if device == Device.CPU else Architecture.GPU
        applied_patterns = [
            pattern(make_field_descriptor(domain_descriptor, f, arch=arch)) for f in sliced_fields
        ]
        handle = self._comm.exchange(applied_patterns)
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' initiated.")
        return MultiNodeResult(handle, applied_patterns)

    def exchange_and_wait(self, dim: Dimension, *fields: tuple):
        res = self.exchange(dim, *fields)
        res.wait()
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' done.")

    def __call__(self, *args, **kwargs) -> Optional[MultiNodeResult]:
        """Perform a halo exchange operation.

        Args:
            args: The fields to be exchanged.

        Keyword Args:
            dim: The dimension along which the exchange is performed.
            wait: If True, the operation will block until the exchange is completed (default: True).
        """
        dim = kwargs.get("dim", None)
        if dim is None:
            raise ValueError("Need to define a dimension.")
        wait = kwargs.get("wait", True)

        res = self.exchange(dim, *args)
        if wait:
            res.wait()
        else:
            return res

    if dace:
        # Implementation of DaCe SDFGConvertible interface
        def dace__sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
            if len(args) > GHexMultiNodeExchange.max_num_of_fields_to_communicate_dace:
                raise ValueError(
                    f"Maximum number of fields to communicate is {GHexMultiNodeExchange.max_num_of_fields_to_communicate_dace}. Adapt the max number accordingly."
                )
            dim = kwargs.get("dim", None)
            if dim is None:
                raise ValueError("Need to define a dimension.")
            wait = kwargs.get("wait", True)

            # Build the halo exchange SDFG and return it
            sdfg = dace.SDFG("_halo_exchange_")
            state = sdfg.add_state()

            global_buffers = {
                self.__sdfg_signature__()[0][i]: arg for i, arg in enumerate(args)
            }  # Field name : Data Descriptor

            halo_exchange.add_halo_tasklet(
                sdfg, state, global_buffers, self, dim, id(self), wait, self.num_of_halo_tasklets
            )

            sdfg.arg_names.extend(self.__sdfg_signature__()[0])
            sdfg.arg_names.extend(list(self.__sdfg_closure__().keys()))

            self.num_of_halo_tasklets += 1
            return sdfg

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            # Get the underlying C++ pointers of the GHEX objects and use them in the halo exchange tasklet
            return {ghex_ptr_name: dace.uintp for ghex_ptr_name in halo_exchange.GHEX_PTR_NAMES}

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            args = []
            for i in range(GHexMultiNodeExchange.max_num_of_fields_to_communicate_dace):
                args.append(f"field_{i}")
            return (args, [])

    else:

        def dace__sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
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


@dataclass
class HaloExchangeWait:
    exchange_object: GHexMultiNodeExchange

    buffer_name: ClassVar[str] = "communication_handle"  # DaCe-related

    def __call__(self, communication_handle: MultiNodeResult) -> None:
        """Wait on the communication handle."""
        communication_handle.wait()

    if dace:
        # Implementation of DaCe SDFGConvertible interface
        def dace__sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
            sdfg = dace.SDFG("_halo_exchange_wait_")
            state = sdfg.add_state()

            # The communication handle used in the halo_exchange tasklet is a global variable
            # ghex::communication_handle<communication_handle_type> h_{id(self.exchange_object)};
            # Therefore, this tasklet calls the wait() method on the communication handle -disregards any input-
            tasklet = dace.sdfg.nodes.Tasklet(
                "_halo_exchange_wait_",
                inputs=None,
                outputs=None,
                code=f"h_{id(self.exchange_object)}.wait();\n//__out = 1234;",
                language=dace.dtypes.Language.CPP,
                side_effects=False,
            )
            state.add_node(tasklet)

            # Dummy input to maintain same interface with non-DaCe branch
            buffer_name = HaloExchangeWait.buffer_name
            sdfg.add_scalar(name=buffer_name, dtype=dace.int32)
            buffer = state.add_read(buffer_name)
            tasklet.in_connectors["IN_" + buffer_name] = dace.int32.dtype
            state.add_edge(
                buffer, None, tasklet, "IN_" + buffer_name, dace.Memlet(buffer_name, subset="0")
            )

            """
            # noqa: ERA001

            # Dummy return, otherwise dead-dataflow-elimination kicks in. Return something to generate code.
            sdfg.add_scalar(name="__return", dtype=dace.int32)
            ret = state.add_write("__return")
            state.add_edge(tasklet, "__out", ret, None, dace.Memlet(data="__return", subset="0"))
            tasklet.out_connectors["__out"] = dace.int32
            """

            sdfg.arg_names.extend(self.__sdfg_signature__()[0])

            return sdfg

        def dace__sdfg_closure__(
            self, reevaluate: Optional[dict[str, str]] = None
        ) -> dict[str, Any]:
            return {}

        def dace__sdfg_signature__(self) -> tuple[Sequence[str], Sequence[str]]:
            return (
                [
                    HaloExchangeWait.buffer_name,
                ],
                [],
            )

    else:

        def dace__sdfg__(self, *args, **kwargs) -> dace.sdfg.sdfg.SDFG:
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


@definitions.create_halo_exchange_wait.register(GHexMultiNodeExchange)
def create_multinode_halo_exchange_wait(runtime: GHexMultiNodeExchange) -> HaloExchangeWait:
    return HaloExchangeWait(runtime)


@dataclass
class MultiNodeResult:
    handle: ...
    pattern_refs: ...

    def wait(self):
        self.handle.wait()
        del self.pattern_refs

    def is_ready(self) -> bool:
        return self.handle.is_ready()


@definitions.create_exchange.register(MPICommProcessProperties)
def create_multinode_node_exchange(
    props: MPICommProcessProperties, decomp_info: definitions.DecompositionInfo
) -> definitions.ExchangeRuntime:
    if props.comm_size > 1:
        return GHexMultiNodeExchange(props, decomp_info)
    else:
        return SingleNodeExchange()
