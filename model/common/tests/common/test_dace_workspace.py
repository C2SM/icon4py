# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pickle

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common import dace_workspace, model_backends


def _slab_ptr(slab: object) -> int:
    interface = getattr(slab, "__cuda_array_interface__", None) or getattr(
        slab, "__array_interface__", None
    )
    assert interface is not None, "slab exposes neither array interface"
    return int(interface["data"][0])


class TestIconWorkspaceAllocatorIsSingleton:
    def test_constructor_returns_the_same_instance(self):
        assert dace_workspace.IconWorkspaceAllocator() is dace_workspace.IconWorkspaceAllocator()

    def test_module_level_singleton_is_the_shared_instance(self):
        assert dace_workspace.IconWorkspaceAllocator() is dace_workspace.ICON_WORKSPACE_ALLOCATOR

    def test_is_picklable_and_round_trips_to_the_singleton(self):
        allocator = dace_workspace.IconWorkspaceAllocator()
        assert pickle.loads(pickle.dumps(allocator)) is allocator


class TestIconWorkspaceAllocator:
    def test_satisfies_external_memory_allocator_protocol(self):
        assert isinstance(
            dace_workspace.ICON_WORKSPACE_ALLOCATOR,
            gtx_transformations.ExternalMemoryAllocator,
        )

    @pytest.mark.parametrize("alignment", [256, 512])
    def test_allocate_returns_aligned_buffer_of_requested_size(self, alignment):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        request = gtx_transformations.AllocationRequest(
            nbytes=1024, device=gtx.DeviceType.CPU, alignment=alignment
        )
        workspace = allocator.allocate(request)

        assert workspace.nbytes >= request.nbytes
        assert _slab_ptr(workspace) % alignment == 0

    def test_allocate_reuses_single_slab_across_calls(self):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        request = gtx_transformations.AllocationRequest(
            nbytes=1024, device=gtx.DeviceType.CPU, alignment=256
        )

        first = allocator.allocate(request)
        second = allocator.allocate(request)

        assert first is second

    def test_allocate_reallocates_when_stronger_alignment_is_requested(self):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        weak = allocator.allocate(
            gtx_transformations.AllocationRequest(
                nbytes=1024, device=gtx.DeviceType.CPU, alignment=256
            )
        )
        strong = allocator.allocate(
            gtx_transformations.AllocationRequest(
                nbytes=1024, device=gtx.DeviceType.CPU, alignment=512
            )
        )

        assert weak is not strong
        assert _slab_ptr(strong) % 512 == 0

    def test_allocate_raises_when_request_exceeds_workspace_size(self):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        request = gtx_transformations.AllocationRequest(
            nbytes=dace_workspace._WORKSPACE_SIZE + 1,
            device=gtx.DeviceType.CPU,
            alignment=256,
        )
        with pytest.raises(ValueError, match="exceeds the maximum"):
            allocator.allocate(request)

    def test_deallocate_is_a_no_op_that_keeps_the_slab_alive(self):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        request = gtx_transformations.AllocationRequest(
            nbytes=1024, device=gtx.DeviceType.CPU, alignment=256
        )
        workspace = allocator.allocate(request)

        allocator.deallocate(workspace)

        assert workspace.nbytes >= request.nbytes
        assert (
            dace_workspace.IconWorkspaceAllocator._workspace_slabs[gtx.DeviceType.CPU][0]
            is workspace
        )

    def test_deallocate_does_not_prevent_subsequent_reuse(self):
        dace_workspace.IconWorkspaceAllocator._workspace_slabs.clear()
        allocator = dace_workspace.IconWorkspaceAllocator()
        request = gtx_transformations.AllocationRequest(
            nbytes=1024, device=gtx.DeviceType.CPU, alignment=256
        )
        first = allocator.allocate(request)
        allocator.deallocate(first)
        second = allocator.allocate(request)

        assert first is second


class _RecordingAllocator:
    """Minimal `ExternalMemoryAllocator` used to verify forwarding."""

    def __init__(self):
        self.deallocated = []

    def allocate(self, request: gtx_transformations.AllocationRequest) -> np.ndarray:
        return np.zeros(request.nbytes, dtype=np.uint8)

    def deallocate(self, wsp: gtx_transformations.ExternalWorkspace) -> None:
        self.deallocated.append(wsp)


class TestMakeCustomDaceBackend:
    def test_default_uses_icon_workspace_allocator(self):
        assert model_backends.make_custom_dace_backend(device=model_backends.CPU) is not None

    def test_accepts_custom_external_memory_allocator(self):
        backend = model_backends.make_custom_dace_backend(
            device=model_backends.CPU, external_memory_allocator=_RecordingAllocator()
        )
        assert backend is not None
