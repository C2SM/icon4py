# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import sys
from collections.abc import Iterator

import numpy as np
import pytest

from icon4py.model.common import backend_configuration as bc


CPU = bc.gtx.DeviceType.CPU


@pytest.fixture
def allocator() -> Iterator[bc.IconWorkspaceAllocator]:
    bc.IconWorkspaceAllocator._workspace_slabs.clear()
    yield bc.ICON_WORKSPACE_ALLOCATOR
    bc.IconWorkspaceAllocator._workspace_slabs.clear()


class TestBackendConfig:
    @pytest.mark.parametrize("alignment", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    def test_valid_construction(self, alignment: int) -> None:
        config = bc.BackendConfig(workspace_size=1024, workspace_alignment=alignment)
        assert config.workspace_size == 1024
        assert config.workspace_alignment == alignment

    @pytest.mark.parametrize("size", [0, -1, -1024])
    def test_invalid_size_raises(self, size: int) -> None:
        with pytest.raises(ValueError, match="workspace_size"):
            bc.BackendConfig(workspace_size=size, workspace_alignment=256)

    @pytest.mark.parametrize("alignment", [0, -1, -256])
    def test_non_positive_alignment_raises(self, alignment: int) -> None:
        with pytest.raises(ValueError, match="workspace_alignment"):
            bc.BackendConfig(workspace_size=1024, workspace_alignment=alignment)

    @pytest.mark.parametrize("alignment", [3, 5, 6, 7, 9, 100, 255, 300])
    def test_non_power_of_two_alignment_raises(self, alignment: int) -> None:
        with pytest.raises(ValueError, match="workspace_alignment"):
            bc.BackendConfig(workspace_size=1024, workspace_alignment=alignment)

    def test_is_frozen(self) -> None:
        config = bc.BackendConfig(workspace_size=1024, workspace_alignment=256)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.workspace_size = 2048  # type: ignore[misc]


class TestBackendConfigFromEnv:
    def test_returns_none_when_size_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_SIZE", raising=False)
        assert bc.backend_config_from_env() is None

    def test_uses_default_alignment_when_alignment_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ICON4PY_BACKEND_WORKSPACE_SIZE", "4096")
        monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_ALIGNMENT", raising=False)
        config = bc.backend_config_from_env()
        assert config is not None
        assert config.workspace_size == 4096
        assert config.workspace_alignment == bc._DEFAULT_ALIGNMENT

    def test_reads_both_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICON4PY_BACKEND_WORKSPACE_SIZE", "8192")
        monkeypatch.setenv("ICON4PY_BACKEND_WORKSPACE_ALIGNMENT", "512")
        config = bc.backend_config_from_env()
        assert config is not None
        assert config.workspace_size == 8192
        assert config.workspace_alignment == 512


class TestArrayBasePtr:
    def test_returns_int_for_numpy_array(self) -> None:
        arr = np.empty(64, dtype=np.uint8)
        ptr = bc._array_base_ptr(arr)
        assert isinstance(ptr, int)
        assert ptr == int(arr.__array_interface__["data"][0])

    def test_raises_for_object_without_array_interface(self) -> None:
        with pytest.raises(ValueError, match="no array interface"):
            bc._array_base_ptr(object())


class TestAlignedSlab:
    @pytest.mark.parametrize("alignment", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    def test_slab_is_aligned(self, alignment: int) -> None:
        slab = bc._aligned_slab(1000, alignment, CPU)
        assert bc._array_base_ptr(slab) % alignment == 0

    @pytest.mark.parametrize("nbytes", [1, 100, 1000, 4096])
    def test_slab_has_correct_size(self, nbytes: int) -> None:
        slab = bc._aligned_slab(nbytes, 256, CPU)
        assert slab.nbytes == nbytes

    def test_raises_runtime_error_when_cupy_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.raises(RuntimeError, match="cupy is not available:"):
            bc._aligned_slab(1000, 256, bc.gtx.DeviceType.CUDA)


class TestIconWorkspaceAllocator:
    def test_is_singleton(self) -> None:
        assert bc.IconWorkspaceAllocator() is bc.IconWorkspaceAllocator()
        assert bc.ICON_WORKSPACE_ALLOCATOR is bc.IconWorkspaceAllocator()

    def test_allocate_single_device(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp = allocator.allocate(CPU, size=512, alignment=256)
        assert CPU in wsp
        assert np.asarray(wsp[CPU]).nbytes == 512

    def test_allocate_iterable_of_devices(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp = allocator.allocate([CPU], size=512, alignment=256)
        assert CPU in wsp
        assert np.asarray(wsp[CPU]).nbytes == 512

    def test_allocate_uses_default_alignment(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp = allocator.allocate(CPU, size=512)
        assert bc._array_base_ptr(wsp[CPU]) % bc._DEFAULT_ALIGNMENT == 0

    def test_cache_hit_returns_same_slab(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp1 = allocator.allocate(CPU, size=512, alignment=256)
        wsp2 = allocator.allocate(CPU, size=512, alignment=256)
        assert wsp1[CPU] is wsp2[CPU]

    def test_cache_hit_size_mismatch_raises(self, allocator: bc.IconWorkspaceAllocator) -> None:
        allocator.allocate(CPU, size=512, alignment=256)
        with pytest.raises(ValueError, match="size mismatch"):
            allocator.allocate(CPU, size=1024, alignment=256)

    def test_cache_hit_alignment_mismatch_raises(
        self,
        allocator: bc.IconWorkspaceAllocator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        allocator.allocate(CPU, size=512, alignment=256)
        # Force the cached slab's base pointer to look non-aligned to 512.
        monkeypatch.setattr(bc, "_array_base_ptr", lambda _buf: 1)
        with pytest.raises(ValueError, match="alignment mismatch"):
            allocator.allocate(CPU, size=512, alignment=512)
