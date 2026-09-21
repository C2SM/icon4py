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
    def test_valid_construction(self) -> None:
        config = bc.BackendConfig(workspace_size=1024)
        assert config.workspace_size == 1024

    @pytest.mark.parametrize("size", [0, -1, -1024])
    def test_invalid_size_raises(self, size: int) -> None:
        with pytest.raises(ValueError, match="workspace_size"):
            bc.BackendConfig(workspace_size=size)

    def test_is_frozen(self) -> None:
        config = bc.BackendConfig(workspace_size=1024)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.workspace_size = 2048  # type: ignore[misc]


class TestBackendConfigFromEnv:
    def test_returns_none_when_size_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ICON4PY_BACKEND_WORKSPACE_SIZE", raising=False)
        assert bc.backend_config_from_env() is None

    def test_reads_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICON4PY_BACKEND_WORKSPACE_SIZE", "8192")
        config = bc.backend_config_from_env()
        assert config is not None
        assert config.workspace_size == 8192


class TestGetSlab:
    @pytest.mark.parametrize("nbytes", [1, 100, 1000, 4096])
    def test_slab_has_correct_size(self, nbytes: int) -> None:
        slab = bc._get_slab(nbytes, CPU)
        assert slab.nbytes == nbytes

    def test_raises_runtime_error_when_cupy_not_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "cupy", None)
        with pytest.raises(RuntimeError, match="cupy is not available:"):
            bc._get_slab(1000, bc.gtx.DeviceType.CUDA)


class TestIconWorkspaceAllocator:
    def test_is_singleton(self) -> None:
        assert bc.IconWorkspaceAllocator() is bc.IconWorkspaceAllocator()
        assert bc.ICON_WORKSPACE_ALLOCATOR is bc.IconWorkspaceAllocator()

    def test_allocate_single_device(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp = allocator.allocate(CPU, size=512)
        assert CPU in wsp
        assert np.asarray(wsp[CPU]).nbytes == 512

    def test_allocate_iterable_of_devices(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp = allocator.allocate([CPU], size=512)
        assert CPU in wsp
        assert np.asarray(wsp[CPU]).nbytes == 512

    def test_cache_hit_returns_same_slab(self, allocator: bc.IconWorkspaceAllocator) -> None:
        wsp1 = allocator.allocate(CPU, size=512)
        wsp2 = allocator.allocate(CPU, size=512)
        assert wsp1[CPU] is wsp2[CPU]

    def test_cache_hit_size_mismatch_raises(self, allocator: bc.IconWorkspaceAllocator) -> None:
        allocator.allocate(CPU, size=512)
        with pytest.raises(ValueError, match="size mismatch"):
            allocator.allocate(CPU, size=1024)
