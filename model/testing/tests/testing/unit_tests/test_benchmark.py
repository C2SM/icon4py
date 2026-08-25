# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.testing import benchmark


@pytest.fixture
def _clear_rank_env(monkeypatch):
    for var in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("source", ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"])
def test_resolve_rank_reads_all_sources(_clear_rank_env, monkeypatch, source):
    monkeypatch.setenv(source, "7")
    assert benchmark.resolve_rank() == 7


def test_resolve_rank_precedence(_clear_rank_env, monkeypatch):
    monkeypatch.setenv("PMI_RANK", "1")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("SLURM_PROCID", "3")
    assert benchmark.resolve_rank() == 1


def test_resolve_rank_returns_none_when_missing(_clear_rank_env):
    assert benchmark.resolve_rank() is None


def test_is_upload_rank_true_for_rank_zero(_clear_rank_env, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert benchmark.is_upload_rank() is True


def test_is_upload_rank_false_for_nonzero(_clear_rank_env, monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
    assert benchmark.is_upload_rank() is False


def test_is_upload_rank_explicit_none():
    assert benchmark.is_upload_rank(None) is True


def test_is_upload_rank_from_env_when_missing(_clear_rank_env):
    assert benchmark.is_upload_rank(benchmark.resolve_rank()) is True


def test_validate_grid_override_allows_same_grid():
    benchmark.validate_grid_override("R02B04_GLOBAL", "R02B04_GLOBAL", 5)


def test_validate_grid_override_allows_different_grid_single_step():
    benchmark.validate_grid_override("R02B04_GLOBAL", "R02B06_GLOBAL", 1)


def test_validate_grid_override_rejects_different_grid_multi_step():
    with pytest.raises(ValueError, match="dtime rescaling"):
        benchmark.validate_grid_override("R02B04_GLOBAL", "R02B06_GLOBAL", 5)
