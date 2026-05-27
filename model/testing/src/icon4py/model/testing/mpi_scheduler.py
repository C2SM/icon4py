# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""MPI subcommunicator scheduler for parallel pytest execution.

When running MPI tests with `mpirun -np <total_ranks> pytest --with-mpi --mpi-subcomm-size <n>`,
this plugin splits COMM_WORLD into non-overlapping subcommunicators of size <n>.
Each subcomm runs a disjoint subset of MPI tests concurrently.

Example: `mpirun -np 8 pytest --with-mpi --mpi-subcomm-size 4` runs two MPI test
suites in parallel, each using 4 ranks.
"""

from __future__ import annotations

import contextlib

import pytest


class MPISubcommScheduler:
    """Splits MPI_COMM_WORLD into subcommunicators for parallel test execution."""

    def __init__(self, subcomm_size: int):
        from mpi4py import MPI

        if subcomm_size <= 0:
            raise ValueError("--mpi-subcomm-size must be a positive integer")

        self.world = MPI.COMM_WORLD
        self.world_size = self.world.Get_size()
        self.world_rank = self.world.Get_rank()
        self.subcomm_size = subcomm_size
        self._finalized = False

        if self.world_size % self.subcomm_size != 0:
            raise ValueError(
                f"MPI world size ({self.world_size}) must be divisible by "
                f"--mpi-subcomm-size ({self.subcomm_size})"
            )

        self.num_groups = self.world_size // self.subcomm_size
        self.group_id = self.world_rank // self.subcomm_size
        self.subcomm = self.world.Split(self.group_id, self.world_rank)

        # Monkeypatch _get_process_properties to inject subcomm by default.
        # Tests that explicitly pass a comm_id are unaffected.
        from icon4py.model.common.decomposition import mpi_decomposition

        self._original_get_props = mpi_decomposition._get_process_properties

        def _patched_get_props(with_mpi=False, comm_id=None, **kwargs):
            if with_mpi and comm_id is None:
                comm_id = self.subcomm
            return self._original_get_props(with_mpi=with_mpi, comm_id=comm_id, **kwargs)

        mpi_decomposition._get_process_properties = _patched_get_props

    def filter_items(self, items: list[pytest.Item]) -> list[pytest.Item]:
        """Return only the test items assigned to this subcomm group."""
        # Sort by nodeid for deterministic partitioning regardless of collection order.
        mpi_items = sorted(
            (i for i in items if i.get_closest_marker("mpi")),
            key=lambda i: i.nodeid,
        )
        non_mpi_items = [i for i in items if not i.get_closest_marker("mpi")]

        # Skip MPI tests whose min_size exceeds our subcomm size.
        # pytest-mpi checks against COMM_WORLD, which may be larger than the subcomm.
        # We use get_closest_marker to match pytest-mpi semantics exactly.
        valid_mpi_items = [
            item
            for item in mpi_items
            if (mark := item.get_closest_marker("mpi")) is not None
            and mark.kwargs.get("min_size", 1) <= self.subcomm_size
        ]

        # Round-robin assignment of MPI tests across groups.
        assigned_mpi = [
            item
            for idx, item in enumerate(valid_mpi_items)
            if idx % self.num_groups == self.group_id
        ]

        # Only group 0 runs non-MPI tests to avoid races on shared resources.
        if self.group_id == 0:
            return non_mpi_items + assigned_mpi
        return assigned_mpi

    def finalize(self) -> None:
        """Free the subcommunicator and restore patched functions."""
        if self._finalized:
            return
        self._finalized = True

        try:
            if self.subcomm is not None and self.subcomm != self.world:
                self.subcomm.Free()
        finally:
            # Un-monkeypatch even if Free() raises.
            from icon4py.model.common.decomposition import mpi_decomposition

            mpi_decomposition._get_process_properties = self._original_get_props


@pytest.hookimpl(trylast=True)
def pytest_addoption(parser: pytest.Parser) -> None:
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--mpi-subcomm-size",
            action="store",
            type=int,
            default=None,
            help="Size of MPI subcommunicators for parallel test execution. "
            "Total ranks must be a multiple of this value.",
        )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    subcomm_size = config.getoption("--mpi-subcomm-size", default=None)
    if subcomm_size is None:
        return

    # MPI must be initialized before we can split communicators.
    # pytest_hooks.py calls init_mpi() earlier in its pytest_configure;
    # we rely on that having happened.
    from mpi4py import MPI

    if not MPI.Is_initialized():
        raise pytest.UsageError(
            "--mpi-subcomm-size requires MPI to be initialized. Make sure --with-mpi is passed."
        )

    scheduler = MPISubcommScheduler(subcomm_size)
    config._mpi_scheduler = scheduler

    # Log on each subcomm's rank 0
    if scheduler.subcomm.Get_rank() == 0:
        start_rank = scheduler.group_id * scheduler.subcomm_size
        end_rank = (scheduler.group_id + 1) * scheduler.subcomm_size - 1
        print(
            f"\n[MPI Scheduler] Group {scheduler.group_id}/{scheduler.num_groups}: "
            f"world ranks {start_rank}-{end_rank}, subcomm size {scheduler.subcomm_size}"
        )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    scheduler = getattr(config, "_mpi_scheduler", None)
    if scheduler is not None:
        items[:] = scheduler.filter_items(items)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    scheduler = getattr(session.config, "_mpi_scheduler", None)
    if scheduler is not None:
        scheduler.finalize()
