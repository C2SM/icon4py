# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared between benchmark tests and the nox/bencher driver sessions."""

import os


__all__ = ["is_upload_rank", "resolve_rank"]


def resolve_rank() -> int | None:
    """Return the MPI rank from the runtime environment, or None when not set.

    The precedence mirrors ``.cscs-ci/scripts/ci-mpi-wrapper.sh``:
    ``PMI_RANK`` -> ``OMPI_COMM_WORLD_RANK`` -> ``SLURM_PROCID``.
    """
    rank = (
        os.environ.get("PMI_RANK")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("SLURM_PROCID")
    )
    if rank is None:
        return None
    return int(rank)


def is_upload_rank(rank: int | None = None) -> bool:
    """Return True if this rank is responsible for uploading benchmark results.

    In MPI runs only rank 0 uploads; in single-rank runs the resolved rank is
    treated as rank 0.
    """
    if rank is None:
        rank = resolve_rank()
    return rank == 0 or rank is None
