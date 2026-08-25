# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared between benchmark tests and the nox/bencher driver sessions."""

import os


__all__ = ["is_upload_rank", "resolve_rank", "validate_grid_override"]


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


def validate_grid_override(
    experiment_grid_name: str,
    override_grid_name: str,
    num_steps: int,
) -> None:
    """Fail fast when a grid override would run multi-step without dtime rescaling.

    The experiment config (including ``dtime``) is authored for
    ``experiment_grid_name``. Running it on a different grid without rescaling
    the timestep can violate CFL stability. Grid overrides are therefore
    restricted to single-step runs until dtime rescaling is implemented.
    """
    if experiment_grid_name != override_grid_name and num_steps != 1:
        raise ValueError(
            f"Grid override '{override_grid_name}' differs from experiment grid "
            f"'{experiment_grid_name}'; multi-step overrides require dtime "
            f"rescaling. Use --driver-benchmark-steps=1 or implement dtime rescaling."
        )


def is_upload_rank(rank: int | None = None) -> bool:
    """Return True if this rank is responsible for uploading benchmark results.

    In MPI runs only rank 0 uploads; in single-rank runs the resolved rank is
    treated as rank 0.
    """
    if rank is None:
        rank = resolve_rank()
    return rank == 0 or rank is None
