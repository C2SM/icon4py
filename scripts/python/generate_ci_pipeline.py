#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate GitLab CI child pipeline YAML from pipeline variables.

Reads the pipeline variables (SESSIONS, MODEL_SUBPACKAGES, MODEL_MPI_SUBPACKAGES,
BACKENDS, LEVELS, GRIDS, MODEL_SUBSETS) or corresponding command-line options
and writes a child pipeline that includes ``ci/base.yml`` and instantiates only
the test jobs whose matrix entries match the requested filter.

Each SESSION value maps to a single job template that corresponds to a nox
session. Parameters like MODEL_SUBPACKAGES and MODEL_SUBSETS are passed as nox
session parameters.

This script exists because using rules:if with regexes that are dynamically
generated does not work (gitlab does not expand variables in patterns). This
script does the filtering at generation time instead.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Annotated

import typer
import yaml


cli = typer.Typer(no_args_is_help=True, help=__doc__)

ALL_SESSIONS = ["model", "tools", "mpi"]
ALL_MODEL_SUBSETS = ["stencils", "datatest", "basic"]
ALL_MODEL_MPI_SUBSETS = ["basic", "datatest"]
ALL_MODEL_SUBPACKAGES = [
    "advection",
    "diffusion",
    "dycore",
    "microphysics",
    "muphys",
    "common",
    "driver",
    "standalone_driver",
]

ALL_MODEL_MPI_SUBPACKAGES = [
    "advection",
    "diffusion",
    "dycore",
    "common",
    "standalone_driver",
]
ALL_BACKENDS = ["embedded", "dace_cpu", "dace_gpu", "gtfn_cpu", "gtfn_gpu"]
ALL_GRIDS = ["simple", "icon_regional"]
ALL_LEVELS = ["any", "unit", "integration"]


def _parse_list(raw: str | None) -> list[str]:
    """Parse a colon- or comma-separated string into a list of tokens.

    Colons are supported as separators because commas can't be used as
    separators in CSCS CI variables when triggering jobs. The comma is reserved
    for separating pipeline names in cscs-ci run pipeline1,pipeline2.
    """
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,:]", raw) if x.strip()]


def _intersect(constraint: list[str], candidates: list[str]) -> list[str]:
    """Return *candidates* that are in *constraint*, preserving *candidates* order."""
    constraint_set = set(constraint)
    return [v for v in candidates if v in constraint_set]


def _validate_tokens(name: str, tokens: list[str], valid: list[str]) -> None:
    """Validate that all tokens are members of valid.

    Exits with a descriptive error message and status 1 if any token is not
    recognised, helping users catch typos early instead of silently producing
    an empty pipeline.
    """
    invalid = [t for t in tokens if t not in valid]
    if invalid:
        accepted = ", ".join(sorted(valid))
        print(
            f"ERROR: invalid {name} values: {', '.join(invalid)}",
            file=sys.stderr,
        )
        print(f"Accepted values: {accepted}", file=sys.stderr)
        sys.exit(1)


def _resolve_filter(cli_value: str | None, env_var: str, default: str) -> list[str]:
    """Resolve a filter value from CLI arg, env var, or built-in default.

    When *cli_value* is provided (including empty string) it takes
    precedence.  Otherwise the environment variable is consulted,
    falling back to *default*.
    """
    if cli_value is not None:
        return _parse_list(cli_value)
    return _parse_list(os.environ.get(env_var)) or _parse_list(default)


def _generate_child_pipeline(
    *,
    sessions: str | None = None,
    model_subsets: str | None = None,
    model_subpackages: str | None = None,
    model_mpi_subpackages: str | None = None,
    model_mpi_subsets: str | None = None,
    backends: str | None = None,
    levels: str | None = None,
    grids: str | None = None,
) -> str:
    """Return the child pipeline YAML as a string."""
    # Fallback defaults match ci/default.yml pipeline variables.
    requested_sessions = _resolve_filter(sessions, "SESSIONS", "model:tools:mpi")
    _validate_tokens("SESSIONS", requested_sessions, ALL_SESSIONS)

    requested_model_subsets = _resolve_filter(model_subsets, "MODEL_SUBSETS", "stencils:datatest")
    _validate_tokens("MODEL_SUBSETS", requested_model_subsets, ALL_MODEL_SUBSETS)

    requested_model_subpackages = _resolve_filter(
        model_subpackages,
        "MODEL_SUBPACKAGES",
        "advection:diffusion:dycore:microphysics:muphys:common:driver:standalone_driver",
    )
    _validate_tokens("MODEL_SUBPACKAGES", requested_model_subpackages, ALL_MODEL_SUBPACKAGES)

    requested_model_mpi_subpackages = _resolve_filter(
        model_mpi_subpackages,
        "MODEL_MPI_SUBPACKAGES",
        "advection:diffusion:dycore:common:standalone_driver",
    )
    _validate_tokens(
        "MODEL_MPI_SUBPACKAGES", requested_model_mpi_subpackages, ALL_MODEL_MPI_SUBPACKAGES
    )

    requested_model_mpi_subsets = _resolve_filter(
        model_mpi_subsets, "MODEL_MPI_SUBSETS", "basic:datatest"
    )
    _validate_tokens("MODEL_MPI_SUBSETS", requested_model_mpi_subsets, ALL_MODEL_MPI_SUBSETS)

    requested_backends = _resolve_filter(backends, "BACKENDS", "dace_gpu")
    _validate_tokens("BACKENDS", requested_backends, ALL_BACKENDS)

    requested_levels = _resolve_filter(levels, "LEVELS", "integration")
    _validate_tokens("LEVELS", requested_levels, ALL_LEVELS)

    requested_grids = _resolve_filter(grids, "GRIDS", "simple:icon_regional")
    _validate_tokens("GRIDS", requested_grids, ALL_GRIDS)

    pipeline: dict = {
        "include": [{"local": "ci/base.yml"}],
    }

    if "model" in requested_sessions:
        filtered_subpackages = _intersect(requested_model_subpackages, ALL_MODEL_SUBPACKAGES)
        filtered_backends = _intersect(requested_backends, ALL_BACKENDS)
        filtered_subsets = _intersect(requested_model_subsets, ALL_MODEL_SUBSETS)

        if filtered_subpackages and filtered_backends and filtered_subsets:
            matrix: list[dict] = []

            # Stencils subset uses GRID dimension
            if "stencils" in filtered_subsets:
                filtered_grids = _intersect(requested_grids, ALL_GRIDS)
                if filtered_grids:
                    matrix.append(
                        {
                            "MODEL_SUBPACKAGE": filtered_subpackages,
                            "MODEL_SUBSET": ["stencils"],
                            "BACKEND": filtered_backends,
                            "GRID": filtered_grids,
                        }
                    )

            # Datatest and basic subsets need LEVEL dimension
            level_subsets = [s for s in filtered_subsets if s in ("datatest", "basic")]
            if level_subsets:
                filtered_levels = _intersect(requested_levels, ALL_LEVELS)
                if filtered_levels:
                    matrix.append(
                        {
                            "MODEL_SUBPACKAGE": filtered_subpackages,
                            "MODEL_SUBSET": level_subsets,
                            "BACKEND": filtered_backends,
                            "LEVEL": filtered_levels,
                        }
                    )

            if matrix:
                pipeline["test_model_aarch64"] = {
                    "extends": ".test_model_aarch64",
                    "parallel": {"matrix": matrix},
                }

    if "tools" in requested_sessions:
        pipeline["test_tools_aarch64"] = {
            "extends": ".test_tools_aarch64",
            "parallel": {
                "matrix": [
                    {"SELECTION": ["datatest", "unittest"]},
                ]
            },
        }

    if "mpi" in requested_sessions:
        filtered_subpackages = _intersect(
            requested_model_mpi_subpackages, ALL_MODEL_MPI_SUBPACKAGES
        )
        filtered_backends = _intersect(requested_backends, ALL_BACKENDS)
        filtered_levels = _intersect(requested_levels, ALL_LEVELS)
        filtered_mpi_subsets = _intersect(requested_model_mpi_subsets, ALL_MODEL_MPI_SUBSETS)
        if filtered_subpackages and filtered_backends and filtered_levels and filtered_mpi_subsets:
            pipeline["test_model_mpi_aarch64"] = {
                "extends": ".test_model_mpi_aarch64",
                "parallel": {
                    "matrix": [
                        {
                            "MODEL_MPI_SUBPACKAGE": filtered_subpackages,
                            "BACKEND": filtered_backends,
                            "LEVEL": filtered_levels,
                            "SELECTION": filtered_mpi_subsets,
                        }
                    ]
                },
            }

    test_jobs = [k for k in pipeline if k != "include"]
    if not test_jobs:
        print(
            "ERROR: no test jobs matched the filter",
            file=sys.stderr,
        )
        sys.exit(1)

    return yaml.dump(pipeline)


@cli.command()
def generate_ci_pipeline(  # noqa: PLR0917 [too-many-positional-arguments]
    sessions: Annotated[
        str | None,
        typer.Option(
            "--sessions",
            help="Colon/comma-separated nox session filter (model, tools, mpi)",
        ),
    ] = None,
    model_subpackages: Annotated[
        str | None,
        typer.Option(
            "--model-subpackages",
            help="Colon/comma-separated model subpackage filter",
        ),
    ] = None,
    model_mpi_subpackages: Annotated[
        str | None,
        typer.Option(
            "--model-mpi-subpackages",
            help="Colon/comma-separated MPI subpackage filter",
        ),
    ] = None,
    model_mpi_subsets: Annotated[
        str | None,
        typer.Option(
            "--model-mpi-subsets",
            help="Colon/comma-separated MPI test subset filter (basic, datatest)",
        ),
    ] = None,
    model_subsets: Annotated[
        str | None,
        typer.Option(
            "--model-subsets",
            help="Colon/comma-separated model test subset filter (stencils, datatest, basic)",
        ),
    ] = None,
    backends: Annotated[
        str | None,
        typer.Option("--backends", help="Colon/comma-separated backend filter"),
    ] = None,
    levels: Annotated[
        str | None,
        typer.Option("--levels", help="Colon/comma-separated level filter"),
    ] = None,
    grids: Annotated[
        str | None,
        typer.Option("--grids", help="Colon/comma-separated grid filter"),
    ] = None,
) -> None:
    """Generate child pipeline YAML to stdout.

    Reads pipeline variables from environment or CLI options and writes
    a child pipeline to standard output that includes only the test jobs
    matching the requested filter.
    """
    sys.stdout.write(
        _generate_child_pipeline(
            sessions=sessions,
            model_subpackages=model_subpackages,
            model_mpi_subpackages=model_mpi_subpackages,
            model_mpi_subsets=model_mpi_subsets,
            model_subsets=model_subsets,
            backends=backends,
            levels=levels,
            grids=grids,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
