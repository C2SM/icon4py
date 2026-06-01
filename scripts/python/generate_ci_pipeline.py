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

Reads the pipeline variables (COMPONENTS, BACKENDS, LEVELS, GRIDS,
SELECTION) and writes a child pipeline that includes ``ci/base.yml``
and instantiates only the test jobs whose matrix entries match the
requested filter.

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

# Full matrix dimensions
STENCIL_COMPONENTS = [
    "advection",
    "diffusion",
    "dycore",
    "microphysics",
    "muphys",
    "common",
    "driver",
]

DATATEST_COMPONENTS = [
    "advection",
    "diffusion",
    "dycore",
    "microphysics",
    "muphys",
    "common",
    "driver",
    "standalone_driver",
]

MPI_COMPONENTS = [
    "atmosphere/diffusion",
    "atmosphere/dycore",
    "common",
    "standalone_driver",
]

ALL_BACKENDS = ["embedded", "dace_cpu", "dace_gpu", "gtfn_cpu", "gtfn_gpu"]
ALL_GRIDS = ["simple", "icon_regional"]
ALL_LEVELS = ["unit", "integration"]

INTEGRATION_LEVEL = ["integration"]


def _parse_list(raw: str | None) -> list[str]:
    """Parse a colon- or comma-separated string into a list of tokens."""
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[,:]", raw) if x.strip()]


def _intersect(constraint: list[str], candidates: list[str]) -> list[str]:
    """Return *candidates* that are in *constraint*, preserving *candidates* order."""
    constraint_set = set(constraint)
    return [v for v in candidates if v in constraint_set]


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
    components: str | None = None,
    backends: str | None = None,
    levels: str | None = None,
    grids: str | None = None,
    selection: str | None = None,
) -> str:
    """Return the child pipeline YAML as a string."""
    # Fallback defaults match ci/default.yml pipeline variables.
    requested_selections = _resolve_filter(selection, "SELECTION", "stencils:datatest:mpi:tools")
    requested_components = _resolve_filter(
        components,
        "COMPONENTS",
        (
            "advection:diffusion:dycore:microphysics:muphys:"
            "common:driver:standalone_driver:"
            "atmosphere/diffusion:atmosphere/dycore"
        ),
    )
    requested_backends = _resolve_filter(backends, "BACKENDS", "dace_gpu")
    requested_levels = _resolve_filter(levels, "LEVELS", "integration")
    requested_grids = _resolve_filter(grids, "GRIDS", "simple:icon_regional")

    pipeline: dict = {
        "include": [{"local": "ci/base.yml"}],
    }

    # Stencil tests
    if "stencils" in requested_selections:
        filtered_components = _intersect(requested_components, STENCIL_COMPONENTS)
        filtered_backends = _intersect(requested_backends, ALL_BACKENDS)
        filtered_grids = _intersect(requested_grids, ALL_GRIDS)
        if filtered_components and filtered_backends and filtered_grids:
            pipeline["test_stencils_aarch64"] = {
                "extends": ".test_stencils_aarch64",
                "parallel": {
                    "matrix": [
                        {
                            "COMPONENT": filtered_components,
                            "BACKEND": filtered_backends,
                            "GRID": filtered_grids,
                        }
                    ]
                },
            }

    # Serial datatest tests
    if "datatest" in requested_selections:
        filtered_components = _intersect(requested_components, DATATEST_COMPONENTS)
        filtered_backends = _intersect(requested_backends, ALL_BACKENDS)
        filtered_levels = _intersect(requested_levels, ALL_LEVELS)
        if filtered_components and filtered_backends and filtered_levels:
            pipeline["test_datatests_serial_aarch64"] = {
                "extends": ".test_datatests_serial_aarch64",
                "parallel": {
                    "matrix": [
                        {
                            "COMPONENT": filtered_components,
                            "BACKEND": filtered_backends,
                            "LEVEL": filtered_levels,
                        }
                    ]
                },
            }

    # Tools test (single job, no matrix)
    if "tools" in requested_selections:
        pipeline["test_tools_aarch64"] = {
            "extends": ".test_tools_aarch64",
        }

    # MPI datatest tests
    if "mpi" in requested_selections:
        filtered_components = _intersect(requested_components, MPI_COMPONENTS)
        filtered_backends = _intersect(requested_backends, ALL_BACKENDS)
        filtered_levels = _intersect(requested_levels, INTEGRATION_LEVEL)
        if filtered_components and filtered_backends and filtered_levels:
            pipeline["test_datatests_mpi_aarch64"] = {
                "extends": ".test_datatests_mpi_aarch64",
                "parallel": {
                    "matrix": [
                        {
                            "COMPONENT": filtered_components,
                            "BACKEND": filtered_backends,
                            "LEVEL": filtered_levels,
                        }
                    ]
                },
            }

    test_jobs = [k for k in pipeline if k != "include"]
    if not test_jobs:
        print(
            "WARNING: no test jobs matched the filter",
            file=sys.stderr,
        )

    return yaml.dump(pipeline, default_flow_style=False, sort_keys=False)


@cli.command()
def generate_ci_pipeline(
    components: Annotated[
        str | None,
        typer.Option("--components", help="Colon/comma-separated component filter"),
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
    selection: Annotated[
        str | None,
        typer.Option("--selection", help="Colon/comma-separated selection filter"),
    ] = None,
) -> None:
    """Generate child pipeline YAML to stdout.

    Reads pipeline variables from environment or CLI options and writes
    a child pipeline to standard output that includes only the test jobs
    matching the requested filter.
    """
    sys.stdout.write(
        _generate_child_pipeline(
            components=components,
            backends=backends,
            levels=levels,
            grids=grids,
            selection=selection,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
