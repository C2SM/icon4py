# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Shared test-selection helpers used by nox sessions and CI generators."""

from __future__ import annotations

from typing import Literal, TypeAlias


ModelTestsSubset: TypeAlias = Literal["datatest", "stencils", "basic"]


def _selection_to_pytest_args(selection: ModelTestsSubset) -> list[str]:
    """Return pytest CLI flags for a model test subset."""
    match selection:
        case "datatest":
            return ["--datatest-only"]
        case "stencils":
            return ["-k", "stencil_tests"]
        case "basic":
            return [
                "--datatest-skip",
                "-k",
                "not stencil_tests and not benchmark_only",
            ]
        case _:
            raise AssertionError(f"Invalid selection: {selection}")
