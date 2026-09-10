# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Package-level test configuration of tracer_advection.

Floating-point contraction. Bit-exactness against a Fortran capture needs FMA contraction
off on both sides (the Jocksch cylinder capture was built with -Kieee -Mnofma -gpu=nofma).
GT4Py has no knob of its own, but its build systems honour the environment: 'CXXFLAGS'
(gtfn: CMake's initial C++ flags; DaCe: replaces 'compiler.cpu.args' wholesale, see
gt4py.next.program_processors.runners.dace.workflow.common.set_dace_config) and
'NVCC_APPEND_FLAGS' (appended by nvcc itself, so it reaches both build systems). These are
the variables ci/base.yml exports for the bit-reproducibility jobs; setting
'ICON4PY_FP_CONTRACT_OFF=1' makes this conftest export the same values, unless the caller
set them already. A porting tool, not a production setting: the build cache key ignores
compiler flags, so a run with the flags MUST use its own 'GT4PY_BUILD_CACHE_DIR' (enforced
below), or the contracted and the uncontracted builds silently mix.
"""

import os

import pytest

from icon4py.model.common.utils import env

# the savepoint selection fixture 'advection_init/exit_savepoint' (fixtures.py) depend on;
# tests override it by a parametrized argument of the same name
from .fixtures import step


#: the values of ci/base.yml (MPI-reproducibility jobs)
NO_FMA_CXXFLAGS = "-ffp-contract=off"
NO_FMA_NVCC_APPEND_FLAGS = "--fmad=false"


def pytest_configure(config: pytest.Config) -> None:
    if not env.flag_to_bool("ICON4PY_FP_CONTRACT_OFF", False):
        return
    if "GT4PY_BUILD_CACHE_DIR" not in os.environ:
        raise pytest.UsageError(
            "'ICON4PY_FP_CONTRACT_OFF' needs a dedicated 'GT4PY_BUILD_CACHE_DIR': the build "
            "cache does not key on compiler flags, a shared cache mixes the two builds."
        )
    os.environ.setdefault("CXXFLAGS", NO_FMA_CXXFLAGS)
    os.environ.setdefault("NVCC_APPEND_FLAGS", NO_FMA_NVCC_APPEND_FLAGS)
