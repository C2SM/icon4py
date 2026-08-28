# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test configuration for the turbulence granule.

The turbulence stencils are validated bit-for-bit against serialized ICON output, which is only
attainable if neither side contracts multiply-adds. Both sides had to be arranged for it:

- The reference: 'build_serialize' is compiled with '-Kieee -Mnofma -gpu=nofma' on the four
  turbulence translation units (see 'docs/superpowers/notes/2026-08-28-serialization-recipe.md',
  section 11). Without it nvhpc fuses, and the capture carries one fewer rounding than the
  expression as written.
- The port: the compiled backends contract by default -- GCC at '-ffp-contract=fast', nvcc at
  '--fmad=true'. GT4Py exposes no knob for this ('gt4py.next.config' has only CMAKE_BUILD_TYPE),
  but its CMake toolchain honours the standard environment variables, which is what is set below.

Measured on section 1b, which is the first section containing an 'a*b + c*d': without the flag
'gtfn_cpu' misses bit-exactness by exactly one rounding on 'frh' and 'frm'; with it, all four
dates agree exactly. 'embedded' needs no flag, numpy not contracting.

Setting these here rather than documenting them keeps a passing test suite from depending on
somebody remembering an environment variable. They are 'setdefault', so an explicit setting in
the environment still wins -- which is how you would measure the contracted behaviour on purpose.

'pytest_configure' below is there for the same reason and for a related defect: GT4Py's build
cache key names neither these flags nor the backend, so one persistent cache directory shared
between backends serves a CPU-compiled program to a GPU run. It gives each backend its own
subdirectory; the flags still need one directory per flag set, which is what the name of
'GT4PY_BUILD_CACHE_DIR' in the run scripts records.
"""

import os
import pathlib
import re

import pytest


#: GCC/Clang for the 'gtfn_cpu' and 'dace_cpu' backends. Verified: flips section 1b from
#: one-rounding-off to bit-exact on all four dates.
os.environ.setdefault("CXXFLAGS", "-ffp-contract=off")

#: nvcc for the 'gtfn_gpu' and 'dace_gpu' backends. VERIFIED on 2026-08-28: section 1b's
#: 'compute_thermal_forcing' -- the 'a*b + c*d' whose rounding this decides -- is bit-exact on
#: both GPU backends, so the variable does reach nvcc. It reaches it by two different routes,
#: and neither is CMake's usual environment pickup alone:
#:
#: - 'gtfn_gpu' goes through GT4Py's CMake project, which lets CMake initialise CMAKE_CUDA_FLAGS
#:   from the environment as usual;
#: - 'dace_gpu' would not, because DaCe passes '-DCMAKE_CUDA_FLAGS' explicitly on its cmake
#:   command line (dace/codegen/targets/cuda.py) and would thereby override the environment.
#:   GT4Py reads CUDAFLAGS itself and writes it into DaCe's 'compiler.cuda.args'
#:   (gt4py/next/program_processors/runners/dace/workflow/common.py), which is also what
#:   displaces DaCe's default '--use_fast_math'.
#:
#: So the gate registry needs no per-backend entry. Note that setting this REPLACES DaCe's whole
#: CUDA argument string, '-O3' included; correctness is what these tests are for, not speed.
os.environ.setdefault("CUDAFLAGS", "--fmad=false")


def pytest_configure(config: pytest.Config) -> None:
    """Give each backend its own subdirectory of the persistent GT4Py build cache.

    GT4Py's build cache is keyed by the program alone. 'fingerprint_compilable_program'
    (gt4py/next/otf/stages.py) hashes the ITIR program, the offset provider and the column axis
    and nothing else -- not the target device, not the backend, not the compiler flags -- and
    every DaCe backend shares one 'translation_cache/' directory under 'BUILD_CACHE_DIR'. So with
    'GT4PY_BUILD_CACHE_LIFETIME=persistent' and one cache directory for all backends, the first
    backend to lower a program decides for every later one:

    - a 'dace_cpu' run writes an SDFG whose arrays carry 'StorageType.Default'. A later
      'dace_gpu' run reads it back, hands its GPU-resident CuPy arrays to a host-storage SDFG,
      and DaCe fails in argument marshalling with "'ndarray' object has no attribute
      '__array_interface__'";
    - a 'gtfn_cpu' run writes C++ that includes 'gridtools/fn/backend/naive.hpp'. A later
      'gtfn_gpu' run compiles that as host code and calls it with device pointers, which
      segfaults.

    Measured on 2026-08-28: 16 of 44 'dace_gpu' failures and the whole 'gtfn_gpu' SIGSEGV came
    from exactly this, from a cache directory a CPU run had warmed first.

    Scoping the directory by '--backend' here rather than in whatever script submits the job is
    the same choice as the two flags above: a passing test suite should not depend on somebody
    remembering to point 'GT4PY_BUILD_CACHE_DIR' somewhere different for each backend. Only the
    persistent cache needs this; the default session cache is a fresh temporary directory per
    interpreter and cannot collide.

    'gt4py.next.config.BUILD_CACHE_DIR' is read at each cache lookup but derived from
    'GT4PY_BUILD_CACHE_DIR' once, when 'gt4py.next.config' is imported -- which has already
    happened by the time any conftest runs -- so the module attribute is what has to be set.
    """
    from gt4py.next import config as gtx_config  # noqa: PLC0415 [import-outside-top-level]

    if gtx_config.BUILD_CACHE_LIFETIME is not gtx_config.BuildCacheLifetime.PERSISTENT:
        return

    backend = str(config.getoption("backend", "embedded"))
    # A backend may be named as 'path.to.module:symbol', which is not a directory name.
    scoped = pathlib.Path(gtx_config.BUILD_CACHE_DIR) / re.sub(r"[^0-9A-Za-z._-]", "_", backend)
    scoped.mkdir(parents=True, exist_ok=True)
    gtx_config.BUILD_CACHE_DIR = scoped
