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
"""

import os


#: GCC/Clang for the 'gtfn_cpu' and 'dace_cpu' backends. Verified: flips section 1b from
#: one-rounding-off to bit-exact on all four dates.
os.environ.setdefault("CXXFLAGS", "-ffp-contract=off")

#: nvcc for the 'gtfn_gpu' and 'dace_gpu' backends. NOT yet verified -- 'cupy' is not installed
#: in this environment, so no GPU backend has been run. If a GPU run shows section 1b failing its
#: Exact() gate while its 'up_to_one_contraction' test still passes, this variable is not reaching
#: nvcc (DaCe in particular may configure CUDA compilation through its own settings) and the gate
#: registry needs a per-backend entry instead.
os.environ.setdefault("CUDAFLAGS", "--fmad=false")
