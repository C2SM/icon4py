# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained @py2fgen.export-decorated test functions.

Stand-in for ``icon4py.bindings.simple`` used by the py2fgen test suite so
that ``tools/tests/`` no longer reaches into the ``bindings`` package.
The Fortran driver (``fortran_samples/test_square.f90``) only checks that
``square_from_function`` squares a 2-D float64 array and returns 1, so a
plain ``np.square`` body is sufficient.
"""

from __future__ import annotations

import numpy as np

from icon4py.tools import py2fgen


_FLOAT64_2D = py2fgen.ArrayParamDescriptor(
    rank=2,
    dtype=py2fgen.FLOAT64,
    memory_space=py2fgen.MemorySpace.HOST,
    is_optional=False,
)


# A bare ``annotation_mapping_hook`` that always returns ``None`` — this falls
# back to ``_conversion.default_mapping``, which translates the raw
# ``ArrayInfo`` tuple py2fgen passes from CFFI into a NumPy view.
def _default_only(_: object, __: py2fgen.ParamDescriptor) -> None:
    return None


@py2fgen.export(
    param_descriptors={"inp": _FLOAT64_2D, "result": _FLOAT64_2D},
    annotation_mapping_hook=_default_only,
)
def square_from_function(inp: np.ndarray, result: np.ndarray) -> None:
    np.square(inp, out=result)


@py2fgen.export(
    param_descriptors={"inp": _FLOAT64_2D, "result": _FLOAT64_2D},
    annotation_mapping_hook=_default_only,
)
def square_error(inp: np.ndarray, result: np.ndarray) -> None:
    raise Exception("Exception foo occurred")
