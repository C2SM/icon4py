# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal, TypeAlias

import gt4py.next as gtx


DEFAULT_PRECISION = "double"

wpfloat: TypeAlias = gtx.float64  # noqa: UP040
vpfloat: type[gtx.float32] | type[gtx.float64] = wpfloat
type anyfloat = gtx.float32 | gtx.float64

precision = os.environ.get("FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed", "single"]) -> None:
    global precision  # noqa: PLW0603 [global-statement]
    global vpfloat  # noqa: PLW0603 [global-statement]
    global wpfloat  # noqa: PLW0603 [global-statement]

    precision = new_precision.lower()
    match precision:
        case "double":
            vpfloat = wpfloat
        case "mixed":
            vpfloat = gtx.float32
        case "single":
            vpfloat = gtx.float32
            wpfloat = gtx.float32
        case _:
            raise ValueError("Only 'double', 'mixed' and 'single' precision are supported.")


set_precision(precision)

#: Precision of the quantities the ICON Fortran declares REAL(sp) while the rest of the
#: computation is REAL(wp): ICON's 'lsq_error' (mo_intp_data_strc.f90 83) and, in the WENO
#: schemes of A. Jocksch's icon-exclaim branch (mo_advection_hflux.f90), the smoothness path
#: `zlc, z_lsq_smooth, area` (2643, 3246, 3925) and the hybrid's fit residual `lsqe` (3246).
#: Decision of 2026-09-10: the working precision until the single-precision option (PR #970)
#: is merged; set it to gtx.float32 to reproduce the Fortran's single-precision arithmetic.
#: Init-time numpy code stays float64 regardless; the cast happens where the fields are built.
fortran_sp_float: type[gtx.float32] | type[gtx.float64] = wpfloat
#: Kind of the Fortran's unsuffixed real literals (e.g. `5e-5`, `1e-10` at mo_advection_hflux.f90
#: 3574): always REAL(sp), independent of the REAL(sp) *variables* fortran_sp_float stands in for.
fortran_sp_literal: type[gtx.float32] = gtx.float32


def dataclass_scalars_to_wp(self, attributes: list[str] | None = None):
    for name in attributes or []:
        if not isinstance(v := object.__getattribute__(self, name), wpfloat):
            object.__setattr__(self, name, wpfloat(v))
