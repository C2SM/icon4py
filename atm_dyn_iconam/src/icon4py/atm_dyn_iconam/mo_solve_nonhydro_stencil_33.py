# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, broadcast

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_33() -> tuple[
    Field[[EdgeDim, KDim], float], Field[[EdgeDim, KDim], float]
]:
    vn_traj = broadcast(0.0, (EdgeDim, KDim))
    mass_flx_me = broadcast(0.0, (EdgeDim, KDim))
    return vn_traj, mass_flx_me


@field_operator
def _mo_solve_nonhydro_stencil_33_vn_traj() -> Field[[EdgeDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_33()[0]


@field_operator
def _mo_solve_nonhydro_stencil_33_mass_flx_me() -> Field[[EdgeDim, KDim], float]:
    return _mo_solve_nonhydro_stencil_33()[1]


@program
def mo_solve_nonhydro_stencil_33(
    vn_traj: Field[[EdgeDim, KDim], float],
    mass_flx_me: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_33_vn_traj(out=vn_traj)
    _mo_solve_nonhydro_stencil_33_mass_flx_me(out=mass_flx_me)
