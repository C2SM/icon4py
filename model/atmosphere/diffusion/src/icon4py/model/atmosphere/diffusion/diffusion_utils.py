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
from typing import Tuple

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import (
    broadcast,
    minimum,
)

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.common.math.smagorinsky import _en_smag_fac_for_zero_nshift
from icon4py.model.common.settings import backend, xp


@gtx.field_operator
def _identity_c_k(field: fa.CellKField[float]) -> fa.CellKField[float]:
    return field


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def copy_field(old_f: fa.CellKField[float], new_f: fa.CellKField[float]):
    _identity_c_k(old_f, out=new_f)


@gtx.field_operator
def _identity_e_k(field: fa.EdgeKField[float]) -> fa.EdgeKField[float]:
    return field


@gtx.field_operator
def _scale_k(field: fa.KField[float], factor: float) -> fa.KField[float]:
    return field * factor


@gtx.program(backend=backend)
def scale_k(field: fa.KField[float], factor: float, scaled_field: fa.KField[float]):
    _scale_k(field, factor, out=scaled_field)


@gtx.field_operator
def _init_zero_v_k() -> gtx.Field[[dims.VertexDim, dims.KDim], float]:
    return broadcast(0.0, (VertexDim, KDim))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def init_zero_v_k(field: gtx.Field[[dims.VertexDim, dims.KDim], float]):
    _init_zero_v_k(out=field)


@gtx.field_operator
def _setup_smag_limit(diff_multfac_vn: fa.KField[float]) -> fa.KField[float]:
    return 0.125 - 4.0 * diff_multfac_vn


@gtx.field_operator
def _setup_runtime_diff_multfac_vn(k4: float, dyn_substeps: float) -> fa.KField[float]:
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


@gtx.field_operator
def _setup_initial_diff_multfac_vn(k4: float, hdiff_efdt_ratio: float) -> fa.KField[float]:
    return broadcast(k4 / 3.0 * hdiff_efdt_ratio, (KDim,))


@gtx.field_operator
def _setup_fields_for_initial_step(
    k4: float, hdiff_efdt_ratio: float
) -> Tuple[fa.KField[float], fa.KField[float]]:
    diff_multfac_vn = _setup_initial_diff_multfac_vn(k4, hdiff_efdt_ratio)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    return diff_multfac_vn, smag_limit


@gtx.program(backend=backend)
def setup_fields_for_initial_step(
    k4: float,
    hdiff_efdt_ratio: float,
    diff_multfac_vn: fa.KField[float],
    smag_limit: fa.KField[float],
):
    _setup_fields_for_initial_step(k4, hdiff_efdt_ratio, out=(diff_multfac_vn, smag_limit))


@gtx.field_operator
def _init_diffusion_local_fields_for_regular_timestemp(
    k4: float,
    dyn_substeps: float,
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: fa.KField[float],
) -> tuple[fa.KField[float], fa.KField[float], fa.KField[float]]:
    diff_multfac_vn = _setup_runtime_diff_multfac_vn(k4, dyn_substeps)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    enh_smag_fac = _en_smag_fac_for_zero_nshift(
        vect_a,
        hdiff_smag_fac,
        hdiff_smag_fac2,
        hdiff_smag_fac3,
        hdiff_smag_fac4,
        hdiff_smag_z,
        hdiff_smag_z2,
        hdiff_smag_z3,
        hdiff_smag_z4,
    )
    return (
        diff_multfac_vn,
        smag_limit,
        enh_smag_fac,
    )


@gtx.program(backend=backend)
def init_diffusion_local_fields_for_regular_timestep(
    k4: float,
    dyn_substeps: float,
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    vect_a: fa.KField[float],
    diff_multfac_vn: fa.KField[float],
    smag_limit: fa.KField[float],
    enh_smag_fac: fa.KField[float],
):
    _init_diffusion_local_fields_for_regular_timestemp(
        k4,
        dyn_substeps,
        hdiff_smag_fac,
        hdiff_smag_fac2,
        hdiff_smag_fac3,
        hdiff_smag_fac4,
        hdiff_smag_z,
        hdiff_smag_z2,
        hdiff_smag_z3,
        hdiff_smag_z4,
        vect_a,
        out=(
            diff_multfac_vn,
            smag_limit,
            enh_smag_fac,
        ),
    )


def init_nabla2_factor_in_upper_damping_zone(
    k_size: int, nrdmax: gtx.int32, nshift: int, physical_heights: fa.KField[float]
) -> fa.KField[float]:
    """
    Calculate diff_multfac_n2w.

    numpy version, since gt4py does not allow non-constant indexing into fields

    Args
        k_size: number of vertical levels
        nrdmax: index of the level where rayleigh dampint starts
        nshift:
        physcial_heights: vector of physical heights [m] of the height levels
    """
    # TODO(Magdalena): fix with as_offset in gt4py
    heights = physical_heights.ndarray
    buffer = xp.zeros(k_size)
    buffer[1 : nrdmax + 1] = (
        1.0
        / 12.0
        * (
            (heights[1 + nshift : nrdmax + 1 + nshift] - heights[nshift + nrdmax + 1])
            / (heights[1] - heights[nshift + nrdmax + 1])
        )
        ** 4
    )
    return gtx.as_field((dims.KDim,), buffer)
