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

import numpy as np
from gt4py.next.common import Dimension, Field
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import (  # noqa: A004 # import gt4py builtin
    abs,
    broadcast,
    int32,
    maximum,
    minimum,
)
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, Koff, VertexDim


def zero_field(grid, *dims: Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    if is_halfdim:
        assert len(shapex) == 2
        shapex = (shapex[0], shapex[1] + 1)
    return np_as_located_field(*dims)(np.zeros(shapex, dtype=dtype))


def indices_field(dim: Dimension, grid, is_halfdim, dtype=int):
    shapex = grid.size[dim] + 1 if is_halfdim else grid.size[dim]
    return np_as_located_field(dim)(np.arange(shapex, dtype=dtype))


def _allocate(*dims: Dimension, grid, is_halfdim=False, dtype=float):
    return zero_field(grid, *dims, is_halfdim=is_halfdim, dtype=dtype)


def _allocate_indices(*dims: Dimension, grid, is_halfdim=False, dtype=int32):
    return indices_field(*dims, grid=grid, is_halfdim=is_halfdim, dtype=dtype)


@field_operator
def _scale_k(field: Field[[KDim], float], factor: float) -> Field[[KDim], float]:
    return field * factor


@program
def scale_k(field: Field[[KDim], float], factor: float, scaled_field: Field[[KDim], float]):
    _scale_k(field, factor, out=scaled_field)


@field_operator
def _set_zero_v_k() -> Field[[VertexDim, KDim], float]:
    return broadcast(0.0, (VertexDim, KDim))


@program
def set_zero_v_k(field: Field[[VertexDim, KDim], float]):
    _set_zero_v_k(out=field)


@field_operator
def _set_zero_e_k() -> Field[[EdgeDim, KDim], float]:
    return broadcast(0.0, (EdgeDim, KDim))


@program
def set_zero_e_k(
    field: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_zero_e_k(
        out=field,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _set_zero_c_k() -> Field[[CellDim, KDim], float]:
    return broadcast(0.0, (CellDim, KDim))


@program
def set_zero_c_k(
    field: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _set_zero_c_k(
        out=field,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]) -> Field[[KDim], float]:
    return 0.125 - 4.0 * diff_multfac_vn


@field_operator
def _setup_initial_diff_multfac_vn(k4: float, hdiff_efdt_ratio: float) -> Field[[KDim], float]:
    return broadcast(k4 / 3.0 * hdiff_efdt_ratio, (KDim,))


@field_operator
def _setup_fields_for_initial_step(
    k4: float, hdiff_efdt_ratio: float
) -> Tuple[Field[[KDim], float], Field[[KDim], float]]:
    diff_multfac_vn = _setup_initial_diff_multfac_vn(k4, hdiff_efdt_ratio)
    smag_limit = _setup_smag_limit(diff_multfac_vn)
    return diff_multfac_vn, smag_limit


@program
def setup_fields_for_initial_step(
    k4: float,
    hdiff_efdt_ratio: float,
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
):
    _setup_fields_for_initial_step(k4, hdiff_efdt_ratio, out=(diff_multfac_vn, smag_limit))


@field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: Field[[KDim], float], nudge_max_coeff: float, dbl_eps: float
) -> Field[[KDim], float]:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


# TODO (magdalena) copied from diffusion/diffusion_utils/_en_smag_fac_for_zero_nshift makes sense?
@field_operator
def _en_smag_fac_for_zero_nshift(
    vect_a: Field[[KDim], float],
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
) -> Field[[KDim], float]:
    dz21 = hdiff_smag_z2 - hdiff_smag_z
    alin = (hdiff_smag_fac2 - hdiff_smag_fac) / dz21
    df32 = hdiff_smag_fac3 - hdiff_smag_fac2
    df42 = hdiff_smag_fac4 - hdiff_smag_fac2
    dz32 = hdiff_smag_z3 - hdiff_smag_z2
    dz42 = hdiff_smag_z4 - hdiff_smag_z2

    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (vect_a + vect_a(Koff[1]))

    dzlin = minimum(dz21, maximum(0.0, zf - hdiff_smag_z))
    dzqdr = minimum(dz42, maximum(0.0, zf - hdiff_smag_z2))
    enh_smag_fac = hdiff_smag_fac + (dzlin * alin) + dzqdr * (aqdr + dzqdr * bqdr)
    return enh_smag_fac


@field_operator
def _compute_z_raylfac(rayleigh_w: Field[[KDim], float], dtime: float) -> Field[[KDim], float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@program
def compute_z_raylfac(
    rayleigh_w: Field[[KDim], float], dtime: float, z_raylfac: Field[[KDim], float]
):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)


@field_operator
def _scal_divdamp_calcs(
    enh_divdamp_fac: Field[[KDim], float], mean_cell_area: float
) -> Field[[KDim], float]:
    return -enh_divdamp_fac * mean_cell_area**2.0


@program
def scal_divdamp_calcs(
    enh_divdamp_fac: Field[[KDim], float],
    out: Field[[KDim], float],
    mean_cell_area: float,
):
    _scal_divdamp_calcs(enh_divdamp_fac, mean_cell_area, out=out)
