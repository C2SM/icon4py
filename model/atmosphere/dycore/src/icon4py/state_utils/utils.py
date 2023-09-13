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
from gt4py.next.ffront.fbuiltins import abs, broadcast, int32, maximum, minimum
from gt4py.next.iterator.embedded import np_as_located_field

from icon4py.common.dimension import CellDim, EdgeDim, KDim, Koff, VertexDim


# TODO [@Magdalena] fix duplication: duplicated from test testutils/utils.py
def zero_field(mesh, *dims: Dimension, is_halfdim=False, dtype=float):
    shapex = tuple(map(lambda x: mesh.size[x], dims))
    shapex = tuple({shapex[0], shapex[1] + 1}) if is_halfdim else shapex
    return np_as_located_field(*dims)(np.zeros(shapex, dtype=dtype))


@field_operator
def _identity_c_k(
    field: Field[[CellDim, KDim], float]
) -> Field[[CellDim, KDim], float]:
    return field


@program
def copy_field(
    old_f: Field[[CellDim, KDim], float], new_f: Field[[CellDim, KDim], float]
):
    _identity_c_k(old_f, out=new_f)


@field_operator
def _identity_e_k(
    field: Field[[EdgeDim, KDim], float]
) -> Field[[EdgeDim, KDim], float]:
    return field


def indices_field(dim: Dimension, mesh, is_halfdim, dtype=int):
    shapex = mesh.size[dim] + 1 if is_halfdim else mesh.size[dim]
    return np_as_located_field(dim)(np.arange(shapex, dtype=dtype))


def _allocate(*dims: Dimension, mesh, is_halfdim=False, dtype=float):
    return zero_field(mesh, *dims, is_halfdim=is_halfdim, dtype=dtype)


def _allocate_indices(*dims: Dimension, mesh, is_halfdim=False, dtype=int32):
    return indices_field(*dims, mesh=mesh, is_halfdim=is_halfdim, dtype=dtype)


@program
def copy_diagnostic_and_prognostics(
    hdef_ic_new: Field[[CellDim, KDim], float],
    hdef_ic: Field[[CellDim, KDim], float],
    div_ic_new: Field[[CellDim, KDim], float],
    div_ic: Field[[CellDim, KDim], float],
    dwdx_new: Field[[CellDim, KDim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy_new: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
    vn_new: Field[[EdgeDim, KDim], float],
    vn: Field[[EdgeDim, KDim], float],
    w_new: Field[[CellDim, KDim], float],
    w: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
):
    _identity_c_k(hdef_ic_new, out=hdef_ic)
    _identity_c_k(div_ic_new, out=div_ic)
    _identity_c_k(dwdx_new, out=dwdx)
    _identity_c_k(dwdy_new, out=dwdy)
    _identity_e_k(vn_new, out=vn)
    _identity_c_k(w_new, out=w)
    _identity_c_k(exner_new, out=exner)
    _identity_c_k(theta_v_new, out=theta_v)


@field_operator
def _scale_k(field: Field[[KDim], float], factor: float) -> Field[[KDim], float]:
    return field * factor


@program
def scale_k(
    field: Field[[KDim], float], factor: float, scaled_field: Field[[KDim], float]
):
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
def _set_bool_c_k() -> Field[[CellDim, KDim], bool]:
    return broadcast(False, (CellDim, KDim))


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]) -> Field[[KDim], float]:
    return 0.125 - 4.0 * diff_multfac_vn


@field_operator
def _setup_runtime_diff_multfac_vn(
    k4: float, dyn_substeps: float
) -> Field[[KDim], float]:
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


@field_operator
def _setup_initial_diff_multfac_vn(
    k4: float, hdiff_efdt_ratio: float
) -> Field[[KDim], float]:
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
    _setup_fields_for_initial_step(
        k4, hdiff_efdt_ratio, out=(diff_multfac_vn, smag_limit)
    )


@field_operator
def _calculate_bdy_divdamp(
    scal_divdamp: Field[[KDim], float], nudge_max_coeff: float, dbl_eps: float
) -> Field[[KDim], float]:
    return 0.75 / (nudge_max_coeff + dbl_eps) * abs(scal_divdamp)


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
    vect_a: Field[[KDim], float],
) -> tuple[Field[[KDim], float], Field[[KDim], float], Field[[KDim], float]]:
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


@program
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
    vect_a: Field[[KDim], float],
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
    enh_smag_fac: Field[[KDim], float],
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


@field_operator
def _compute_z_raylfac(
    rayleigh_w: Field[[KDim], float], dtime: float
) -> Field[[KDim], float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@program
def compute_z_raylfac(
    rayleigh_w: Field[[KDim], float], dtime: float, z_raylfac: Field[[KDim], float]
):
    _compute_z_raylfac(rayleigh_w, dtime, out=z_raylfac)


def init_nabla2_factor_in_upper_damping_zone(
    k_size: int, nrdmax: int32, nshift: int, physical_heights: np.ndarray
) -> Field[[KDim], float]:
    """
    Calculate diff_multfac_n2w.

    numpy version, since gt4py does not allow non-constant indexing into fields
    TODO: @magdalena fix this once IndexedFields are implemented

    Args
        k_size: number of vertical levels
        nrdmax: index of the level where rayleigh dampint starts
        nshift:
        physcial_heights: vector of physical heights [m] of the height levels
    """
    buffer = np.zeros(k_size)
    buffer[1 : nrdmax + 1] = (
        1.0
        / 12.0
        * (
            (
                physical_heights[1 + nshift : nrdmax + 1 + nshift]
                - physical_heights[nshift + nrdmax + 1]
            )
            / (physical_heights[1] - physical_heights[nshift + nrdmax + 1])
        )
        ** 4
    )
    return np_as_located_field(KDim)(buffer)


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
