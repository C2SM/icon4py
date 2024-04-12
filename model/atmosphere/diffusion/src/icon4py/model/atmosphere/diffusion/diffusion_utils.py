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

from gt4py.next import as_field
from gt4py.next.common import Dimension, Field, GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, minimum

from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.math.smagorinsky import _en_smag_fac_for_zero_nshift
from icon4py.model.common.settings import backend, xp


# TODO(Magdalena): fix duplication: duplicated from test testutils/utils.py
def zero_field(grid, *dims: Dimension, dtype=float):
    shapex = tuple(map(lambda x: grid.size[x], dims))
    return as_field(dims, xp.zeros(shapex, dtype=dtype))


@field_operator
def _identity_c_k(field: Field[[CellDim, KDim], float]) -> Field[[CellDim, KDim], float]:
    return field


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def copy_field(old_f: Field[[CellDim, KDim], float], new_f: Field[[CellDim, KDim], float]):
    _identity_c_k(old_f, out=new_f)


@field_operator
def _identity_e_k(field: Field[[EdgeDim, KDim], float]) -> Field[[EdgeDim, KDim], float]:
    return field


@field_operator
def _scale_k(field: Field[[KDim], float], factor: float) -> Field[[KDim], float]:
    return field * factor


@program(backend=backend)
def scale_k(field: Field[[KDim], float], factor: float, scaled_field: Field[[KDim], float]):
    _scale_k(field, factor, out=scaled_field)


@field_operator
def _init_zero_v_k() -> Field[[VertexDim, KDim], float]:
    return broadcast(0.0, (VertexDim, KDim))


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def init_zero_v_k(field: Field[[VertexDim, KDim], float]):
    _init_zero_v_k(out=field)


@field_operator
def _setup_smag_limit(diff_multfac_vn: Field[[KDim], float]) -> Field[[KDim], float]:
    return 0.125 - 4.0 * diff_multfac_vn


@field_operator
def _setup_runtime_diff_multfac_vn(k4: float, dyn_substeps: float) -> Field[[KDim], float]:
    con = 1.0 / 128.0
    dyn = k4 * dyn_substeps / 3.0
    return broadcast(minimum(con, dyn), (KDim,))


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


@program(backend=backend)
def setup_fields_for_initial_step(
    k4: float,
    hdiff_efdt_ratio: float,
    diff_multfac_vn: Field[[KDim], float],
    smag_limit: Field[[KDim], float],
):
    _setup_fields_for_initial_step(k4, hdiff_efdt_ratio, out=(diff_multfac_vn, smag_limit))


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


@program(backend=backend)
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


def init_nabla2_factor_in_upper_damping_zone(
    k_size: int, nrdmax: int32, nshift: int, physical_heights: Field[[KDim], float]
) -> Field[[KDim], float]:
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
    return as_field((KDim,), buffer)
