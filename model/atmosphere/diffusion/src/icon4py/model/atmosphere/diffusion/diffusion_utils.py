# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Tuple

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import broadcast, minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim, VertexDim
from icon4py.model.common.math.smagorinsky import _en_smag_fac_for_zero_nshift
from icon4py.model.common.settings import backend


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


@gtx.field_operator
def _init_nabla2_factor_in_upper_damping_zone(
    physical_heights: fa.KField[float],
    k_field: fa.KField[gtx.int32],
    nrdmax: gtx.int32,
    nshift: gtx.int32,
    heights_nrd_shift: float,
    heights_1: float,
) -> fa.KField[float]:
    height_sliced = where(
        (k_field >= (1 + nshift)) & (k_field < (nshift + nrdmax + 1)), physical_heights, 0.0
    )
    diff_multfac_n2w = (
        1.0 / 12.0 * ((height_sliced - heights_nrd_shift) / (heights_1 - heights_nrd_shift)) ** 4
    )
    return diff_multfac_n2w


@gtx.program
def init_nabla2_factor_in_upper_damping_zone(
    physical_heights: fa.KField[float],
    k_field: fa.KField[gtx.int32],
    diff_multfac_n2w: fa.KField[float],
    nrdmax: gtx.int32,
    nshift: gtx.int32,
    heights_nrd_shift: float,
    heights_1: float,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Calculate diff_multfac_n2w.

    numpy version, since gt4py does not allow non-constant indexing into fields

    Args
        physcial_heights: vector of physical heights [m] of the height levels
        k_field: field of k levels
        nrdmax: index of the level where rayleigh damping starts
        nshift: 0
        heights_nrd_shift: physcial_heights at nrdmax + nshift + 1,
        heights_1: physcial_heights at 1st level,
        vertical_start: vertical lower bound,
        vertical_end: vertical upper bound,
    """
    _init_nabla2_factor_in_upper_damping_zone(
        physical_heights=physical_heights,
        k_field=k_field,
        nrdmax=nrdmax,
        nshift=nshift,
        heights_nrd_shift=heights_nrd_shift,
        heights_1=heights_1,
        out=diff_multfac_n2w,
        domain={dims.KDim: (vertical_start, vertical_end)},
    )
