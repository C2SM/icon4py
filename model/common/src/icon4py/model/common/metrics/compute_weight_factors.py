# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from types import ModuleType

import gt4py.next as gtx
import numpy as np
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.experimental import concat_where
from gt4py.next.program_processors.runners.gtfn import run_gtfn

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


@field_operator
def _compute_wgtfac_c_nlev(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_0(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[+1]) - z_ifc) / (z_ifc(Koff[+2]) - z_ifc)
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c_inner(
    z_ifc: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    z_wgtfac_c = (z_ifc(Koff[-1]) - z_ifc) / (z_ifc(Koff[-1]) - z_ifc(Koff[+1]))
    return z_wgtfac_c


@field_operator
def _compute_wgtfac_c(
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    wgt_fac_c = concat_where(
        (0 < dims.KDim) & (dims.KDim < nlev), _compute_wgtfac_c_inner(z_ifc), z_ifc
    )
    wgt_fac_c = concat_where(dims.KDim == 0, _compute_wgtfac_c_0(z_ifc=z_ifc), wgt_fac_c)
    wgt_fac_c = concat_where(dims.KDim == nlev, _compute_wgtfac_c_nlev(z_ifc=z_ifc), wgt_fac_c)

    return wgt_fac_c


# TODO(halungge): missing test?
@program(grid_type=GridType.UNSTRUCTURED, backend=run_gtfn)
def compute_wgtfac_c(
    wgtfac_c: fa.CellKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
):
    _compute_wgtfac_c(
        z_ifc,
        nlev,
        out=wgtfac_c,
    )


def _compute_z1_z2_z3(
    z_ifc: data_alloc.NDArray, i1: int, i2: int, i3: int, i4: int
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    z1 = 0.5 * (z_ifc[:, i2] - z_ifc[:, i1])
    z2 = 0.5 * (z_ifc[:, i2] + z_ifc[:, i3]) - z_ifc[:, i1]
    z3 = 0.5 * (z_ifc[:, i3] + z_ifc[:, i4]) - z_ifc[:, i1]
    return z1, z2, z3


def compute_wgtfacq_c_dsl(
    z_ifc: data_alloc.NDArray,
    nlev: int,
    array_ns: ModuleType = np,
) -> data_alloc.NDArray:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        z_ifc: Field[CellDim, KDim] (half levels), geometric height at the vertical interface of cells.
        nlev: int, last k level
    Returns:
    Field[CellDim, KDim] (full levels)
    """
    wgtfacq_c = array_ns.zeros((z_ifc.shape[0], nlev + 1))
    wgtfacq_c_dsl = array_ns.zeros((z_ifc.shape[0], nlev))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)

    wgtfacq_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    wgtfacq_c[:, 1] = (z1 - wgtfacq_c[:, 2] * (z1 - z3)) / (z1 - z2)
    wgtfacq_c[:, 0] = 1.0 - (wgtfacq_c[:, 1] + wgtfacq_c[:, 2])

    wgtfacq_c_dsl[:, nlev - 1] = wgtfacq_c[:, 0]
    wgtfacq_c_dsl[:, nlev - 2] = wgtfacq_c[:, 1]
    wgtfacq_c_dsl[:, nlev - 3] = wgtfacq_c[:, 2]

    return wgtfacq_c_dsl


def compute_wgtfacq_e_dsl(
    e2c,
    z_ifc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    wgtfacq_c_dsl: data_alloc.NDArray,
    n_edges: int,
    nlev: int,
    array_ns: ModuleType = np,
):
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        e2c: Edge to Cell offset
        z_ifc: geometric height at the vertical interface of cells.
        wgtfacq_c_dsl: weighting factor for quadratic interpolation to surface
        c_lin_e: interpolation field
        n_edges: number of edges
        nlev: int, last k level
    Returns:
    Field[EdgeDim, KDim] (full levels)
    """
    wgtfacq_e_dsl = array_ns.zeros(shape=(n_edges, nlev + 1))
    z_aux_c = array_ns.zeros((z_ifc.shape[0], 6))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)
    z_aux_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 1] = (z1 - wgtfacq_c_dsl[:, nlev - 3] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 0] = 1.0 - (wgtfacq_c_dsl[:, nlev - 2] + wgtfacq_c_dsl[:, nlev - 3])

    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, 0, 1, 2, 3)
    z_aux_c[:, 5] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 4] = (z1 - z_aux_c[:, 5] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 3] = 1.0 - (z_aux_c[:, 4] + z_aux_c[:, 5])

    c_lin_e = c_lin_e[:, :, array_ns.newaxis]
    z_aux_e = array_ns.sum(c_lin_e * z_aux_c[e2c], axis=1)

    wgtfacq_e_dsl[:, nlev] = z_aux_e[:, 0]
    wgtfacq_e_dsl[:, nlev - 1] = z_aux_e[:, 1]
    wgtfacq_e_dsl[:, nlev - 2] = z_aux_e[:, 2]

    return wgtfacq_e_dsl
