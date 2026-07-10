# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter stencils of the TmxState (dycore -> tmx input translation).

``compute_air_mass``: mair = rho * dz (``diag%airmass_new`` bound to
``field%mair`` in mo_interface_iconam_aes.f90; shallow atmosphere).
``compute_cv_air``: port of ``get_cvair`` in mo_aes_phy_diag.f90.
"""

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_air_mass(
    rho: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    return rho * ddqz_z_full


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_air_mass(
    rho: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_air_mass(
        rho=rho,
        ddqz_z_full=ddqz_z_full,
        out=air_mass,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_cv_air(
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    # locals, not module globals: see tmx/docs/gt4py_patterns.md (gtfn backend)
    cvd = PhysicsConstants.cvd
    cvv = PhysicsConstants.cvv
    clw = PhysicsConstants.cpl
    ci = PhysicsConstants.cpi
    qliq = qc + qr
    qice = qi + qs + qg
    qtot = qv + qliq + qice
    cv = cvd * (wpfloat(1.0) - qtot) + cvv * qv + clw * qliq + ci * qice
    return cv * air_mass


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cv_air(
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_cv_air(
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        air_mass=air_mass,
        out=cv_air,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _apply_tendency_on_edge_k(
    field_a: fa.EdgeKField[wpfloat],
    coeff: wpfloat,
    field_b: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    return field_a + coeff * field_b


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_tendency_on_edge_k(
    field_a: fa.EdgeKField[wpfloat],
    coeff: wpfloat,
    field_b: fa.EdgeKField[wpfloat],
    output_field: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_tendency_on_edge_k(
        field_a=field_a,
        coeff=coeff,
        field_b=field_b,
        out=output_field,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
