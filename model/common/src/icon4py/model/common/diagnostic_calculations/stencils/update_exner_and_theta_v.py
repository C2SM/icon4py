# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import exp, log

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_exner_and_theta_v(
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """Update exner and theta_v from a physics virtual-temperature tendency.

    Mirrors ICON's physics-to-dynamics thermodynamic update in
    ``mo_interface_iconam_aes.f90``: with the density held fixed by fast physics,
    recompute exner from the new virtual temperature via the exact equation of
    state and diagnose ``theta_v = Tv / exner``, so the exner/rho/theta_v trio
    stays EOS-consistent::

        Tv_new = Tv + dtime * dTv / dt
        exner_new = (rd / p0ref * rho * Tv_new) ** (rd / cpd)
        theta_v = Tv_new / exner_new

    Args:
        rho: air density [kg m-3]
        virtual_temperature: virtual temperature before the physics update [K]
        virtual_temperature_tendency: physics virtual-temperature tendency [K s-1]
        dtime: time step [s]
    Returns:
        (new exner function, new virtual potential temperature theta_v [K])
    """
    new_virtual_temperature = virtual_temperature + virtual_temperature_tendency * dtime
    new_exner = exp(
        PhysicsConstants.rd_o_cpd * log(PhysicsConstants.rd_o_p0ref * rho * new_virtual_temperature)
    )
    new_theta_v = new_virtual_temperature / new_exner
    return new_exner, new_theta_v


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_exner_and_theta_v(
    rho: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    dtime: ta.wpfloat,
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_exner_and_theta_v(
        rho=rho,
        virtual_temperature=virtual_temperature,
        virtual_temperature_tendency=virtual_temperature_tendency,
        dtime=dtime,
        out=(exner, theta_v),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
