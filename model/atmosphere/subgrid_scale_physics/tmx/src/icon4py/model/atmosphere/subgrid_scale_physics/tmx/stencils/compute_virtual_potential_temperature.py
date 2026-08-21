# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import power

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_virtual_potential_temperature(
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Compute the virtual potential temperature at full-level cell centers.

    Port of ``get_virtual_potential_temperature`` in ICON's ``mo_vdf_atmo.f90``:

        theta_v = tv * (p0ref / p)**rd_o_cpd

    with tv the virtual temperature (``ptvm1``) and p the pressure (``papm1``).
    The constants match the Fortran ``mo_physical_constants.f90`` values:
    ``p0ref = 100000.0`` Pa, ``rd_o_cpd = rd / cpd = 287.04 / 1004.64``
    (available as ``PhysicsConstants.p0ref`` / ``PhysicsConstants.rd_o_cpd``).

    The tmx call site (``Compute_diagnostics`` in ``mo_vdf_atmo.f90``) uses
    ``rl_start = 3``, ``rl_end = min_rlcell_int``, which maps to the horizontal
    domain ``(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3, h_grid.Zone.LOCAL)``, and
    all full levels (Fortran jk = 1..nlev -> k = 0..nlev-1).

    Args:
        virtual_temperature: virtual temperature at full levels [K]
        pressure: air pressure at full levels [Pa]

    Returns:
        virtual potential temperature at full levels [K]
    """
    return virtual_temperature * power(PhysicsConstants.p0ref / pressure, PhysicsConstants.rd_o_cpd)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_virtual_potential_temperature(
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_virtual_potential_temperature(
        virtual_temperature=virtual_temperature,
        pressure=pressure,
        out=theta_v,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
