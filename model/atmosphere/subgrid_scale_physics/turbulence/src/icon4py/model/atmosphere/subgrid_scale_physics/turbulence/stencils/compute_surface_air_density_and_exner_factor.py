# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Air density and Exner factor at the lower boundary of the Prandtl layer.

Translated from 'icon/src/atm_phy_schemes/turb_vertdiff.f90', SUBROUTINE 'vertdiff'
(:536-542 at icon commit 26d6b98cce), the loop headed "Berechnung der Luftdichte und des
Exner-Faktors am Unterrand" -- "calculation of the air density and of the Exner factor at the
lower boundary". The scientific commentary in that file is by Matthias Raschendorfer (DWD).

    virt = 1 + rvd_m_o*qv_s(i)
    rhon(i,ke1) = ps(i)/(r_d*virt*t_g(i))
    eprs(i,ke1) = zexner(ps(i))

One Fortran loop, one program: the two statements share the surface pressure and the row they
write, and the row is one deep, so a second kernel launch would be nearly all overhead.

'rhon' is 'INTENT(INOUT)' and this is the only place 'vertdiff' writes it; every other row is
what 'turbdiff' left. The Fortran's own note is worth carrying: in the turbulence model
'rhon(:,ke1)' belongs to the lower boundary of the Prandtl layer rather than to the surface
level, but 'vert_grad_diff' uses it as a surface-level value.

'eprs' is declared '(nvec, ke1:ke1)' -- a one-level slab -- and is kept a full half-level field
here, written on the surface row only, so that its consumers can read it without a rank change.
"""

import gt4py.next as gtx
from gt4py.next import broadcast, exp, log

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_surface_air_density(
    surface_pressure: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'rhon(:,ke1)': the ideal-gas density of moist air at the ground [kg/m3].

    The virtual factor is written and multiplied exactly as the Fortran does -- 'r_d*virt*t_g'
    associates to the left, and the quotient is formed once -- because any regrouping of the
    denominator is a rounding this port would not be able to explain away.
    """
    virtual_factor = wpfloat("1.0") + ThermoConstants.RVD_M_O * surface_specific_humidity
    return broadcast(
        surface_pressure / (ThermoConstants.RD * virtual_factor * surface_temperature),
        (dims.CellDim, dims.KDim),
    )


@gtx.field_operator
def _compute_surface_exner_factor(
    surface_pressure: fa.CellField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """'eprs(:,ke1) = zexner(ps)': the Exner factor '(p_s/p0)**(R_d/c_pd)' [-].

    'EXP(rdocp*LOG(...))' rather than '**', as 'thermodynamic_functions._exner_factor' explains
    -- and as the Fortran itself writes it. This is the one transcendental in 'vertdiff'.
    """
    return broadcast(
        exp(ThermoConstants.RDOCP * log(surface_pressure / ThermoConstants.P0REF)),
        (dims.CellDim, dims.KDim),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_air_density_and_exner_factor(
    surface_pressure: fa.CellField[wpfloat],
    surface_specific_humidity: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    air_density: fa.CellKField[wpfloat],
    surface_exner_factor: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Write the surface row of the half-level air density and of the Exner factor.

    Args:
        surface_pressure: 'ps' [Pa].
        surface_specific_humidity: 'qv_s' [kg/kg].
        surface_temperature: 't_g', the weighted surface temperature [K].
        air_density: In-out, 'rhon' on half levels [kg/m3]; only the surface row is written.
        surface_exner_factor: Output, 'eprs' [-]; only the surface row is written.
        horizontal_start: First column, 'ivstart'.
        horizontal_end: End of the columns, 'ivend'.
        vertical_start: The surface half level, 'nlev' (Fortran 'ke1').
        vertical_end: 'nlev + 1'.
    """
    _compute_surface_air_density(
        surface_pressure=surface_pressure,
        surface_specific_humidity=surface_specific_humidity,
        surface_temperature=surface_temperature,
        out=air_density,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _compute_surface_exner_factor(
        surface_pressure=surface_pressure,
        out=surface_exner_factor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
