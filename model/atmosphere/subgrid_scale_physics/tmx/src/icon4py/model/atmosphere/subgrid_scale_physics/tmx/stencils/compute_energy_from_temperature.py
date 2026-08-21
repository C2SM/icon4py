# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.physics.thermodynamics import _internal_energy
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_energy_from_temperature(
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
    use_internal_energy: bool,
) -> fa.CellKField[wpfloat]:
    """
    Compute the energy diffused by the tmx heat diffusion from the temperature.

    Port of 'temp_to_energy' (mo_vdf_atmo.f90):

    - dry static energy (``energy_type = 1``, ``use_internal_energy = False``,
      'compute_static_energy'):

          energy = cpd * temperature + grav * height_above_ground

    - internal energy + geopotential above ground (``energy_type = 2``,
      ``use_internal_energy = True``, 'compute_internal_energy'):

          energy = internal_energy(temperature, qv, qc + qr, qi + qs + qg,
                                   rho = 1, dz = 1)
                   + grav * height_above_ground * cvd / cpd

      with ``internal_energy`` from mo_aes_thermo.f90 (ported in
      :mod:`icon4py.model.common.physics.thermodynamics`). The moisture state
      passed by the caller selects between the old and the new (updated by the
      hydrometeor diffusion) tracers (``use_new_moisture_state`` in the
      Fortran); qr, qs and qg are not diffused and have no new state.

    ``use_internal_energy`` is a scalar configuration flag; it can be passed as
    a static (compile-time) argument so that only the selected variant is
    compiled.

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        temperature: air temperature at full levels [K]
        qv: specific humidity [kg/kg]
        qc: cloud water mixing ratio [kg/kg]
        qi: cloud ice mixing ratio [kg/kg]
        qr: rain mixing ratio [kg/kg]
        qs: snow mixing ratio [kg/kg]
        qg: graupel mixing ratio [kg/kg]
        height_above_ground: height of the full levels above the surface [m]
        grav: gravitational acceleration [m/s^2]
        use_internal_energy: True for internal energy, False for dry static energy

    Returns:
        energy at full levels [J/kg]
    """
    if use_internal_energy:
        one = broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim))
        q_liquid = qc + qr
        q_solid = qi + qs + qg
        energy = (
            _internal_energy(t=temperature, qv=qv, qliq=q_liquid, qice=q_solid, rho=one, dz=one)
            + grav * height_above_ground * PhysicsConstants.cvd / PhysicsConstants.cpd
        )
    else:
        energy = PhysicsConstants.cpd * temperature + grav * height_above_ground
    return energy


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_energy_from_temperature(
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    energy: fa.CellKField[wpfloat],
    grav: wpfloat,
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_energy_from_temperature(
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        height_above_ground=height_above_ground,
        grav=grav,
        use_internal_energy=use_internal_energy,
        out=energy,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
