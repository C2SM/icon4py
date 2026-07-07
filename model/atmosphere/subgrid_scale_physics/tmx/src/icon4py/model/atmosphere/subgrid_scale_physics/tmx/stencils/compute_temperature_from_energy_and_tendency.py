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
from icon4py.model.common.physics.thermodynamics import _T_from_internal_energy
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_temperature_from_energy_and_tendency(
    energy: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """
    Compute the new temperature from the diffused energy and the resulting
    temperature tendency.

    Port of 'energy_to_temp' (mo_vdf_atmo.f90) and the final tendency loop of
    'Compute_diffusion_temperature' (mo_vdf.f90):

    - dry static energy (``energy_type = 1``, ``use_internal_energy = False``,
      'compute_temp_from_static_energy'):

          new_temperature = (energy - grav * height_above_ground) / cpd

    - internal energy + geopotential above ground (``energy_type = 2``,
      ``use_internal_energy = True``,
      'compute_temperature_from_internal_energy'):

          u               = energy - grav * height_above_ground * cvd / cpd
          new_temperature = T_from_internal_energy(u, qv, qc + qr,
                                                   qi + qs + qg,
                                                   rho = 1, dz = 1)

      with ``T_from_internal_energy`` from mo_aes_thermo.f90 (ported in
      :mod:`icon4py.model.common.physics.thermodynamics`). The Fortran call
      site uses the *new* moisture state (``use_new_moisture_state=.TRUE.``,
      the tracers updated by the hydrometeor diffusion); qr, qs and qg are not
      diffused and have no new state.

    In both cases the temperature tendency is

          tend_temperature = (new_temperature - temperature) / dtime

    ``use_internal_energy`` is a scalar configuration flag; it can be passed as
    a static (compile-time) argument so that only the selected variant is
    compiled.

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        energy: diffused (new) energy at full levels [J/kg]
        temperature: air temperature before the diffusion at full levels [K]
        qv: new specific humidity [kg/kg]
        qc: new cloud water mixing ratio [kg/kg]
        qi: new cloud ice mixing ratio [kg/kg]
        qr: rain mixing ratio [kg/kg]
        qs: snow mixing ratio [kg/kg]
        qg: graupel mixing ratio [kg/kg]
        height_above_ground: height of the full levels above the surface [m]
        grav: gravitational acceleration [m/s^2]
        dtime: time step [s]
        use_internal_energy: True for internal energy, False for dry static energy

    Returns:
        (new temperature, temperature tendency) at full levels
    """
    if use_internal_energy:
        one = broadcast(wpfloat("1.0"), (dims.CellDim, dims.KDim))
        q_liquid = qc + qr
        q_solid = qi + qs + qg
        u = energy - grav * height_above_ground * PhysicsConstants.cvd / PhysicsConstants.cpd
        new_temperature = _T_from_internal_energy(
            u=u, qv=qv, qliq=q_liquid, qice=q_solid, rho=one, dz=one
        )
    else:
        new_temperature = (energy - grav * height_above_ground) / PhysicsConstants.cpd
    rdtime = wpfloat("1.0") / dtime
    tend_temperature = (new_temperature - temperature) * rdtime
    return new_temperature, tend_temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_temperature_from_energy_and_tendency(
    energy: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    grav: wpfloat,
    dtime: wpfloat,
    use_internal_energy: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_temperature_from_energy_and_tendency(
        energy=energy,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        qr=qr,
        qs=qs,
        qg=qg,
        height_above_ground=height_above_ground,
        grav=grav,
        dtime=dtime,
        use_internal_energy=use_internal_energy,
        out=(new_temperature, tend_temperature),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
