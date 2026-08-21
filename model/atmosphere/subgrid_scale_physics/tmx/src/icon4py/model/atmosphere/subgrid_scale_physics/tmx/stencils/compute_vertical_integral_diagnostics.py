# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import _compute_vertical_integral
from icon4py.model.common.physics.thermodynamics import _internal_energy
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_vertical_integral_diagnostics(
    static_energy: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    dz: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Compute the running vertical integrals of the tmx energy diagnostics.

    Port of the vertical-integral part of 'Update_diagnostics' in
    mo_vdf_atmo.f90 ('compute_internal_energy_vi' and the accumulation loop):

        ctgzvi             = sum_k ctgz(k) * rho(k) * dz(k)
        dissip_ke_vi       = sum_k dissip_ke(k)
        int_energy_vi      = sum_k internal_energy(new state, k)
        int_energy_vi_tend = (int_energy_vi - sum_k internal_energy(old state, k))
                             / dtime

    with ``internal_energy`` from mo_aes_thermo.f90 (ported in
    :mod:`icon4py.model.common.physics.thermodynamics`; the liquid phase is
    qc + qr, the solid phase qi + qs + qg). qr, qs and qg are not diffused
    and have no new state. The Fortran diagnostics are 2D surface fields;
    here each output holds the running top-down sum, so its value at the last
    full level (k = nlev - 1) is the column integral the caller extracts.

    Domains (Fortran): jk = 1..nlev; the tmx ``t_domain`` cell range
    (``grf_bdywidth_c + 1`` to ``min_rlcell_int``), which maps to the
    horizontal domain ``(h_grid.Zone.NUDGING, h_grid.Zone.LOCAL)``.

    Args:
        static_energy: dry static energy cpd*T + g*ghf from the *new*
            temperature (``ctgz``) [J/kg]
        dissip_ke: kinetic energy dissipation per layer [W/m^2]
        rho: air density [kg/m^3]
        dz: layer thickness [m]
        temperature: old air temperature [K]
        qv: old specific humidity [kg/kg]
        qc: old cloud water mixing ratio [kg/kg]
        qi: old cloud ice mixing ratio [kg/kg]
        new_temperature: updated air temperature [K]
        new_qv: updated specific humidity [kg/kg]
        new_qc: updated cloud water mixing ratio [kg/kg]
        new_qi: updated cloud ice mixing ratio [kg/kg]
        qr: rain mixing ratio [kg/kg]
        qs: snow mixing ratio [kg/kg]
        qg: graupel mixing ratio [kg/kg]
        dtime: time step [s]

    Returns:
        running vertical integrals of the static energy [J/m^2], the kinetic
        energy dissipation [W/m^2], the internal energy of the new state
        [J/m^2] and the internal energy tendency [W/m^2]
    """
    int_energy_old = _internal_energy(
        t=temperature, qv=qv, qliq=qc + qr, qice=qi + qs + qg, rho=rho, dz=dz
    )
    int_energy_new = _internal_energy(
        t=new_temperature,
        qv=new_qv,
        qliq=new_qc + qr,
        qice=new_qi + qs + qg,
        rho=rho,
        dz=dz,
    )
    cptgz_vi = _compute_vertical_integral(static_energy * rho * dz)
    dissip_ke_vi = _compute_vertical_integral(dissip_ke)
    int_energy_vi = _compute_vertical_integral(int_energy_new)
    int_energy_vi_old = _compute_vertical_integral(int_energy_old)
    int_energy_vi_tend = (int_energy_vi - int_energy_vi_old) / dtime
    return cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_integral_diagnostics(
    static_energy: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    dz: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    cptgz_vi: fa.CellKField[wpfloat],
    dissip_ke_vi: fa.CellKField[wpfloat],
    int_energy_vi: fa.CellKField[wpfloat],
    int_energy_vi_tend: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_integral_diagnostics(
        static_energy=static_energy,
        dissip_ke=dissip_ke,
        rho=rho,
        dz=dz,
        temperature=temperature,
        qv=qv,
        qc=qc,
        qi=qi,
        new_temperature=new_temperature,
        new_qv=new_qv,
        new_qc=new_qc,
        new_qi=new_qi,
        qr=qr,
        qs=qs,
        qg=qg,
        dtime=dtime,
        out=(cptgz_vi, dissip_ke_vi, int_energy_vi, int_energy_vi_tend),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
