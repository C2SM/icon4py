# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stencils of the tmx energy update and end-of-step diagnostics."""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import (
    _accumulate_from_top,
    _copy_half_level_below_to_model_levels_on_cells,
)
from icon4py.model.common.physics.thermodynamics.compute_energy import (
    _compute_dry_static_energy,
    compute_internal_energy_per_area,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_temperature_and_compute_end_of_step_diagnostics(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    q_snocpymlt: fa.CellField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    dissipation_factor: wpfloat,
    grav: wpfloat,
    dtime: wpfloat,
    nlev: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Update the temperature with the dissipation heating and compute the energy diagnostics of
    the updated state.

    `tend_temperature` is the heat-diffusion tendency, `new_u` and `new_v` are the diffused
    winds. The heating is the kinetic energy dissipated by the wind diffusion, less the heat
    used to melt snow on the canopy (`q_snocpymlt`) at the lowest level. qr, qs and qg are not
    diffused and count in both the old and the new internal energy.

    The vertical integrals are running sums from the top: their last level is the column
    integral.

    Returns:
        the dissipated kinetic energy, the heating, the updated temperature, the dry static energy of the updated temperature and the vertical
        integrals of the dry static energy, of the dissipated kinetic energy, of the internal
        energy and of its tendency
    """
    dissip_ke = (
        wpfloat("0.5")
        * air_mass
        * dissipation_factor
        * (wpfloat("1.0") / dtime)
        * (u * u - new_u * new_u + v * v - new_v * new_v)
    )
    # `nlev`, not `vertical_end`: the surface term belongs to the lowest model level whatever
    # the domain, and dace fails when a domain bound is also a `concat_where` operand
    heating = concat_where(dims.KDim < nlev - 1, dissip_ke, dissip_ke - q_snocpymlt)
    new_temperature = temperature + (tend_temperature + heating / cv_air) * dtime

    cptgz = _compute_dry_static_energy(new_temperature, height_above_ground, grav)
    cptgz_vi = _accumulate_from_top(cptgz * rho * ddqz_z_full)
    dissip_ke_vi = _accumulate_from_top(dissip_ke)
    int_energy_vi = _accumulate_from_top(
        compute_internal_energy_per_area(
            new_temperature, new_qv, new_qc + qr, new_qi + qs + qg, rho, ddqz_z_full
        )
    )
    old_int_energy_vi = _accumulate_from_top(
        compute_internal_energy_per_area(temperature, qv, qc + qr, qi + qs + qg, rho, ddqz_z_full)
    )
    int_energy_vi_tend = (int_energy_vi - old_int_energy_vi) / dtime
    return (
        dissip_ke,
        heating,
        new_temperature,
        cptgz,
        cptgz_vi,
        dissip_ke_vi,
        int_energy_vi,
        int_energy_vi_tend,
    )


@gtx.field_operator
def _add_heating_to_temperature_tendency(
    tend_temperature: fa.CellKField[wpfloat],
    heating: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    return tend_temperature + heating / cv_air


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_and_compute_end_of_step_diagnostics(
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    new_u: fa.CellKField[wpfloat],
    new_v: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    cv_air: fa.CellKField[wpfloat],
    temperature: fa.CellKField[wpfloat],
    tend_temperature: fa.CellKField[wpfloat],
    q_snocpymlt: fa.CellField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    new_qv: fa.CellKField[wpfloat],
    new_qc: fa.CellKField[wpfloat],
    new_qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    km_ic: fa.CellKHalfField[wpfloat],
    kh_ic: fa.CellKHalfField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    dissip_ke: fa.CellKField[wpfloat],
    heating: fa.CellKField[wpfloat],
    new_temperature: fa.CellKField[wpfloat],
    cptgz: fa.CellKField[wpfloat],
    cptgz_vi: fa.CellKField[wpfloat],
    dissip_ke_vi: fa.CellKField[wpfloat],
    int_energy_vi: fa.CellKField[wpfloat],
    int_energy_vi_tend: fa.CellKField[wpfloat],
    km: fa.CellKField[wpfloat],
    kh: fa.CellKField[wpfloat],
    dissipation_factor: wpfloat,
    grav: wpfloat,
    dtime: wpfloat,
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Update the temperature with the dissipation heating and compute the end-of-step
    diagnostics.

    `tend_temperature` is updated in place. `km` and `kh` are written above the lowest level
    only; the lowest level is the surface exchange coefficient.
    """
    _update_temperature_and_compute_end_of_step_diagnostics(
        u=u,
        v=v,
        new_u=new_u,
        new_v=new_v,
        air_mass=air_mass,
        cv_air=cv_air,
        temperature=temperature,
        tend_temperature=tend_temperature,
        q_snocpymlt=q_snocpymlt,
        qv=qv,
        qc=qc,
        qi=qi,
        new_qv=new_qv,
        new_qc=new_qc,
        new_qi=new_qi,
        qr=qr,
        qs=qs,
        qg=qg,
        rho=rho,
        ddqz_z_full=ddqz_z_full,
        height_above_ground=height_above_ground,
        dissipation_factor=dissipation_factor,
        grav=grav,
        dtime=dtime,
        nlev=nlev,
        out=(
            dissip_ke,
            heating,
            new_temperature,
            cptgz,
            cptgz_vi,
            dissip_ke_vi,
            int_energy_vi,
            int_energy_vi_tend,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    # TODO(havogt): write it in a way that tend_temperature is not inout.
    _add_heating_to_temperature_tendency(
        tend_temperature,
        heating,
        cv_air,
        out=tend_temperature,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _copy_half_level_below_to_model_levels_on_cells(
        km_ic,
        out=km,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _copy_half_level_below_to_model_levels_on_cells(
        kh_ic,
        out=kh,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
