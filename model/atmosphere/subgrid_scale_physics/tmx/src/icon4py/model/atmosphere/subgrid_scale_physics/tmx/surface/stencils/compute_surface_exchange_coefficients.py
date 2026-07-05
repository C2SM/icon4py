# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import arctan, log, maximum, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


# Port of 'sfc_exchange_coefficients' (mo_vdf_diag_smag.f90:395-420): a bulk
# Richardson first guess (Holtslag & Boville 1992) followed by a fixed
# 5-iteration Monin-Obukhov / Businger-Dyer loop. To keep the gtfn translation
# unit small, the iteration is NOT unrolled in the DSL: the Businger step is a
# standalone program (compiled once) that the granule calls 5 times over
# ping-pong scratch fields.
#
# Businger-Dyer / Zeng et al. (1997) constants (bsm=bsh=5, bum=buh=16;
# smag:56-59), von Karman constant ckap=0.4, pi/2, ln 2, and a small positive
# floor used to keep the not-taken 'where' branches finite (gt4py evaluates both
# branches, unlike the Fortran IF). Defined locally inside each field operator
# because gtfn rejects module-level constants.


@gtx.field_operator
def _stability_function_mom(
    richardson: fa.CellField[wpfloat],
    height_ratio: fa.CellField[wpfloat],
    neutral_coeff: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """First-guess momentum stability function (Holtslag & Boville 1992; smag:546-554)."""
    stable = wpfloat(1.0) / (
        wpfloat(1.0) + wpfloat(10.0) * richardson * (wpfloat(1.0) + wpfloat(8.0) * richardson)
    )
    height_factor = (
        maximum(height_ratio, wpfloat(1.0)) ** (wpfloat(1.0) / wpfloat(3.0)) - wpfloat(1.0)
    ) ** wpfloat(1.5)
    abs_richardson = maximum(richardson, -richardson)
    unstable = wpfloat(1.0) + wpfloat(10.0) * abs_richardson / (
        wpfloat(1.0) + wpfloat(75.0) * neutral_coeff * height_factor * sqrt(abs_richardson)
    )
    return where(richardson >= wpfloat(0.0), stable, unstable)


@gtx.field_operator
def _stability_function_heat(
    richardson: fa.CellField[wpfloat],
    height_ratio: fa.CellField[wpfloat],
    neutral_coeff: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """First-guess heat stability function (Holtslag & Boville 1992; smag:570-577)."""
    stable = wpfloat(1.0) / (
        wpfloat(1.0) + wpfloat(10.0) * richardson * (wpfloat(1.0) + wpfloat(8.0) * richardson)
    )
    height_factor = (
        maximum(height_ratio, wpfloat(1.0)) ** (wpfloat(1.0) / wpfloat(3.0)) - wpfloat(1.0)
    ) ** wpfloat(1.5)
    abs_richardson = maximum(richardson, -richardson)
    unstable = wpfloat(1.0) + wpfloat(15.0) * abs_richardson / (
        wpfloat(1.0) + wpfloat(75.0) * neutral_coeff * height_factor * sqrt(abs_richardson)
    )
    return where(richardson >= wpfloat(0.0), stable, unstable)


@gtx.field_operator
def _businger_mom(
    z0: fa.CellField[wpfloat],
    z1: fa.CellField[wpfloat],
    obukhov_length: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """Integrated Businger-Dyer momentum profile (smag:599-625)."""
    ckap = wpfloat(0.4)
    bsm = wpfloat(5.0)
    bum = wpfloat(16.0)
    half_pi = wpfloat(1.5707963267948966)
    ln2 = wpfloat(0.6931471805599453)
    eps = wpfloat(1.0e-12)
    log_ratio = log(z1 / z0)
    length_safe = where(obukhov_length == wpfloat(0.0), wpfloat(1.0), obukhov_length)
    zeta = z1 / length_safe
    zeta0 = z0 / length_safe
    # stable regime, Obukhov length positive: Zeng matching branch for zeta above one, linear below
    psi_zeng = (
        -bsm + bsm * zeta0 + (wpfloat(1.0) - bsm) * log(maximum(zeta, eps)) - zeta + wpfloat(1.0)
    )
    stable = where(
        zeta > wpfloat(1.0),
        (log_ratio - psi_zeng) / ckap,
        (log_ratio + bsm * zeta - bsm * zeta0) / ckap,
    )
    # unstable regime, Obukhov length negative
    lamda = sqrt(sqrt(maximum(wpfloat(1.0) - bum * zeta, eps)))
    lamda0 = sqrt(sqrt(maximum(wpfloat(1.0) - bum * zeta0, eps)))
    psi = (
        wpfloat(2.0) * log(wpfloat(1.0) + lamda)
        + log(wpfloat(1.0) + lamda * lamda)
        - wpfloat(2.0) * arctan(lamda)
        + half_pi
        - wpfloat(3.0) * ln2
    )
    psi0 = (
        wpfloat(2.0) * log(wpfloat(1.0) + lamda0)
        + log(wpfloat(1.0) + lamda0 * lamda0)
        - wpfloat(2.0) * arctan(lamda0)
        + half_pi
        - wpfloat(3.0) * ln2
    )
    unstable = (log_ratio - psi + psi0) / ckap
    neutral = log_ratio / ckap
    return where(
        obukhov_length > wpfloat(0.0),
        stable,
        where(obukhov_length < wpfloat(0.0), unstable, neutral),
    )


@gtx.field_operator
def _businger_heat(
    z0: fa.CellField[wpfloat],
    z1: fa.CellField[wpfloat],
    obukhov_length: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """Integrated Businger-Dyer heat profile (smag:645-665)."""
    ckap = wpfloat(0.4)
    bsh = wpfloat(5.0)
    buh = wpfloat(16.0)
    ln2 = wpfloat(0.6931471805599453)
    eps = wpfloat(1.0e-12)
    log_ratio = log(z1 / z0)
    length_safe = where(obukhov_length == wpfloat(0.0), wpfloat(1.0), obukhov_length)
    zeta = z1 / length_safe
    zeta0 = z0 / length_safe
    psi_zeng = (
        -bsh + bsh * zeta0 + (wpfloat(1.0) - bsh) * log(maximum(zeta, eps)) - zeta + wpfloat(1.0)
    )
    stable = where(
        zeta > wpfloat(1.0),
        (log_ratio - psi_zeng) / ckap,
        (log_ratio + bsh * zeta - bsh * zeta0) / ckap,
    )
    lamda = sqrt(maximum(wpfloat(1.0) - buh * zeta, eps))
    lamda0 = sqrt(maximum(wpfloat(1.0) - buh * zeta0, eps))
    psi = wpfloat(2.0) * (log(wpfloat(1.0) + lamda) - ln2)
    psi0 = wpfloat(2.0) * (log(wpfloat(1.0) + lamda0) - ln2)
    unstable = (log_ratio - psi + psi0) / ckap
    neutral = log_ratio / ckap
    return where(
        obukhov_length > wpfloat(0.0),
        stable,
        where(obukhov_length < wpfloat(0.0), unstable, neutral),
    )


@gtx.field_operator
def _compute_surface_exchange_first_guess(
    theta_atm: fa.CellField[wpfloat],
    theta_sfc: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    rough_m: fa.CellField[wpfloat],
    dz: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """Bulk Richardson first guess for the transfer coefficients (smag:397-402).

    ``dz`` is the surface-layer thickness (``ddqz_z_half`` at the surface); the
    similarity-theory reference height is ``0.5 * dz`` (smag:517).
    """
    ckap = wpfloat(0.4)
    reference_height = wpfloat(0.5) * dz
    height_ratio = reference_height / rough_m
    richardson = (
        PhysicsConstants.grav
        * (theta_atm - theta_sfc)
        * (reference_height - rough_m)
        / (theta_sfc * wind_rel * wind_rel)
    )
    neutral_coeff_arg = ckap / log(height_ratio)
    neutral_coeff = neutral_coeff_arg * neutral_coeff_arg
    transfer_mom = neutral_coeff * _stability_function_mom(richardson, height_ratio, neutral_coeff)
    transfer_heat = neutral_coeff * _stability_function_heat(
        richardson, height_ratio, neutral_coeff
    )
    return transfer_mom, transfer_heat


@gtx.field_operator
def _obukhov_businger_step(
    transfer_mom: fa.CellField[wpfloat],
    transfer_heat: fa.CellField[wpfloat],
    theta_atm: fa.CellField[wpfloat],
    theta_sfc: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    qa: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    rough_m: fa.CellField[wpfloat],
    dz: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """One Obukhov / Businger fixed-point iteration (smag:406-415). ``dz`` is halved (smag:517)."""
    ckap = wpfloat(0.4)
    reference_height = wpfloat(0.5) * dz
    sensible = transfer_heat * wind_rel * (theta_sfc - theta_atm)
    latent = transfer_heat * wind_rel * (qsat_sfc - qa)
    buoyancy_flux = sensible + PhysicsConstants.rv_o_rd_minus_1 * theta_sfc * latent
    ustar = sqrt(transfer_mom) * wind_rel
    obukhov_length = (
        -(ustar * ustar * ustar) * theta_sfc / (PhysicsConstants.grav * ckap * buoyancy_flux)
    )
    inv_bus_mom = wpfloat(1.0) / _businger_mom(rough_m, reference_height, obukhov_length)
    new_transfer_heat = inv_bus_mom / _businger_heat(rough_m, reference_height, obukhov_length)
    new_transfer_mom = inv_bus_mom * inv_bus_mom
    return new_transfer_mom, new_transfer_heat


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_exchange_first_guess(
    theta_atm: fa.CellField[wpfloat],
    theta_sfc: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    rough_m: fa.CellField[wpfloat],
    dz: fa.CellField[wpfloat],
    km: fa.CellField[wpfloat],
    kh: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_exchange_first_guess(
        theta_atm=theta_atm,
        theta_sfc=theta_sfc,
        wind_rel=wind_rel,
        rough_m=rough_m,
        dz=dz,
        out=(km, kh),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def obukhov_businger_step(
    km_in: fa.CellField[wpfloat],
    kh_in: fa.CellField[wpfloat],
    theta_atm: fa.CellField[wpfloat],
    theta_sfc: fa.CellField[wpfloat],
    qsat_sfc: fa.CellField[wpfloat],
    qa: fa.CellField[wpfloat],
    wind_rel: fa.CellField[wpfloat],
    rough_m: fa.CellField[wpfloat],
    dz: fa.CellField[wpfloat],
    km_out: fa.CellField[wpfloat],
    kh_out: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    """One Obukhov / Businger iteration; call 5 times over ping-pong km/kh buffers."""
    _obukhov_businger_step(
        km_in,
        kh_in,
        theta_atm,
        theta_sfc,
        qsat_sfc,
        qa,
        wind_rel,
        rough_m,
        dz,
        out=(km_out, kh_out),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
        },
    )
