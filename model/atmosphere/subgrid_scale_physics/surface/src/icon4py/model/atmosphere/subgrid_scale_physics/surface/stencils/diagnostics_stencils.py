# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stencils of the surface diagnostics.

Ports ``Compute_diagnostics`` of ``t_vdf_sfc`` (mo_vdf_sfc.f90) for the sea surface: the
routines of mo_vdf_diag_smag.f90 and mo_tmx_surface.f90 it calls, in the order it calls them.
"""

import gt4py.next as gtx
from gt4py.next import abs, arctan, broadcast, log, maximum, power, sqrt, where  # noqa: A004

from icon4py.model.atmosphere.subgrid_scale_physics.surface.constants import LouisConstants
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.dimension import CellDim
from icon4py.model.common.physics.thermodynamics.compute_moisture import _specific_humidity_on_cells
from icon4py.model.common.physics.thermodynamics.compute_pressure import _sat_pres_water_on_cells
from icon4py.model.common.physics.thermodynamics.compute_temperature import (
    _compute_potential_temperature,
)
from icon4py.model.common.type_alias import wpfloat


# ---------------------------------------------------------------------------
# Surface-layer diagnostics: pointwise, one pass
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_wind_speed(
    u: fa.CellField[wpfloat],
    v: fa.CellField[wpfloat],
    minimum_surface_wind: wpfloat,
) -> fa.CellField[wpfloat]:
    """``compute_wind_speed`` (mo_vdf_diag_smag.f90): the lowest-level wind speed, bounded below."""
    return maximum(minimum_surface_wind, sqrt(u * u + v * v))


@gtx.field_operator
def _compute_air_potential_temperatures(
    temperature: fa.CellField[wpfloat],
    virtual_temperature: fa.CellField[wpfloat],
    pressure: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """``compute_atm_potential_temperature`` (mo_vdf_diag_smag.f90), at the lowest level."""
    return (
        _compute_potential_temperature(temperature, pressure),
        _compute_potential_temperature(virtual_temperature, pressure),
    )


@gtx.field_operator
def _compute_surface_saturation_specific_humidity(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """``compute_sfc_sat_spec_humidity`` (mo_tmx_surface.f90), water surface."""
    return _specific_humidity_on_cells(
        _sat_pres_water_on_cells(surface_temperature), surface_pressure
    )


@gtx.field_operator
def _compute_surface_density(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """``compute_sfc_density`` (mo_vdf_diag_smag.f90): air density at the surface."""
    return surface_pressure / (
        PhysicsConstants.rd
        * surface_temperature
        * (wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * saturation_specific_humidity)
    )


@gtx.field_operator
def _compute_surface_potential_temperatures(
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """``compute_sfc_potential_temperature`` (mo_vdf_diag_smag.f90)."""
    theta = _compute_potential_temperature(surface_temperature, surface_pressure)
    theta_v = theta * (
        wpfloat("1.0") + PhysicsConstants.rv_o_rd_minus_1 * saturation_specific_humidity
    )
    return theta, theta_v


@gtx.field_operator
def _compute_moist_richardson_number(
    z_mc: fa.CellField[wpfloat],
    virtual_potential_temperature: fa.CellField[wpfloat],
    surface_virtual_potential_temperature: fa.CellField[wpfloat],
    wind_speed: fa.CellField[wpfloat],
    surface_layer_fraction: wpfloat,
    grav: wpfloat,
) -> fa.CellField[wpfloat]:
    """
    ``compute_moist_richardson`` (mo_vdf_diag_smag.f90).

    The mean virtual potential temperature of the surface layer weights the air with
    ``surface_layer_fraction``; the denominator carries the wind speed, not its square, as
    in ICON.
    """
    theta_v_mid = (
        surface_layer_fraction * virtual_potential_temperature
        + (wpfloat("1.0") - surface_layer_fraction) * surface_virtual_potential_temperature
    )
    return (
        z_mc
        * grav
        * (virtual_potential_temperature - surface_virtual_potential_temperature)
        / (theta_v_mid * wind_speed)
    )


@gtx.field_operator
def _compute_surface_layer_diagnostics(
    u: fa.CellField[wpfloat],
    v: fa.CellField[wpfloat],
    temperature: fa.CellField[wpfloat],
    virtual_temperature: fa.CellField[wpfloat],
    pressure: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    z_mc: fa.CellField[wpfloat],
    minimum_surface_wind: wpfloat,
    surface_layer_fraction: wpfloat,
    grav: wpfloat,
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    wind_speed = _compute_wind_speed(u, v, minimum_surface_wind)
    theta, theta_v = _compute_air_potential_temperatures(temperature, virtual_temperature, pressure)
    qsat = _compute_surface_saturation_specific_humidity(surface_temperature, surface_pressure)
    density = _compute_surface_density(surface_temperature, surface_pressure, qsat)
    theta_sfc, theta_v_sfc = _compute_surface_potential_temperatures(
        surface_temperature, surface_pressure, qsat
    )
    richardson_number = _compute_moist_richardson_number(
        z_mc, theta_v, theta_v_sfc, wind_speed, surface_layer_fraction, grav
    )
    return wind_speed, theta, theta_v, qsat, density, theta_sfc, theta_v_sfc, richardson_number


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_layer_diagnostics(
    u: fa.CellField[wpfloat],
    v: fa.CellField[wpfloat],
    temperature: fa.CellField[wpfloat],
    virtual_temperature: fa.CellField[wpfloat],
    pressure: fa.CellField[wpfloat],
    surface_temperature: fa.CellField[wpfloat],
    surface_pressure: fa.CellField[wpfloat],
    z_mc: fa.CellField[wpfloat],
    minimum_surface_wind: wpfloat,
    surface_layer_fraction: wpfloat,
    grav: wpfloat,
    wind_speed: fa.CellField[wpfloat],
    potential_temperature: fa.CellField[wpfloat],
    virtual_potential_temperature: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
    density: fa.CellField[wpfloat],
    surface_potential_temperature: fa.CellField[wpfloat],
    surface_virtual_potential_temperature: fa.CellField[wpfloat],
    richardson_number: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_layer_diagnostics(
        u,
        v,
        temperature,
        virtual_temperature,
        pressure,
        surface_temperature,
        surface_pressure,
        z_mc,
        minimum_surface_wind,
        surface_layer_fraction,
        grav,
        out=(
            wind_speed,
            potential_temperature,
            virtual_potential_temperature,
            saturation_specific_humidity,
            density,
            surface_potential_temperature,
            surface_virtual_potential_temperature,
            richardson_number,
        ),
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


# ---------------------------------------------------------------------------
# Roughness length of the sea: reads the previous step's momentum coefficient
# ---------------------------------------------------------------------------
@gtx.field_operator
def _compute_sea_surface_roughness(
    wind_speed: fa.CellField[wpfloat],
    exchange_coefficient_momentum: fa.CellField[wpfloat],
    is_initial_time: bool,
    ocean_roughness_length: wpfloat,
    minimum_roughness_length: wpfloat,
    grav: wpfloat,
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    ``compute_sfc_roughness`` (mo_tmx_surface.f90) for the sea.

    The ocean roughness length at the initial time, afterwards the Charnock relation with
    the momentum exchange coefficient of the previous step; bounded below. The sea has the
    same roughness length for momentum and heat.
    """
    if is_initial_time:
        roughness_length = broadcast(ocean_roughness_length, (CellDim,))
    else:
        roughness_length = (
            wind_speed
            * exchange_coefficient_momentum
            * PhysicsConstants.charnock
            * (wpfloat("1.0") / grav)
        )
    roughness_length = maximum(minimum_roughness_length, roughness_length)
    return roughness_length, roughness_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_sea_surface_roughness(
    wind_speed: fa.CellField[wpfloat],
    exchange_coefficient_momentum: fa.CellField[wpfloat],
    is_initial_time: bool,
    ocean_roughness_length: wpfloat,
    minimum_roughness_length: wpfloat,
    grav: wpfloat,
    roughness_length_momentum: fa.CellField[wpfloat],
    roughness_length_heat: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_sea_surface_roughness(
        wind_speed,
        exchange_coefficient_momentum,
        is_initial_time,
        ocean_roughness_length,
        minimum_roughness_length,
        grav,
        out=(roughness_length_momentum, roughness_length_heat),
        domain={CellDim: (horizontal_start, horizontal_end)},
    )


# ---------------------------------------------------------------------------
# Exchange coefficients: Louis (1979) with Businger-Dyer profiles
# ---------------------------------------------------------------------------
@gtx.field_operator
def _stability_function_momentum(
    richardson_number: fa.CellField[wpfloat],
    height_ratio: fa.CellField[wpfloat],
    neutral_coefficient: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """``stability_function_mom`` (mo_vdf_diag_smag.f90): Holtslag & Boville (1992)."""
    height_factor = power(
        power(maximum(height_ratio, wpfloat("1.0")), wpfloat("1.0") / wpfloat("3.0"))
        - wpfloat("1.0"),
        wpfloat("1.5"),
    )
    return where(
        richardson_number >= wpfloat("0.0"),
        wpfloat("1.0")
        / (
            wpfloat("1.0")
            + wpfloat("10.0")
            * richardson_number
            * (wpfloat("1.0") + wpfloat("8.0") * richardson_number)
        ),
        wpfloat("1.0")
        + wpfloat("10.0")
        * abs(richardson_number)
        / (
            wpfloat("1.0")
            + wpfloat("75.0") * neutral_coefficient * height_factor * sqrt(abs(richardson_number))
        ),
    )


@gtx.field_operator
def _stability_function_heat(
    richardson_number: fa.CellField[wpfloat],
    height_ratio: fa.CellField[wpfloat],
    neutral_coefficient: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """``stability_function_heat`` (mo_vdf_diag_smag.f90): Holtslag & Boville (1992)."""
    height_factor = power(
        power(maximum(height_ratio, wpfloat("1.0")), wpfloat("1.0") / wpfloat("3.0"))
        - wpfloat("1.0"),
        wpfloat("1.5"),
    )
    return where(
        richardson_number >= wpfloat("0.0"),
        wpfloat("1.0")
        / (
            wpfloat("1.0")
            + wpfloat("10.0")
            * richardson_number
            * (wpfloat("1.0") + wpfloat("8.0") * richardson_number)
        ),
        wpfloat("1.0")
        + wpfloat("15.0")
        * abs(richardson_number)
        / (
            wpfloat("1.0")
            + wpfloat("75.0") * neutral_coefficient * height_factor * sqrt(abs(richardson_number))
        ),
    )


@gtx.field_operator
def _businger_momentum(
    roughness_length: fa.CellField[wpfloat],
    height: fa.CellField[wpfloat],
    obukhov_length: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    ``businger_mom`` (mo_vdf_diag_smag.f90): the Businger-Dyer momentum profile integrated
    from the roughness length to ``height``, over the von Karman constant.
    """
    zeta = height / obukhov_length
    zeta0 = roughness_length / obukhov_length
    log_height_ratio = log(height / roughness_length)
    # stable: Zeng et al. (1997) beyond zeta = 1; ICON adds bsh, not bsm, there
    psi_very_stable = -LouisConstants.businger_stable_momentum * log(zeta) - zeta + wpfloat("1.0")
    psi0_very_stable = (
        -LouisConstants.businger_stable_momentum * log(zeta0) - zeta0 + wpfloat("1.0")
    )
    psi_stable = -LouisConstants.businger_stable_momentum * zeta
    psi0_stable = -LouisConstants.businger_stable_momentum * zeta0
    stable = where(
        zeta > wpfloat("1.0"),
        log(obukhov_length / roughness_length)
        + LouisConstants.businger_stable_heat
        - psi_very_stable
        + psi0_very_stable,
        log_height_ratio - psi_stable + psi0_stable,
    )
    # unstable
    lamda = sqrt(sqrt(wpfloat("1.0") - LouisConstants.businger_unstable_momentum * zeta))
    lamda0 = sqrt(sqrt(wpfloat("1.0") - LouisConstants.businger_unstable_momentum * zeta0))
    psi_unstable = (
        wpfloat("2.0") * log(wpfloat("1.0") + lamda)
        + log(wpfloat("1.0") + lamda * lamda)
        - wpfloat("2.0") * arctan(lamda)
        + LouisConstants.half_pi
        - wpfloat("3.0") * LouisConstants.ln_2
    )
    psi0_unstable = (
        wpfloat("2.0") * log(wpfloat("1.0") + lamda0)
        + log(wpfloat("1.0") + lamda0 * lamda0)
        - wpfloat("2.0") * arctan(lamda0)
        + LouisConstants.half_pi
        - wpfloat("3.0") * LouisConstants.ln_2
    )
    unstable = log_height_ratio - psi_unstable + psi0_unstable
    factor = where(
        obukhov_length > wpfloat("0.0"),
        stable,
        where(obukhov_length < wpfloat("0.0"), unstable, log_height_ratio),
    )
    return factor / PhysicsConstants.von_karman


@gtx.field_operator
def _businger_heat(
    roughness_length: fa.CellField[wpfloat],
    height: fa.CellField[wpfloat],
    obukhov_length: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """
    ``businger_heat`` (mo_vdf_diag_smag.f90): the Businger-Dyer heat profile integrated
    from the roughness length to ``height``, over the von Karman constant.
    """
    zeta = height / obukhov_length
    zeta0 = roughness_length / obukhov_length
    log_height_ratio = log(height / roughness_length)
    # stable: Zeng et al. (1997) beyond zeta = 1
    psi_very_stable = -LouisConstants.businger_stable_heat * log(zeta) - zeta + wpfloat("1.0")
    psi0_very_stable = -LouisConstants.businger_stable_heat * log(zeta0) - zeta0 + wpfloat("1.0")
    psi_stable = -LouisConstants.businger_stable_heat * zeta
    psi0_stable = -LouisConstants.businger_stable_heat * zeta0
    stable = where(
        zeta > wpfloat("1.0"),
        log(obukhov_length / roughness_length)
        + LouisConstants.businger_stable_heat
        - psi_very_stable
        + psi0_very_stable,
        log_height_ratio - psi_stable + psi0_stable,
    )
    # unstable
    lamda = sqrt(wpfloat("1.0") - LouisConstants.businger_unstable_heat * zeta)
    lamda0 = sqrt(wpfloat("1.0") - LouisConstants.businger_unstable_heat * zeta0)
    psi_unstable = wpfloat("2.0") * (log(wpfloat("1.0") + lamda) - LouisConstants.ln_2)
    psi0_unstable = wpfloat("2.0") * (log(wpfloat("1.0") + lamda0) - LouisConstants.ln_2)
    unstable = log_height_ratio - psi_unstable + psi0_unstable
    factor = where(
        obukhov_length > wpfloat("0.0"),
        stable,
        where(obukhov_length < wpfloat("0.0"), unstable, log_height_ratio),
    )
    return factor / PhysicsConstants.von_karman


@gtx.field_operator
def _louis_iteration(
    transfer_coefficient_momentum: fa.CellField[wpfloat],
    transfer_coefficient_heat: fa.CellField[wpfloat],
    specific_humidity: fa.CellField[wpfloat],
    potential_temperature: fa.CellField[wpfloat],
    wind_speed: fa.CellField[wpfloat],
    roughness_length: fa.CellField[wpfloat],
    surface_potential_temperature: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
    reference_height: fa.CellField[wpfloat],
    grav: wpfloat,
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """One step of the iteration in ``sfc_exchange_coefficients`` (mo_vdf_diag_smag.f90)."""
    sensible_heat_flux = (
        transfer_coefficient_heat
        * wind_speed
        * (surface_potential_temperature - potential_temperature)
    )
    latent_heat_flux = (
        transfer_coefficient_heat * wind_speed * (saturation_specific_humidity - specific_humidity)
    )
    buoyancy_flux = (
        sensible_heat_flux
        + PhysicsConstants.rv_o_rd_minus_1 * surface_potential_temperature * latent_heat_flux
    )
    ustar = sqrt(transfer_coefficient_momentum) * wind_speed
    obukhov_length = -(
        ustar * ustar * ustar * surface_potential_temperature * (wpfloat("1.0") / grav)
    ) / (PhysicsConstants.von_karman * buoyancy_flux)
    inverse_businger_momentum = wpfloat("1.0") / _businger_momentum(
        roughness_length, reference_height, obukhov_length
    )
    transfer_coefficient_heat = inverse_businger_momentum / _businger_heat(
        roughness_length, reference_height, obukhov_length
    )
    transfer_coefficient_momentum = inverse_businger_momentum * inverse_businger_momentum
    return transfer_coefficient_momentum, transfer_coefficient_heat


@gtx.field_operator
def _compute_surface_exchange_coefficients(
    ddqz_z_half: fa.CellField[wpfloat],
    specific_humidity: fa.CellField[wpfloat],
    potential_temperature: fa.CellField[wpfloat],
    wind_speed: fa.CellField[wpfloat],
    roughness_length: fa.CellField[wpfloat],
    surface_potential_temperature: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
    grav: wpfloat,
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    """
    ``compute_sfc_exchange_coefficients`` and ``sfc_exchange_coefficients``
    (mo_vdf_diag_smag.f90): the Louis (1979) exchange coefficients of momentum and heat.

    A bulk first guess from the Richardson number with the Holtslag & Boville stability
    functions, then five iterations on the Obukhov length with the Businger-Dyer profiles.
    The reference height of the surface layer is half the lowest layer thickness
    ``ddqz_z_half``, as in ICON's wrapper.

    Returns:
        the exchange coefficients of momentum and heat, and their neutral counterparts
    """
    reference_height = ddqz_z_half * wpfloat("0.5")
    ckap = PhysicsConstants.von_karman
    height_ratio = reference_height / roughness_length
    log_height_ratio = log(height_ratio)
    richardson_number = (
        grav
        * (potential_temperature - surface_potential_temperature)
        * (reference_height - roughness_length)
        / (surface_potential_temperature * wind_speed * wind_speed)
    )
    neutral_momentum = (ckap / log_height_ratio) * (ckap / log_height_ratio)
    transfer_momentum = neutral_momentum * _stability_function_momentum(
        richardson_number, height_ratio, neutral_momentum
    )
    neutral_heat = ckap * ckap / (log_height_ratio * log_height_ratio)
    transfer_heat = neutral_heat * _stability_function_heat(
        richardson_number, height_ratio, neutral_heat
    )
    transfer_momentum, transfer_heat = _louis_iteration(
        transfer_momentum,
        transfer_heat,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length,
        surface_potential_temperature,
        saturation_specific_humidity,
        reference_height,
        grav,
    )
    transfer_momentum, transfer_heat = _louis_iteration(
        transfer_momentum,
        transfer_heat,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length,
        surface_potential_temperature,
        saturation_specific_humidity,
        reference_height,
        grav,
    )
    transfer_momentum, transfer_heat = _louis_iteration(
        transfer_momentum,
        transfer_heat,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length,
        surface_potential_temperature,
        saturation_specific_humidity,
        reference_height,
        grav,
    )
    transfer_momentum, transfer_heat = _louis_iteration(
        transfer_momentum,
        transfer_heat,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length,
        surface_potential_temperature,
        saturation_specific_humidity,
        reference_height,
        grav,
    )
    transfer_momentum, transfer_heat = _louis_iteration(
        transfer_momentum,
        transfer_heat,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length,
        surface_potential_temperature,
        saturation_specific_humidity,
        reference_height,
        grav,
    )
    neutral_coefficient_momentum = ckap / maximum(
        LouisConstants.minimum_sqrt_neutral_coefficient, sqrt(neutral_momentum)
    )
    neutral_coefficient_heat = ckap / maximum(
        LouisConstants.minimum_sqrt_neutral_coefficient, sqrt(neutral_heat)
    )
    return transfer_momentum, transfer_heat, neutral_coefficient_momentum, neutral_coefficient_heat


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_surface_exchange_coefficients(
    ddqz_z_half: fa.CellField[wpfloat],
    specific_humidity: fa.CellField[wpfloat],
    potential_temperature: fa.CellField[wpfloat],
    wind_speed: fa.CellField[wpfloat],
    roughness_length_momentum: fa.CellField[wpfloat],
    surface_potential_temperature: fa.CellField[wpfloat],
    saturation_specific_humidity: fa.CellField[wpfloat],
    grav: wpfloat,
    exchange_coefficient_momentum: fa.CellField[wpfloat],
    exchange_coefficient_heat: fa.CellField[wpfloat],
    exchange_coefficient_momentum_neutral: fa.CellField[wpfloat],
    exchange_coefficient_heat_neutral: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_surface_exchange_coefficients(
        ddqz_z_half,
        specific_humidity,
        potential_temperature,
        wind_speed,
        roughness_length_momentum,
        surface_potential_temperature,
        saturation_specific_humidity,
        grav,
        out=(
            exchange_coefficient_momentum,
            exchange_coefficient_heat,
            exchange_coefficient_momentum_neutral,
            exchange_coefficient_heat_neutral,
        ),
        domain={CellDim: (horizontal_start, horizontal_end)},
    )
