# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.surface.stencils.diagnostics_stencils import (
    compute_sea_surface_roughness,
    compute_surface_exchange_coefficients,
    compute_surface_layer_diagnostics,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure
from icon4py.model.testing import stencil_tests


def _potential_temperature(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    # as potential_temperature in mo_aes_thermo.f90
    return temperature * np.exp(constants.RD_O_CPD * np.log(constants.P0REF / pressure))


def _restrict(outputs: dict[str, np.ndarray], horizontal_start: int, horizontal_end: int) -> dict:
    """Zero outside the program's domain, where the output fields keep their initial zeros."""
    cells = slice(horizontal_start, horizontal_end)
    restricted = {}
    for name, values in outputs.items():
        restricted[name] = np.zeros_like(values)
        restricted[name][cells] = values[cells]
    return restricted


class TestComputeSurfaceLayerDiagnostics(stencil_tests.StencilTest):
    PROGRAM = compute_surface_layer_diagnostics
    OUTPUTS = (
        "wind_speed",
        "potential_temperature",
        "virtual_potential_temperature",
        "saturation_specific_humidity",
        "density",
        "surface_potential_temperature",
        "surface_virtual_potential_temperature",
        "richardson_number",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        u: np.ndarray,
        v: np.ndarray,
        temperature: np.ndarray,
        virtual_temperature: np.ndarray,
        pressure: np.ndarray,
        surface_temperature: np.ndarray,
        surface_pressure: np.ndarray,
        z_mc: np.ndarray,
        minimum_surface_wind: float,
        surface_layer_fraction: float,
        grav: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        # ports compute_wind_speed and compute_atm_potential_temperature of mo_vdf_diag_smag.f90
        wind_speed = np.maximum(minimum_surface_wind, np.sqrt(u**2 + v**2))
        theta = _potential_temperature(temperature, pressure)
        theta_v = _potential_temperature(virtual_temperature, pressure)
        # ports compute_sfc_sat_spec_humidity for a water surface, and compute_sfc_density
        qsat = compute_moisture.specific_humidity(
            compute_pressure.sat_pres_water(surface_temperature), surface_pressure
        )
        density = surface_pressure / (
            constants.RD * surface_temperature * (1.0 + constants.RV_O_RD_MINUS_1 * qsat)
        )
        # ports compute_sfc_potential_temperature and compute_moist_richardson
        theta_sfc = _potential_temperature(surface_temperature, surface_pressure)
        theta_v_sfc = theta_sfc * (1.0 + constants.RV_O_RD_MINUS_1 * qsat)
        theta_v_mid = (
            surface_layer_fraction * theta_v + (1.0 - surface_layer_fraction) * theta_v_sfc
        )
        richardson_number = z_mc * grav * (theta_v - theta_v_sfc) / (theta_v_mid * wind_speed)
        return _restrict(
            {
                "wind_speed": wind_speed,
                "potential_temperature": theta,
                "virtual_potential_temperature": theta_v,
                "saturation_specific_humidity": qsat,
                "density": density,
                "surface_potential_temperature": theta_sfc,
                "surface_virtual_potential_temperature": theta_v_sfc,
                "richardson_number": richardson_number,
            },
            horizontal_start,
            horizontal_end,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            u=data_alloc.random_field(dims.CellDim, low=-15.0, high=15.0),
            v=data_alloc.random_field(dims.CellDim, low=-15.0, high=15.0),
            temperature=data_alloc.random_field(dims.CellDim, low=260.0, high=305.0),
            virtual_temperature=data_alloc.random_field(dims.CellDim, low=260.0, high=310.0),
            pressure=data_alloc.random_field(dims.CellDim, low=9.0e4, high=1.02e5),
            surface_temperature=data_alloc.random_field(dims.CellDim, low=270.0, high=310.0),
            surface_pressure=data_alloc.random_field(dims.CellDim, low=9.0e4, high=1.05e5),
            z_mc=data_alloc.random_field(dims.CellDim, low=10.0, high=40.0),
            minimum_surface_wind=0.3,
            surface_layer_fraction=0.4,
            grav=constants.GRAV,
            wind_speed=data_alloc.zero_field(dims.CellDim),
            potential_temperature=data_alloc.zero_field(dims.CellDim),
            virtual_potential_temperature=data_alloc.zero_field(dims.CellDim),
            saturation_specific_humidity=data_alloc.zero_field(dims.CellDim),
            density=data_alloc.zero_field(dims.CellDim),
            surface_potential_temperature=data_alloc.zero_field(dims.CellDim),
            surface_virtual_potential_temperature=data_alloc.zero_field(dims.CellDim),
            richardson_number=data_alloc.zero_field(dims.CellDim),
            # narrower than the field, which the grid's zones are not
            horizontal_start=1,
            horizontal_end=grid.num_cells - 1,
        )


def _roughness_inputs(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, *, is_initial_time: bool
) -> dict:
    return dict(
        wind_speed=data_alloc.random_field(dims.CellDim, low=0.3, high=20.0),
        exchange_coefficient_momentum=data_alloc.random_field(
            dims.CellDim, low=1.0e-4, high=5.0e-3
        ),
        is_initial_time=is_initial_time,
        ocean_roughness_length=1.0e-3,
        minimum_roughness_length=1.5e-5,
        grav=constants.GRAV,
        roughness_length_momentum=data_alloc.zero_field(dims.CellDim),
        roughness_length_heat=data_alloc.zero_field(dims.CellDim),
        horizontal_start=1,
        horizontal_end=grid.num_cells - 1,
    )


class TestComputeSeaSurfaceRoughness(stencil_tests.StencilTest):
    """The Charnock relation of the steps after the first."""

    PROGRAM = compute_sea_surface_roughness
    OUTPUTS = ("roughness_length_momentum", "roughness_length_heat")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        wind_speed: np.ndarray,
        exchange_coefficient_momentum: np.ndarray,
        is_initial_time: bool,
        ocean_roughness_length: float,
        minimum_roughness_length: float,
        grav: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        # ports compute_sfc_roughness of mo_tmx_surface.f90 for the ocean
        if is_initial_time:
            roughness = np.full_like(wind_speed, ocean_roughness_length)
        else:
            roughness = (
                wind_speed * exchange_coefficient_momentum * constants.CHARNOCK_CONSTANT / grav
            )
        roughness = np.maximum(minimum_roughness_length, roughness)
        return _restrict(
            {"roughness_length_momentum": roughness, "roughness_length_heat": roughness},
            horizontal_start,
            horizontal_end,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _roughness_inputs(data_alloc, grid, is_initial_time=False)


class TestComputeSeaSurfaceRoughnessAtInitialTime(TestComputeSeaSurfaceRoughness):
    """The ocean roughness length of the first step."""

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _roughness_inputs(data_alloc, grid, is_initial_time=True)


# Louis (1979) exchange coefficients, transcribed from mo_vdf_diag_smag.f90
_BSM, _BUM, _BSH, _BUH = 5.0, 16.0, 5.0, 16.0  # Businger-Dyer coefficients
_ZEPSEC = 0.028


def _stability_function(
    richardson: np.ndarray, height_ratio: np.ndarray, neutral: np.ndarray, unstable_factor: float
) -> np.ndarray:
    # ports stability_function_mom and stability_function_heat, Holtslag and Boville
    height_factor = (np.maximum(height_ratio, 1.0) ** (1.0 / 3.0) - 1.0) ** 1.5
    return np.where(
        richardson >= 0.0,
        1.0 / (1.0 + 10.0 * richardson * (1.0 + 8.0 * richardson)),
        1.0
        + unstable_factor
        * np.abs(richardson)
        / (1.0 + 75.0 * neutral * height_factor * np.sqrt(np.abs(richardson))),
    )


def _businger_momentum(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        zeta, zeta0 = z1 / length, z0 / length
        log_ratio = np.log(z1 / z0)
        psi_vs, psi0_vs = -_BSM * np.log(zeta) - zeta + 1.0, -_BSM * np.log(zeta0) - zeta0 + 1.0
        stable = np.where(
            zeta > 1.0,
            np.log(length / z0) + _BSH - psi_vs + psi0_vs,
            log_ratio - (-_BSM * zeta) + (-_BSM * zeta0),
        )
        lamda, lamda0 = np.sqrt(np.sqrt(1.0 - _BUM * zeta)), np.sqrt(np.sqrt(1.0 - _BUM * zeta0))
        psi = (
            2.0 * np.log(1.0 + lamda)
            + np.log(1.0 + lamda * lamda)
            - 2.0 * np.arctan(lamda)
            + np.pi / 2.0
            - 3.0 * np.log(2.0)
        )
        psi0 = (
            2.0 * np.log(1.0 + lamda0)
            + np.log(1.0 + lamda0 * lamda0)
            - 2.0 * np.arctan(lamda0)
            + np.pi / 2.0
            - 3.0 * np.log(2.0)
        )
        unstable = log_ratio - psi + psi0
        factor = np.where(length > 0.0, stable, np.where(length < 0.0, unstable, log_ratio))
    return factor / constants.CKAP


def _businger_heat(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        zeta, zeta0 = z1 / length, z0 / length
        log_ratio = np.log(z1 / z0)
        psi_vs, psi0_vs = -_BSH * np.log(zeta) - zeta + 1.0, -_BSH * np.log(zeta0) - zeta0 + 1.0
        stable = np.where(
            zeta > 1.0,
            np.log(length / z0) + _BSH - psi_vs + psi0_vs,
            log_ratio - (-_BSH * zeta) + (-_BSH * zeta0),
        )
        lamda, lamda0 = np.sqrt(1.0 - _BUH * zeta), np.sqrt(1.0 - _BUH * zeta0)
        psi = 2.0 * (np.log(1.0 + lamda) - np.log(2.0))
        psi0 = 2.0 * (np.log(1.0 + lamda0) - np.log(2.0))
        unstable = log_ratio - psi + psi0
        factor = np.where(length > 0.0, stable, np.where(length < 0.0, unstable, log_ratio))
    return factor / constants.CKAP


class TestComputeSurfaceExchangeCoefficients(stencil_tests.StencilTest):
    PROGRAM = compute_surface_exchange_coefficients
    OUTPUTS = (
        "exchange_coefficient_momentum",
        "exchange_coefficient_heat",
        "exchange_coefficient_momentum_neutral",
        "exchange_coefficient_heat_neutral",
    )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        ddqz_z_half: np.ndarray,
        specific_humidity: np.ndarray,
        potential_temperature: np.ndarray,
        wind_speed: np.ndarray,
        roughness_length_momentum: np.ndarray,
        surface_potential_temperature: np.ndarray,
        saturation_specific_humidity: np.ndarray,
        grav: float,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        ckap = constants.CKAP
        z0, theta_sfc = roughness_length_momentum, surface_potential_temperature
        # compute_sfc_exchange_coefficients: dz is twice the reference height
        z = 0.5 * ddqz_z_half
        # sfc_exchange_coefficients: bulk first guess
        rib = grav * (potential_temperature - theta_sfc) * (z - z0) / (theta_sfc * wind_speed**2)
        tcn_mom = (ckap / np.log(z / z0)) ** 2
        tcm = tcn_mom * _stability_function(rib, z / z0, tcn_mom, 10.0)
        tcn_heat = ckap**2 / (np.log(z / z0) * np.log(z / z0))
        tch = tcn_heat * _stability_function(rib, z / z0, tcn_heat, 15.0)
        for _ in range(5):
            shfl = tch * wind_speed * (theta_sfc - potential_temperature)
            lhfl = tch * wind_speed * (saturation_specific_humidity - specific_humidity)
            bflx = shfl + constants.RV_O_RD_MINUS_1 * theta_sfc * lhfl
            ustar = np.sqrt(tcm) * wind_speed
            obukhov_length = -(ustar**3) * theta_sfc / grav / (ckap * bflx)
            inv_bus_mom = 1.0 / _businger_momentum(z0, z, obukhov_length)
            tch = inv_bus_mom / _businger_heat(z0, z, obukhov_length)
            tcm = inv_bus_mom * inv_bus_mom
        return _restrict(
            {
                "exchange_coefficient_momentum": tcm,
                "exchange_coefficient_heat": tch,
                "exchange_coefficient_momentum_neutral": ckap
                / np.maximum(_ZEPSEC, np.sqrt(tcn_mom)),
                "exchange_coefficient_heat_neutral": ckap / np.maximum(_ZEPSEC, np.sqrt(tcn_heat)),
            },
            horizontal_start,
            horizontal_end,
        )

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            ddqz_z_half=data_alloc.random_field(dims.CellDim, low=20.0, high=80.0),
            specific_humidity=data_alloc.random_field(dims.CellDim, low=2.0e-3, high=2.0e-2),
            # stable and unstable surface layers: the air warmer or colder than the sea
            potential_temperature=data_alloc.random_field(dims.CellDim, low=280.0, high=310.0),
            wind_speed=data_alloc.random_field(dims.CellDim, low=1.0, high=15.0),
            roughness_length_momentum=data_alloc.random_field(
                dims.CellDim, low=1.5e-5, high=1.0e-3
            ),
            surface_potential_temperature=data_alloc.random_field(
                dims.CellDim, low=285.0, high=305.0
            ),
            saturation_specific_humidity=data_alloc.random_field(
                dims.CellDim, low=1.0e-2, high=3.0e-2
            ),
            grav=constants.GRAV,
            exchange_coefficient_momentum=data_alloc.zero_field(dims.CellDim),
            exchange_coefficient_heat=data_alloc.zero_field(dims.CellDim),
            exchange_coefficient_momentum_neutral=data_alloc.zero_field(dims.CellDim),
            exchange_coefficient_heat_neutral=data_alloc.zero_field(dims.CellDim),
            horizontal_start=1,
            horizontal_end=grid.num_cells - 1,
        )
