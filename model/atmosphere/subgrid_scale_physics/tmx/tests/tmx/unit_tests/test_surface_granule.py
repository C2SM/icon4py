# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface import surface, surface_states
from icon4py.model.common import (
    constants,
    dimension as dims,
    model_backends,
    thermodynamic_functions as thermo_host,
    type_alias as ta,
)
from icon4py.model.common.constants import SeaIceConstants as SI
from icon4py.model.common.physics.thermodynamics import ThermodynamicConstants as TC
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import base


_CKAP = 0.4
_EPS = 1.0e-12
_DTIME = 1800.0
_TMELT = float(constants.MELTING_TEMPERATURE)


def _stability(rib: np.ndarray, hz0: np.ndarray, tc: np.ndarray, coeff: float) -> np.ndarray:
    stable = 1.0 / (1.0 + 10.0 * rib * (1.0 + 8.0 * rib))
    hf = (np.maximum(hz0, 1.0) ** (1.0 / 3.0) - 1.0) ** 1.5
    unstable = 1.0 + coeff * np.abs(rib) / (1.0 + 75.0 * tc * hf * np.sqrt(np.abs(rib)))
    return np.where(rib >= 0.0, stable, unstable)


def _businger_mom(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    bsm, bum, half_pi, ln2 = 5.0, 16.0, np.pi / 2.0, np.log(2.0)
    lr = np.log(z1 / z0)
    ls = np.where(length == 0.0, 1.0, length)
    zeta, zeta0 = z1 / ls, z0 / ls
    zeng = -bsm + bsm * zeta0 + (1.0 - bsm) * np.log(np.maximum(zeta, _EPS)) - zeta + 1.0
    stable = np.where(zeta > 1.0, (lr - zeng) / _CKAP, (lr + bsm * zeta - bsm * zeta0) / _CKAP)
    lam = np.sqrt(np.sqrt(np.maximum(1.0 - bum * zeta, _EPS)))
    lam0 = np.sqrt(np.sqrt(np.maximum(1.0 - bum * zeta0, _EPS)))
    psi = (
        2.0 * np.log(1.0 + lam)
        + np.log(1.0 + lam * lam)
        - 2.0 * np.arctan(lam)
        + half_pi
        - 3.0 * ln2
    )
    psi0 = (
        2.0 * np.log(1.0 + lam0)
        + np.log(1.0 + lam0 * lam0)
        - 2.0 * np.arctan(lam0)
        + half_pi
        - 3.0 * ln2
    )
    unstable = (lr - psi + psi0) / _CKAP
    return np.where(length > 0.0, stable, np.where(length < 0.0, unstable, lr / _CKAP))


def _businger_heat(z0: np.ndarray, z1: np.ndarray, length: np.ndarray) -> np.ndarray:
    bsh, buh, ln2 = 5.0, 16.0, np.log(2.0)
    lr = np.log(z1 / z0)
    ls = np.where(length == 0.0, 1.0, length)
    zeta, zeta0 = z1 / ls, z0 / ls
    zeng = -bsh + bsh * zeta0 + (1.0 - bsh) * np.log(np.maximum(zeta, _EPS)) - zeta + 1.0
    stable = np.where(zeta > 1.0, (lr - zeng) / _CKAP, (lr + bsh * zeta - bsh * zeta0) / _CKAP)
    lam = np.sqrt(np.maximum(1.0 - buh * zeta, _EPS))
    lam0 = np.sqrt(np.maximum(1.0 - buh * zeta0, _EPS))
    psi = 2.0 * (np.log(1.0 + lam) - ln2)
    psi0 = 2.0 * (np.log(1.0 + lam0) - ln2)
    unstable = (lr - psi + psi0) / _CKAP
    return np.where(length > 0.0, stable, np.where(length < 0.0, unstable, lr / _CKAP))


def _exchange_solver(  # noqa: PLR0917 [too-many-positional-arguments] numpy reference helper
    theta_atm: np.ndarray,
    theta_sfc: np.ndarray,
    qsat: np.ndarray,
    qa: np.ndarray,
    wind_rel: np.ndarray,
    rough_m: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    grav, vtmpc1 = constants.GRAV, constants.RV_O_RD_MINUS_1
    z_mc = 0.5 * dz
    height_ratio = z_mc / rough_m
    rib = grav * (theta_atm - theta_sfc) * (z_mc - rough_m) / (theta_sfc * wind_rel**2)
    neutral = (_CKAP / np.log(height_ratio)) ** 2
    km = neutral * _stability(rib, height_ratio, neutral, 10.0)
    kh = neutral * _stability(rib, height_ratio, neutral, 15.0)
    for _ in range(5):
        buoyancy = kh * wind_rel * (theta_sfc - theta_atm) + vtmpc1 * theta_sfc * (
            kh * wind_rel * (qsat - qa)
        )
        ustar = np.sqrt(km) * wind_rel
        length = -(ustar**3) * theta_sfc / (grav * _CKAP * buoyancy)
        inv = 1.0 / _businger_mom(rough_m, z_mc, length)
        kh = inv / _businger_heat(rough_m, z_mc, length)
        km = inv * inv
    return km, kh


def _ocean_reference(
    inp: dict[str, np.ndarray],
    ocean_km_in: np.ndarray,
    config: surface.TmxSurfaceConfig,
    params: surface.TmxSurfaceParams,
) -> dict[str, np.ndarray]:
    grav, rd, p0ref = constants.GRAV, constants.RD, constants.P0REF
    rd_o_cpd, vtmpc1 = constants.RD_O_CPD, constants.RV_O_RD_MINUS_1
    wind_rel = np.maximum(
        config.min_sfc_wind,
        np.sqrt((inp["ua"] - inp["ocean_u"]) ** 2 + (inp["va"] - inp["ocean_v"]) ** 2),
    )
    qsat = thermo_host.specific_humidity(thermo_host.sat_pres_water(inp["sst"]), inp["psfc"])
    rho = inp["psfc"] / (rd * inp["sst"] * (1.0 + vtmpc1 * qsat))
    theta_atm = inp["ta"] * (p0ref / inp["pa"]) ** rd_o_cpd
    theta_sfc = inp["sst"] * (p0ref / inp["psfc"]) ** rd_o_cpd
    rough_m = np.maximum(
        config.z0m_min,
        wind_rel**2 * ocean_km_in * params.charnock / grav
        + params.viscous_coeff
        * np.minimum(0.01, params.air_kinematic_viscosity / (np.sqrt(ocean_km_in) * wind_rel)),
    )
    km, kh = _exchange_solver(theta_atm, theta_sfc, qsat, inp["qa"], wind_rel, rough_m, inp["dz"])
    gust = np.sqrt(config.wind_g**2 + wind_rel**2)
    return dict(
        evapotranspiration=rho * kh * gust * (inp["qa"] - qsat),
        sensible_heat_flux=float(TC.cvd) * rho * kh * gust * (inp["ta"] - inp["sst"]),
        u_stress=rho * km * wind_rel * (inp["ua"] - inp["ocean_u"]),
        v_stress=rho * km * wind_rel * (inp["va"] - inp["ocean_v"]),
        ocean_km=km,
    )


def _land_reference(inp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rd, p0ref = constants.RD, constants.P0REF
    rd_o_cpd, vtmpc1 = constants.RD_O_CPD, constants.RV_O_RD_MINUS_1
    min_sfc_wind = 0.3  # TmxSurfaceConfig default; land reference velocity is zero
    wind_rel = np.maximum(min_sfc_wind, np.sqrt(inp["ua"] ** 2 + inp["va"] ** 2))
    qsat = inp["land_qsat_star"]  # config.land_qsat_from_star default True
    rho = inp["psfc"] / (rd * inp["land_tskin"] * (1.0 + vtmpc1 * qsat))
    theta_atm = inp["ta"] * (p0ref / inp["pa"]) ** rd_o_cpd
    theta_sfc = inp["land_tskin"] * (p0ref / inp["psfc"]) ** rd_o_cpd
    km, _kh = _exchange_solver(
        theta_atm, theta_sfc, qsat, inp["qa"], wind_rel, inp["land_rough_m"], inp["dz"]
    )
    return dict(
        evapotranspiration=inp["land_evapotrans"],
        sensible_heat_flux=inp["land_sensible_hflx"],
        u_stress=rho * km * wind_rel * inp["ua"],
        v_stress=rho * km * wind_rel * inp["va"],
        q_snocpymlt=inp["land_q_snocpymlt"],
    )


def _ice_reference(
    inp: dict[str, np.ndarray],
    ice: dict[str, np.ndarray],
    config: surface.TmxSurfaceConfig,
) -> dict[str, np.ndarray]:
    rd, p0ref = constants.RD, constants.P0REF
    rd_o_cpd, vtmpc1 = constants.RD_O_CPD, constants.RV_O_RD_MINUS_1
    ki, ks, rhoi, ci_slab = float(SI.ki), float(SI.ks), float(SI.rhoi), float(SI.ci)
    tsurf_old, hi, hs = ice["tsurf_ice_old"], inp["ice_thickness"], ice["snow_thickness"]
    # Semtner zero-layer skin-temperature update (nfg_flag = 0 by default)
    nfg = 0.0 if config.use_no_flux_gradients else 1.0
    tfw = config.ocean_freezing_temperature
    nonsolar = inp["lwflx_net"] + ice["lagged_ice_lhflx"] + ice["lagged_ice_shflx"]
    dnonsolardt = -4.0 * inp["emissivity"] * float(SI.stbo) * tsurf_old**3
    tsurf_degc = tsurf_old - _TMELT
    c_icelayer = rhoi * config.ice_layer_heat_capacity_thickness * ci_slab / _DTIME
    k_eff = ki * ks / np.maximum(ks * hi + ki * hs, 1.0e-12)
    f_s = -k_eff * (tsurf_degc - tfw)
    f_a = -inp["swflx_net"] - nonsolar
    denom = k_eff - nfg * dnonsolardt + c_icelayer
    dtemp = np.where(tsurf_degc + (f_s - f_a) / denom > 0.0, -tsurf_degc, (f_s - f_a) / denom)
    tsurf_new = np.where(hi > 0.0, tsurf_degc + dtemp + _TMELT, tfw + _TMELT)
    # exchange coefficients and bulk fluxes on the new skin temperature (no gustiness)
    wind_rel = np.maximum(
        config.min_sfc_wind,
        np.sqrt((inp["ua"] - inp["ice_u"]) ** 2 + (inp["va"] - inp["ice_v"]) ** 2),
    )
    qsat = thermo_host.specific_humidity(thermo_host.sat_pres_ice(tsurf_new), inp["psfc"])
    rho = inp["psfc"] / (rd * tsurf_new * (1.0 + vtmpc1 * qsat))
    theta_atm = inp["ta"] * (p0ref / inp["pa"]) ** rd_o_cpd
    theta_sfc = tsurf_new * (p0ref / inp["psfc"]) ** rd_o_cpd
    rough_m = np.full_like(hi, config.z0m_ice)
    km, kh = _exchange_solver(theta_atm, theta_sfc, qsat, inp["qa"], wind_rel, rough_m, inp["dz"])
    return dict(
        evapotranspiration=rho * kh * wind_rel * (inp["qa"] - qsat),
        sensible_heat_flux=float(TC.cvd) * rho * kh * wind_rel * (inp["ta"] - tsurf_new),
        u_stress=rho * km * wind_rel * (inp["ua"] - inp["ice_u"]),
        v_stress=rho * km * wind_rel * (inp["va"] - inp["ice_v"]),
        tsurf_ice_new=tsurf_new,
    )


def _build_input_state(
    grid: base.Grid,
    allocator: model_backends.BackendLike,
    frac_oce: float,
    frac_ice: float,
    frac_lnd: float,
) -> tuple[surface_states.SurfaceInputState, dict[str, np.ndarray]]:
    def rand(low: float, high: float) -> object:
        return data_alloc.random_field(grid, dims.CellDim, low=low, high=high, dtype=ta.wpfloat)

    def const(value: float) -> object:
        field = data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator)
        field.ndarray[...] = value
        return field

    active = {
        "ta": rand(240.0, 305.0),
        "qa": rand(1.0e-3, 0.02),
        "ua": rand(-20.0, 20.0),
        "va": rand(-20.0, 20.0),
        "pa": rand(9.0e4, 1.0e5),
        "psfc": rand(9.5e4, 1.05e5),
        "sst": rand(271.0, 305.0),
        "ocean_u": rand(-1.0, 1.0),
        "ocean_v": rand(-1.0, 1.0),
        "dz": rand(20.0, 100.0),
        # prescribed sea-ice inputs and radiation forcing
        "ice_u": rand(-0.5, 0.5),
        "ice_v": rand(-0.5, 0.5),
        "ice_thickness": rand(0.1, 3.0),
        "lwflx_net": rand(-100.0, 0.0),
        "swflx_net": rand(0.0, 300.0),
        "emissivity": rand(0.9, 1.0),
        # prescribed land inputs (kept valid so the land tile never produces NaN)
        "land_tskin": rand(250.0, 305.0),
        "land_rough_m": rand(0.01, 0.5),
        "land_qsat_star": rand(1.0e-3, 0.03),
        "land_evapotrans": rand(-1.0e-4, 1.0e-4),
        "land_sensible_hflx": rand(-200.0, 200.0),
        "land_q_snocpymlt": rand(0.0, 5.0),
        "frac_oce": const(frac_oce),
        "frac_ice": const(frac_ice),
        "frac_lnd": const(frac_lnd),
    }
    field_names = {f.name for f in dataclasses.fields(surface_states.SurfaceInputState)}
    zero = lambda: data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator)  # noqa: E731
    fields = {name: (active[name] if name in active else zero()) for name in field_names}
    input_state = surface_states.SurfaceInputState(**fields)
    active_np = {name: field.asnumpy() for name, field in active.items()}
    return input_state, active_np


def _run_granule(
    grid: base.Grid,
    backend_like: model_backends.BackendLike,
    frac_oce: float,
    frac_ice: float,
    frac_lnd: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    surface.TmxSurfaceConfig,
    surface.TmxSurfaceParams,
]:
    allocator = model_backends.get_allocator(backend_like)
    config = surface.TmxSurfaceConfig()
    params = surface.TmxSurfaceParams()
    granule = surface.TmxSurface(grid=grid, config=config, params=params, backend=backend_like)

    input_state, active_np = _build_input_state(grid, allocator, frac_oce, frac_ice, frac_lnd)
    surface_state = surface_states.SurfaceState.allocate(grid, allocator=allocator)
    flux_state = tmx_states.TmxSurfaceFluxState.allocate(grid, allocator=allocator)

    ocean_km_in = data_alloc.random_field(
        grid, dims.CellDim, low=1.0e-3, high=0.05, dtype=ta.wpfloat
    )
    surface_state.ocean_km.ndarray[...] = ocean_km_in.ndarray
    ocean_km_in_np = ocean_km_in.asnumpy()

    # seed the persistent ice state (old skin temperature, snow, lagged fluxes)
    ice_seed = {
        "tsurf_ice_old": data_alloc.random_field(
            grid, dims.CellDim, low=250.0, high=272.0, dtype=ta.wpfloat
        ),
        "snow_thickness": data_alloc.random_field(
            grid, dims.CellDim, low=0.0, high=0.4, dtype=ta.wpfloat
        ),
        "lagged_ice_lhflx": data_alloc.random_field(
            grid, dims.CellDim, low=-50.0, high=10.0, dtype=ta.wpfloat
        ),
        "lagged_ice_shflx": data_alloc.random_field(
            grid, dims.CellDim, low=-100.0, high=50.0, dtype=ta.wpfloat
        ),
    }
    ice_init_np = {}
    for name, field in ice_seed.items():
        getattr(surface_state, name).ndarray[...] = field.ndarray
        ice_init_np[name] = field.asnumpy()

    granule.run(
        input_state=input_state, surface_state=surface_state, flux_state=flux_state, dtime=_DTIME
    )

    produced = {
        name: getattr(flux_state, name).asnumpy()
        for name in (
            "evapotranspiration",
            "sensible_heat_flux",
            "u_stress",
            "v_stress",
            "q_snocpymlt",
        )
    }
    produced["ocean_km"] = surface_state.ocean_km.asnumpy()
    produced["tsurf_ice_new"] = surface_state.tsurf_ice_new.asnumpy()
    return produced, active_np, ocean_km_in_np, ice_init_np, config, params


def test_surface_granule_ocean_run(
    grid: base.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    produced, active_np, ocean_km_in, _ice_init, config, params = _run_granule(
        grid, backend_like, frac_oce=1.0, frac_ice=0.0, frac_lnd=0.0
    )
    ocean = _ocean_reference(active_np, ocean_km_in, config, params)
    for name in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress"):
        np.testing.assert_allclose(produced[name], ocean[name], rtol=1e-9, err_msg=name)
    # the fifth iteration stores the new km into ocean_km for the next step's Charnock lag
    np.testing.assert_allclose(produced["ocean_km"], ocean["ocean_km"], rtol=1e-9)


def test_surface_granule_ocean_land_run(
    grid: base.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    frac_oce, frac_lnd = 0.6, 0.4
    produced, active_np, ocean_km_in, _ice_init, config, params = _run_granule(
        grid, backend_like, frac_oce=frac_oce, frac_ice=0.0, frac_lnd=frac_lnd
    )
    ocean = _ocean_reference(active_np, ocean_km_in, config, params)
    land = _land_reference(active_np)
    for name in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress"):
        expected = frac_oce * ocean[name] + frac_lnd * land[name]
        np.testing.assert_allclose(produced[name], expected, rtol=1e-9, err_msg=name)
    np.testing.assert_allclose(produced["q_snocpymlt"], frac_lnd * land["q_snocpymlt"], rtol=1e-9)


def test_surface_granule_ice_run(
    grid: base.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    produced, active_np, _ocean_km_in, ice_init, config, _params = _run_granule(
        grid, backend_like, frac_oce=0.0, frac_ice=1.0, frac_lnd=0.0
    )
    ice = _ice_reference(active_np, ice_init, config)
    # the prognostic skin temperature is written back into the surface state
    np.testing.assert_allclose(produced["tsurf_ice_new"], ice["tsurf_ice_new"], rtol=1e-11)
    for name in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress"):
        np.testing.assert_allclose(produced[name], ice[name], rtol=1e-9, err_msg=name)


def test_surface_granule_ocean_ice_land_run(
    grid: base.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    frac_oce, frac_ice, frac_lnd = 0.5, 0.2, 0.3
    produced, active_np, ocean_km_in, ice_init, config, params = _run_granule(
        grid, backend_like, frac_oce=frac_oce, frac_ice=frac_ice, frac_lnd=frac_lnd
    )
    ocean = _ocean_reference(active_np, ocean_km_in, config, params)
    ice = _ice_reference(active_np, ice_init, config)
    land = _land_reference(active_np)
    # the skin-temperature update is independent of the tile fraction
    np.testing.assert_allclose(produced["tsurf_ice_new"], ice["tsurf_ice_new"], rtol=1e-11)
    for name in ("evapotranspiration", "sensible_heat_flux", "u_stress", "v_stress"):
        expected = frac_oce * ocean[name] + frac_ice * ice[name] + frac_lnd * land[name]
        np.testing.assert_allclose(produced[name], expected, rtol=1e-9, err_msg=name)
    np.testing.assert_allclose(produced["q_snocpymlt"], frac_lnd * land["q_snocpymlt"], rtol=1e-9)
