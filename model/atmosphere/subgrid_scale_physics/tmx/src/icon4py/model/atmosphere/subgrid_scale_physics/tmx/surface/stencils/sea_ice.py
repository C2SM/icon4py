# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.constants import PhysicsConstants, SeaIceConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_ice_nonsolar_forcing(
    lwflx_net: fa.CellField[wpfloat],
    lhflx: fa.CellField[wpfloat],
    shflx: fa.CellField[wpfloat],
    tsurf_old: fa.CellField[wpfloat],
    emissivity: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Assemble the non-solar surface flux and its temperature derivative for the ice model.

    Port of the forcing build in 'update_sea_ice' (mo_tmx_surface.f90:350-351):
    ``nonsolar = lwflx_net + lhflx + shflx`` and
    ``dnonsolar/dT = -4 * emissivity * stbo * Tsurf^3`` (Tsurf in Kelvin).

    Args:
        lwflx_net: net longwave radiation at the surface [W/m^2]
        lhflx: ice-tile latent heat flux (lagged one step) [W/m^2]
        shflx: ice-tile sensible heat flux (lagged one step) [W/m^2]
        tsurf_old: ice surface temperature at the current step [K]
        emissivity: surface longwave emissivity [-]

    Returns:
        (nonsolar, dnonsolardt): non-solar flux [W/m^2] and its d/dT [W/(m^2 K)]
    """
    nonsolar = lwflx_net + lhflx + shflx
    dnonsolardt = (
        -wpfloat("4.0") * emissivity * SeaIceConstants.stbo * tsurf_old * tsurf_old * tsurf_old
    )
    return nonsolar, dnonsolardt


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ice_nonsolar_forcing(
    lwflx_net: fa.CellField[wpfloat],
    lhflx: fa.CellField[wpfloat],
    shflx: fa.CellField[wpfloat],
    tsurf_old: fa.CellField[wpfloat],
    emissivity: fa.CellField[wpfloat],
    nonsolar: fa.CellField[wpfloat],
    dnonsolardt: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _compute_ice_nonsolar_forcing(
        lwflx_net=lwflx_net,
        lhflx=lhflx,
        shflx=shflx,
        tsurf_old=tsurf_old,
        emissivity=emissivity,
        out=(nonsolar, dnonsolardt),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _set_ice_temp_zerolayer(
    tsurf_old: fa.CellField[wpfloat],
    hi: fa.CellField[wpfloat],
    hs: fa.CellField[wpfloat],
    swnet: fa.CellField[wpfloat],
    nonsolar: fa.CellField[wpfloat],
    dnonsolardt: fa.CellField[wpfloat],
    freezing_temperature: wpfloat,
    heat_capacity_thickness: wpfloat,
    nonsolar_gradient_flag: wpfloat,
    dtime: wpfloat,
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Semtner zero-layer sea-ice surface temperature update.

    Port of 'set_ice_temp_zerolayer' (mo_ice_zerolayer.f90:296-325). The surface
    temperature is carried in Kelvin; the update works in degC internally.

        c_icelayer = rhoi * hci_layer * ci / dt
        k_eff      = ki*ks / (ks*hi + ki*hs)
        F_S        = -k_eff * (Tsurf - Tfw)
        F_A        = -SWnet - nonsolar
        denom      = k_eff - nfg_flag * dnonsolardt + c_icelayer
        deltaT     = (F_S - F_A) / denom, clamped so Tsurf does not exceed 0 degC
        Qtop       = -F_A + F_S - denom * deltaT
        Qbot       = -F_S + k_eff * deltaT

    Open water (``hi <= 0``): Tsurf is reset to the freezing temperature and Qtop,
    Qbot are zero. ``nonsolar_gradient_flag`` is 0 when ``use_no_flux_gradients``
    is True (the default), dropping the dnonsolardt term.

    Args:
        tsurf_old: ice surface temperature at the current step [K]
        hi: sea-ice thickness [m]
        hs: snow thickness [m]
        swnet: net shortwave flux at the surface [W/m^2]
        nonsolar: non-solar flux [W/m^2]
        dnonsolardt: temperature derivative of the non-solar flux [W/(m^2 K)]
        freezing_temperature: sea-water freezing temperature (Tf) [degC]
        heat_capacity_thickness: ice heat-capacity slab thickness (hci_layer) [m]
        nonsolar_gradient_flag: 1 to keep the dnonsolardt term, 0 to drop it
        dtime: time step [s]

    Returns:
        (tsurf_new, qtop, qbot): new surface temperature [K], surface-melt and
        bottom-melt heat fluxes [W/m^2]
    """
    tmelt = PhysicsConstants.tmelt
    tsurf_degc = tsurf_old - tmelt
    c_icelayer = SeaIceConstants.rhoi * heat_capacity_thickness * SeaIceConstants.ci / dtime
    k_eff = (
        SeaIceConstants.ki
        * SeaIceConstants.ks
        / maximum(SeaIceConstants.ks * hi + SeaIceConstants.ki * hs, wpfloat("1.0e-12"))
    )
    conductive_flux = -k_eff * (tsurf_degc - freezing_temperature)
    atmospheric_flux = -swnet - nonsolar
    denominator = k_eff - nonsolar_gradient_flag * dnonsolardt + c_icelayer
    delta_temperature_raw = (conductive_flux - atmospheric_flux) / denominator
    # clamp to melting: surface temperature may not exceed 0 degC
    is_melting = tsurf_degc + delta_temperature_raw > wpfloat("0.0")
    delta_temperature = where(is_melting, -tsurf_degc, delta_temperature_raw)
    has_ice = hi > wpfloat("0.0")
    tsurf_new = where(has_ice, tsurf_degc + delta_temperature + tmelt, freezing_temperature + tmelt)
    # Qtop is zero unless the surface is clamped to melting; computing it uniformly
    # (as the Fortran does) leaves ~1e-14 cancellation noise where unclamped, since
    # denominator * delta_temperature == F_S - F_A there.
    qtop = where(
        has_ice & is_melting,
        -atmospheric_flux + conductive_flux - denominator * delta_temperature,
        wpfloat("0.0"),
    )
    qbot = where(has_ice, -conductive_flux + k_eff * delta_temperature, wpfloat("0.0"))
    return tsurf_new, qtop, qbot


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_ice_temp_zerolayer(
    tsurf_old: fa.CellField[wpfloat],
    hi: fa.CellField[wpfloat],
    hs: fa.CellField[wpfloat],
    swnet: fa.CellField[wpfloat],
    nonsolar: fa.CellField[wpfloat],
    dnonsolardt: fa.CellField[wpfloat],
    tsurf_new: fa.CellField[wpfloat],
    qtop: fa.CellField[wpfloat],
    qbot: fa.CellField[wpfloat],
    freezing_temperature: wpfloat,
    heat_capacity_thickness: wpfloat,
    nonsolar_gradient_flag: wpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _set_ice_temp_zerolayer(
        tsurf_old,
        hi,
        hs,
        swnet,
        nonsolar,
        dnonsolardt,
        freezing_temperature,
        heat_capacity_thickness,
        nonsolar_gradient_flag,
        dtime,
        out=(tsurf_new, qtop, qbot),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _set_ice_albedo(
    tsurf_new: fa.CellField[wpfloat],
    hi: fa.CellField[wpfloat],
    hs: fa.CellField[wpfloat],
) -> tuple[
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
    fa.CellField[wpfloat],
]:
    """
    Sea-ice albedo, scheme 1 (temperature-weighted; mo_ice_parameterizations.f90:89-113).

    ``albflag = 1 / (1 + albtrans * Tsurf_degC^2)`` blends the melting and cold
    values for snow (``hs > 1 cm``) or bare ice. All four spectral bands are equal.

    Args:
        tsurf_new: ice surface temperature [K]
        hi: sea-ice thickness [m]
        hs: snow thickness [m]

    Returns:
        (albvisdir, albvisdif, albnirdir, albnirdif): surface albedos [-]
    """
    albtrans = wpfloat("0.5")
    albs = wpfloat("0.85")
    albsm = wpfloat("0.70")
    albi = wpfloat("0.75")
    albim = wpfloat("0.70")
    tsurf_degc = tsurf_new - PhysicsConstants.tmelt
    albflag = wpfloat("1.0") / (wpfloat("1.0") + albtrans * tsurf_degc * tsurf_degc)
    snow_albedo = albflag * albsm + (wpfloat("1.0") - albflag) * albs
    ice_albedo = albflag * albim + (wpfloat("1.0") - albflag) * albi
    albedo = where(
        hi > wpfloat("0.0"),
        where(hs > wpfloat("0.01"), snow_albedo, ice_albedo),
        wpfloat("0.0"),
    )
    return albedo, albedo, albedo, albedo


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_ice_albedo(
    tsurf_new: fa.CellField[wpfloat],
    hi: fa.CellField[wpfloat],
    hs: fa.CellField[wpfloat],
    albvisdir: fa.CellField[wpfloat],
    albvisdif: fa.CellField[wpfloat],
    albnirdir: fa.CellField[wpfloat],
    albnirdif: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _set_ice_albedo(
        tsurf_new,
        hi,
        hs,
        out=(albvisdir, albvisdif, albnirdir, albnirdif),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
