# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import maximum, minimum, power, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import g_ct, idx, t_d
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    _deposition_auto_conversion,
    _deposition_factor,
    _ice_deposition_nucleation,
    _ice_mass,
    _ice_number,
    _ice_sticking,
    _snow_lambda,
    _snow_number,
    _vel_scale_factor_default,
    _vel_scale_factor_ice,
    _vel_scale_factor_snow,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import (
    _internal_energy,
    _qsat_ice_rho,
    _qsat_rho,
    _qsat_rho_tmelt,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import (
    _cloud_to_graupel,
    _cloud_to_rain,
    _cloud_to_snow,
    _cloud_x_ice,
    _graupel_to_rain,
    _ice_to_graupel,
    _ice_to_snow,
    _rain_to_graupel,
    _rain_to_vapor,
    _snow_to_graupel,
    _snow_to_rain,
    _vapor_x_graupel,
    _vapor_x_ice,
    _vapor_x_snow,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(0.0, 0.0, 0.0, False))
def _precip(
    state: tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, bool],
    prefactor: ta.wpfloat,  # param[0] of fall_speed
    exponent: ta.wpfloat,  # param[1] of fall_speed
    offset: ta.wpfloat,  # param[1] of fall_speed
    zeta: ta.wpfloat,  # dt/(2dz)
    vc: ta.wpfloat,  # state dependent fall speed correction
    q: ta.wpfloat,  # specific mass of hydrometeor
    q_kp1: ta.wpfloat,  # specific mass in next lower cell
    rho: ta.wpfloat,  # density
    mask: bool,  # k-level located in cloud
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, bool]:  # updates
    _, flx, vt, is_level_activated = state
    is_level_activated = is_level_activated | mask
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0 * flx
    #   Inlined calculation using _fall_speed_scalar
    flx_partial = minimum(rho_x * vc * prefactor * power((rho_x + offset), exponent), flx_eff)
    if is_level_activated:
        update0 = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)  # q update
        update1 = (update0 * rho * vt + flx_partial) * 0.5  # flux
        rho_x = (update0 + q_kp1) * 0.5 * rho
        # Inlined calculation using _fall_speed_scalar
        update2 = vc * prefactor * power((rho_x + offset), exponent)  # vt
    else:
        update0 = q
        update1 = 0.0
        update2 = 0.0
    return update0, update1, update2, is_level_activated


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(0.0, 0.0, False))
def _temperature_update(
    state: tuple[ta.wpfloat, ta.wpfloat, bool],
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,
    ei_old: ta.wpfloat,
    pr: ta.wpfloat,  # precipitable rain
    pflx_tot: ta.wpfloat,  # total precipitation flux
    qv: ta.wpfloat,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,  # density
    dz: ta.wpfloat,
    dt: ta.wpfloat,
    mask: bool,
) -> tuple[ta.wpfloat, ta.wpfloat, bool]:
    _, eflx, is_level_activated = state
    is_level_activated = is_level_activated | mask
    if is_level_activated:
        e_int = ei_old + eflx

        eflx = dt * (
            pr * (t_d.clw * t - t_d.cvd * t_kp1 - g_ct.lvc)
            + (pflx_tot) * (g_ct.ci * t - t_d.cvd * t_kp1 - g_ct.lsc)
        )
        e_int = e_int - eflx

        #  Inlined calculation using T_from_internal_energy_scalar
        #  in order to avoid scan_operator -> field_operator
        qtot = qliq + qice + qv  # total water specific mass
        cv = (
            (t_d.cvd * (1.0 - qtot) + t_d.cvv * qv + t_d.clw * qliq + g_ct.ci * qice) * rho * dz
        )  # Moist isometric specific heat
        t = (e_int + rho * dz * (qliq * g_ct.lvc + qice * g_ct.lsc)) / cv

    return t, eflx, is_level_activated


@gtx.field_operator
def _q_t_update(  # noqa: PLR0915 [too-many-statements]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    q: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
) -> tuple[
    Q,
    fa.CellKField[ta.wpfloat],
]:
    mask = where(
        (maximum(q.c, maximum(q.g, maximum(q.i, maximum(q.r, q.s)))) > g_ct.qmin)
        | ((t < g_ct.tfrz_het2) & (q.v > _qsat_ice_rho(t, rho))),
        True,
        False,
    )
    is_sig_present = maximum(q.g, maximum(q.i, q.s)) > g_ct.qmin

    dvsw = q.v - _qsat_rho(t, rho)
    qvsi = _qsat_ice_rho(t, rho)
    dvsi = q.v - qvsi
    n_snow = _snow_number(t, rho, q.s)

    l_snow = _snow_lambda(rho, q.s, n_snow)

    # Define conversion 'matrix'
    sx2x_c_r = _cloud_to_rain(t, q.c, q.r, qnc)
    sx2x_r_v = _rain_to_vapor(t, rho, q.c, q.r, dvsw, dt)
    sx2x_c_i = _cloud_x_ice(t, q.c, q.i, dt)
    sx2x_i_c = -minimum(sx2x_c_i, 0.0)
    sx2x_c_i = maximum(sx2x_c_i, 0.0)

    sx2x_c_s = _cloud_to_snow(t, q.c, q.s, n_snow, l_snow)
    sx2x_c_g = _cloud_to_graupel(t, rho, q.c, q.g)

    t_below_tmelt = t < t_d.tmelt
    t_at_least_tmelt = not t_below_tmelt

    n_ice = _ice_number(t, rho)
    m_ice = _ice_mass(q.i, n_ice)
    x_ice = _ice_sticking(t)

    eta = where(t_below_tmelt & is_sig_present, _deposition_factor(t, qvsi), 0.0)
    sx2x_v_i = where(
        t_below_tmelt & is_sig_present, _vapor_x_ice(q.i, m_ice, eta, dvsi, rho, dt), 0.0
    )
    sx2x_i_v = where(t_below_tmelt & is_sig_present, -minimum(sx2x_v_i, 0.0), 0.0)
    sx2x_v_i = where(t_below_tmelt & is_sig_present, maximum(sx2x_v_i, 0.0), sx2x_i_v)

    ice_dep = where(t_below_tmelt & is_sig_present, minimum(sx2x_v_i, dvsi / dt), 0.0)
    # TODO(): _deposition_auto_conversion yields roundoff differences in sx2x_i_s
    sx2x_i_s = where(
        t_below_tmelt & is_sig_present,
        _deposition_auto_conversion(q.i, m_ice, ice_dep) + _ice_to_snow(q.i, n_snow, l_snow, x_ice),
        0.0,
    )
    sx2x_i_g = where(
        t_below_tmelt & is_sig_present, _ice_to_graupel(rho, q.r, q.g, q.i, x_ice), 0.0
    )
    sx2x_s_g = where(t_below_tmelt & is_sig_present, _snow_to_graupel(t, rho, q.c, q.s), 0.0)
    sx2x_r_g = where(
        t_below_tmelt & is_sig_present,
        _rain_to_graupel(t, rho, q.c, q.r, q.i, q.s, m_ice, dvsw, dt),
        0.0,
    )

    sx2x_v_i = where(
        t_below_tmelt, sx2x_v_i + _ice_deposition_nucleation(t, q.c, q.i, n_ice, dvsi, dt), 0.0
    )  # 0.0 or sx2x_v_i both OK
    sx2x_c_r = where(t_at_least_tmelt, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r)
    sx2x_c_s = where(t_at_least_tmelt, 0.0, sx2x_c_s)
    sx2x_c_g = where(t_at_least_tmelt, 0.0, sx2x_c_g)
    ice_dep = where(t_at_least_tmelt, 0.0, ice_dep)
    eta = where(t_at_least_tmelt, 0.0, eta)

    dvsw0 = where(is_sig_present, q.v - _qsat_rho_tmelt(rho), 0.0)
    sx2x_v_s = where(
        is_sig_present,
        _vapor_x_snow(t, p, rho, q.s, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt),
        0.0,
    )
    sx2x_s_v = where(is_sig_present, -minimum(sx2x_v_s, 0.0), 0.0)
    sx2x_v_s = where(is_sig_present, maximum(sx2x_v_s, 0.0), 0.0)

    sx2x_v_g = where(is_sig_present, _vapor_x_graupel(t, p, rho, q.g, dvsw, dvsi, dvsw0, dt), 0.0)
    sx2x_g_v = where(is_sig_present, -minimum(sx2x_v_g, 0.0), 0.0)
    sx2x_v_g = where(is_sig_present, maximum(sx2x_v_g, 0.0), 0.0)

    sx2x_s_r = where(is_sig_present, _snow_to_rain(t, p, rho, dvsw0, q.s), 0.0)
    sx2x_g_r = where(is_sig_present, _graupel_to_rain(t, p, rho, dvsw0, q.g), 0.0)

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    # SINK calculation

    # if (is_sig_present[j]) or (qx_ind[ix] == lqc) or (qx_ind[ix] == lqv) or (qx_ind[ix] == lqr)
    sink_v = sx2x_v_s + sx2x_v_i + sx2x_v_g  # Missing sx2x_v_c + sx2x_v_r
    sink_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g  # Missing  sx2x_c_v
    sink_r = sx2x_r_v + sx2x_r_g  # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
    sink_s = where(
        is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, 0.0
    )  # Missing: sx2x_s_c + sx2x_s_i
    sink_i = where(
        is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, 0.0
    )  # Missing: sx2x_i_r
    sink_g = where(
        is_sig_present, sx2x_g_v + sx2x_g_r, 0.0
    )  # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    #  if ((sink[qx_ind[ix]] > stot) && (q[qx_ind[ix]].x[oned_vec_index] > qmin))
    stot = q.v / dt
    sink_v_saturated = (sink_v > stot) & (q.v > g_ct.qmin)
    sx2x_v_s = where(sink_v_saturated, sx2x_v_s * stot / sink_v, sx2x_v_s)
    sx2x_v_i = where(sink_v_saturated, sx2x_v_i * stot / sink_v, sx2x_v_i)
    sx2x_v_g = where(sink_v_saturated, sx2x_v_g * stot / sink_v, sx2x_v_g)
    sink_v = where(
        sink_v_saturated, sx2x_v_s + sx2x_v_i + sx2x_v_g, sink_v
    )  # Missing: sx2x_v_c + sx2x_v_r

    stot = q.c / dt
    sink_c_saturated = (sink_c > stot) & (q.c > g_ct.qmin)
    sx2x_c_r = where(sink_c_saturated, sx2x_c_r * stot / sink_c, sx2x_c_r)
    sx2x_c_s = where(sink_c_saturated, sx2x_c_s * stot / sink_c, sx2x_c_s)
    sx2x_c_i = where(sink_c_saturated, sx2x_c_i * stot / sink_c, sx2x_c_i)
    sx2x_c_g = where(sink_c_saturated, sx2x_c_g * stot / sink_c, sx2x_c_g)
    sink_c = where(
        sink_c_saturated, sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g, sink_c
    )  # Missing: sx2x_c_v

    stot = q.r / dt
    sink_r_saturated = (sink_r > stot) & (q.r > g_ct.qmin)
    sx2x_r_v = where(sink_r_saturated, sx2x_r_v * stot / sink_r, sx2x_r_v)
    sx2x_r_g = where(sink_r_saturated, sx2x_r_g * stot / sink_r, sx2x_r_g)
    sink_r = where(
        sink_r_saturated, sx2x_r_v + sx2x_r_g, sink_r
    )  # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i

    stot = q.s / dt
    sink_s_saturated = (sink_s > stot) & (q.s > g_ct.qmin)
    sx2x_s_v = where(sink_s_saturated, sx2x_s_v * stot / sink_s, sx2x_s_v)
    sx2x_s_r = where(sink_s_saturated, sx2x_s_r * stot / sink_s, sx2x_s_r)
    sx2x_s_g = where(sink_s_saturated, sx2x_s_g * stot / sink_s, sx2x_s_g)
    sink_s = where(
        sink_s_saturated, sx2x_s_v + sx2x_s_r + sx2x_s_g, sink_s
    )  # Missing: sx2x_s_c + sx2x_s_i

    stot = q.i / dt
    sink_i_saturated = (sink_i > stot) & (q.i > g_ct.qmin)
    sx2x_i_v = where(sink_i_saturated, sx2x_i_v * stot / sink_i, sx2x_i_v)
    sx2x_i_c = where(sink_i_saturated, sx2x_i_c * stot / sink_i, sx2x_i_c)
    sx2x_i_s = where(sink_i_saturated, sx2x_i_s * stot / sink_i, sx2x_i_s)
    sx2x_i_g = where(sink_i_saturated, sx2x_i_g * stot / sink_i, sx2x_i_g)
    sink_i = where(
        sink_i_saturated, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, sink_i
    )  # Missing: sx2x_i_r

    stot = q.g / dt
    sink_g_saturated = (sink_g > stot) & (q.g > g_ct.qmin)
    sx2x_g_v = where(sink_g_saturated, sx2x_g_v * stot / sink_g, sx2x_g_v)
    sx2x_g_r = where(sink_g_saturated, sx2x_g_r * stot / sink_g, sx2x_g_r)
    sink_g = where(
        sink_g_saturated, sx2x_g_v + sx2x_g_r, sink_g
    )  # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    # water content updates:
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    dqdt_v = sx2x_r_v + sx2x_s_v + sx2x_i_v + sx2x_g_v - sink_v  # Missing: sx2x_c_v
    qv = where(mask, maximum(0.0, q.v + dqdt_v * dt), q.v)
    dqdt_c = sx2x_i_c - sink_c  # Missing: sx2x_v_c, sx2x_r_c, sx2x_s_c, sx2x_g_c
    qc = where(mask, maximum(0.0, q.c + dqdt_c * dt), q.c)
    dqdt_r = sx2x_c_r + sx2x_s_r + sx2x_g_r - sink_r  # Missing: sx2x_v_r + sx2x_i_r
    qr = where(mask, maximum(0.0, q.r + dqdt_r * dt), q.r)
    dqdt_s = sx2x_v_s + sx2x_c_s + sx2x_i_s - sink_s  # Missing: sx2x_r_s + sx2x_g_s
    qs = where(mask, maximum(0.0, q.s + dqdt_s * dt), q.s)
    dqdt_i = sx2x_v_i + sx2x_c_i - sink_i  # Missing: sx2x_r_i + sx2x_s_i + sx2x_g_i
    qi = where(mask, maximum(0.0, q.i + dqdt_i * dt), q.i)
    dqdt_g = sx2x_v_g + sx2x_c_g + sx2x_r_g + sx2x_s_g + sx2x_i_g - sink_g
    qg = where(mask, maximum(0.0, q.g + dqdt_g * dt), q.g)

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv = (
        t_d.cvd
        + (t_d.cvv - t_d.cvd) * qtot
        + (t_d.clw - t_d.cvv) * qliq
        + (g_ct.ci - t_d.cvv) * qice
    )
    t = where(
        mask,
        t
        + dt
        * (
            (dqdt_c + dqdt_r) * (g_ct.lvc - (t_d.clw - t_d.cvv) * t)
            + (dqdt_i + dqdt_s + dqdt_g) * (g_ct.lsc - (g_ct.ci - t_d.cvv) * t)
        )
        / cv,
        t,
    )
    return Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg), t


@gtx.field_operator
def _precipitation_effects(
    last_lev: gtx.int32,
    kmin_r: fa.CellKField[bool],  # rain minimum level
    kmin_i: fa.CellKField[bool],  # ice minimum level
    kmin_s: fa.CellKField[bool],  # snow minimum level
    kmin_g: fa.CellKField[bool],  # graupel minimum level
    q_in: Q,
    t: fa.CellKField[ta.wpfloat],  # temperature,
    rho: fa.CellKField[ta.wpfloat],  # density
    dz: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    # Store current fields for later temperature update
    qliq = q_in.c + q_in.r
    qice = q_in.s + q_in.i + q_in.g
    ei_old = _internal_energy(t, q_in.v, qliq, qice, rho, dz)
    zeta = dt / (2.0 * dz)
    xrho = sqrt(g_ct.rho_00 / rho)

    vc_r = _vel_scale_factor_default(xrho)
    vc_s = _vel_scale_factor_snow(xrho, rho, t, q_in.s)
    vc_i = _vel_scale_factor_ice(xrho)
    vc_g = _vel_scale_factor_default(xrho)

    q_kp1 = concat_where(dims.KDim < last_lev, q_in.r(Koff[1]), q_in.r)
    qr, pr, _, _ = _precip(
        idx.prefactor_r, idx.exponent_r, idx.offset_r, zeta, vc_r, q_in.r, q_kp1, rho, kmin_r
    )
    q_kp1 = concat_where(dims.KDim < last_lev, q_in.s(Koff[1]), q_in.s)
    qs, ps, _, _ = _precip(
        idx.prefactor_s, idx.exponent_s, idx.offset_s, zeta, vc_s, q_in.s, q_kp1, rho, kmin_s
    )
    q_kp1 = concat_where(dims.KDim < last_lev, q_in.i(Koff[1]), q_in.i)
    qi, pi, _, _ = _precip(
        idx.prefactor_i, idx.exponent_i, idx.offset_i, zeta, vc_i, q_in.i, q_kp1, rho, kmin_i
    )
    q_kp1 = concat_where(dims.KDim < last_lev, q_in.g(Koff[1]), q_in.g)
    qg, pg, _, _ = _precip(
        idx.prefactor_g, idx.exponent_g, idx.offset_g, zeta, vc_g, q_in.g, q_kp1, rho, kmin_g
    )

    qliq = q_in.c + qr
    qice = qs + qi + qg
    p_sig = ps + pi + pg
    t_kp1 = concat_where(dims.KDim < last_lev, t(Koff[1]), t)
    kmin_rsig = kmin_r | kmin_s | kmin_i | kmin_g
    t, eflx, _ = _temperature_update(
        t, t_kp1, ei_old, pr, p_sig, q_in.v, qliq, qice, rho, dz, dt, kmin_rsig
    )

    return qr, qs, qi, qg, t, p_sig + pr, pr, ps, pi, pg, eflx / dt


@gtx.field_operator
def _graupel_run(
    last_lev: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    Q,
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    kmin_r = where(q_in.r > g_ct.qmin, True, False)
    kmin_i = where(q_in.i > g_ct.qmin, True, False)
    kmin_s = where(q_in.s > g_ct.qmin, True, False)
    kmin_g = where(q_in.g > g_ct.qmin, True, False)
    q, t = _q_t_update(te, p, rho, q_in, dt, qnc)
    qr, qs, qi, qg, t, pflx, pr, ps, pi, pg, pre = _precipitation_effects(
        last_lev, kmin_r, kmin_i, kmin_s, kmin_g, q, t, rho, dz, dt
    )

    return t, Q(v=q.v, c=q.c, r=qr, s=qs, i=qi, g=qg), pflx, pr, ps, pi, pg, pre


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_run(
    last_lev: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: Q,
    dt: ta.wpfloat,  # Time step
    qnc: ta.wpfloat,
    q_out: Q,
    t_out: fa.CellKField[ta.wpfloat],  # Revised temperature
    pflx: fa.CellKField[ta.wpfloat],  # Total precipitation flux
    pr: fa.CellKField[ta.wpfloat],  # Precipitation of rain
    ps: fa.CellKField[ta.wpfloat],  # Precipitation of snow
    pi: fa.CellKField[ta.wpfloat],  # Precipitation of ice
    pg: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    pre: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _graupel_run(
        last_lev,  # TODO vertical_end - 1
        dz,
        te,
        p,
        rho,
        q_in,
        dt,
        qnc,
        out=(t_out, q_out, pflx, pr, ps, pi, pg, pre),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
