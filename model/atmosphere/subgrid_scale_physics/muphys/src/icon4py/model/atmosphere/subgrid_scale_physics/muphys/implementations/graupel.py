# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum, minimum, power, sqrt
from gt4py.next.ffront.experimental import concat_where
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.frozen import idx, g_ct, t_d
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import _fall_speed, _snow_number, _snow_lambda, _ice_number, _ice_mass, _ice_sticking, _deposition_factor, _deposition_auto_conversion, _ice_deposition_nucleation, _vel_scale_factor_default, _vel_scale_factor_ice, _vel_scale_factor_snow, _fall_speed_scalar
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import _qsat_rho, _qsat_ice_rho, _internal_energy
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import _cloud_to_rain, _rain_to_vapor, _cloud_x_ice, _cloud_to_snow, _cloud_to_graupel, _vapor_x_ice, _ice_to_snow, _ice_to_graupel, _rain_to_graupel, _snow_to_graupel, _vapor_x_snow, _vapor_x_graupel, _snow_to_rain, _graupel_to_rain

K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)

@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0, 0.0))
def _precip(
    state:     tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat],
    prefactor: ta.wpfloat,             # param[0] of fall_speed
    exponent:  ta.wpfloat,             # param[1] of fall_speed
    offset:    ta.wpfloat,             # param[1] of fall_speed
    zeta:      ta.wpfloat,             # dt/(2dz)
    vc:        ta.wpfloat,             # state dependent fall speed correction
    q:         ta.wpfloat,             # specific mass of hydrometeor
    q_kp1:     ta.wpfloat,             # specific mass in next lower cell
    rho:       ta.wpfloat,             # density
    mask:      bool                    # k-level located in cloud
) -> tuple[ta.wpfloat,ta.wpfloat,ta.wpfloat]:   # updates
    _, flx, vt = state
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0*flx
#    flx_partial = minimum(rho_x * vc * _fall_speed_scalar(rho_x, prefactor, offset, exponent), flx_eff)
    flx_partial = minimum(rho_x * vc * prefactor * power((rho_x+offset), exponent), flx_eff)
    if( mask ):
        update0 = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)    # q update
        update1 = (update0 * rho * vt + flx_partial) * 0.5                        # flux
        rho_x   = (update0 + q_kp1) * 0.5 * rho
#    update2 = vc * _fall_speed_scalar(rho_x, prefactor, offset, exponent)           # vt
        update2 = vc * prefactor * power((rho_x+offset), exponent)                # vt
    else:
        update0  = q
        update1  = 0.0
        update2  = 0.0
    return update0, update1, update2

@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0))
def _temperature_update(
    state:     tuple[ta.wpfloat, ta.wpfloat],
    t:         ta.wpfloat,
    t_kp1:     ta.wpfloat,
    ei_old:    ta.wpfloat,
    pr:        ta.wpfloat,             # precipitable rain
    pflx_tot:  ta.wpfloat,             # total precipitation flux
    qv:        ta.wpfloat,
    qliq:      ta.wpfloat,
    qice:      ta.wpfloat,
    rho:       ta.wpfloat,             # density
    dz:        ta.wpfloat,
    dt:        ta.wpfloat,
) -> tuple[ta.wpfloat,ta.wpfloat]:
    _, eflx = state
    e_int   = ei_old + eflx

    eflx    = dt * ( pr * ( t_d.clw * t - t_d.cvd * t_kp1 - g_ct.lvc ) + (pflx_tot) * ( g_ct.ci * t - t_d.cvd * t_kp1 - g_ct.lsc ) )
    e_int   = e_int - eflx

#   t       = T_from_internal_energy_scalar( min_k, e_int, qv, qliq, qice, rho, dz )
#   inlined in order to avoid scan_operator -> field_operator
    qtot    = qliq + qice + qv                          # total water specific mass
    cv      = ( t_d.cvd * ( 1.0 - qtot ) + t_d.cvv * qv + t_d.clw * qliq + g_ct.ci * qice ) * rho * dz # Moist isometric specific heat
    t       = ( e_int + rho * dz * ( qliq * g_ct.lvc + qice * g_ct.lsc )) / cv

    return t, eflx

@gtx.field_operator
def _graupel_mask(
    t:      fa.CellKField[ta.wpfloat],             # Temperature
    rho:    fa.CellKField[ta.wpfloat],             # Density
    qv:     fa.CellKField[ta.wpfloat],             # Q vapor content
    qc:     fa.CellKField[ta.wpfloat],             # Q cloud content
    qg:     fa.CellKField[ta.wpfloat],             # Q graupel content
    qi:     fa.CellKField[ta.wpfloat],             # Q ice content
    qr:     fa.CellKField[ta.wpfloat],             # Q rain content
    qs:     fa.CellKField[ta.wpfloat],             # Q snow content
) -> tuple[fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool]]:

    mask = where( (maximum( qc, maximum(qg, maximum(qi, maximum(qr, qs)))) > g_ct.qmin) | ((t < g_ct.tfrz_het2) & (qv > _qsat_ice_rho(t, rho) ) ), True, False )
    is_sig_present = where( maximum( qg, maximum(qi, qs)) > g_ct.qmin, True, False )
    kmin_r = where( qr > g_ct.qmin, True, False )
    kmin_i = where( qi > g_ct.qmin, True, False )
    kmin_s = where( qs > g_ct.qmin, True, False )
    kmin_g = where( qg > g_ct.qmin, True, False )
    return mask, is_sig_present, kmin_r, kmin_i, kmin_s, kmin_g

@gtx.field_operator
def _q_t_update(
    t:              fa.CellKField[ta.wpfloat],
    p:              fa.CellKField[ta.wpfloat],
    rho:            fa.CellKField[ta.wpfloat],
    qv:             fa.CellKField[ta.wpfloat],             # Q vapor content
    qc:             fa.CellKField[ta.wpfloat],             # Q cloud content
    qr:             fa.CellKField[ta.wpfloat],             # Q rain content
    qs:             fa.CellKField[ta.wpfloat],             # Q snow content
    qi:             fa.CellKField[ta.wpfloat],             # Q ice content
    qg:             fa.CellKField[ta.wpfloat],             # Q graupel content
    mask:           fa.CellKField[bool],
    is_sig_present: fa.CellKField[bool],
    dt:             ta.wpfloat,
    qnc:            ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:

    dvsw = qv - _qsat_rho( t, rho )
    qvsi = _qsat_ice_rho( t, rho )
    dvsi = qv - qvsi

    n_snow = _snow_number( t, rho, qs )

    l_snow = _snow_lambda( rho, qs, n_snow )

    # Define conversion 'matrix'
    sx2x_c_r = _cloud_to_rain( t, qc, qr, qnc )
    sx2x_r_v = _rain_to_vapor( t, rho, qc, qr, dvsw, dt )
    sx2x_c_i = _cloud_x_ice( t, qc, qi, dt )
    sx2x_i_c = -minimum(sx2x_c_i, 0.0)
    sx2x_c_i = maximum(sx2x_c_i, 0.0)

    sx2x_c_s = where( t < t_d.tmelt, _cloud_to_snow( t, qc, qs, n_snow, l_snow ), 0.0 )
    sx2x_c_g = where( t < t_d.tmelt, _cloud_to_graupel( t, rho, qc, qg ), 0.0 )

    n_ice    = where( t < t_d.tmelt, _ice_number( t, rho ), 0.0 )
    m_ice    = where( t < t_d.tmelt, _ice_mass( qi, n_ice ), 0.0 )
    x_ice    = where( t < t_d.tmelt, _ice_sticking( t ), 0.0 )

    eta      = where( (t < t_d.tmelt) & is_sig_present, _deposition_factor( t, qvsi ), 0.0 )
    sx2x_v_i = where( (t < t_d.tmelt) & is_sig_present, _vapor_x_ice( qi, m_ice, eta, dvsi, rho, dt ), 0.0 )
    sx2x_i_v = where( (t < t_d.tmelt) & is_sig_present, -minimum( sx2x_v_i, 0.0 ) , 0.0 )
    sx2x_v_i = where( (t < t_d.tmelt) & is_sig_present, maximum( sx2x_v_i, 0.0 ) , 0.0 )#    ice_dep  = where( (t < t_d.tmelt) & is_sig_present, minimum( sx2x_v_i, dvsi/dt ) , 0.0 )

    ice_dep  = where( t < t_d.tmelt, minimum( sx2x_v_i, dvsi / dt), 0.0 )
    # TODO: _deposition_auto_conversion yields roundoff differences in q
    sx2x_i_s = where( (t < t_d.tmelt) & is_sig_present, _deposition_auto_conversion( qi, m_ice, ice_dep ) + _ice_to_snow( qi, n_snow, l_snow, x_ice ), 0.0 )
    sx2x_i_g = where( (t < t_d.tmelt) & is_sig_present, _ice_to_graupel( rho, qr, qg, qi, x_ice ), 0.0 )
    sx2x_s_g = where( (t < t_d.tmelt) & is_sig_present, _snow_to_graupel( t, rho, qc, qs ), 0.0 )
    sx2x_r_g = where( (t < t_d.tmelt) & is_sig_present, _rain_to_graupel( t, rho, qc, qr, qi, qs, m_ice, dvsw, dt ), 0.0 )

    sx2x_v_i = where( t < t_d.tmelt, sx2x_v_i + _ice_deposition_nucleation(t, qc, qi, n_ice, dvsi, dt ), 0.0 )
    sx2x_c_r = where( t >= t_d.tmelt, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r )

    dvsw0    = where( is_sig_present, qv - _qsat_rho( t, rho ), 0.0 )  # TODO: new qsat_rho_tmelt, TODO: use dvsw ??
    sx2x_v_s = where( is_sig_present, _vapor_x_snow( t, p, rho, qs, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_s_v = where( is_sig_present, -minimum( sx2x_v_s, 0.0 ), 0.0 )
    sx2x_v_s = where( is_sig_present, maximum( sx2x_v_s, 0.0 ), 0.0 )

    sx2x_v_g = where( is_sig_present, _vapor_x_graupel( t, p, rho, qg, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_g_v = where( is_sig_present, -minimum( sx2x_v_g, 0.0 ), 0.0 )
    sx2x_v_g = where( is_sig_present, maximum( sx2x_v_g, 0.0 ), 0.0 )

    sx2x_s_r = where( is_sig_present, _snow_to_rain( t, p, rho, dvsw0, qs ), 0.0 )
    sx2x_g_r = where( is_sig_present, _graupel_to_rain( t, p, rho, dvsw0, qg ), 0.0 )

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    # SINK calculation

    # if (is_sig_present[j]) or (qx_ind[ix] == lqc) or (qx_ind[ix] == lqv) or (qx_ind[ix] == lqr)
    sink_v   = sx2x_v_s + sx2x_v_i + sx2x_v_g   # Missing sx2x_v_c + sx2x_v_r
    sink_c   = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g   # Missing  sx2x_c_v
    sink_r   = sx2x_r_v + sx2x_r_g # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
    sink_s   = where ( is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, 0.0 ) # Missing: sx2x_s_c + sx2x_s_i
    sink_i   = where ( is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, 0.0 ) # Missing: sx2x_i_r
    sink_g   = where ( is_sig_present, sx2x_g_v + sx2x_g_r, 0.0 ) # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    #  if ((sink[qx_ind[ix]] > stot) && (q[qx_ind[ix]].x[oned_vec_index] > qmin))

    stot     = qv / dt
    sx2x_v_s = where( (sink_v > stot) & (qv > g_ct.qmin), sx2x_v_s * stot / sink_v, sx2x_v_s )
    sx2x_v_i = where( (sink_v > stot) & (qv > g_ct.qmin), sx2x_v_i * stot / sink_v, sx2x_v_i )
    sx2x_v_g = where( (sink_v > stot) & (qv > g_ct.qmin), sx2x_v_g * stot / sink_v, sx2x_v_g )
    sink_v   = where( (sink_v > stot) & (qv > g_ct.qmin), sx2x_v_s + sx2x_v_i + sx2x_v_g, sink_v) # Missing: sx2x_v_c + sx2x_v_r

    stot     = qc / dt
    sx2x_c_r = where( (sink_c > stot) & (qc > g_ct.qmin), sx2x_c_r * stot / sink_c, sx2x_c_r )
    sx2x_c_s = where( (sink_c > stot) & (qc > g_ct.qmin), sx2x_c_s * stot / sink_c, sx2x_c_s )
    sx2x_c_i = where( (sink_c > stot) & (qc > g_ct.qmin), sx2x_c_i * stot / sink_c, sx2x_c_i )
    sx2x_c_g = where( (sink_c > stot) & (qc > g_ct.qmin), sx2x_c_g * stot / sink_c, sx2x_c_g )
    sink_c   = where( (sink_c > stot) & (qc > g_ct.qmin), sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g, sink_c) # Missing: sx2x_c_v

    stot     = qr / dt
    sx2x_r_v = where( (sink_r > stot) & (qr > g_ct.qmin), sx2x_r_v * stot / sink_r, sx2x_r_v )
    sx2x_r_g = where( (sink_r > stot) & (qr > g_ct.qmin), sx2x_r_g * stot / sink_r, sx2x_r_g )
    sink_r   = where( (sink_r > stot) & (qr > g_ct.qmin), sx2x_r_v + sx2x_r_g, sink_r) # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i

    stot     = qs / dt
    sx2x_s_v = where( (sink_s > stot) & (qs > g_ct.qmin), sx2x_s_v * stot / sink_s, sx2x_s_v )
    sx2x_s_r = where( (sink_s > stot) & (qs > g_ct.qmin), sx2x_s_r * stot / sink_s, sx2x_s_r )
    sx2x_s_g = where( (sink_s > stot) & (qs > g_ct.qmin), sx2x_s_g * stot / sink_s, sx2x_s_g )
    sink_s   = where( (sink_s > stot) & (qs > g_ct.qmin), sx2x_s_v + sx2x_s_r + sx2x_s_g, sink_s) # Missing: sx2x_s_c + sx2x_s_i

    stot     = qi / dt
    sx2x_i_v = where( (sink_i > stot) & (qi > g_ct.qmin), sx2x_i_v * stot / sink_i, sx2x_i_v )
    sx2x_i_c = where( (sink_i > stot) & (qi > g_ct.qmin), sx2x_i_c * stot / sink_i, sx2x_i_c )
    sx2x_i_s = where( (sink_i > stot) & (qi > g_ct.qmin), sx2x_i_s * stot / sink_i, sx2x_i_s )
    sx2x_i_g = where( (sink_i > stot) & (qi > g_ct.qmin), sx2x_i_g * stot / sink_i, sx2x_i_g )
    sink_i   = where( (sink_i > stot) & (qi > g_ct.qmin), sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, sink_i) # Missing: sx2x_i_r

    stot     = qg / dt
    sx2x_g_v = where( (sink_g > stot) & (qg > g_ct.qmin), sx2x_g_v * stot / sink_g, sx2x_g_v )
    sx2x_g_r = where( (sink_g > stot) & (qg > g_ct.qmin), sx2x_g_r * stot / sink_g, sx2x_g_r )
    sink_g   = where( (sink_g > stot) & (qg > g_ct.qmin), sx2x_g_v + sx2x_g_r, sink_g) # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    # water content updates:
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    dqdt_v = sx2x_r_v + sx2x_s_v + sx2x_i_v + sx2x_g_v - sink_v                    # Missing: sx2x_c_v
    qv     = where( mask, maximum( 0.0, qv + dqdt_v * dt ), qv )
    dqdt_c = sx2x_i_c - sink_c                     # Missing: sx2x_v_c, sx2x_r_c, sx2x_s_c, sx2x_g_c
    qc     = where( mask, maximum( 0.0, qc + dqdt_c * dt ), qc )
    dqdt_r = sx2x_c_r + sx2x_s_r + sx2x_g_r - sink_r                     # Missing: sx2x_v_r + sx2x_i_r
    qr     = where( mask, maximum( 0.0, qr + dqdt_r * dt ), qr )
    dqdt_s = sx2x_v_s + sx2x_c_s + sx2x_i_s - sink_s                     # Missing: sx2x_r_s + sx2x_g_s
    qs     = where( mask, maximum( 0.0, qs + dqdt_s * dt ), qs )
    dqdt_i = sx2x_v_i + sx2x_c_i - sink_i                    # Missing: sx2x_r_i + sx2x_s_i + sx2x_g_i
    qi     = where( mask, maximum( 0.0, qi + dqdt_i * dt ), qi )
    dqdt_g = sx2x_v_g + sx2x_c_g + sx2x_r_g + sx2x_s_g + sx2x_i_g - sink_g
    qg     = where( mask, maximum( 0.0, qg + dqdt_g * dt ), qg )

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv   = t_d.cvd + (t_d.cvv - t_d.cvd) * qtot + (t_d.clw - t_d.cvv) * qliq + (g_ct.ci - t_d.cvv) * qice
    t = where( mask, t + dt * ( (dqdt_c + dqdt_r) * (g_ct.lvc - (t_d.clw - t_d.cvv)*t) + \
                                (dqdt_i + dqdt_s + dqdt_g) * (g_ct.lsc - (g_ct.ci - t_d.cvv)*t ) ) / cv, t )
    return qv, qc, qr, qs, qi, qg, t

@gtx.field_operator
def _precipitation_effects(
    k: fa.KField[gtx.int32],
    last_lev: gtx.int32,
    kmin_r: fa.CellKField[bool],                   # rain minimum level
    kmin_i: fa.CellKField[bool],                   # ice minimum level
    kmin_s: fa.CellKField[bool],                   # snow minimum level
    kmin_g: fa.CellKField[bool],                   # graupel minimum level
    qv:     fa.CellKField[ta.wpfloat],             # Q vapor content
    qc:     fa.CellKField[ta.wpfloat],             # Q cloud content
    qr:     fa.CellKField[ta.wpfloat],             # Q rain content
    qs:     fa.CellKField[ta.wpfloat],             # Q snow content
    qi:     fa.CellKField[ta.wpfloat],             # Q ice content
    qg:     fa.CellKField[ta.wpfloat],             # Q graupel content    qv,
    t:      fa.CellKField[ta.wpfloat],             # temperature,
    rho:    fa.CellKField[ta.wpfloat],             # density
    dz:     fa.CellKField[ta.wpfloat],
    dt:     ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],\
           fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],\
           fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:

    # Store current fields for later temperature update
    qliq    = qc + qr
    qice    = qs + qi + qg
    ei_old  = _internal_energy( t, qv, qliq, qice, rho, dz )

    zeta  = dt / (2.0 * dz )
    xrho  = sqrt( g_ct.rho_00 / rho )

    # NP = 4    qp_ind[] = {lqr, lqi, lqs, lqg};
    vc_r = where( kmin_r, _vel_scale_factor_default( xrho ), 0.0 )
    vc_i = where( kmin_i, _vel_scale_factor_ice( xrho ), 0.0 )
    vc_s = where( kmin_s, _vel_scale_factor_snow( xrho, rho, t, qs ), 0.0 )
    vc_g = where( kmin_g, _vel_scale_factor_default( xrho ), 0.0 )

    q_kp1     = concat_where( k < last_lev, qr(Koff[1]), qr )
    qr, pr, _ = _precip( idx.prefactor_r, idx.exponent_r, idx.offset_r, zeta, vc_r, qr, q_kp1, rho, True )
    q_kp1     = concat_where( k < last_lev, qi(Koff[1]), qi )
    qi, pi, _ = _precip( idx.prefactor_i, idx.exponent_i, idx.offset_i, zeta, vc_i, qi, q_kp1, rho, True )
    q_kp1     = concat_where( k < last_lev, qs(Koff[1]), qs )
    qs, ps, _ = _precip( idx.prefactor_s, idx.exponent_s, idx.offset_s, zeta, vc_s, qs, q_kp1, rho, True )
    q_kp1     = concat_where( k < last_lev, qg(Koff[1]), qg )
    qg, pg, _ = _precip( idx.prefactor_g, idx.exponent_g, idx.offset_g, zeta, vc_g, qg, q_kp1, rho, True )

    qliq      = qc + qr
    qice      = qs + qi + qg
    p_sig     = ps + pi + pg
    t_kp1     = concat_where( k < last_lev, t(Koff[1]), t )
    t, eflx   = _temperature_update( t, t_kp1, ei_old, pr, p_sig, qv, qliq, qice, rho, dz, dt )
    # TODO: here we have to return a single layer for pre_gsp
    return qr, qs, qi, qg, t, p_sig+pr, pr, ps, pi, pg, eflx/dt

@gtx.field_operator
def _output_calculation(
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qx_hold:   fa.CellKField[ta.wpfloat],             # TBD
    qx:        fa.CellKField[ta.wpfloat],             # TBD
    Tx_hold:   fa.CellKField[ta.wpfloat],             # TBD
    Tx:        fa.CellKField[ta.wpfloat],             # TBD
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:                       # Internal energy

    te  = where( ( qve+qce <= qx_hold ), Tx_hold, Tx )
    qce = where( ( qve+qce <= qx_hold ), 0.0, maximum(qve+qce-qx, 0.0) )
    qve = where( ( qve+qce <= qx_hold ), qve+qce, qx )
    return te, qve, qce

@gtx.field_operator
def _graupel_run(
    k: fa.KField[gtx.int32],
    last_lev:  gtx.int32,
    dz:        fa.CellKField[ta.wpfloat],             #
    te:        fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Pressure
    rho:       fa.CellKField[ta.wpfloat],             # Density containing dry air and water constituents
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellKField[ta.wpfloat],             # Specific rain water
    qse:       fa.CellKField[ta.wpfloat],             # Specific snow water
    qie:       fa.CellKField[ta.wpfloat],             # Specific ice water content
    qge:       fa.CellKField[ta.wpfloat],             # Specific graupel water content
    dt:        ta.wpfloat,
    qnc:       ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat], \
           fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat], \
           fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat], \
           fa.CellKField[ta.wpfloat]]:
    mask, is_sig_present, kmin_r, kmin_i, kmin_s, kmin_g = _graupel_mask(te, rho, qve, qce, qge, qie, qre, qse )
    qv, qc, qr, qs, qi, qg, t = _q_t_update( te, p, rho, qve, qce, qre, qse, qie, qge, mask, is_sig_present, dt, qnc )
    qr, qs, qi, qg, t, pflx, pr, ps, pi, pg, pre = \
        _precipitation_effects( k, last_lev, kmin_r, kmin_i, kmin_s, kmin_g, qv, qc, qr, qs, qi, qg, t, rho, dz, dt )

    return t, qv, qc, qr, qs, qi, qg, pflx, pr, ps, pi, pg, pre

# TODO : program  needs to be called with offset_provider={"Koff": K}
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_run(
    k: fa.KField[gtx.int32],
    last_lev:  gtx.int32,
    dz:        fa.CellKField[ta.wpfloat],             #
    te:        fa.CellKField[ta.wpfloat],             # Temperature
    p:         fa.CellKField[ta.wpfloat],             # Pressure
    rho:       fa.CellKField[ta.wpfloat],             # Density containing dry air and water constituents
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidityn
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellKField[ta.wpfloat],             # Specific rain water
    qse:       fa.CellKField[ta.wpfloat],             # Specific snow water
    qie:       fa.CellKField[ta.wpfloat],             # Specific ice water content
    qge:       fa.CellKField[ta.wpfloat],             # Specific graupel water content
    dt:        ta.wpfloat,                            # Time step
    qnc:       ta.wpfloat,                            #
    t_out:     fa.CellKField[ta.wpfloat],             # Revised temperature
    qv_out:    fa.CellKField[ta.wpfloat],             # Revised humidity
    qc_out:    fa.CellKField[ta.wpfloat],             # Revised cloud water
    qr_out:    fa.CellKField[ta.wpfloat],             # Revised rain water
    qs_out:    fa.CellKField[ta.wpfloat],             # Revised snow water
    qi_out:    fa.CellKField[ta.wpfloat],             # Revised ice water
    qg_out:    fa.CellKField[ta.wpfloat],             # Revised graupel water
    pflx:      fa.CellKField[ta.wpfloat],             # Total precipitation flux
    pr:        fa.CellKField[ta.wpfloat],             # Precipitation of rain
    ps:        fa.CellKField[ta.wpfloat],             # Precipitation of snow
    pi:        fa.CellKField[ta.wpfloat],             # Precipitation of ice
    pg:        fa.CellKField[ta.wpfloat],             # Precipitation of graupel
    pre:       fa.CellKField[ta.wpfloat],             # Precipitation of graupel
):
    _graupel_run(k, last_lev, dz, te, p, rho, qve, qce, qre, qse, qie, qge, dt, qnc, \
                 out=(t_out, qv_out, qc_out, qr_out, qs_out, qi_out, qg_out, pflx, pr, ps, pi, pg, pre) )
