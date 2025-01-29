# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum, minimum, power, sqrt
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import _fall_speed, _snow_number, _snow_lambda, _ice_number, _ice_mass, _ice_sticking, _deposition_factor, _deposition_auto_conversion, _ice_deposition_nucleation, _vel_scale_factor_default, _vel_scale_factor_ice, _vel_scale_factor_snow
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import _qsat_rho, _qsat_ice_rho
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
) -> tuple[ta.wpfloat,ta.wpfloat,ta.wpfloat]:   # updates
    _, flx, vt = state
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0*flx
#    flx_partial = minimum(rho_x * vc * _fall_speed(rho_x, prefactor, offset, exponent), flx_eff) 
    flx_partial = minimum(rho_x * vc * prefactor * power((rho_x+offset), exponent), flx_eff) 
    update0 = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)   # q update
    update1 = (update0 * rho * vt + flx_partial) * 0.5                       # flux
    rho_x   = (update0 + q_kp1) * 0.5 * rho
#    update2 = vc * _fall_speed(rho_x, prefactor, offset, exponent)           # vt
    update2 = vc * prefactor * power((rho_x+offset), exponent)                # vt
    return update0, update1, update2

@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0, 0.0))
def _temperature_update(
    state:     tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat],
    t:         ta.wpfloat,
    t_kp1:     ta.wpfloat,
    qv_old:    ta.wpfloat,
    qliq_old:  ta.wpfloat,
    qice_old:  ta.wpfloat,
    qr:        ta.wpfloat,
    qv:        ta.wpfloat,
    qliq:      ta.wpfloat,
    qice:      ta.wpfloat,
    rho:       ta.wpfloat,             # density
    dz:        ta.wpfloat,
    dt:        ta.wpfloat,
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LSC:       ta.wpfloat,
    LVC:       ta.wpfloat,
) -> tuple[ta.wpfloat,ta.wpfloat,ta.wpfloat]:
    _, eflx, pflx = state

#    e_int   = internal_energy( t_old, qv_old, qliq_old, qice_old, rho, dz, CI, CLW, CVD, CVV, LSC, LVC ) + eflx
#   inlined in order to avoid scan_operator -> field_operator
    qtot    = qliq_old + qice_old + qv_old
    cv      = CVD * ( 1.0 - qtot ) + CVV * qv_old + CLW * qliq_old + CI * qice_old
    e_int   = rho * dz * ( cv * t - qliq * LVC - qice * LSC ) + eflx

    eflx    = dt * ( qr * ( CLW * t - CVD * t_kp1 - LVC ) + pflx * ( CI * t - CVD * t_kp1 - LSC ) )
    e_int   = e_int - eflx

    pflx    = pflx + qr  # pflx[oned_vec_index] = pflx[oned_vec_index] + q[lqr].p[iv];

#    t       = T_from_internal_energy( min_k, e_int, qv, qliq, qice, rho, dz, CI, CLW, CVD, CVV, LSC, LVC )
#   inlined in order to avoid scan_operator -> field_operator
    qtot    = qliq + qice + qv                          # total water specific mass
    cv      = ( CVD * ( 1.0 - qtot ) + CVV * qv + CLW * qliq + CI * qice ) * rho * dz # Moist isometric specific heat
    t       = ( e_int + rho * dz * ( qliq * LVC + qice * LSC )) / cv

    return t, eflx, pflx

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
    QMIN:      ta.wpfloat,                           # threshold Q
    TFRZ_HET2: ta.wpfloat,                           # TBD
    TMELT:     ta.wpfloat,                           # TBD
    RV:        ta.wpfloat,                           # TBD
) -> tuple[fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool],fa.CellKField[bool]]:

    mask = where( (maximum( qc, maximum(qg, maximum(qi, maximum(qr, qs)))) > QMIN) | ((t < TFRZ_HET2) & (qv > _qsat_ice_rho(t, rho, TMELT, RV) ) ), True, False )
    is_sig_present = where( maximum( qg, maximum(qi, qs)) > QMIN, True, False )
    kmin_r = where( qr > QMIN, True, False )
    kmin_i = where( qi > QMIN, True, False )
    kmin_s = where( qs > QMIN, True, False )
    kmin_g = where( qg > QMIN, True, False )
    return mask, is_sig_present, kmin_r, kmin_i, kmin_s, kmin_g
 
@gtx.field_operator
def _graupel_loop2(
# TODO: arguments
    qv:             fa.CellKField[ta.wpfloat],             # Q vapor content
    qc:             fa.CellKField[ta.wpfloat],             # Q cloud content
    qr:             fa.CellKField[ta.wpfloat],             # Q rain content
    qs:             fa.CellKField[ta.wpfloat],             # Q snow content
    qi:             fa.CellKField[ta.wpfloat],             # Q ice content
    qg:             fa.CellKField[ta.wpfloat],             # Q graupel content
    qnc:            fa.CellKField[ta.wpfloat],
    t:              fa.CellKField[ta.wpfloat],
    p:              fa.CellKField[ta.wpfloat],
    rho:            fa.CellKField[ta.wpfloat],
    is_sig_present: fa.CellKField[bool],
    dt:             ta.wpfloat,
    CI:             ta.wpfloat,
    CLW:            ta.wpfloat,
    CVD:            ta.wpfloat,
    CVV:            ta.wpfloat,
    QMIN:           ta.wpfloat,
    LSC:            ta.wpfloat,
    LVC:            ta.wpfloat,
    TMELT:          ta.wpfloat,
    RD:             ta.wpfloat,
    RV:             ta.wpfloat,
    ALS:            ta.wpfloat,
    AMS:            ta.wpfloat,
    BMS:            ta.wpfloat,
    M0_ICE:         ta.wpfloat,
    TFRZ_HET1:      ta.wpfloat,
    TFRZ_HET2:      ta.wpfloat,
    TFRZ_HOM:       ta.wpfloat,
    TX:             ta.wpfloat,
    V0S:            ta.wpfloat,
    V1S:            ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:

    dvsw = qv - _qsat_rho( t, rho, TMELT, RV )
    qvsi = _qsat_ice_rho( t, rho, TMELT, RV )
    dvsi = qv - qvsi

    n_snow = _snow_number( t, rho, qs, QMIN, AMS, TMELT )
    
    l_snow = _snow_lambda( rho, qs, n_snow, QMIN, AMS, BMS )

    # Define conversion 'matrix'
    sx2x_c_r = _cloud_to_rain( t, qc, qr, qnc, TFRZ_HOM )
    sx2x_r_v = _rain_to_vapor( t, rho, qc, qr, dvsw, dt, QMIN, TMELT )
    sx2x_c_i = _cloud_x_ice( t, qc, qi, dt, TFRZ_HOM, QMIN, TMELT )
    sx2x_i_c = -minimum(sx2x_c_i, 0.0)
    sx2x_c_i = maximum(sx2x_c_i, 0.0)

    sx2x_c_s = _cloud_to_snow( t, qc, qs, n_snow, l_snow, V0S, V1S, TFRZ_HOM, QMIN )
    sx2x_c_g = _cloud_to_graupel( t, rho, qc, qg, TFRZ_HOM, QMIN )

    n_ice = where( t < TMELT, _ice_number( t, rho, TMELT ), 0.0 )
    m_ice = where( t < TMELT, _ice_mass( qi, n_ice, M0_ICE ), 0.0 )
    x_ice = where( t < TMELT, _ice_sticking( t, TMELT ), 0.0 )

    eta          = where( (t < TMELT) & is_sig_present, _deposition_factor( t, qvsi, QMIN, ALS, RD, RV, TMELT ), 0.0 )
    sx2x_v_i = where( (t < TMELT) & is_sig_present, _vapor_x_ice( qi, m_ice, eta, dvsi, rho, dt, QMIN ), 0.0 )
    sx2x_i_v = where( (t < TMELT) & is_sig_present, -minimum( sx2x_v_i, 0.0 ) , 0.0 )
    sx2x_v_i = where( (t < TMELT) & is_sig_present, maximum( sx2x_v_i, 0.0 ) , 0.0 )
    ice_dep      = where( (t < TMELT) & is_sig_present, minimum( sx2x_v_i, dvsi/dt ) , 0.0 )

    sx2x_i_s = where( (t < TMELT) & is_sig_present, _deposition_auto_conversion( qi, m_ice, ice_dep, QMIN ) + _ice_to_snow( qi, n_snow, l_snow, x_ice, QMIN, V0S, V1S ), 0.0 )
    sx2x_i_g = where( (t < TMELT) & is_sig_present, _ice_to_graupel( rho, qr, qg, qi, m_ice, QMIN ), 0.0 )
    sx2x_s_g = where( (t < TMELT) & is_sig_present, _snow_to_graupel( t, rho, qc, qs, QMIN, TFRZ_HOM ), 0.0 )
    sx2x_r_g = where( (t < TMELT) & is_sig_present, _rain_to_graupel( t, rho, qc, qr, qi, qs, m_ice, dvsw, dt, QMIN, TFRZ_HOM, TMELT ), 0.0 )

    sx2x_v_i = where( t < TMELT, sx2x_v_i + _ice_deposition_nucleation(t, qc, qi, n_ice, dvsi, dt, QMIN, M0_ICE, TFRZ_HET1, TFRZ_HET2 ), 0.0 )

    sx2x_c_r = where( t >= TMELT, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r )

    dvsw0    = where( is_sig_present, qv - _qsat_rho( t, rho, TMELT, RV ), 0.0 )  # TODO: new qsat_rho_tmelt, TODO: use dvsw ??
    sx2x_v_s = where( is_sig_present, _vapor_x_snow( t, p, rho, qs, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt, QMIN, TX, TMELT, V0S, V1S ), 0.0 )
    sx2x_s_v = where( is_sig_present, -minimum( sx2x_v_s, 0.0 ), 0.0 )
    sx2x_v_s = where( is_sig_present, maximum( sx2x_v_s, 0.0 ), 0.0 )

    sx2x_v_g = where( is_sig_present, _vapor_x_graupel( t, p, rho, qg, dvsw, dvsi, dvsw0, dt, QMIN, TX, TMELT ), 0.0 )
    sx2x_g_v = where( is_sig_present, -minimum( sx2x_v_g, 0.0 ), 0.0 )
    sx2x_v_g = where( is_sig_present, maximum( sx2x_v_g, 0.0 ), 0.0 )

    sx2x_s_r = where( is_sig_present, _snow_to_rain( t, p, rho, dvsw0, qs, QMIN, TX, TMELT ), 0.0 )
    sx2x_g_r = where( is_sig_present, _graupel_to_rain( t, p, rho, dvsw0, qg, QMIN, TX, TMELT ), 0.0 )

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
                          
    # SINK calculation
    
    # if (is_sig_present[j]) or (qx_ind[ix] == lqc) or (qx_ind[ix] == lqv) or (qx_ind[ix] == lqr)
    sink_v = sx2x_v_s + sx2x_v_i + sx2x_v_g   # Missing sx2x_v_c + sx2x_v_r
    sink_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g   # Missing  sx2x_c_v
    sink_r = sx2x_r_v + sx2x_r_g # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
    sink_s = where ( is_sig_present, sx2x_s_v + sx2x_s_r + sx2x_s_g, 0.0 ) # Missing: sx2x_s_c + sx2x_s_i
    sink_i = where ( is_sig_present, sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, 0.0 ) # Missing: sx2x_i_r
    sink_g = where ( is_sig_present, sx2x_g_v + sx2x_g_r, 0.0 ) # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i
    
    #  if ((sink[qx_ind[ix]] > stot) && (q[qx_ind[ix]].x[oned_vec_index] > qmin))

    stot = qv / dt
    sx2x_v_s = where( (sink_v > stot) & (qv > QMIN), sx2x_v_s * stot / sink_s, sx2x_v_s )
    sx2x_v_i = where( (sink_v > stot) & (qv > QMIN), sx2x_v_i * stot / sink_i, sx2x_v_i )
    sx2x_v_g = where( (sink_v > stot) & (qv > QMIN), sx2x_v_g * stot / sink_g, sx2x_v_g )
    sink_v = where( (sink_v > stot) & (qv > QMIN), sx2x_v_s + sx2x_v_i + sx2x_v_g, sink_v) # Missing: sx2x_v_c + sx2x_v_r

    stot = qc / dt
    sx2x_c_r = where( (sink_c > stot) & (qc > QMIN), sx2x_c_r * stot / sink_r, sx2x_c_r )
    sx2x_c_s = where( (sink_c > stot) & (qc > QMIN), sx2x_c_s * stot / sink_s, sx2x_c_s )
    sx2x_c_i = where( (sink_c > stot) & (qc > QMIN), sx2x_c_i * stot / sink_i, sx2x_c_i )
    sx2x_c_g = where( (sink_c > stot) & (qc > QMIN), sx2x_c_g * stot / sink_g, sx2x_c_g )
    sink_c = where( (sink_c > stot) & (qc > QMIN), sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g, sink_c) # Missing: sx2x_c_v

    stot = qr / dt 
    sx2x_r_v = where( (sink_r > stot) & (qr > QMIN), sx2x_r_v * stot / sink_v, sx2x_r_v )
    sx2x_r_g = where( (sink_r > stot) & (qr > QMIN), sx2x_r_g * stot / sink_r, sx2x_r_g )
    sink_r = where( (sink_r > stot) & (qr > QMIN), sx2x_r_v + sx2x_r_g, sink_r) # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
     
    stot = qs / dt
    sx2x_s_v = where( (sink_s > stot) & (qs > QMIN), sx2x_s_v * stot / sink_v, sx2x_s_v )
    sx2x_s_r = where( (sink_s > stot) & (qs > QMIN), sx2x_s_r * stot / sink_s, sx2x_s_r )
    sx2x_s_g = where( (sink_s > stot) & (qs > QMIN), sx2x_s_g * stot / sink_g, sx2x_s_g )
    sink_s = where( (sink_s > stot) & (qs > QMIN), sx2x_s_v + sx2x_s_r + sx2x_s_g, sink_s) # Missing: sx2x_s_c + sx2x_s_i

    stot = qi / dt
    sx2x_i_v = where( (sink_i > stot) & (qi > QMIN), sx2x_i_v * stot / sink_v, sx2x_i_v )
    sx2x_i_c = where( (sink_i > stot) & (qi > QMIN), sx2x_i_c * stot / sink_i, sx2x_i_c )
    sx2x_i_s = where( (sink_i > stot) & (qi > QMIN), sx2x_i_s * stot / sink_s, sx2x_i_s )
    sx2x_i_g = where( (sink_i > stot) & (qi > QMIN), sx2x_i_g * stot / sink_g, sx2x_i_g )
    sink_i = where( (sink_i > stot) & (qi > QMIN), sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g, sink_i) # Missing: sx2x_i_r

    stot = qg / dt
    sx2x_g_v = where( (sink_g > stot) & (qg > QMIN), sx2x_g_v * stot / sink_v, sx2x_g_v )
    sx2x_g_r = where( (sink_g > stot) & (qg > QMIN), sx2x_g_r * stot / sink_g, sx2x_g_r )
    sink_g = where( (sink_g > stot) & (qg > QMIN), sx2x_g_v + sx2x_g_r, sink_g) # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    # water content updates:
    qv     = maximum( 0.0, qv + ( sx2x_v_s + sx2x_v_i + sx2x_v_g - sink_v ) * dt ) # Missing: sx2x_v_c + sx2x_v_r
    dqdt_c = sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g - sink_c                    # Missing: sx2x_c_v
    qc     = maximum( 0.0, qc + dqdt_c * dt )
    dqdt_r = sx2x_r_v + sx2x_r_g - sink_r                                          # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
    qr     = maximum( 0.0, qr + dqdt_r * dt )
    dqdt_s = sx2x_s_v + sx2x_s_r + sx2x_s_g - sink_s                               # Missing: sx2x_s_c + sx2x_s_i
    qs     = maximum( 0.0, qs + dqdt_s * dt )
    dqdt_i = sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g - sink_i                    # Missing: sx2x_i_r
    qi     = maximum( 0.0, qi + dqdt_i * dt )
    dqdt_g = sx2x_g_v + sx2x_g_r - sink_g                                          # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i
    qg     = maximum( 0.0, qg + dqdt_g * dt ) 

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv   = CVD + (CVV - CVD) * qtot + (CLW - CVV) * qliq + (CI - CVV) * qice
    t = t + dt * ( (dqdt_c + dqdt_r) * (LVC - (CLW - CVV)*t) + (dqdt_i + dqdt_s + dqdt_g) * (LSC - (CI - CVV)*t ) ) / cv

    return qv, qc, qr, qs, qi, qg, t

@gtx.field_operator
def _graupel_loop3_if_lrain(
# TODO: arguments
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
    is_sig_present : fa.CellKField[bool],
    RHO_00: ta.wpfloat,
    QMIN:   ta.wpfloat,
    AMS:    ta.wpfloat,
    TMELT:  ta.wpfloat,
) -> [fa.CellKField[ta.wpfloat],fa.CellKField[ta.wpfloat]]:

    # Store current fields for later temperature update
    qv_old   = qv
    t_old    = t
    qliq_old = qc + qr
    qice_old = qs + qi + qg

    zeta  = dt / (2.0 * dz )
    xrho  = sqrt( RHO_00 / rho )

    # NP = 4    qp_ind[] = {lqr, lqi, lqs, lqg};
    vc_r = where( kmin_r, _vel_scale_factor_default( xrho ), 0.0 )
    vc_i = where( kmin_i, _vel_scale_factor_ice( xrho ), 0.0 )
    vc_s = where( kmin_s, _vel_scale_factor_snow( xrho, rho, t, qs, QMIN, AMS, TMELT ), 0.0 )
    vc_g = where( kmin_g, _vel_scale_factor_default( xrho ), 0.0 )

    q_kp1    = qr(Koff[1])
    qr, pr, _ = _precip( prefactor_r, exponent_r, offset_r, zeta, vc_r, qr, q_kp1, rho )
    q_kp1    = qi(Koff[1])
    qi, pi, _ = _precip( prefactor_i, exponent_i, offset_i, zeta, vi_r, qi, q_kp1, rho )
    q_kp1    = qs(Koff[1])
    qs, ps, _ = _precip( prefactor_s, exponent_s, offset_s, zeta, vc_r, qs, q_kp1, rho )
    q_kp1    = qg(Koff[1])
    qg, pg, _ = _precip( prefactor_g, exponent_g, offset_g, zeta, vc_g, qg, q_kp1, rho )

    pflx      = ps + pi + pg   # pflx[oned_vec_index] = q[lqs].p[iv] + q[lqi].p[iv] + q[lqg].p[iv];

    qliq  = qc + qr
    qice  = qs + qi + qg

    t_kp1   = t(Koff[1])
    t, eflx = _temperature_update( t, t_kp1, qv_old, qliq_old, qice_old, qv, qliq, qice, rho, dz )

    # TODO: here we have to return a single layer for pre_gsp
    return qr, qi, qs, qg
    
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

# TODO : program  needs to be called with offset_provider={"Koff": K}  
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def graupel_run(
    te:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:       fa.CellKField[ta.wpfloat],             # Density containing dry air and water constituents
    qve:       fa.CellKField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellKField[ta.wpfloat],             # Specific cloud water content
    qre:       fa.CellKField[ta.wpfloat],             # Specific rain water
    qti:       fa.CellKField[ta.wpfloat],             # Specific mass of all ice species (total-ice)
    QMIN:      ta.wpfloat,
    CI:        ta.wpfloat,
    CLW:       ta.wpfloat,
    CVD:       ta.wpfloat,
    CVV:       ta.wpfloat,
    LVC:       ta.wpfloat,
    TFRZ_HET2: ta.wpfloat,
    TMELT:     ta.wpfloat,
    RV:        ta.wpfloat,
    mask_out:  fa.CellKField[bool],                      # Temporary mask for > QMIN points
    is_sig_present_out:  fa.CellKField[bool],            # Temporary mask S, I, or G > QMIN
    kmin_r_out:  fa.CellKField[bool],                    # Specific rainwater content
    kmin_i_out:  fa.CellKField[bool],                    # Specific cloud water content
    kmin_s_out:  fa.CellKField[bool],                    # Specific cloud water content
    kmin_g_out:  fa.CellKField[bool],                    # Specific cloud water content
):

    _graupel_mask(te, rho, qve, qce, qge, qie, qre, qse, QMIN, TFRZ_HET2, TMELT, RV, out=(mask_out, is_sig_present_out, kmin_r_out, kmin_i_out, kmin_s_out, kmin_g_out) )
