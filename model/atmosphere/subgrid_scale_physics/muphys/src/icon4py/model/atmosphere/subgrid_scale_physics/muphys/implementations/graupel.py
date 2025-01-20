\1;95;0c# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.qsat_rho import _qsat_rho
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo.dqsatdT_rho import _dqsatdT_rho

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
    flx_partial = minimum(rho_x * vc * fall_speed(rho_x, prefactor, offset, exponent), flx_eff) 
    update0 = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)   # q update
    update1 = (update0 * rho * vt + flx_partial) * 0.5                       # flux
    rho_x   = (update0 + q_kp1) * 0.5 * rho
    update2 = vc * fall_speed(rho_x, prefactor, offset, exponent)            # vt
    return update0, update1, update2

@gtx.scan_operator(axis=K, forward=True, init=(0.0, 0.0, 0.0))
def _temperature_update((
    state:     tuple[ta.wpfloat, ta.wpfloat],
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
) -> tuple[ta.wpfloat,ta.wpfloat]:
    _, eflx = state

    e_int   = internal_energy( t_old, qv_old, qliq_old, qice_old, rho, dz, OTHERS ) + eflx
    eflx    = dt * ( qr * ( CLW * t - CVD * t_kp1 - LVC ) + pflx * ( CI * t - CVD * t_kp1 - LSC ) )
    e_int   = e_int - eflx
    t       = T_from_internal_energy( min_k, e_int, qv, qliq, qice, rho, dz )

    return t, eflx

@gtx.field_operator
def _graupel_mask(
    qc:     fa.CellField[ta.wpfloat],             # Q cloud content
    qg:     fa.CellField[ta.wpfloat],             # Q graupel content
    qi:     fa.CellField[ta.wpfloat],             # Q ice content
    qr:     fa.CellField[ta.wpfloat],             # Q rain content
    qs:     fa.CellField[ta.wpfloat],             # Q snow content
    QMIN:      ta.wpfloat,                           # threshold Q
    TFRZ_HET2: ta.wpfloat,                           # TBD
    TMELT:     ta.wpfloat,                           # TBD
    RV:        ta.wpfloat,                           # TBD
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:

    mask = where( (maximum( qc, maximum(qg, maximum(qi, maximum(qr, qs)))) > QMIN) | ((t < tfrz_het2) & (qv > qsat_ice_rho(t, rho, TMELT, RV) ) ), 1.0, 0.0 )
    is_sig_present = where( maximum( qg, maximum(qi, qs)) > QMIN, 1.0, 0.0 )
    return mask, is_sig_present
 
@gtx.field_operator
def _graupel_loop2(
# TODO: arguments
    qv:     fa.CellField[ta.wpfloat],             # Q vapor content
    qc:     fa.CellField[ta.wpfloat],             # Q cloud content
    qr:     fa.CellField[ta.wpfloat],             # Q rain content
    qs:     fa.CellField[ta.wpfloat],             # Q snow content
    qi:     fa.CellField[ta.wpfloat],             # Q ice content
    qg:     fa.CellField[ta.wpfloat],             # Q graupel content
    qnc:    fa.CellField[ta.wpfloat],
    t :     fa.CellField[ta.wpfloat],
    rho:    fa.CellField[ta.wpfloat],
    is_sig_present: ,
    dt      ta.wpfloat,
    TMELT   ta.wpfloat,
    OTHERS
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:

    dvsw = qv - qsat_rho( t, rho, TMELT, RV )
    qvsi = qsat_ice_rho( t, rho, TMELT, RV )
    dvsi = qv - qvsi

    n_snow = snow_number( t, rho, qs, QMIN, AMS, TMELT )
    l_snow = snow_lambda( rho, qs, n_snow, QMIN, AMS, BMS )

    # Define conversion 'matrix'
    sx2x_c_r = cloud_to_rain( t, qc, qr, qnc )
    sx2x_r_v = rain_to_vapor( t, rho, qc, qr, dvsw, dt )
    sx2x_c_i = cloud_x_ice( t, qc, qi, dt )
    sx2x_i_c = -minimum(sx2x_c_i, 0.0)
    sx2x_c_i = maximum(sx2x_c_i, 0.0)

    sx2x_c_s = cloud_to_snow( t, qc, qs, n_snow, l_snow )
    sx2x_c_g = cloud_to_graupel( t, rho, qc, qg )

    n_ice = where( t < TMELT, ice_number( t, rho, OTHERS ), 0.0 )
    m_ice = where( t < TMELT, ice_max( qi, OTHERS ), 0.0 )
    x_ice = where( t < TMELT, ice_sticking( t, OTHERS ), 0.0 )

    eta          = where( (t < TMELT) & is_sig_present, deposition_factor( t, qvsi, OTHERS ), 0.0 )
    sx2x_v_i = where( (t < TMELT) & is_sig_present, vapor_x_ice( qi, m_ice, eta, dvsi, rho, dt, OTHERS ), 0.0 )
    sx2x_i_v = where( (t < TMELT) & is_sig_present, -minimum( sx2x_v_i, 0.0 ) , 0.0 )
    sx2x_v_i = where( (t < TMELT) & is_sig_present, maximum( sx2x_v_i, 0.0 ) , 0.0 )
    ice_dep      = where( (t < TMELT) & is_sig_present, minimum( sx2x_v_i, dvsi/dt ) , 0.0 )

    sx2x_i_s = where( (t < TMELT) & is_sig_present, deposition_auto_conversion( qi, m_ice, ice_dep, OTHERS ) + ice_to_snow( qi, n_snow, l_snow, x_ice, OTHERS), 0.0 )
    sx2x_i_g = where( (t < TMELT) & is_sig_present, ice_to_graupel( rho, qr, qg, qi, m_ice, OTHERS ), 0.0 )
    sx2x_s_g = where( (t < TMELT) & is_sig_present, snow_to_graupel( t, rho, qc, qs, OTHERS ), 0.0 )
    sx2x_r_g = where( (t < TMELT) & is_sig_present, rain_to_graupel( t, rho, qc, qr, qi, qs, m_ice, dvsw, dt, OTHERS ), 0.0 )

    sx2x_v_i = where( t < TMELT, sx2x_v_i + ice_deposition_nucleation(t, qc, qi, n_ice, dvsi, dt, OTHERS ), 0.0 )

    sx2x_c_r = where( t >= TMELT, sx2x_c_r + sx2x_c_s + sx2x_c_g, sx2x_c_r )

    dvsw0        = where( is_sig_present, qv - qsat_rho( tmelt, rho, OTHERS )  # TODO: new qsat_rho_tmelt
    sx2x_v_s = where( is_sig_present, vapor_x_snow( t, p, rho, qs, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_s_v = where( is_sig_present, -minimum( sx2x_v_s, 0.0 ), 0.0 )
    sx2x_v_s = where( is_sig_present, maximum( sx2x_v_s, 0.0 ), 0.0 )

    sx2x_v_g = where( is_sig_present, vapor_x_graupel( t, p, rho, qg, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_g_v = where( is_sig_present, -minimum( sx2x_v_g, 0.0 ), 0.0 )
    sx2x_v_g = where( is_sig_present, maximum( sx2x_v_g, 0.0 ), 0.0 )

    sx2x_s_r = where( is_sig_present, snow_to_rain( t, p, rho, dvsw0, qs, OTHERS ), 0.0 )
    sx2x_g_r = where( is_sig_present, graupel_to_rain( t, p, rho, dvsw0, qg, OTHERS ), 0.0 )

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
    qv  = maximum( 0.0, qv + ( sx2x_v_s + sx2x_v_i + sx2x_v_g - sink_v ) * dt ) # Missing: sx2x_v_c + sx2x_v_r
    qc  = maximum( 0.0, qc + ( sx2x_c_r + sx2x_c_s + sx2x_c_i + sx2x_c_g - sink_c ) * dt ) # Missing: sx2x_c_v
    qr  = maximum( 0.0, qr + ( sx2x_r_v + sx2x_r_g - sink_r ) * dt ) # Missing: sx2x_r_c + sx2x_r_s + sx2x_r_i
    qs  = maximum( 0.0, qs + ( sx2x_s_v + sx2x_s_r + sx2x_s_g - sink_s ) * dt ) # Missing: sx2x_s_c + sx2x_s_i
    qi  = maximum( 0.0, qi + ( sx2x_i_v + sx2x_i_c + sx2x_i_s + sx2x_i_g - sink_i ) * dt ) # Missing: sx2x_i_r
    qg  = maximum( 0.0, qg + ( sx2x_g_v + sx2x_g_r - sink_g ) * dt ) # Missing: sx2x_g_c + sx2x_g_s + sx2x_g_i

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv   = CVD + (CVV - CVD) * qtot + (CLW - CVV) * qliq + (CI - CVV) * qice
    t = t + dt * ( (dqdt_c + dqdt_r) * (LVC - (CLW - CVV)*t) + (dqdt_i + dqdt_s + dqdt_g) * (LSC - (CI - CVV)*t ) ) / cv

    return qv, qc, qr, qs, qi, qg, t

@gtx.field_operator
def _graupel_loop3_if_lrain(
# TODO: arguments
    kmin_r: fa.CellField[ta.int],                 # rain minimum level
    kmin_i: fa.CellField[ta.int],                 # ice minimum level
    kmin_s: fa.CellField[ta.int],                 # snow minimum level
    kmin_g: fa.CellField[ta.int],                 # graupel minimum level
    qv:     fa.CellField[ta.wpfloat],             # Q vapor content
    qc:     fa.CellField[ta.wpfloat],             # Q cloud content
    qr:     fa.CellField[ta.wpfloat],             # Q rain content
    qs:     fa.CellField[ta.wpfloat],             # Q snow content
    qi:     fa.CellField[ta.wpfloat],             # Q ice content
    qg:     fa.CellField[ta.wpfloat],             # Q graupel content    qv,
    t:      fa.CellField[ta.wpfloat],             # temperature,
    rho:    fa.CellField[ta.wpfloat],             # density
    dz:     fa.CellField[ta.wpfloat], 
    dt:     ta.wpfloat,
    is_sig_present,
    RHO_00: ta.wpfloat,
    B_I:    ta.wpfloat,
    B_S:    ta.wpfloat,
    QMIN:   ta.wpfloat,
    AMS:    ta.wpfloat,
    TMELT:  ta.wpfloat,
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:

    # Store current fields for later temperature update
    qv_old   = qv
    t_old    = t
    qliq_old = qc + qr
    qice_old = qs + qi + qg

    zeta  = dt / (2.0 * dz )
    xrho  = sqrt( RHO_00 / rho )

    # NP = 4    qp_ind[] = {lqr, lqi, lqs, lqg};
    vc_r = where( kmin_r, vel_scale_factor_others( xrho ), 0.0 )
    vc_i = where( kmin_i, vel_scale_factor_i( xrho, B_I ), 0.0 )
    vc_s = where( kmin_s, vel_scale_factor_s( xrho, rho, t, q_s, B_S, QMIN, AMS, TMELT ), 0.0 )
    vc_g = where( kmin_g, vel_scale_factor_others( xrho ), 0.0 )

    q_kp1    = qr(Koff[1])
    qr, _, _ = _precip( prefactor_r, exponent_r, offset_r, zeta, vc_r, qr, q_kp1, rho )
    q_kp1    = qi(Koff[1])
    qi, _, _ = _precip( prefactor_i, exponent_i, offset_i, zeta, vi_r, qi, q_kp1, rho )
    q_kp1    = qs(Koff[1])
    qs, _, _ = _precip( prefactor_s, exponent_s, offset_s, zeta, vc_r, qs, q_kp1, rho )
    q_kp1    = qg(Koff[1])
    qg, _, _ = _precip( prefactor_g, exponent_g, offset_g, zeta, vc_g, qg, q_kp1, rho )
    # TODO:  Other species

    qliq  = qc + qr
    qice  = qs + qi + qg

    t_kp1   = t(Koff[1])
    t, eflx = _temperature_update( t, t_kp1, qv_old, qliq_old, qice_old, qv, qliq, qice, rho, dz )

    # TODO: here we have to return a single layer for pre_gsp
    return qr, qi, qs, qg
    
@gtx.field_operator
def _output_calculation(
    qve:       fa.CellField[ta.wpfloat],             # Specific humidity
    qce:       fa.CellField[ta.wpfloat],             # Specific cloud water content
    qx_hold:   fa.CellField[ta.wpfloat],             # TBD
    qx:        fa.CellField[ta.wpfloat],             # TBD
    Tx_hold:   fa.CellField[ta.wpfloat],             # TBD
    Tx:        fa.CellField[ta.wpfloat],             # TBD
) -> tuple[fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:                       # Internal energy

    te  = where( ( qve+qce <= qx_hold ), Tx_hold, Tx )
    qce = where( ( qve+qce <= qx_hold ), 0.0, maximum(qve+qce-qx, 0.0) )
    qve = where( ( qve+qce <= qx_hold ), qve+qce, qx )
    return te, qve, qce

# TODO : program  needs to be called with offset_provider={"Koff": K}  
graupel_implementation(

):
    graupel_loop2 ...
_graupel_loop3_if_lrain(qv,
        qc,
        qr,
        qs,
        qi,
        qg,
        t,
        rho,
        dz,
        dvsw,
        dt,
        is_sig_present,
        kmin,
        TMELT,
    OTHERS

                        
    
