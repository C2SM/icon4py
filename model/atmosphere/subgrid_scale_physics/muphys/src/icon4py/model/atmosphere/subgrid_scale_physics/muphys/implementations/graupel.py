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
    flx:       ta.wpfloat,             # flux into cell from above
    vt:        ta.wpfloat,             # terminal velocity
    q:         ta.wpfloat,             # specific mass of hydrometeor
    q_kp1:     ta.wpfloat,             # specific mass in next lower cell
    rho:       ta.wpfloat,             # density
) -> tuple[ta.wpfloat,ta.wpfloat,ta.wpfloat]:   # updates
    qx, vx, vt = state
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + 2.0*flx
    flx_partial = minimum(rho_x * vc * fall_speed(rho_x, prefactor, offset, exponent), flx_eff) 
    update0 = (zeta * (flx_eff - flx_partial)) / ((1.0 + zeta * vt) * rho)   # q update
    update1 = (update0 * rho * vt + flx_partial) * 0.5                       # flux
    update2 = vc * fall_speed(rho_x, prefactor, offset, exponent)            # vt
    return update0, update1, update2

@gtx.field_operator
def _graupel_mask(
    q_lqc:     fa.CellField[ta.wpfloat],             # Q cloud content
    q_lqg:     fa.CellField[ta.wpfloat],             # Q graupel content
    q_lqi:     fa.CellField[ta.wpfloat],             # Q ice content
    q_lqr:     fa.CellField[ta.wpfloat],             # Q rain content
    q_lqs:     fa.CellField[ta.wpfloat],             # Q snow content
    QMIN:      ta.wpfloat,                           # threshold Q
    TFRZ_HET2: ta.wpfloat,                           # TBD
    TMELT:     ta.wpfloat,                           # TBD
    RV:        ta.wpfloat,                           # TBD
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:

    mask = where( (maximum( q_lqc, maximum(q_lqg, maximum(q_lqi, maximum(q_lqr, q_lqs)))) > QMIN) | ((t < tfrz_het2) & (q_lqv > qsat_ice_rho(t, rho, TMELT, RV) ) ), 1.0, 0.0 )
    is_sig_present = where( maximum( q_lqg, maximum(q_lqi, q_lqs)) > QMIN, 1.0, 0.0 )
    return mask, is_sig_present
 
@gtx.field_operator
def _graupel_loop2(
# TODO: arguments
        t,
        qnc,
        dvsw,
        dt,
        is_sig_present,
        TMELT,
    OTHERS
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:

    dvsw = q_lqv_x - qsat_rho( t, rho, OTHERS )
    qvsi = qsat_ice_rho( t, rho, OTHERS )
    dvsi = q_lqv_x = qvsi

    n_snow = snow_number( t, rho, q_lqs_x, OTHERS )
    l_snow = snow_lambda( rho, q_lqs_x, n_snow, OTHERS )

    # Define conversion 'matrix'
    sx2x_lqc_lqr = cloud_to_rain( t, q_lqc_x, q_lqr_x, qnc )
    sx2x_lqr_lqv = rain_to_vapor( t, rho, q_lqc_x, q_lqr_x, dvsw, dt )
    sx2x_lqc_lqi = cloud_x_ice( t, q_lqc_x, q_lqi_x, dt )
    sx2x_lqi_lqc = -minimum(sx2x_lqc_lqi, 0.0)
    sx2x_lqc_lqi = maximum(sx2x_lqc_lqi, 0.0)

    sx2x_lqc_lqs = cloud_to_snow( t, q_lqc_x, q_lqs_x, n_snow, l_snow )
    sx2x_lqc_lqg = cloud_to_graupel( t, rho, q_lqc_x, q_lqg_x )

    n_ice = where( t < TMELT, ice_number( t, rho, OTHERS ), 0.0 )
    m_ice = where( t < TMELT, ice_max( q_lqi_x, OTHERS ), 0.0 )
    x_ice = where( t < TMELT, ice_sticking( t, OTHERS ), 0.0 )

    eta          = where( (t < TMELT) & is_sig_present, deposition_factor( t, qvsi, OTHERS ), 0.0 )
    sx2x_lqv_lqi = where( (t < TMELT) & is_sig_present, vapor_x_ice( q_lqi_x, m_ice, eta, dvsi, rho, dt, OTHERS ), 0.0 )
    sx2x_lqi_lqv = where( (t < TMELT) & is_sig_present, -minimum( sx2x_lqv_lqi, 0.0 ) , 0.0 )
    sx2x_lqv_lqi = where( (t < TMELT) & is_sig_present, maximum( sx2x_lqv_lqi, 0.0 ) , 0.0 )
    ice_dep      = where( (t < TMELT) & is_sig_present, minimum( sx2x_lqv_lqi, dvsi/dt ) , 0.0 )

    sx2x_lqi_lqs = where( (t < TMELT) & is_sig_present, deposition_auto_conversion( q_lqi_x, m_ice, ice_dep, OTHERS ) + ice_to_snow( q_lqi_x, n_snow, l_snow, x_ice, OTHERS), 0.0 )
    sx2x_lqi_lqg = where( (t < TMELT) & is_sig_present, ice_to_graupel( rho, q_lqr_x, q_lqg_x, q_lqi_x, m_ice, OTHERS ), 0.0 )
    sx2x_lqs_lqg = where( (t < TMELT) & is_sig_present, snow_to_graupel( t, rho, q_lqc_x, q_lqs_x, OTHERS ), 0.0 )
    sx2x_lqr_lqg = where( (t < TMELT) & is_sig_present, rain_to_graupel( t, rho, q_lqc_x, q_lqr_x, q_lqi_x, q_lqs_x, m_ice, dvsw, dt, OTHERS ), 0.0 )

    sx2x_lqv_lqi = where( t < TMELT, sx2x_lqv_lqi + ice_deposition_nucleation(t, q_lqc_x, q_lqi_x, n_ice, dvsi, dt, OTHERS ), 0.0 )

    sx2x_lqc_lqr = where( t >= TMELT, sx2x_lqc_lqr + sx2x_lqc_lqs + sx2x_lqc_lqg, sx2x_lqc_lqr )

    dvsw0        = where( is_sig_present, q_lqv_x - qsat_rho( tmelt, rho, OTHERS )  # TODO: new qsat_rho_tmelt
    sx2x_lqv_lqs = where( is_sig_present, vapor_x_snow( t, p, rho, q_lqs_x, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_lqs_lqv = where( is_sig_present, -minimum( sx2x_lqv_lqs, 0.0 ), 0.0 )
    sx2x_lqv_lqs = where( is_sig_present, maximum( sx2x_lqv_lqs, 0.0 ), 0.0 )

    sx2x_lqv_lqg = where( is_sig_present, vapor_x_graupel( t, p, rho, q_lqg_x, dvsw, dvsi, dvsw0, dt ), 0.0 )
    sx2x_lqg_lqv = where( is_sig_present, -minimum( sx2x_lqv_lqg, 0.0 ), 0.0 )
    sx2x_lqv_lqg = where( is_sig_present, maximum( sx2x_lqv_lqg, 0.0 ), 0.0 )

    sx2x_lqs_lqr = where( is_sig_present, snow_to_rain( t, p, rho, dvsw0, q_lqs_x, OTHERS ), 0.0 )
    sx2x_lqg_lqr = where( is_sig_present, graupel_to_rain( t, p, rho, dvsw0, q_lqg_x, OTHERS ), 0.0 )

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
                          
    # SINK calculation
    
    # if (is_sig_present[j]) or (qx_ind[ix] == lqc) or (qx_ind[ix] == lqv) or (qx_ind[ix] == lqr)
    sink_lqv = sx2x_lqv_lqs + sx2x_lqv_lqi + sx2x_lqv_lqg   # Missing sx2x_lqv_lqc + sx2x_lqv_lqr
    sink_lqc = sx2x_lqc_lqr + sx2x_lqc_lqs + sx2x_lqc_lqi + sx2x_lqc_lqg   # Missing  sx2x_lqc_lqv
    sink_lqr = sx2x_lqr_lqv + sx2x_lqr_lqg # Missing: sx2x_lqr_lqc + sx2x_lqr_lqs + sx2x_lqr_lqi
    sink_lqs = where ( is_sig_present, sx2x_lqs_lqv + sx2x_lqs_lqr + sx2x_lqs_lqg, 0.0 ) # Missing: sx2x_lqs_lqc + sx2x_lqs_lqi
    sink_lqi = where ( is_sig_present, sx2x_lqi_lqv + sx2x_lqi_lqc + sx2x_lqi_lqs + sx2x_lqi_lqg, 0.0 ) # Missing: sx2x_lqi_lqr
    sink_lqg = where ( is_sig_present, sx2x_lqg_lqv + sx2x_lqg_lqr, 0.0 ) # Missing: sx2x_lqg_lqc + sx2x_lqg_lqs + sx2x_lqg_lqi
    
    #  if ((sink[qx_ind[ix]] > stot) && (q[qx_ind[ix]].x[oned_vec_index] > qmin))

    sx2x_lqv_lqs = where( (sink_lqv > STOT) & (q_lqv_x > QMIN), sx2x_lqv_lqs * STOT / sink_lqs, sx2x_lqv_lqs )
    sx2x_lqv_lqi = where( (sink_lqv > STOT) & (q_lqv_x > QMIN), sx2x_lqv_lqi * STOT / sink_lqi, sx2x_lqv_lqi )
    sx2x_lqv_lqg = where( (sink_lqv > STOT) & (q_lqv_x > QMIN), sx2x_lqv_lqg * STOT / sink_lqg, sx2x_lqv_lqg )

    sx2x_lqc_lqr = where( (sink_lqc > STOT) & (q_lqc_x > QMIN), sx2x_lqc_lqr * STOT / sink_lqr, sx2x_lqc_lqr )
    sx2x_lqc_lqs = where( (sink_lqc > STOT) & (q_lqc_x > QMIN), sx2x_lqc_lqs * STOT / sink_lqs, sx2x_lqc_lqs )
    sx2x_lqc_lqi = where( (sink_lqc > STOT) & (q_lqc_x > QMIN), sx2x_lqc_lqi * STOT / sink_lqi, sx2x_lqc_lqi )
    sx2x_lqc_lqg = where( (sink_lqc > STOT) & (q_lqc_x > QMIN), sx2x_lqc_lqg * STOT / sink_lqg, sx2x_lqc_lqg )

    sx2x_lqr_lqv = where( (sink_lqr > STOT) & (q_lqr_x > QMIN), sx2x_lqr_lqv * STOT / sink_lqv, sx2x_lqr_lqv )
    sx2x_lqr_lqg = where( (sink_lqr > STOT) & (q_lqr_x > QMIN), sx2x_lqr_lqg * STOT / sink_lqr, sx2x_lqr_lqg )
     
    sx2x_lqs_lqv = where( (sink_lqs > STOT) & (q_lqs_x > QMIN), sx2x_lqs_lqv * STOT / sink_lqv, sx2x_lqs_lqv )
    sx2x_lqs_lqr = where( (sink_lqs > STOT) & (q_lqs_x > QMIN), sx2x_lqs_lqr * STOT / sink_lqs, sx2x_lqs_lqr )
    sx2x_lqs_lqg = where( (sink_lqs > STOT) & (q_lqs_x > QMIN), sx2x_lqs_lqg * STOT / sink_lqg, sx2x_lqs_lqg )

    sx2x_lqi_lqv = where( (sink_lqi > STOT) & (q_lqi_x > QMIN), sx2x_lqi_lqv * STOT / sink_lqv, sx2x_lqi_lqv )
    sx2x_lqi_lqc = where( (sink_lqi > STOT) & (q_lqi_x > QMIN), sx2x_lqi_lqc * STOT / sink_lqi, sx2x_lqi_lqc )
    sx2x_lqi_lqs = where( (sink_lqi > STOT) & (q_lqi_x > QMIN), sx2x_lqi_lqs * STOT / sink_lqs, sx2x_lqi_lqs )
    sx2x_lqi_lqg = where( (sink_lqi > STOT) & (q_lqi_x > QMIN), sx2x_lqi_lqg * STOT / sink_lqg, sx2x_lqi_lqg )

    sx2x_lqg_lqv = where( (sink_lqg > STOT) & (q_lqg_x > QMIN), sx2x_lqg_lqv * STOT / sink_lqv, sx2x_lqg_lqv )
    sx2x_lqg_lqr = where( (sink_lqg > STOT) & (q_lqg_x > QMIN), sx2x_lqg_lqr * STOT / sink_lqg, sx2x_lqg_lqr )
     
    sink_lqv = where( (sink_lqv > STOT) & (q_lqv_x > QMIN), sx2x_lqv_lqs + sx2x_lqv_lqi + sx2x_lqv_lqg, sink_lqv) # Missing: sx2x_lqv_lqc + sx2x_lqv_lqr
    sink_lqc = where( (sink_lqc > STOT) & (q_lqc_x > QMIN), sx2x_lqc_lqr + sx2x_lqc_lqs + sx2x_lqc_lqi + sx2x_lqc_lqg, sink_lqc) # Missing: sx2x_lqc_lqv
    sink_lqr = where( (sink_lqr > STOT) & (q_lqr_x > QMIN), sx2x_lqr_lqv + sx2x_lqr_lqg, sink_lqr) # Missing: sx2x_lqr_lqc + sx2x_lqr_lqs + sx2x_lqr_lqi
    sink_lqs = where( (sink_lqs > STOT) & (q_lqs_x > QMIN), sx2x_lqs_lqv + sx2x_lqs_lqr + sx2x_lqs_lqg, sink_lqs) # Missing: sx2x_lqs_lqc + sx2x_lqs_lqi
    sink_lqi = where( (sink_lqi > STOT) & (q_lqi_x > QMIN), sx2x_lqi_lqv + sx2x_lqi_lqc + sx2x_lqi_lqs + sx2x_lqi_lqg, sink_lqi) # Missing: sx2x_lqi_lqr
    sink_lqg = where( (sink_lqg > STOT) & (q_lqg_x > QMIN), sx2x_lqg_lqv + sx2x_lqg_lqr, sink_lqg) # Missing: sx2x_lqg_lqc + sx2x_lqg_lqs + sx2x_lqg_lqi

    # water content updates:
    q_lqv_x  = maximum( 0.0, q_lqv_x + ( sx2x_lqv_lqs + sx2x_lqv_lqi + sx2x_lqv_lqg - sink_lqv ) * dt ) # Missing: sx2x_lqv_lqc + sx2x_lqv_lqr
    q_lqc_x  = maximum( 0.0, q_lqc_x + ( sx2x_lqc_lqr + sx2x_lqc_lqs + sx2x_lqc_lqi + sx2x_lqc_lqg - sink_lqc ) * dt ) # Missing: sx2x_lqc_lqv
    q_lqr_x  = maximum( 0.0, q_lqr_x + ( sx2x_lqr_lqv + sx2x_lqr_lqg - sink_lqr ) * dt ) # Missing: sx2x_lqr_lqc + sx2x_lqr_lqs + sx2x_lqr_lqi
    q_lqs_x  = maximum( 0.0, q_lqs_x + ( sx2x_lqs_lqv + sx2x_lqs_lqr + sx2x_lqs_lqg - sink_lqs ) * dt ) # Missing: sx2x_lqs_lqc + sx2x_lqs_lqi
    q_lqi_x  = maximum( 0.0, q_lqi_x + ( sx2x_lqi_lqv + sx2x_lqi_lqc + sx2x_lqi_lqs + sx2x_lqi_lqg - sink_lqi ) * dt ) # Missing: sx2x_lqi_lqr
    q_lqg_x  = maximum( 0.0, q_lqg_x + ( sx2x_lqg_lqv + sx2x_lqg_lqr - sink_lqg ) * dt ) # Missing: sx2x_lqg_lqc + sx2x_lqg_lqs + sx2x_lqg_lqi

    # Copy q_lq*_x to a level offset version

    qice = q_lqs_x + q_lqi_x + q_lqg_x
    qliq = q_lqc_x + q_lqr_x
    qtot = q_lqv_x + qice + qliq

    cv   = CVD + (CVV - CVD) * qtot + (CLW - CVV) * qliq + (CI - CVV) * qice
    t = t + dt * ( (dqdt_lqc + dqdt_lqr) * (LVC - (CLW - CVV)*t) + (dqdt_lqi + dqdt_lqs + dqdt_lqg) * (LSC - (CI - CVV)*t ) ) / cv

@gtx.field_operator
def _graupel_loop3_if_lrain(
# TODO: arguments
        q_lqv_x,
        q_lqc_x,
        q_lqr_x,
        q_lqs_x,
        q_lqi_x,
        q_lqg_x,
        t,
        rho,
        dz, 
        dvsw,
        dt,
        is_sig_present,
        kmin,
        TMELT,
    OTHERS
) -> [fa.CellField[ta.wpfloat],fa.CellField[ta.wpfloat]]:
    qliq = where( kmin, q_lqc_x + q_lqr_x, 0.0 )
    qice = where( kmin, q_lqs_x + q_lqi_x + q_lqg_x, 0.0 )
    e_int = internal_energy( t, q_lqv_x, qliq, qice, rho, dz )
    zeta  = dt / (2.0 * dz )
    xrho  = sqrt( rho00 / rho )

    # NP = 4    qp_ind[] = {lqr, lqi, lqs, lqg};
    vc_lqr = where( kmin_lqr, vel_scale_factor_others( xrho ), 0.0 )
    vc_lqi = where( kmin_lqi, vel_scale_factor_lqi( xrho, OTHERS ), 0.0 )
    vc_lqs = where( kmin_lqs, vel_scale_factor_lqs( xrho, rho, t, q_lqs, dz, OTHERS ), 0.0 )
    vc_lqg = where( kmin_lqg, vel_scale_factor_others( xrho ), 0.0 )

    q_lqr_x, _, _ = _precip( prefactor_r, exponent_r, offset_r, zeta, vc, q_lqr_p, vt, q_lqr_x, q_lqr_x_kp1, rho )

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
