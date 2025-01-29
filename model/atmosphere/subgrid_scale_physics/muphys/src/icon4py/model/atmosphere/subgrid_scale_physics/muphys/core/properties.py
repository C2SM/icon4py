# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, maximum, minimum, power, exp
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _deposition_auto_conversion(
    qi:           fa.CellKField[ta.wpfloat],             # Ice specific mass
    m_ice:        fa.CellKField[ta.wpfloat],             # Ice crystal mass
    ice_dep:      fa.CellKField[ta.wpfloat],             # Rate of ice deposition (some to snow)
    QMIN:         ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                          # Conversion rate
    M0_S     = 3.0e-9                                    # Initial mass of snow crystals
    B_DEP    = 0.666666666666666667                      # Exponent
    XCRIT    = 1.0                                       # Critical threshold parameter
    
    return where( (qi > QMIN), maximum(0.0, ice_dep) * B_DEP / (power((M0_S/m_ice), B_DEP) - XCRIT), 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_auto_conversion(
    qi:           fa.CellKField[ta.wpfloat],             # Ice specific mass
    m_ice:        fa.CellKField[ta.wpfloat],             # Ice crystal mass
    ice_dep:      fa.CellKField[ta.wpfloat],             # Rate of ice deposition (some to snow)
    QMIN:         ta.wpfloat,
    conversion_rate: fa.CellKField[ta.wpfloat],          # output
):
    _deposition_auto_conversion(qi, m_ice, ice_dep, QMIN, out=conversion_rate)

@gtx.field_operator
def _deposition_factor(
    t:            fa.CellKField[ta.wpfloat],             # Temperature
    qvsi:         fa.CellKField[ta.wpfloat],             # Saturation (ice) specific vapor mass
    QMIN:         ta.wpfloat,
    ALS:          ta.wpfloat,
    RD:           ta.wpfloat, 
    RV:           ta.wpfloat, 
    TMELT:        ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                          # Deposition factor
    KAPPA    = 2.40e-2                                  # Thermal conductivity of dry air
    B        = 1.94                                     # Exponent
    A        = ALS*ALS / (KAPPA*RV)                     # TBD
    CX       = 2.22e-5 * power(TMELT, (-B)) * 101325.0  # TBD

    x = CX / RD * power(t, B-1.0)
    return  ( CX / RD * power(t, B-1.0) ) / (1.0 + A*x*qvsi / (t*t) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def deposition_factor(
    t:            fa.CellKField[ta.wpfloat],             # Temperature
    qvsi:         fa.CellKField[ta.wpfloat],             # Saturation (ice) specific vapor mass
    QMIN:         ta.wpfloat,
    ALS:          ta.wpfloat,
    RD:	      	  ta.wpfloat,
    RV:	      	  ta.wpfloat,
    TMELT:        ta.wpfloat,
    deposition_factor: fa.CellKField[ta.wpfloat],        # output
):
    _deposition_factor(t, qvsi, QMIN, ALS, RD, RV, TMELT, out=deposition_factor)

@gtx.field_operator
def _fall_speed_scalar(
    density:      gtx.Field[[], ta.wpfloat],                            # Density of species
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat, 
    QMIN:         ta.wpfloat,
    V0S:          ta.wpfloat,    
    V1S:          ta.wpfloat,
) -> gtx.Field[[], ta.wpfloat]:                          # Fall speed

    return prefactor * power((density+offset), exponent)

@gtx.field_operator
def _fall_speed(
    density:      fa.CellKField[ta.wpfloat],             # Density of species
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat, 
    QMIN:         ta.wpfloat,
    V0S:          ta.wpfloat,    
    V1S:          ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                          # Fall speed

    return prefactor * power((density+offset), exponent)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fall_speed_scalar(
    density:      gtx.Field[[], ta.wpfloat],                            # Density of species
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat,              
    QMIN:         ta.wpfloat,
    V0S:          ta.wpfloat,    
    V1S:          ta.wpfloat,
    fall_speed:   gtx.Field[[], ta.wpfloat],                            # output
):
    _fall_speed_scalar(density, prefactor, offset, exponent, QMIN, V0S, V1S, out=fall_speed)

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def fall_speed(
    density:      fa.CellKField[ta.wpfloat],             # Density of species
    prefactor:    ta.wpfloat,
    offset:       ta.wpfloat,
    exponent:     ta.wpfloat,              
    QMIN:         ta.wpfloat,
    V0S:          ta.wpfloat,    
    V1S:          ta.wpfloat,
    fall_speed:   fa.CellKField[ta.wpfloat],             # output
):
    _fall_speed(density, prefactor, offset, exponent, QMIN, V0S, V1S, out=fall_speed)

@gtx.field_operator
def _ice_deposition_nucleation(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    qc:        fa.CellKField[ta.wpfloat],             # Specific humidity of cloud
    qi:        fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellKField[ta.wpfloat],             # Ice crystal number
    dvsi:      fa.CellKField[ta.wpfloat],             # Vapor excess with respect to ice sat
    dt:        ta.wpfloat,                           # Time step
    QMIN:      ta.wpfloat,
    M0_ICE:    ta.wpfloat,
    TFRZ_HET1: ta.wpfloat,
    TFRZ_HET2: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # Rate of vapor deposition for new ice
    return where( ( (qi <= QMIN) & ((t < TFRZ_HET2) & (dvsi > 0.0)) ) | ( (t <= TFRZ_HET1) & (qc > QMIN) ), minimum(M0_ICE * ni, maximum(0.0, dvsi)) / dt, 0.0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_deposition_nucleation(
    t:         fa.CellKField[ta.wpfloat],             # Temperature
    qc:        fa.CellKField[ta.wpfloat],             # Specific humidity of cloud
    qi:	       fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    ni:	       fa.CellKField[ta.wpfloat],             # Ice crystal number
    dvsi:      fa.CellKField[ta.wpfloat],             # Vapor excess with respect to ice sat
    dt:	       ta.wpfloat,                           # Time step 
    QMIN:      ta.wpfloat,
    M0_ICE:    ta.wpfloat,
    TFRZ_HET1: ta.wpfloat,
    TFRZ_HET2: ta.wpfloat,
    vapor_deposition_rate: fa.CellKField[ta.wpfloat]  # output
):
    _ice_deposition_nucleation( t, qc, qi, ni, dvsi, dt, QMIN, M0_ICE, TFRZ_HET1, TFRZ_HET2, out=vapor_deposition_rate )

@gtx.field_operator
def _ice_mass(
    qi:        fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellKField[ta.wpfloat],             # Ice crystal number
    M0_ICE:    ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # Ice mass
    MI_MAX = 1.0e-9
    return maximum(M0_ICE*ni, minimum(qi/ni, MI_MAX))

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_mass(
    qi:        fa.CellKField[ta.wpfloat],             # Specific humidity of ice
    ni:        fa.CellKField[ta.wpfloat],             # Ice crystal number
    M0_ICE:    ta.wpfloat,
    ice_mass: fa.CellKField[ta.wpfloat]  # output
):
    _ice_mass( qi, ni, M0_ICE, out=ice_mass )

@gtx.field_operator
def _ice_number(
    t:         fa.CellKField[ta.wpfloat],             # Ambient temperature
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    TMELT:     ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                       # Ice number
    A_COOP = 5.000                                   # Parameter in cooper fit
    B_COOP = 0.304                                   # Parameter in cooper fit
    NIMAX  = 250.e+3                                 # Maximal number of ice crystals
    return minimum(NIMAX, A_COOP * exp( B_COOP * (TMELT - t) ) ) / rho

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_number(
    t:         fa.CellKField[ta.wpfloat],             # Ambient temperature
    rho:       fa.CellKField[ta.wpfloat],             # Ambient density
    TMELT:     ta.wpfloat,
    ice_number: fa.CellKField[ta.wpfloat]             # output
):
    _ice_number( t, rho, TMELT, out=ice_number )

@gtx.field_operator
def _ice_sticking(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    TMELT:    ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:                      # Ice sticking
    A_FREEZ   = 0.09         # Scale factor for freezing depression
    B_MAX_EXP = 1.00         # Maximum for exponential temperature factor
    EFF_MIN   = 0.075        # Minimum sticking efficiency
    EFF_FAC   = 3.5e-3       # Scaling factor [1/K] for cloud ice sticking efficiency
    TCRIT     = TMELT - 85.0 # Temperature at which cloud ice autoconversion starts

    return maximum( maximum( minimum( exp( A_FREEZ * (t-TMELT) ), B_MAX_EXP ), EFF_MIN ), EFF_FAC * (t-TCRIT) )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_sticking(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    TMELT:    ta.wpfloat,
    ice_sticking: fa.CellKField[ta.wpfloat]  # output
):
    _ice_sticking( t, TMELT, out=ice_sticking )

@gtx.field_operator
def _snow_lambda(
    rho:        fa.CellKField[ta.wpfloat],           # Ambient density
    qs:         fa.CellKField[ta.wpfloat],           # Snow specific mass
    ns:         fa.CellKField[ta.wpfloat],           # Snow number
    QMIN:       ta.wpfloat,                         # 
    AMS:        ta.wpfloat,                         # 
    BMS:        ta.wpfloat,                         # 
) -> fa.CellKField[ta.wpfloat]:                      # Riming snow rate
    A2     = AMS * 2.0            # (with ams*gam(bms+1.0_wp) where gam(3) = 2)
    LMD_0  = 1.0e+10              # no snow value of lambda
    BX     = 1.0 / ( BMS + 1.0 )  # Exponent
    QSMIN  = 0.0e-6               # TODO: Check with Georgiana that this value is correct

    return where( qs > QMIN, power( (A2*ns / ((qs + QSMIN) * rho)), BX ), LMD_0 )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_lambda(
    rho:        fa.CellKField[ta.wpfloat],           # Ambient density
    qs:         fa.CellKField[ta.wpfloat],           # Snow specific mass
    ns:         fa.CellKField[ta.wpfloat],           # Snow number
    QMIN:       ta.wpfloat,                         # 
    AMS:        ta.wpfloat,                         # 
    BMS:        ta.wpfloat,                         # 
    riming_snow_rate: fa.CellKField[ta.wpfloat]  # output
):
    _snow_lambda( rho, qs, ns, QMIN, AMS, BMS, out=riming_snow_rate )

@gtx.field_operator
def _snow_number(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient air density
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,                           # 
    AMS:      ta.wpfloat,                           # 
    TMELT:    ta.wpfloat,                           # 
) -> fa.CellKField[ta.wpfloat]:                      # Snow number
    TMIN = TMELT - 40.0
    TMAX = TMELT
    QSMIN = 2.0e-6
    XA1 = -1.65e+0
    XA2 = 5.45e-2
    XA3 = 3.27e-4
    XB1 = 1.42e+0
    XB2 = 1.19e-2
    XB3 = 9.60e-5
    N0S0 = 8.00e+5
    N0S1 = 13.5 * 5.65e+05
    N0S2 = -0.107
    N0S3 = 13.5
    N0S4 = 0.5 * N0S1
    N0S5 = 1.e6
    N0S6 = 1.e2 * N0S1
    N0S7 = 1.e9

    # TODO: see if these can be incorporated into WHERE statement
    tc   = maximum( minimum( t, TMAX), TMIN ) - TMELT
    alf  = power( 10.0, ( XA1 + tc * (XA2 + tc * XA3)) )
    bet  = XB1 + tc * ( XB2  + tc * XB3 )
    n0s  = N0S3 * power( ( ( qs + QSMIN ) * rho / AMS), ( 4.0 - 3.0 * bet ) ) / ( alf * alf * alf )
    y    = exp( N0S2 * tc )
    n0smn= maximum( N0S4 * y, N0S5 )
    n0smx= minimum( N0S6 * y, N0S7 )
    return where( qs > QMIN, minimum( n0smx, maximum( n0smn, n0s ) ) , N0S0  )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def snow_number(
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    rho:      fa.CellKField[ta.wpfloat],             # Ambient air density
    qs:       fa.CellKField[ta.wpfloat],             # Snow specific mass
    QMIN:     ta.wpfloat,                           # 
    AMS:      ta.wpfloat,                           # 
    TMELT:    ta.wpfloat,                           # 
    snow_number: fa.CellKField[ta.wpfloat]           # output
):
    _snow_number( t, rho, qs, QMIN, AMS, TMELT, out=snow_number )

@gtx.field_operator
def _vel_scale_factor_ice(
    xrho:     fa.CellKField[ta.wpfloat],             # sqrt(rho_00/rho)
) -> fa.CellKField[ta.wpfloat]:                      # Snow number
    B_I = 0.66666666666666667
    return power( xrho, B_I )

@gtx.field_operator
def _vel_scale_factor_snow(
    xrho:     fa.CellKField[ta.wpfloat],             # sqrt(rho_00/rho)
    rho:      fa.CellKField[ta.wpfloat],             # Density of condensate
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    qs:       fa.CellKField[ta.wpfloat],             # Specific mass
    QMIN:     ta.wpfloat,                           #
    AMS:      ta.wpfloat,                           #
    TMELT:    ta.wpfloat,                           #
) -> fa.CellKField[ta.wpfloat]:                      # Scale factor
    B_S = -0.16666666666666667
    return xrho * power( _snow_number( t, rho, qs, QMIN, AMS, TMELT ),  B_S )

@gtx.field_operator
def _vel_scale_factor_default(
    xrho:     fa.CellKField[ta.wpfloat],             # sqrt(rho_00/rho)
) -> fa.CellKField[ta.wpfloat]:                      # Scale factor
    return xrho

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vel_scale_factor_ice(
    xrho:     fa.CellKField[ta.wpfloat],             # sqrt(rho_00/rho)
    scale_factor: fa.CellKField[ta.wpfloat]          # output
):
    _vel_scale_factor_ice( xrho, out=scale_factor )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def vel_scale_factor_snow(
    xrho:     fa.CellKField[ta.wpfloat],             # sqrt(rho_00/rho)
    rho:      fa.CellKField[ta.wpfloat],             # Density of condensate
    t:        fa.CellKField[ta.wpfloat],             # Temperature
    qs:       fa.CellKField[ta.wpfloat],             # Specific mass
    QMIN:     ta.wpfloat,                           #
    AMS:      ta.wpfloat,                           #
    TMELT:    ta.wpfloat,                           #
    scale_factor: fa.CellKField[ta.wpfloat]          # output
):
    _vel_scale_factor_snow( xrho, rho, t, qs, QMIN, AMS, TMELT, out=scale_factor )
