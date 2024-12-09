# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from gt4py.next.ffront.fbuiltins import where, power, maximum, minimum
from icon4py.model.common import field_type_aliases as fa, type_alias as ta

@gtx.field_operator
def _cloud_to_rain(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass
    qr:       fa.CellField[ta.wpfloat],             # Rain water specific mass
    nc:       fa.CellField[ta.wpfloat],             # Cloud water number concentration
    TFRZ_HOM: ta.wpfloat,
) -> fa.CellField[ta.wpfloat]:                      # Return: Riming graupel rate
    QMIN_AC       = 1.0e-6                          # threshold for auto conversion
    TAU_MAX       = 0.90e0                            # maximum allowed value of tau
    TAU_MIN       = 1.0e-30                         # minimum allowed value of tau
    A_PHI         = 6.0e2                           # constant in phi-function for autoconversion
    B_PHI         = 0.68e0                          # exponent in phi-function for autoconversion
    C_PHI         = 5.0e-5                          # exponent in phi-function for accretion
    AC_KERNEL     = 5.25e0                          # kernel coeff for SB2001 accretion
    X3            = 2.0e0                           # gamma exponent for cloud distribution
    X2            = 2.6e-10                         # separating mass between cloud and rain
    X1            = 9.44e9                          # kernel coeff for SB2001 autoconversion
    AU_KERNEL     = X1/(20.0*X2) * (X3+2.0) * (X3+4.0) / ((X3+1.0)*(X3+1.0))

    # TO-DO: put as much of this into the WHERE statement as possible
    tau = maximum(TAU_MIN, minimum(1.0-qc/(qc+qr), TAU_MAX))  # temporary cannot go in where
    phi = power(tau,B_PHI)
    phi = A_PHI * phi * power(1.0-phi, 3.0)
    xau = AU_KERNEL * power(qc*qc/nc, 2.0) * (1.0 + phi/power(1.0-tau,2.0))
    xac = AC_KERNEL * qc * qr * power(tau/(tau+C_PHI),4.0)
    return where( (qc > QMIN_AC) & (t > TFRZ_HOM), xau+xac , 0. )

@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_to_rain(
    t:        fa.CellField[ta.wpfloat],             # Temperature
    qc:       fa.CellField[ta.wpfloat],             # Cloud specific mass 
    qr:       fa.CellField[ta.wpfloat],             # Rain water specific mass
    nc:       fa.CellField[ta.wpfloat],             # Cloud water number concentration
    TFRZ_HOM: ta.wpfloat,
    conversion_rate:  fa.CellField[ta.wpfloat],     # output
):
    _cloud_to_rain(t, qc, qr, nc, TFRZ_HOM, out=conversion_rate)
