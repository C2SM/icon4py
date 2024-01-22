# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Mathematical constants.

Values are taken from glibc 2.2.5: /usr/include/math.h
These constants are provided with more significant digits than necessary
for a 64-bit double precision number; they may be used for other purposes where
the extra precision is necessary or useful.

"""

from typing import Final

from gt4py.eve.utils import FrozenNamespace


class MathConstants(FrozenNamespace):
    """Mathematical constants."""

    euler = 2.71828182845904523536028747135266250
    """e"""
    log2e = 1.44269504088896340735992468100189214
    """log2(e)"""
    log10e = 0.434294481903251827651128918916605082
    """log10(e)"""
    ln2 = 0.693147180559945309417232121458176568
    """ln(2)"""
    ln10 = 2.30258509299404568401799145468436421
    """ln(10)"""
    pi = 3.14159265358979323846264338327950288
    """Ratio of a circle's circumference to its diameter"""
    pi2 = pi * 2.0
    """2*pi"""
    pi_2 = 1.57079632679489661923132169163975144
    """pi/2"""
    pi_4 = 0.785398163397448309615660845819875721
    """pi/4"""
    rpi = 0.318309886183790671537767526745028724  # 1/pi
    """1/pi"""
    rpi_2 = 0.636619772367581343075535053490057448
    """2/pi"""
    rsqrtpi_2 = 1.12837916709551257389615890312154517
    """2/(sqrt(pi)"""
    sqrt2 = 1.41421356237309504880168872420969808
    """Square root of 2"""
    sqrt1_2 = 0.707106781186547524400844362104849039
    """1/sqrt(2)"""
    sqrt3 = 1.7320508075688772935274463415058723
    """Square root of 3"""
    sqrt1_3 = 0.5773502691896257645091487805019575
    """1/sqrt(3)"""
    cos45 = sqrt1_2
    """Cosine of 45°"""
    one_third = 1.0 / 3.0

    # some more useful constants
    pi_5 = pi * 0.2
    """Half angle of pentagon"""

    rad2deg = 180.0 / pi
    """Conversion factor from radians to degree"""
    deg2rad = pi / 180.0
    """Conversion factor from degree to radians"""
    eps = 1.0e-8  # DL: Actually 1.19209e-07, no?
    """Residual bound for solvers """

    dbl_eps = abs(7.0 / 3 - 4.0 / 3 - 1)
    """Machine epsilon in single and double precision"""
    pi_180 = pi / 180.0

    phi0 = 0.46364760900080614903  # phi0 = 0.5*pi - 2.*acos(1.0/(2.*sin(pi/5.)))
    """latitude of the lowest major triangle corner and the latitude of the major hexagonal faces centers"""


math_const: Final = MathConstants()
