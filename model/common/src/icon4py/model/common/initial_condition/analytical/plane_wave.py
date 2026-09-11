# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""A complex plane wave on cell centres and the numerical frequency it acquires in one step.

The dispersion analysis of Jocksch et al. (PPAM 2026, section 4): the wave
``q = exp(-i alpha (x u_x + y u_y) 2 / a)`` with the non-dimensional wavenumber ``alpha``
(``a`` the edge length, so ``a / 2`` is the length scale, ``alpha = pi`` the 2-cell wave along
the wind ``(u_x, u_y) = (cos theta, sin theta)``), advected once, gives the numerical
frequency ``omega = -ln(q_new / q_now) i (a / 2) / dt`` per cell, whose real part is the phase
speed (exact: ``omega = alpha``) and whose negative imaginary part the growth rate (exact: 0).
The reference implementation is the live block in A. Jocksch's icon-exclaim
(mo_nh_stepping.f90, 'if (.false.)' dispersion block): the wave at lines 3427-3437 (theta = 0,
the x coordinate only; the theta = 30 degree block uses ``x u_x + y u_y``), the frequency and
the exact-translation check at 3462-3478. Numpy only: the real and imaginary parts are two
real tracers of the linear schemes, whose responses superpose.
"""

from __future__ import annotations

import numpy as np


def plane_wave(
    *,
    alpha: float,
    wind_angle: float,
    cell_center_x: np.ndarray,
    cell_center_y: np.ndarray,
    edge_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary part of ``exp(-i alpha (x cos(theta) + y sin(theta)) 2 / a)``.

    ``wind_angle`` (theta) is in radians; for ``wind_angle = 0`` the phase is the Fortran's
    ``-i x 2 / length alpha`` (mo_nh_stepping.f90:3429-3432), with ``length`` the edge length.
    """
    u_x = np.cos(wind_angle)
    u_y = np.sin(wind_angle)
    phase = -(cell_center_x * u_x + cell_center_y * u_y) * 2.0 / edge_length * alpha
    wave = np.exp(1j * phase)
    return wave.real, wave.imag


def numerical_frequency(
    *,
    q_now: np.ndarray,
    q_new: np.ndarray,
    dtime: float,
    edge_length: float,
) -> np.ndarray:
    """``omega = -ln(q_new / q_now) i / dt (a / 2)``, the Fortran's mo_nh_stepping.f90:3469-3473.

    The principal branch of the logarithm, as the Fortran's ``log``: ``Re omega`` is
    ``arg(q_new / q_now) (a / 2) / dt`` in ``(-pi, pi] (a / 2) / dt`` (aliased for
    ``2 CFL alpha > pi``, not unwrapped, the tables carry the raw value) and ``-Im omega`` is
    ``ln |q_new / q_now| (a / 2) / dt``, positive where the wave grows.
    """
    return -np.log(q_new / q_now) * 1j / dtime * (edge_length / 2)


def exact_translation_error(
    *,
    q_now: np.ndarray,
    q_new: np.ndarray,
    cfl: float,
    alpha: float | np.ndarray,
) -> np.ndarray:
    """``|q_new - q_now / exp(-i 2 CFL alpha)|``, the Fortran's ``diff`` (mo_nh_stepping.f90:3474-3478).

    The exact one-step solution is the wave translated by ``2 CFL`` half-edges, i.e. the
    phase advanced by ``2 CFL alpha`` (the 1.9999999999 of the Fortran's time step is not
    in this check, as it is not in the Fortran's).
    """
    return np.abs(q_new - q_now / np.exp(-1j * 2.0 * cfl * alpha))
