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
The reference implementation is A. Jocksch's pair of dispersion blocks in
src/atm_dyn_iconam/mo_nh_stepping.f90 of icon-exclaim branch transport_ajocksch (pristine
commit dacecf46aa: theta = 0 at lines 3322-3466, theta = 30 degrees at 3467-3678). Line
numbers below are of the capture branch transport_ajocksch_capture at commit db7a1f149d,
which runs both blocks behind the ICON_DISPERSION switch and produced the Fortran tables
of the port: theta = 0 at 3397-3550 (the wave 3488-3495, x only; the frequency and the
exact-translation check 3528-3543), theta = 30 degrees at 3551-3791 (the wave
``x u_x + y u_y`` 3656-3672, the chequerboard re-imposition 3714-3753). Numpy only: the
real and imaginary parts are two real tracers of the linear schemes, whose responses
superpose.
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
    ``-i x 2 / length alpha`` (mo_nh_stepping.f90:3490-3493 at db7a1f149d), with ``length`` the
    edge length, and for 30 degrees ``-i (x u_x + y u_y) 2 / length alpha`` (:3662-3669), the
    operations in the Fortran's order (``((-r) 2) / length) alpha``, the real part zero).
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
    """``omega = -ln(q_new / q_now) i / dt (a / 2)``, the Fortran's mo_nh_stepping.f90:3534-3538.

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
    """``|q_new - q_now / exp(-i 2 CFL alpha)|``, the Fortran's ``diff`` (mo_nh_stepping.f90:3539-3543).

    The exact one-step solution is the wave translated by ``2 CFL`` half-edges, i.e. the
    phase advanced by ``2 CFL alpha`` (the 1.9999999999 of the Fortran's time step is not
    in this check, as it is not in the Fortran's).
    """
    return np.abs(q_new - q_now / np.exp(-1j * 2.0 * cfl * alpha))


def chequerboard_phase_factor(
    *,
    alpha: np.ndarray,
    wind_angle: float,
    cell_center_x: np.ndarray,
    cell_center_y: np.ndarray,
    reference_cell: int,
    edge_length: float,
) -> np.ndarray:
    """``exp(-i ((x_j - x_ref) u_x + (y_j - y_ref) u_y) 2 / a alpha)`` per (cell, alpha).

    The phase factor of the theta = 30 degree block's re-imposition
    (mo_nh_stepping.f90:3735-3749 at db7a1f149d): cell ``j`` (1-based) is shifted from the
    reference cell of its parity ``odd_even = mod(j, 2) + 1``, reference 2 being the
    Fortran's cell ``k`` and reference 1 its neighbour ``k + 1`` (:3719-3736). Here
    ``reference_cell`` is ``k`` 0-based (694 for the Fortran's 695), so the cells of even
    0-based index take reference 2 and the odd ones reference 1. Independent of the
    iteration, so computed once per wind and grid.
    """
    u_x = np.cos(wind_angle)
    u_y = np.sin(wind_angle)
    parity_reference = _parity_reference(cell_center_x.size, reference_cell)
    x_ref = cell_center_x[parity_reference]
    y_ref = cell_center_y[parity_reference]
    shift = (cell_center_x - x_ref) * u_x + (cell_center_y - y_ref) * u_y
    phase = -shift[:, None] * 2.0 / edge_length * np.asarray(alpha)[None, :]
    return np.exp(1j * phase)


def reimpose_chequerboard_wave(
    *,
    q_new: np.ndarray,
    phase_factor: np.ndarray,
    reference_cell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Real and imaginary part of the wave re-imposed from the advected field (cells, alpha).

    mo_nh_stepping.f90:3731-3749 at db7a1f149d: ``tracer_ref(2)`` is ``q_new`` at the
    reference cell ``k``, ``tracer_ref(1)`` at ``k + 1``; every cell gets
    ``tracer_ref(odd_even) * phase_factor``, whose real and imaginary parts are divided by
    ``|tracer_ref(1)|`` (after taking the parts, as the Fortran does), so the amplitude at
    ``k + 1`` is renormalised to one and the one at ``k`` keeps its ratio to it.

    The complex product is written out as ``(a c - b d, a d + b c)`` in separate real
    operations, the product of the Fortran built without FMA (-Mnofma); numpy's complex
    multiply differs from it in the last bit on aarch64 (contracted). The modulus is libm's
    ``hypot``.
    """
    parity_reference = _parity_reference(q_new.shape[0], reference_cell)
    amplitude = q_new[parity_reference, :]
    reference = q_new[reference_cell + 1, :]
    norm = np.hypot(reference.real, reference.imag)
    a, b = amplitude.real, amplitude.imag
    c, d = phase_factor.real, phase_factor.imag
    wave_re = a * c - b * d
    wave_im = a * d + b * c
    return wave_re / norm[None, :], wave_im / norm[None, :]


def _parity_reference(num_cells: int, reference_cell: int) -> np.ndarray:
    """Per cell the 0-based reference cell of ``odd_even = mod(j, 2) + 1`` (j 1-based)."""
    odd_even = (np.arange(num_cells) + 1) % 2 + 1
    return np.where(odd_even == 2, reference_cell, reference_cell + 1)
