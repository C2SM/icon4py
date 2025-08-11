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

import datetime
import logging
import math
import os
from cmath import sqrt as c_sqrt, exp as c_exp
import scipy

from icon4py.model.common.config import Device
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import xarray as xr
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm

from icon4py.model.common.settings import device, xp
from icon4py.model.driver.initialization_utils import (
    WAVETYPE,
    DivWave,
    configure_logging,
)


log = logging.getLogger(__name__)


def determine_divergence_in_div_converge_experiment(
    lat: xp.ndarray, lon: xp.ndarray, zifc: xp.ndarray, nz: int, div_wave: DivWave
):
    assert zifc.shape[1] == nz + 1
    top_height = zifc[0]
    if div_wave.wave_type == WAVETYPE.SPHERICAL_HARMONICS:
        v_scale = 90.0 / 7.5
        sphere_radius = 6371229.0
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)

        analytic_divergence = div_wave.x_wavenumber_factor * (
            -4.0
            * math.pi
            * 1.0e-5
            * u_factor
            * xp.sin(2.0 * lon * div_wave.x_wavenumber_factor)
            * xp.cos(lat * div_wave.y_wavenumber_factor * v_scale)
            * xp.cos(lat * div_wave.y_wavenumber_factor * v_scale)
            * xp.sin(lat * div_wave.y_wavenumber_factor * v_scale)
        ) + div_wave.y_wavenumber_factor * (
            math.pi
            * 1.0e-5
            * v_factor
            * xp.cos(lon * div_wave.x_wavenumber_factor)
            * xp.cos(2.0 * lat * div_wave.y_wavenumber_factor * v_scale)
        )
    elif div_wave.wave_type == WAVETYPE.X_WAVE:
        analytic_divergence = -2.0 * math.pi * 1.0e-5 * xp.sin(lon * div_wave.x_wavenumber_factor)
    elif div_wave.wave_type == WAVETYPE.Y_WAVE:
        v_scale = 180.0 / 7.5
        analytic_divergence = (
            -2.0 * math.pi * 1.0e-5 * xp.sin(v_scale * lat * div_wave.y_wavenumber_factor)
        )
    elif div_wave.wave_type == WAVETYPE.X_AND_Y_WAVE:
        v_scale = 180.0 / 7.5
        analytic_divergence = -2.0 * math.pi * 1.0e-5 * xp.sin(
            lon * div_wave.x_wavenumber_factor
        ) - 2.0 * math.pi * 1.0e-5 * xp.sin(v_scale * lat * div_wave.y_wavenumber_factor)
    elif div_wave.wave_type == WAVETYPE.Z_WAVE:
        w_factor = math.pi * div_wave.z_wavenumber_factor / top_height
        w = xp.sin(zifc * w_factor) / w_factor
        analytic_divergence = -xp.sin(zifc * w_factor)
    elif div_wave.wave_type == WAVETYPE.X_AND_Z_WAVE:
        w_factor = math.pi * div_wave.z_wavenumber_factor / top_height
        analytic_divergence = -2.0 * math.pi * 1.0e-5 * xp.sin(lon * div_wave.x_wavenumber_factor) - xp.sin(zifc * w_factor)
    elif div_wave.wave_type == WAVETYPE.Y_AND_Z_WAVE:
        w_factor = math.pi * div_wave.z_wavenumber_factor / top_height
        v_scale = 180.0 / 7.5
        analytic_divergence = (
            -2.0 * math.pi * 1.0e-5 * xp.sin(v_scale * lat * div_wave.y_wavenumber_factor)
            -xp.sin(zifc * w_factor)
        )
    else:
        raise NotImplementedError(f"Wave type {div_wave.wave_type} not implemented")
    return analytic_divergence


def cubic_solution(a: float, b: float, c: float, d: float):
    assert a != 0.0, f"First coefficient of cubic equation cannot be zero, {a}"
    p = (3.0 * a * c - b**2) / (3.0 * a**2)
    q = (2.0 * b**3 - 9.0 * a * b * c + 27.0 * d * a**2) / (27.0 * a**3)
    # x = t - b/3a
    determinant = -(4.0 * p**3 + 27.0 * q**2)

    if np.abs(c) <= 1.0e-15 and np.abs(d) <= 1.0e-12:
        first_solution = -b
        second_solution = 0.0
        third_solution = 0.0
        # print("cubic polynomial determinant (c and d are zero) is ", determinant)
    else:
        assert (
            determinant > -1.0e-15
        ), f"Determinant with cubic equation coefficients {a}, {b}, {c}, {d} is negative: {determinant}"
        # print("cubic polynomial determinant is ", a, b, c, d, determinant)

        sqrt_factor = c_sqrt(0.25 * q**2 + 1.0 / 27.0 * p**3)
        u1 = -0.5 * q + sqrt_factor
        u2 = -0.5 * q - sqrt_factor

        cubic_root = 0.5 * (-1 + 1j * math.sqrt(3.0))
        squared_cubic_root = 0.5 * (-1 - 1j * math.sqrt(3.0))
        if sqrt_factor.imag == 0.0:
            u1_cube = np.sign(u1) * np.abs(u1) ** (1 / 3)
            u2_cube = np.sign(u2) * np.abs(u2) ** (1 / 3)
        else:
            u1_cube = u1 ** (1 / 3)
            u2_cube = u2 ** (1 / 3)
        first_solution = u1_cube + u2_cube - b / (3.0 * a)
        second_solution = cubic_root * u1_cube + squared_cubic_root * u2_cube - b / (3.0 * a)
        third_solution = squared_cubic_root * u1_cube + cubic_root * u2_cube - b / (3.0 * a)

        def _cubic_compute(solution):
            return a * solution**3 + b * solution**2 + c * solution + d

        if first_solution.real <= 1.0e-15:
            assert (
                abs(first_solution.imag) < 1.0e-8
            ), f"First solution is not real, {first_solution}"
        else:
            assert (
                np.abs(first_solution.imag) / np.abs(first_solution.real) < 1.0e-6
            ), f"First solution is not real, {first_solution}"
        if second_solution.real <= 1.0e-15:
            assert (
                abs(second_solution.imag) < 1.0e-8
            ), f"Second solution is not real, {second_solution}"
        else:
            assert (
                np.abs(second_solution.imag) / np.abs(second_solution.real) < 1.0e-6
            ), f"Second solution is not real, {second_solution.real}, {second_solution.imag}"
        if third_solution.real <= 1.0e-15:
            assert (
                abs(third_solution.imag) < 1.0e-8
            ), f"Third solution is not real, {third_solution}"
        else:
            assert (
                np.abs(third_solution.imag) / np.abs(third_solution.real) < 1.0e-6
            ), f"Third solution is not real, {third_solution}"

        first_solution = first_solution.real
        second_solution = second_solution.real
        third_solution = third_solution.real

        assert (
            abs(_cubic_compute(first_solution)) < 1.0e-10
        ), f"Cubic polynomial is not zero, first solution is not correct, {_cubic_compute(first_solution)}, {first_solution}"
        assert (
            abs(_cubic_compute(second_solution)) < 1.0e-10
        ), f"Cubic polynomial is not zero, second solution is not correct, {_cubic_compute(second_solution)}, {second_solution}"
        assert (
            abs(_cubic_compute(third_solution)) < 1.0e-10
        ), f"Cubic polynomial is not zero, third solution is not correct, {_cubic_compute(third_solution)}, {third_solution}"

    return (first_solution, second_solution, third_solution)


def eigen_divergence(
    divergence_factor: float, order: int, grid_space: float, div_wave: DivWave, debug: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigen_value = np.zeros(3, dtype=float)
    eigen_vector = np.zeros((3, 3), dtype=float)
    eigen_vector_cartesian = np.zeros((3, 2), dtype=float)

    a = 0.25 * 3.0 * grid_space * 2.0 * math.pi * div_wave.x_wavenumber_factor * 1.0e-5
    b = 0.25 * math.sqrt(3.0) * grid_space * 2.0 * math.pi * div_wave.y_wavenumber_factor * 1.0e-5
    if debug:
        print("a, b: ", a, b)
        print("cos(a+b), cos(a-b), cos(2a): ", math.cos(a + b), math.cos(a - b), math.cos(2.0 * a))

    if order == 1:
        sol1, sol2, sol3 = cubic_solution(
            1.0,
            0.0,
            -(math.cos(a + b) ** 2 + math.cos(a - b) ** 2 + math.cos(2.0 * a) ** 2),
            2.0 * math.cos(a + b) * math.cos(a - b) * math.cos(2.0 * a),
        )
        sol1 = sol1 - 1.0
        sol2 = sol2 - 1.0
        sol3 = sol3 - 1.0
        if debug:
            print("eigen solutions: ", sol1, sol2, sol3)
        # eigen_value[0] = -1.0
        # eigen_value[1] = -1.0 + math.sqrt(1.0 + math.cos(2.0 * a) * (math.cos(2.0 * a) + math.cos(2.0 * b)))
        # eigen_value[2] = -1.0 - math.sqrt(1.0 + math.cos(2.0 * a) * (math.cos(2.0 * a) + math.cos(2.0 * b)))
        eigen_value[0] = sol1
        eigen_value[1] = sol2
        eigen_value[2] = sol3
    elif order == 2:
        if b == 0.0:
            eigen_value[0] = 0.0
            eigen_value[1] = 0.0
            eigen_value[2] = 2.0 * (math.cos(2.0 * a) - 1.0)
        # elif a == 0.0:
        #     g = math.cos(4.0*b) + math.cos(2.0*b) - 2.0
        #     f = (math.cos(4.0*b) - 1.0) * (math.cos(2.0*b) - 1.0) - (math.cos(3.0*b) - math.cos(b))**2
        #     eigen_value[0] = 0.5 * (g + math.sqrt(g**2 - 4.0*f))
        #     eigen_value[1] = 0.5 * (g - math.sqrt(g**2 - 4.0*f))
        #     eigen_value[2] = 0.0
        elif a == b:
            eigen_value[0] = math.cos(4.0 * a) - 1.0
            eigen_value[1] = 0.0
            eigen_value[2] = math.cos(4.0 * a) - 1.0
        else:
            sol1, sol2, sol3 = cubic_solution(
                1.0,
                2.0 * (1.0 + math.sin(2.0 * b) ** 2 - math.cos(2.0 * a) * math.cos(2.0 * b)),
                0.0,  # 12.0 * (math.cos(2.0 * b) + math.cos(2.0 * a))**2,
                2.0
                * math.sin(2.0 * b) ** 2
                * (
                    4.0 * math.sin(a + b) ** 2 * math.sin(a - b) ** 2
                    + (math.cos(2.0 * b) - math.cos(2.0 * a)) ** 2
                    - 4.0
                    * math.sin(a + b)
                    * math.sin(a - b)
                    * (math.cos(2.0 * b) - math.cos(2.0 * a))
                ),
            )
            if debug:
                print("eigen solutions: ", sol1, sol2, sol3)
            eigen_value[0] = sol1
            eigen_value[1] = sol2
            eigen_value[2] = sol3
    else:
        raise NotImplementedError(
            f"computation of eigen values with divergence order {order} is not implemented."
        )

    def _make_matrix_lists(eigenvalue):
        if order == 1:
            diagonal_element = -1.0 - eigenvalue
            list1 = [diagonal_element, math.cos(a + b), math.cos(a - b)]
            list2 = [math.cos(a + b), diagonal_element, -math.cos(2.0 * a)]
            list3 = [math.cos(a - b), -math.cos(2.0 * a), diagonal_element]
        elif order == 2:
            beta = math.cos(a - 3.0 * b) - math.cos(a + b)
            gamma = math.cos(a + 3.0 * b) - math.cos(a - b)
            delta = math.cos(2.0 * b) - math.cos(2.0 * a)
            list1 = [math.cos(4.0 * b) - 1.0 - eigenvalue, beta, gamma]
            list2 = [beta, math.cos(2.0 * a - 2.0 * b) - 1.0 - eigenvalue, delta]
            list3 = [gamma, delta, math.cos(2.0 * a + 2.0 * b) - 1.0 - eigenvalue]
        return list1, list2, list3

    triangle_vector = np.zeros((3, 2), dtype=float)
    triangle_vector[0] = np.array([0.0, -1.0], dtype=float)
    triangle_vector[1] = np.array([0.5 * math.sqrt(3.0), 0.5], dtype=float) / math.sqrt(0.25*3.0 + 0.25)
    triangle_vector[2] = np.array([-0.5 * math.sqrt(3.0), 0.5], dtype=float) / math.sqrt(0.25*3.0 + 0.25)
    # print(triangle_vector.T[0])
    # print(triangle_vector.T[1])
    # print()

    for i in range(3):
        list1, list2, list3 = _make_matrix_lists(eigen_value[i])
        matrix = np.zeros((3, 3), dtype=float)
        matrix[0] = np.array(list1, dtype=float, copy=True)
        matrix[1] = np.array(list2, dtype=float, copy=True)
        matrix[2] = np.array(list3, dtype=float, copy=True)

        rank_space = np.zeros((3, 3), dtype=float)

        # handle special case:
        special_case = False
        if order == 2:
            if b == 0.0:
                if i == 0:
                    eigen_vector[i] = np.array((0.0, 0.0, 0.0), dtype=float)
                elif i == 1:
                    eigen_vector[i] = np.array((0.0, 1.0, 1.0), dtype=float)
                elif i == 2:
                    eigen_vector[i] = np.array((0.0, 1.0, -1.0), dtype=float)
                special_case = True
            # elif a == 0.0:
            #     if i == 0 or i == 1:
            #         minor_rank_space = np.zeros((2, 2), dtype=float)
            #         minor_rank_space[0] = np.array([math.cos(4.0*b) - 1.0 - eigen_value[i], math.cos(3.0*b) - math.cos(b)], dtype=float)
            #         minor_rank_space[0] = minor_rank_space[0] / math.sqrt(np.sum(minor_rank_space[0][:] ** 2))
            #         minor_rank_space[1] = np.array([math.cos(3.0*b) - math.cos(b), math.cos(2.0*b) - 1.0 - eigen_value[i]], dtype=float)
            #         # minor_rank_space[1] = minor_rank_space[1] - np.dot(minor_rank_space[1], minor_rank_space[0]) * minor_rank_space[0]
            #         # minor_rank_space[1] = minor_rank_space[1] / math.sqrt(np.sum(minor_rank_space[1][:] ** 2))
            #         eigen_vector[i,0:2] = np.array((-minor_rank_space[0,1], minor_rank_space[0,0]), dtype=float)
            #         null_space_threshold1 = np.dot(minor_rank_space[0], eigen_vector[i,0:2])
            #         null_space_threshold2 = np.dot(minor_rank_space[1], eigen_vector[i,0:2])
            #         assert (
            #             np.abs(null_space_threshold1) < 1.0e-10
            #         ), f"null space is none, {null_space_threshold1} - {minor_rank_space[0]} - {eigen_vector[i]}"
            #         assert (
            #             np.abs(null_space_threshold2) < 1.0e-10
            #         ), f"null space is none, {null_space_threshold2} - {minor_rank_space[1]} - {eigen_vector[i]}"
            #     elif i == 2:
            #         eigen_vector[i] = np.array((0.0, 0.0, 0.0), dtype=float)
            #     special_case = True
            elif a == b:
                if i == 0:
                    eigen_vector[i] = np.array((1.0, 0.0, 0.0), dtype=float)
                elif i == 1:
                    eigen_vector[i] = np.array((0.0, 1.0, 0.0), dtype=float)
                elif i == 2:
                    eigen_vector[i] = np.array((0.0, 0.0, 1.0), dtype=float)
                special_case = True
        if not special_case:
            rank_space[0] = np.array(list1, dtype=float, copy=True)
            rank_space[0] = rank_space[0] / math.sqrt(np.sum(rank_space[0][:] ** 2))

            rank_space[1] = np.array(list2, dtype=float, copy=True)
            rank_space[1] = rank_space[1] - np.dot(rank_space[1], rank_space[0]) * rank_space[0]
            if math.sqrt(np.sum(rank_space[1][:] ** 2)) == 0.0:
                rank_space[1] = rank_space[0]
            else:
                rank_space[1] = rank_space[1] / math.sqrt(np.sum(rank_space[1][:] ** 2))

            rank_space[2] = np.array(list3, dtype=float, copy=True)
            rank_space[2] = rank_space[2] / math.sqrt(np.sum(rank_space[2][:] ** 2))

            eigen_vector[i] = np.cross(rank_space[0], rank_space[1])
            if math.sqrt(np.sum(eigen_vector[i][:] ** 2)) != 0.0:
                eigen_vector[i] = eigen_vector[i] / math.sqrt(np.sum(eigen_vector[i][:] ** 2))

            null_space_threshold1 = np.dot(matrix[0], eigen_vector[i])
            null_space_threshold2 = np.dot(matrix[1], eigen_vector[i])
            null_space_threshold3 = np.dot(matrix[2], eigen_vector[i])
            assert (
                np.abs(null_space_threshold1) < 1.0e-6
            ), f"null space is none, {null_space_threshold1} - {rank_space[0]} - {rank_space[1]} - {rank_space[2]} - {eigen_vector[i]} - {eigen_value[i]} // {eigen_value[:]}"
            assert (
                np.abs(null_space_threshold2) < 1.0e-6
            ), f"null space is none, {null_space_threshold2} - {rank_space[0]} - {rank_space[1]} - {rank_space[2]} - {eigen_vector[i]} - {eigen_value[i]} // {eigen_value[:]}"
            assert (
                np.abs(null_space_threshold3) < 1.0e-6
            ), f"null space is none, {null_space_threshold3} - {rank_space[0]} - {rank_space[1]} - {rank_space[2]} - {eigen_vector[i]} - {eigen_value[i]} // {eigen_value[:]}"
            if debug:
                print(
                    "checking null space ",
                    null_space_threshold1,
                    null_space_threshold2,
                    null_space_threshold3,
                )

        if debug:
            print("eigenvector of eigenvalue ", eigen_value[i], " is ", eigen_vector[i])
        eigen_vector_cartesian[i][0] = np.dot(eigen_vector[i], triangle_vector.T[0])
        eigen_vector_cartesian[i][1] = np.dot(eigen_vector[i], triangle_vector.T[1])

        if debug:
            print(
                "eigenvector in cartesian before normalization of eigenvalue ",
                eigen_value[i],
                " is ",
                eigen_vector_cartesian[i],
            )
        eigen_vector_cartesian[i] = eigen_vector_cartesian[i] / math.sqrt(
            np.sum(eigen_vector_cartesian[i][:] ** 2)
        )
        if debug:
            print(
                "eigenvector in cartesian of eigenvalue ",
                eigen_value[i],
                " is ",
                eigen_vector_cartesian[i],
            )
            print()

    if debug:
        print()
        print(
            "checking orthogonality of eigenvectors ",
            np.dot(eigen_vector[0], eigen_vector[1]),
            np.dot(eigen_vector[0], eigen_vector[2]),
            np.dot(eigen_vector[1], eigen_vector[2]),
        )

    return eigen_value, eigen_vector, eigen_vector_cartesian


def eigen_vorticity(divergence_factor: float, order: int, grid_space: float, div_wave: DivWave):
    eigen_value = np.zeros(3, dtype=float)
    eigen_vector = np.zeros((3, 3), dtype=float)
    eigen_vector_cartesian = np.zeros((3, 2), dtype=float)

    a = 0.25 * 3.0 * grid_space * 2.0 * math.pi * div_wave.x_wavenumber_factor * 1.0e-5
    b = 0.25 * math.sqrt(3.0) * grid_space * math.pi * div_wave.y_wavenumber_factor * 1.0e-5
    print("a, b: ", a, b)
    print("cos(a+b), cos(a-b), cos(2a): ", math.cos(a + b), math.cos(a - b), math.cos(2.0 * a))
    sol1, sol2, sol3 = cubic_solution(
        1.0,
        0.0,
        -(math.cos(a + b) ** 2 + math.cos(a - b) ** 2 + math.cos(2.0 * a) ** 2),
        2.0 * math.cos(a + b) * math.cos(a - b) * math.cos(2.0 * a),
    )
    sol1 = 1.0 - sol1
    sol2 = 1.0 - sol2
    sol3 = 1.0 - sol3
    print("eigen solutions: ", sol1, sol2, sol3)

    if order == 1:
        # eigen_value[0] = -1.0
        # eigen_value[1] = -1.0 + math.sqrt(1.0 + math.cos(2.0 * a) * (math.cos(2.0 * a) + math.cos(2.0 * b)))
        # eigen_value[2] = -1.0 - math.sqrt(1.0 + math.cos(2.0 * a) * (math.cos(2.0 * a) + math.cos(2.0 * b)))
        eigen_value[0] = sol1
        eigen_value[1] = sol2
        eigen_value[2] = sol3
    else:
        raise NotImplementedError(
            f"computation of eigen values with vorticity order {order} is not implemented."
        )
    print("eigenvalues: ", eigen_value)
    print()

    triangle_vector = np.zeros((3, 2), dtype=float)
    triangle_vector[0] = np.array([0.0, -1.0])
    triangle_vector[1] = np.array([0.5 * math.sqrt(3.0), 0.5])
    triangle_vector[2] = np.array([-0.5 * math.sqrt(3.0), 0.5])
    print(triangle_vector.T[0])
    print(triangle_vector.T[1])
    print()

    for i in range(3):
        diagonal_element = 1.0 - eigen_value[i]

        matrix = np.zeros((3, 3), dtype=float)
        matrix[0] = np.array([diagonal_element, math.cos(a + b), math.cos(a - b)])
        matrix[1] = np.array([math.cos(a + b), diagonal_element, math.cos(2.0 * a)])
        matrix[2] = np.array([math.cos(a - b), math.cos(2.0 * a), diagonal_element])

        rank_space = np.zeros((3, 3), dtype=float)
        rank_space[0] = np.array([diagonal_element, math.cos(a + b), math.cos(a - b)])
        rank_space[0] = rank_space[0] / math.sqrt(np.sum(rank_space[0][:] ** 2))

        rank_space[1] = np.array([math.cos(a + b), diagonal_element, math.cos(2.0 * a)])
        rank_space[1] = rank_space[1] - np.dot(rank_space[1], rank_space[0]) * rank_space[0]
        rank_space[1] = rank_space[1] / math.sqrt(np.sum(rank_space[1][:] ** 2))

        rank_space[2] = np.array([math.cos(a - b), math.cos(2.0 * a), diagonal_element])
        rank_space[2] = rank_space[2] / math.sqrt(np.sum(rank_space[1][:] ** 2))

        eigen_vector[i] = np.cross(rank_space[0], rank_space[1])
        eigen_vector[i] = eigen_vector[i] / math.sqrt(np.sum(eigen_vector[i][:] ** 2))

        null_space_threshold1 = np.dot(matrix[0], eigen_vector[i])
        null_space_threshold2 = np.dot(matrix[1], eigen_vector[i])
        null_space_threshold3 = np.dot(matrix[2], eigen_vector[i])
        assert (
            np.abs(null_space_threshold1) < 1.0e-10
        ), f"null space is none, {null_space_threshold1} - {rank_space[0]} - {eigen_vector[i]}"
        assert (
            np.abs(null_space_threshold2) < 1.0e-10
        ), f"null space is none, {null_space_threshold2} - {rank_space[1]} - {eigen_vector[i]}"
        assert (
            np.abs(null_space_threshold3) < 1.0e-10
        ), f"null space is none, {null_space_threshold3} - {rank_space[2]} - {eigen_vector[i]}"
        print(
            "checking null space ",
            null_space_threshold1,
            null_space_threshold2,
            null_space_threshold3,
        )
        print("eigenvector of eigenvalue ", eigen_value[i], " is ", eigen_vector[i])
        eigen_vector_cartesian[i][0] = np.dot(eigen_vector[i], triangle_vector.T[0])
        eigen_vector_cartesian[i][1] = np.dot(eigen_vector[i], triangle_vector.T[1])

        print(
            "eigenvector in cartesian before normalization of eigenvalue ",
            eigen_value[i],
            " is ",
            eigen_vector_cartesian[i],
        )
        eigen_vector_cartesian[i] = eigen_vector_cartesian[i] / math.sqrt(
            np.sum(eigen_vector_cartesian[i][:] ** 2)
        )
        print(
            "eigenvector in cartesian of eigenvalue ",
            eigen_value[i],
            " is ",
            eigen_vector_cartesian[i],
        )
        print()

    print()
    print(
        "checking orthogonality of eigenvectors ",
        np.dot(eigen_vector[0], eigen_vector[1]),
        np.dot(eigen_vector[0], eigen_vector[2]),
        np.dot(eigen_vector[1], eigen_vector[2]),
    )

    return


def eigen_3d_divergence(order: int, grid_space: float, div_wave: DivWave, full_return: bool = True, method: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    k = 2.0 * math.pi * div_wave.x_wavenumber_factor * 1.0e-5
    l = 2.0 * math.pi * div_wave.y_wavenumber_factor * 1.0e-5
    a = 0.25 * 3.0 * grid_space * k
    b = 0.25 * math.sqrt(3.0) * grid_space * l
    nn = 0.5 * grid_space * math.pi * div_wave.z_wavenumber_factor * 1.0e-5 * 2.0 # n detla_z / 2, the extra factor of 2 at the end may be needed for complete Fourier space

    h = grid_space
    delta_z = grid_space
    d_tri = h / math.sqrt(3.0)
    d_hex = h * math.sqrt(3.0)

    if order == 1:
        constant_2d = 1.0 # 8.0 * KD / h**2
        constant_hw = -math.sqrt(3) / 2.0 * h / delta_z # - 4.0 * math.sqrt(3.0) * KD / h / delta_z
        constant_w = 1.j / math.sqrt(3.0) * h / delta_z # 8.0 / math.sqrt(3.0) * 1.j * KD / h / delta_z

        diagonal_element = -1.0
        h_2d_11 = constant_2d * diagonal_element
        h_2d_12 = constant_2d * math.cos(a + b)
        h_2d_13 = constant_2d * math.cos(a - b)
        h_2d_21 = h_2d_12
        h_2d_22 = constant_2d * diagonal_element
        h_2d_23 = constant_2d * -math.cos(2.0 * a)
        h_2d_31 = h_2d_13
        h_2d_32 = h_2d_23
        h_2d_33 = constant_2d * diagonal_element

        h_1 = constant_hw * math.sin(0.5 * d_tri * l) * math.sin(nn)
        h_2 = constant_hw * math.sin(0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l) * math.sin(nn)
        h_3 = constant_hw * math.sin(-0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l) * math.sin(nn)
        w1 = constant_w * c_exp(-0.5j * d_tri * l) * math.sin(nn)
        w2 = constant_w * c_exp(1.0j * (0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l)) * math.sin(nn)
        w3 = constant_w * c_exp(1.0j * (-0.25 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l - 0.5 * d_tri * l)) * math.sin(nn)
        w4 = -0.5 * math.sin(nn)**2 * h**2 / delta_z**2 # - 4.0 * KD / delta_z**2 * math.sin(nn)**2
    elif order == 2:
        constant_2d = 1.0 # 4.0 * KD / 9.0 / h**2
        constant_hw = -9.0 / math.sqrt(3.0) * h / delta_z # -4.0 * KD / h / delta_z / math.sqrt(3.0)
        constant_w = -18.0 / math.sqrt(3.0) * h / delta_z # -8.0 / math.sqrt(3.0) * KD / h / delta_z

        h_2d_11 = constant_2d * (math.cos(4.0*b) - 1.0)
        h_2d_12 = constant_2d * (math.cos(a - 3.0*b) - math.cos(a + b))
        h_2d_13 = constant_2d * (math.cos(a + 3.0*b) - math.cos(a - b))
        h_2d_21 = h_2d_12
        h_2d_22 = constant_2d * (math.cos(2.0*a - 2.0*b) - 1.0)
        h_2d_23 = constant_2d * (math.cos(2.0*b) - math.cos(2.0*a))
        h_2d_31 = h_2d_13
        h_2d_32 = h_2d_23
        h_2d_33 = constant_2d * (math.cos(2.0*a + 2.0*b) - 1.0)

        hw_1 = math.sin(0.5 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
        hw_2 = math.sin(-0.25 * 3.0 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
        hw_3 = math.sin(0.25 * 3.0 * grid_space * k + 0.25 * math.sqrt(3.0) * grid_space * l) * math.sin(nn)
        h_1 = constant_hw * hw_1
        h_2 = constant_hw * hw_2
        h_3 = constant_hw * hw_3
        w1 = constant_w * hw_1
        w2 = constant_w * hw_2
        w3 = constant_w * hw_3
        w4 = -9.0 * math.sin(nn)**2 * h**2 / delta_z**2 # - 4.0 * KD / delta_z**2 * math.sin(nn)**2


    matrix = np.array(
        [
            [h_2d_11, h_2d_12, h_2d_13, h_1],
            [h_2d_21, h_2d_22, h_2d_23, h_2],
            [h_2d_31, h_2d_32, h_2d_33, h_3],
            [w1, w2, w3, w4],
        ], dtype = complex,
    )
    matrix = matrix.T

    triangle_vector = np.zeros((4, 3), dtype=float)
    triangle_vector[0] = np.array([0.0, -1.0, 0.0], dtype=float)
    triangle_vector[1] = np.array([0.5 * math.sqrt(3.0), 0.5, 0.0], dtype=float) / math.sqrt(0.25*3.0 + 0.25)
    triangle_vector[2] = np.array([-0.5 * math.sqrt(3.0), 0.5, 0.0], dtype=float) / math.sqrt(0.25*3.0 + 0.25)
    triangle_vector[3] = np.array([0.0, 0.0, 1.0], dtype=float)
    
    if method == 1:
        T2, Z2 = scipy.linalg.schur(matrix, output='complex')

        if full_return:
            eigen_value = np.array([T2[0, 0], T2[1, 1], T2[2, 2], T2[3, 3]], dtype=complex)
            eigen_vector = np.zeros((4, 4), dtype=complex)
            eigen_vector[0] = Z2[:, 0]# .real
            eigen_vector[1] = Z2[:, 1]# .real
            eigen_vector[2] = Z2[:, 2]# .real
            eigen_vector[3] = Z2[:, 3]# .real
            eigen_vector_cartesian = np.zeros((4, 3), dtype=float)
            for i in range(4):
                eigen_vector_cartesian[i][0] = np.dot(eigen_vector[i], triangle_vector.T[0])
                eigen_vector_cartesian[i][1] = np.dot(eigen_vector[i], triangle_vector.T[1])
                eigen_vector_cartesian[i][2] = np.dot(eigen_vector[i], triangle_vector.T[2])
        else:
            eigen_value = np.array([T2[0, 0], T2[1, 1], T2[2, 2]], dtype=complex)
            eigen_vector = np.zeros((3, 4), dtype=complex)
            # assert np.abs(Z2[0,:].imag).max() < 1.e-10, f"eigenvector 1 is not real: {Z2[0,:]}, {np.abs(Z2[0,:].imag).max()}"
            # assert np.abs(Z2[1,:].imag).max() < 1.e-10, f"eigenvector 2 is not real: {Z2[1,:]}, {np.abs(Z2[1,:].imag).max()}"
            # assert np.abs(Z2[2,:].imag).max() < 1.e-10, f"eigenvector 3 is not real: {Z2[2,:]}, {np.abs(Z2[2,:].imag).max()}"
            eigen_vector[0] = Z2[:, 0]#.real
            eigen_vector[1] = Z2[:, 1]#.real
            eigen_vector[2] = Z2[:, 2]#.real
            eigen_vector_cartesian = np.zeros((3, 3), dtype=float)
            for i in range(3):
                eigen_vector_cartesian[i][0] = np.dot(eigen_vector[i], triangle_vector.T[0])
                eigen_vector_cartesian[i][1] = np.dot(eigen_vector[i], triangle_vector.T[1])
                eigen_vector_cartesian[i][2] = np.dot(eigen_vector[i], triangle_vector.T[2])
        return eigen_value, eigen_vector, eigen_vector_cartesian

    else:
        eigenvalues, eigenvectors = scipy.linalg.eig(matrix)
        if full_return:
            eigen_vector_cartesian = np.zeros((4, 3), dtype=float)
            for i in range(4):
                eigen_vector_cartesian[i][0] = np.dot(eigenvectors[i], triangle_vector.T[0])
                eigen_vector_cartesian[i][1] = np.dot(eigenvectors[i], triangle_vector.T[1])
                eigen_vector_cartesian[i][2] = np.dot(eigenvectors[i], triangle_vector.T[2])
            return eigenvalues, eigenvectors.T, eigen_vector_cartesian
        else:
            # NOT TESTED!!!
            eigen_vector_cartesian = np.zeros((3, 3), dtype=float)
            for i in range(3):
                eigen_vector_cartesian[i][0] = np.dot(eigenvectors[i], triangle_vector.T[0])
                eigen_vector_cartesian[i][1] = np.dot(eigenvectors[i], triangle_vector.T[1])
                eigen_vector_cartesian[i][2] = np.dot(eigenvectors[i], triangle_vector.T[2])
        
            return eigenvalues[:-1], eigenvectors[:-1].T, eigen_vector_cartesian


def compute_wave_vector_component(
    eigen_vector_list: list[np.ndarray],
    div_wave: DivWave,
    threeD: bool = False,
) -> np.ndarray:
    """
                 /\
                /  \
           u3  /    \  u2
              /      \
             /________\
                  u1
    """
    if threeD:
        u1 = np.array([0.0, 1.0, 0.0], dtype=float)
        u2 = np.array([0.5*math.sqrt(3.0), 0.5, 0.0], dtype=float)
        u3 = np.array([-0.5*math.sqrt(3.0), 0.5, 0.0], dtype=float)
        u4 = np.array([0.0, 0.0, 1.0], dtype=float)
        wave_vector = np.array([div_wave.x_wavenumber_factor, div_wave.y_wavenumber_factor, div_wave.z_wavenumber_factor], dtype=float)
        wave_vector = wave_vector / math.sqrt(div_wave.x_wavenumber_factor**2 + div_wave.y_wavenumber_factor**2 + div_wave.z_wavenumber_factor**2)
        wave_vector_component = np.array(
            [
                np.dot(wave_vector, u1),
                np.dot(wave_vector, u2),
                np.dot(wave_vector, u3),
                np.dot(wave_vector, u4)
            ],
            dtype=float,
        )
        eigen_vector_component = np.zeros(len(eigen_vector_list), dtype=float)
        for i in range(len(eigen_vector_list)):
            eigen_vector_component[i] = np.abs(np.dot(wave_vector_component, eigen_vector_list[i]))
    else:
        u1 = np.array([0.0, 1.0], dtype=float)
        u2 = np.array([0.5*math.sqrt(3.0), 0.5], dtype=float)
        u3 = np.array([-0.5*math.sqrt(3.0), 0.5], dtype=float)
        wave_vector = np.array([div_wave.x_wavenumber_factor, div_wave.y_wavenumber_factor], dtype=float)
        wave_vector = wave_vector / math.sqrt(div_wave.x_wavenumber_factor**2 + div_wave.y_wavenumber_factor**2)
        wave_vector_component = np.array(
            [
                np.dot(wave_vector, u1),
                np.dot(wave_vector, u2),
                np.dot(wave_vector, u3)
            ],
            dtype=float,
        )
        eigen_vector_component = np.zeros(len(eigen_vector_list), dtype=float)
        for i in range(len(eigen_vector_list)):
            eigen_vector_component[i] = np.abs(np.dot(wave_vector_component, eigen_vector_list[i]))
    return eigen_vector_component
    
def create_mask(grid_filename: str) -> np.ndarray:
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    mask = (grid.clat.values > np.deg2rad(-7.4)) & (grid.clat.values < np.deg2rad(7.4))
    # mask = xp.ones(grid.clat.shape[0], dtype = bool)
    voc = grid.vertex_of_cell.T.values - 1
    vx = grid.cartesian_x_vertices.values
    vy = grid.cartesian_y_vertices.values
    vx_c = vx[voc]
    vx_c1 = np.roll(vx_c, shift=1, axis=1)
    vy_c = vy[voc]
    vy_c1 = np.roll(vy_c, shift=1, axis=1)
    vv_distance = np.sum(np.abs(vx_c - vx_c1), axis=1) + np.sum(np.abs(vy_c - vy_c1), axis=1)
    neighbor_cell = grid.neighbor_cell_index.T.values - 1
    threshold = 2.0
    interior_mask_ = vv_distance < threshold * 2.0 * grid.mean_edge_length
    interior_mask = np.array(interior_mask_, copy=True)
    interior_mask = np.where(np.sum(interior_mask[neighbor_cell], axis=1) < 3, False, interior_mask)
    mask = mask & interior_mask

    return mask


def plot_tridata(
    grid_filename: str, data: xp.ndarray, mask: np.ndarray, title: str, output_filename: str
):
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    voc = grid.vertex_of_cell.T[mask].values - 1
    if len(data.shape) == 1:
        number_of_layers = 1
    else:
        number_of_layers = data.shape[1]
    log.info(f"plotting {title} with levels {number_of_layers}, {len(data.shape)}, {data.shape}")

    if device == Device.GPU:
        cell_data0 = data.get()
    else:
        cell_data0 = data
    for k in range(number_of_layers):
        if len(cell_data0.shape) == 1:
            cell_data = cell_data0[mask]
        else:
            cell_data = cell_data0[mask, k]
        data_max, data_min = cell_data.max(), cell_data.min()
        log.info(f"data max min: {data_max}, {data_min}")

        used_vertices = np.unique(voc)
        lat_min = grid.vlat[used_vertices].min().values
        lat_max = grid.vlat[used_vertices].max().values
        lon_min = grid.vlon[used_vertices].min().values
        lon_max = grid.vlon[used_vertices].max().values

        # N = 20
        # cmap = plt.get_cmap('jet', N)
        # norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        # creating ScalarMappable
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        cmap = plt.get_cmap("seismic")
        if data_min * data_max < 0.0:
            norm = TwoSlopeNorm(vmin=data_min - 1.0e-8, vcenter=0.0, vmax=data_max + 1.0e-8)
        else:
            boundaries = np.linspace(data_min - 1.0e-8, data_max + 1.0e-8, 101)
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

        plt.tripcolor(grid.vlon, grid.vlat, voc, cell_data, cmap=cmap, norm=norm)
        plt.title(title)
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        plt.xticks(
            [-math.pi, -0.5 * math.pi, 0, 0.5 * math.pi, math.pi],
            ["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"],
        )
        plt.yticks(
            [-7.5*math.pi/180.0, -7.5*math.pi/360.0, 0, 7.5*math.pi/360.0, 7.5*math.pi/180.0],
            ["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"],
        )

        plt.colorbar()  # sm, ticks=np.linspace(0, 2, N)

        plt.savefig(output_filename + "_at_" + str(k) + "_level.pdf", format="pdf", dpi=500)
        plt.clf()

    return


def plot_vertical_section(
    grid_filename: str,
    data: xp.ndarray,
    zifc: xp.ndarray,
    title: str,
    output_filename: str,
):
    if device == Device.GPU:
        data = data.get()
        zifc = zifc.get()
    
    grid = xr.open_dataset(grid_filename, engine="netcdf4")
    mask = (
        (grid.clat.values > np.deg2rad(0.0))
        & (grid.clat.values < np.deg2rad(0.06))  # 500 m resolution
        # & (grid.clat.values < np.deg2rad(0.03))  # 500 m resolution
        # & (grid.clat.values < np.deg2rad(0.015))  # 250 m resolution
        & (grid.clon.values > np.deg2rad(-180))
        & (grid.clon.values < np.deg2rad(180))
        # (grid.clat.values > -0.065)
        # & (grid.clat.values < -0.045)
        # & (grid.clon.values > np.deg2rad(-170))
        # & (grid.clon.values < np.deg2rad(0))
    )

    used_clat = np.unique(grid.clat[mask].values)
    print(used_clat)
    if len(used_clat) > 1 or len(used_clat) == 0:
        print("there are more than one or no layers to be plotted, please adjust the clat range.", len(used_clat))
        import sys

        sys.exit()

    cell_height = np.zeros((zifc.shape[0], zifc.shape[1] - 1), dtype=float)
    for k in range(cell_height.shape[1]):
        cell_height[:, k] = 0.5 * (zifc[:, k] + zifc[:, k + 1])

    # plt.close()
    # plt.plot(cell_height[1000,:], data[1000, :], color='black', label="1000")
    # plt.plot(cell_height[3000,:], data[3000, :], color='blue', label="3000")
    # plt.plot(cell_height[5000,:], data[5000, :], color='red', label="5000")
    # plt.plot(cell_height[7000,:], data[7000, :], color='green', label="7000")

    # plt.savefig("plot_testing.pdf", format="pdf", dpi=400)

    # cell_data = data.cell_height(height=0).isel[mask].values
    cell_height = cell_height[mask, :]
    clon_masked = grid.clon[mask].values
    for k in range(cell_height.shape[1]):
        plt.plot(grid.clon[mask].values, cell_height[:, k], "or", ms=0.1)
    plt.xlim(-math.pi, math.pi)
    # plt.savefig('figure_testing_tri.png', dpi=400)
    plt.savefig("figure_testing_tri_vertical.pdf", format="pdf", dpi=400)

    

    # cell_data = np.transpose(cell_data)
    xx = grid.clon[mask].values
    argsort = xx.argsort()
    xx = xx[argsort[::-1]]
    # for i in range(len(xx)-1):
    #     print(xx[i+1] - xx[i])
    for k in range(cell_height.shape[1]):
        cell_height[:, k] = cell_height[argsort[::-1], k]
    xx = np.repeat(np.expand_dims(xx, axis=-1), cell_height.shape[1], axis=-1)
    # print(cell_height.shape, xx.shape)


    def section_plot(plot_data, xx, plot_cell_height, cmap, plot_title: str, plot_name: str):
        plt.close()
        seismap = plt.get_cmap(cmap)
        print("plot (inside) data min max: ", plot_data.min(), plot_data.max())
        boundaries = np.linspace(plot_data.min() - 1.e-8, plot_data.max() + 1.e-8, 101)
        lnorm = colors.BoundaryNorm(boundaries, seismap.N, clip=True)
        cp = plt.contourf(xx, plot_cell_height, plot_data, cmap=seismap, levels=boundaries, norm=lnorm)
        cb1 = plt.colorbar(cp, location="right")
        plt.title(plot_title)
        plt.xlim(-math.pi, math.pi)

        plt.savefig(plot_name + ".pdf", format="pdf", dpi=400)

        return

    # print("plot data min max: ", data.min(), data.max())
    plot_data = data
    plot_data = plot_data[mask, :]
    for k in range(plot_data.shape[1]):
        plot_data[:, k] = plot_data[argsort[::-1], k]
    # log.critical(f"DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG: {xx[:, 0].min()} {xx[:, 0].max()}")
    # for i in range(xx.shape[0]):
    #     log.critical(f"   ------------    {i} {xx[i, 0]} {plot_data[i, 0]} {plot_data[i, 10]}")
    section_plot(plot_data, xx, cell_height, "seismic", title, output_filename)


def plot_vertical_section_lat(
    grid_filename: str,
    data: xp.ndarray,
    zifc: xp.ndarray,
    title: str,
    output_filename: str,
):
    if device == Device.GPU:
        data = data.get()
        zifc = zifc.get()
    
    grid = xr.open_dataset(grid_filename, engine="netcdf4")
    mask = (
        (grid.clon.values > np.deg2rad(-0.06))
        & (grid.clon.values < np.deg2rad(0.06))
        & (grid.clat.values > np.deg2rad(-180.0))
        & (grid.clat.values < np.deg2rad(180.0))
    )

    used_clon = np.unique(grid.clon[mask].values)
    print(used_clon)
    if len(used_clon) > 1 or len(used_clon) == 0:
        print("there are more than one or no layers to be plotted, please adjust the clon range.", len(used_clon), np.shape(grid.clon[mask].values))
        # import sys

        # sys.exit()

    top_height = np.mean(zifc[:,0])
    cell_height = np.zeros((zifc.shape[0], zifc.shape[1] - 1), dtype=float)
    for k in range(cell_height.shape[1]):
        cell_height[:, k] = 0.5 * (zifc[:, k] + zifc[:, k + 1])

    cell_height = cell_height[mask, :] * 2.0 * math.pi / top_height
    for k in range(cell_height.shape[1]):
        plt.plot(grid.clat[mask].values, cell_height[:, k], "or", ms=0.1)
    plt.xlim(np.min(grid.clat[mask].values), np.max(grid.clat[mask].values))
    plt.savefig("figure_testing_tri_vertical2.pdf", format="pdf", dpi=400)

    xx = grid.clat[mask].values * 180.0 / 7.5
    argsort = xx.argsort()
    xx = xx[argsort[::-1]]
    # yy = grid.clon[mask].values
    # yy = yy[argsort[::-1]]
    # for i in range(xx.shape[0]):
    #     log.critical(f"   ------------    {i} {xx[i]} {yy[i]}")
    for k in range(cell_height.shape[1]):
        cell_height[:, k] = cell_height[argsort[::-1], k]
    xx = np.repeat(np.expand_dims(xx, axis=-1), cell_height.shape[1], axis=-1)

    def section_plot(plot_data, plot_xx, plot_cell_height, cmap, plot_title: str, plot_name: str):
        plt.close()

        f, ax = plt.subplots(constrained_layout=True)
        seismap = plt.get_cmap(cmap)
        print("plot (inside) data min max: ", plot_data.min(), plot_data.max())
        boundaries = np.linspace(plot_data.min() - 1.e-8, plot_data.max() + 1.e-8, 101)
        lnorm = colors.BoundaryNorm(boundaries, seismap.N, clip=True)
        cp = ax.contourf(plot_xx, plot_cell_height, plot_data, cmap=seismap, levels=boundaries, norm=lnorm)
        cb1 = plt.colorbar(cp, location="right")
        ax.set_title(plot_title)
        ax.set_xlim(plot_xx.min(), plot_xx.max())
        ax.set_ylim(plot_cell_height.min(), plot_cell_height.max())
        tick = [-math.pi, -0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi]
        ax.set_xticks(tick)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = "$-\\pi$"
        labels[1] = "$-\\pi/2$"
        labels[2] = "0"
        labels[3] = "$\\pi/2$"
        labels[4] = "$\\pi$"
        ax.set_xticklabels(labels)

        plt.savefig(plot_name + ".pdf", format="pdf", dpi=400)

        return

    plot_data = data
    plot_data = plot_data[mask, :]
    for k in range(plot_data.shape[1]):
        plot_data[:, k] = plot_data[argsort[::-1], k]
    # log.critical(f"DEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUGDEBUG: {xx[:, 0].min()} {xx[:, 0].max()}")
    # for i in range(xx.shape[0]):
    #     log.critical(f"   ------------    {i} {xx[i, 0]} {plot_data[i, 0]} {plot_data[i, 10]}")
    section_plot(plot_data, xx, cell_height, "seismic", title, output_filename)


def plot_triedgedata(
    grid_filename: str,
    data: xp.ndarray,
    title: str,
    output_filename: str,
    div_wave: DivWave,
    eigen_vector: np.ndarray = None,
    plot_analytic=False,
    z_ind: int = 0,
):
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    elon = np.unique(grid.elon.values)
    mid_point = int(elon.shape[0] / 2)
    mid_point_plus1 = mid_point + 1
    while True:
        if elon[mid_point_plus1] - elon[mid_point] > 1.0e-9:
            break
        else:
            mid_point_plus1 = mid_point_plus1 + 1
    mid_point_plus2 = mid_point_plus1 + 1
    while True:
        if elon[mid_point_plus2] - elon[mid_point_plus1] > 1.0e-9:
            break
        else:
            mid_point_plus2 = mid_point_plus2 + 1
    mid_point_plus3 = mid_point_plus2 + 1
    while True:
        if elon[mid_point_plus3] - elon[mid_point_plus2] > 1.0e-9:
            break
        else:
            mid_point_plus3 = mid_point_plus3 + 1
    mid_point_plus4 = mid_point_plus3 + 1
    while True:
        if elon[mid_point_plus4] - elon[mid_point_plus3] > 1.0e-9:
            break
        else:
            mid_point_plus4 = mid_point_plus4 + 1
    mid_point_plus5 = mid_point_plus4 + 1
    while True:
        if elon[mid_point_plus5] - elon[mid_point_plus4] > 1.0e-9:
            break
        else:
            mid_point_plus5 = mid_point_plus5 + 1
    mask = []
    interior_mask = (grid.elat.values >= -7.4 * math.pi / 180.0) & (
        grid.elat.values <= 7.4 * math.pi / 180.0
    )
    mask.append(
        (grid.elon.values >= elon[mid_point] - 1.0e-9)
        & (grid.elon.values < elon[mid_point_plus1] - 1.0e-9)
        & interior_mask
    )
    mask.append(
        (grid.elon.values >= elon[mid_point_plus1] - 1.0e-9)
        & (grid.elon.values < elon[mid_point_plus2] - 1.0e-9)
        & interior_mask
    )
    mask.append(
        (grid.elon.values >= elon[mid_point_plus2] - 1.0e-9)
        & (grid.elon.values < elon[mid_point_plus3] - 1.0e-9)
        & interior_mask
    )
    mask.append(
        (grid.elon.values >= elon[mid_point_plus3] - 1.0e-9)
        & (grid.elon.values < elon[mid_point_plus4] - 1.0e-9)
        & interior_mask
    )
    mask.append(
        (grid.elon.values >= elon[mid_point_plus4] - 1.0e-9)
        & (grid.elon.values < elon[mid_point_plus5] - 1.0e-9)
        & interior_mask
    )

    plt.close()
    f, ax = plt.subplots(constrained_layout=True)
    if device == Device.GPU:
        plot_data0 = data.get()
    else:
        plot_data0 = data

    font_size = 12
    x_elat = grid.elat[mask[0]].values * 180.0 / 7.5
    argsort = x_elat.argsort()
    x_elat = x_elat[argsort[::-1]]
    plot_data = plot_data0[mask[0], z_ind]
    plot_data = plot_data[argsort[::-1]]
    normal_y = grid.edge_primal_normal_cartesian_y[mask[0]].values
    normal_y = normal_y[argsort[::-1]]
    normal_x = grid.edge_primal_normal_cartesian_x[mask[0]].values
    normal_x = normal_x[argsort[::-1]]
    assert np.abs(normal_x).max() < 1.e-15, "first data is not vertically pointing"
    plot_data = np.where(normal_y < 1.0e-10, -plot_data, plot_data)
    plot_data_min = plot_data.min()
    plot_data_max = plot_data.max()
    ax.plot(x_elat, plot_data, linestyle="solid", color="red", label="$u_1$")
    if plot_analytic:
        v_scale = 180.0 / 7.5
        v_edge = (
            np.cos(v_scale * x_elat * div_wave.y_wavenumber_factor)
            / div_wave.y_wavenumber_factor
        )
        ax.plot(x_elat, v_edge, linestyle="dashed", color="red", label="analytic V 1")
        # ax.scatter(
        #     x_elat, v_edge * np.abs(normal_y), color="purple", label="analytic edge value 1"
        # )
    x_elat = grid.elat[mask[1]].values * 180.0 / 7.5
    argsort = x_elat.argsort()
    x_elat = x_elat[argsort[::-1]]
    plot_data = plot_data0[mask[1], z_ind]
    plot_data = plot_data[argsort[::-1]]
    normal_y = grid.edge_primal_normal_cartesian_y[mask[1]].values
    normal_y = normal_y[argsort[::-1]]
    normal_x = grid.edge_primal_normal_cartesian_x[mask[1]].values
    normal_x = normal_x[argsort[::-1]]
    assert np.abs(normal_x).max() > 0.01, "second data is not diagonally pointing"
    plot_data = np.where(normal_y < 1.0e-10, -plot_data, plot_data)
    plot_data_min = min(plot_data_min, plot_data.min())
    plot_data_max = max(plot_data_max, plot_data.max())
    ax.plot(x_elat, plot_data, linestyle="solid", color="blue", label="$u_2$")
    if plot_analytic:
        v_scale = 180.0 / 7.5
        v_edge = (
            np.cos(v_scale * x_elat * div_wave.y_wavenumber_factor)
            / div_wave.y_wavenumber_factor
        )
        ax.plot(x_elat, v_edge, linestyle="dashed", color="blue", label="analytic V 2")
        # ax.scatter(
        #     x_elat, v_edge * np.abs(normal_y), color="purple", label="analytic edge value 2"
        # )
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel("VN (m s-1)", fontsize=font_size)
    ax.set_xlabel("Latitude", fontsize=font_size)
    ax.set_xlim(-math.pi/2.0, math.pi/2.0)
    if not plot_analytic:
        ax.set_ylim(plot_data_min*1.5, plot_data_max*1.5)
    tick = [-0.5 * math.pi, 0, 0.5 * math.pi]
    ax.set_xticks(tick)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = "$-\\pi/2$"
    labels[1] = "0"
    labels[2] = "$\\pi/2$"
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.legend(fontsize=font_size)
    plt.savefig(output_filename+".pdf", format="pdf", dpi=500)
    plt.clf()

    plt.close()
    for number, item in enumerate(mask):
        f, ax = plt.subplots(constrained_layout=True)
        x_elat = grid.elat[item].values * 180.0 / 7.5
        argsort = x_elat.argsort()
        x_elat = x_elat[argsort[::-1]]
        if device == Device.GPU:
            plot_data = data.get()
        else:
            plot_data = data
        plot_data = plot_data[item, z_ind]
        plot_data = plot_data[argsort[::-1]]
        normal_y = grid.edge_primal_normal_cartesian_y[item].values
        normal_y = normal_y[argsort[::-1]]
        plot_data = np.where(normal_y < 1.0e-10, -plot_data, plot_data)
        ax.plot(x_elat, plot_data, linestyle="solid", color="black", label="numerical")
        if plot_analytic:
            v_scale = 180.0 / 7.5
            v_edge = (
                np.cos(v_scale * x_elat * div_wave.y_wavenumber_factor)
                / div_wave.y_wavenumber_factor
            )
            ax.plot(x_elat, v_edge, linestyle="solid", color="green", label="analytic V")
            ax.scatter(
                x_elat, v_edge * np.abs(normal_y), color="purple", label="analytic edge value"
            )
        if eigen_vector is not None and number == 0:
            x_elat_next = grid.elat[mask[number + 1]].values
            argsort_next = x_elat_next.argsort()
            x_elat_next = x_elat_next[argsort_next[::-1]]
            plot_data_next = data[mask[number + 1], z_ind]
            plot_data_next = plot_data_next[argsort_next[::-1]]
            log.info(f"Number of layers {plot_data_next.shape[0]} ----- {plot_data.shape[0]}")
            for k in range(np.minimum(plot_data_next.shape[0], plot_data.shape[0])):
                log.info(f"{k} == {plot_data_next[k]} ----- {plot_data[k]}")
        ax.set_ylabel("VN (m s-1)")
        ax.set_xlabel("Latitude")
        ax.set_xlim(-math.pi, math.pi)
        tick = [-math.pi, -0.5 * math.pi, 0, 0.5 * math.pi, math.pi]
        ax.set_xticks(tick)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = "$-\\pi$"
        labels[1] = "$-\\pi/2$"
        labels[2] = "0"
        labels[3] = "$\\pi/2$"
        labels[4] = "$\\pi$"
        ax.set_xticklabels(labels)
        ax.set_title(title)
        plt.legend()
        plt.savefig(output_filename + str(number) + ".pdf", format="pdf", dpi=500)
        plt.clf()
    return


def create_globe_mask(grid_filename: str) -> np.ndarray:
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    mask = (grid.clat.values > np.deg2rad(-85)) & (grid.clat.values < np.deg2rad(85))
    voc = grid.vertex_of_cell.T.values - 1
    v_lon = grid.vlon.values  # pi unit
    v_lat = grid.vlat.values  # pi unit
    sphere_radius = 6371229.0
    # vx = sphere_radius * np.cos(v_lat) * np.cos(v_lon)
    # vy = sphere_radius * np.cos(v_lat) * np.cos(v_lat)
    # vz = sphere_radius * np.sin(v_lat)
    vlon_c = v_lon[voc]
    vlon_c1 = np.roll(vlon_c, shift=1, axis=1)
    vlat_c = v_lat[voc]
    vlat_c1 = np.roll(vlat_c, shift=1, axis=1)
    # vv_distance = np.sum(np.abs(vx_c - vx_c1)**2, axis=1) + np.sum(np.abs(vy_c - vy_c1)**2, axis=1) + np.sum(np.abs(vz_c - vz_c1)**2, axis=1)
    # vv_distance = np.sum((vx_c - vx_c1)**2 + (vy_c - vy_c1)**2 + (vz_c - vz_c1)**2, axis=1)
    vv_distance = np.sum(np.abs(vlon_c - vlon_c1) + np.abs(vlat_c - vlat_c1), axis=1)
    # mean_edge_length = np.sum(vv_distance)/float(vv_distance.shape[0])
    mean_edge_length = math.pi
    neighbor_cell = grid.neighbor_cell_index.T.values - 1
    threshold = 1.0
    interior_mask_ = vv_distance < threshold * mean_edge_length
    interior_mask = np.array(interior_mask_, copy=True)
    interior_mask = np.where(np.sum(interior_mask[neighbor_cell], axis=1) < 3, False, interior_mask)
    mask = mask & interior_mask

    return mask


def plot_globetridata(
    grid_filename: str, data: xp.ndarray, mask: np.ndarray, title: str, output_filename: str
):
    grid = xr.open_dataset(grid_filename, engine="netcdf4")

    voc = grid.vertex_of_cell.T[mask].values - 1
    if len(data.shape) == 1:
        number_of_layers = 1
    else:
        number_of_layers = data.shape[1]
    log.info(f"plotting {title} with levels {number_of_layers}, {len(data.shape)}, {data.shape}")

    if device == Device.GPU:
        cell_data0 = data.get()
    else:
        cell_data0 = data
    for k in range(number_of_layers):
        if len(cell_data0.shape) == 1:
            cell_data = cell_data0[mask]
        else:
            cell_data = cell_data0[mask, k]
        data_max, data_min = cell_data.max(), cell_data.min()
        log.info(f"data max min: {data_max}, {data_min}")

        used_vertices = np.unique(voc)
        lat_min = grid.vlat[used_vertices].min().values
        lat_max = grid.vlat[used_vertices].max().values
        lon_min = grid.vlon[used_vertices].min().values
        lon_max = grid.vlon[used_vertices].max().values

        # N = 20
        # cmap = plt.get_cmap('jet', N)
        # norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        # creating ScalarMappable
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])

        cmap = plt.get_cmap("seismic")
        norm = TwoSlopeNorm(vmin=data_min - 1.0e-8, vcenter=0.0, vmax=data_max + 1.0e-8)

        plt.tripcolor(grid.vlon, grid.vlat, voc, cell_data, cmap=cmap, norm=norm)
        plt.title(title)
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)

        plt.colorbar()  # sm, ticks=np.linspace(0, 2, N)

        plt.savefig(output_filename + "_at_" + str(k) + "_level.pdf", format="pdf", dpi=500)
        plt.clf()

    return


def plot_error(
    error: np.ndarray,
    dx: np.ndarray,
    axis_labels: list,
    order: int,
):
    if not isinstance(error, np.ndarray):
        raise TypeError(f"error is not a numpy array, instead it is {type(error)}")
    if len(error.shape) != 1:
        raise ValueError(f"error is not one dimensional, instead it is {error.shape}")

    assert error.shape[0] == len(axis_labels)

    max_xaxis_value = error.shape[0]

    plt.close()

    f, ax = plt.subplots()

    x = np.arange(max_xaxis_value, dtype=float)
    x = x + 1.0
    first_order = 2.0 * error[0] * dx / dx[0]
    second_order = 0.5 * error[0] * dx**2 / dx[0] ** 2
    ax.plot(dx, error, label="error")
    ax.plot(dx, first_order, label="O(dx)")
    ax.plot(dx, second_order, label="O(dx^2)")
    # ax.set_xlim(0.0, max_xaxis_value + 0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = ""
    # for i in range(max_xaxis_value):
    #     labels[i+1] = axis_labels[i]
    # ax.set_xticks(dx, axis_labels)
    # ax.set_xticklabels(axis_labels)
    ax.set_ylabel("l1 error")
    ax.set_xlabel("edge length (m)")
    ax.set_title("Mean error in divergence")
    plt.legend()
    plt.savefig("plot_" + str(order) + "div_error.pdf", format="pdf", dpi=500)


def print_config(config):
    config_vars = vars(config)
    for item in config_vars:
        log.critical(f"{config.__class__}: {item} {config_vars[item]}")

