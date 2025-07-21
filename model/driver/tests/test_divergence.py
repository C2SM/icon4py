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
from cmath import sqrt as c_sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm

from icon4py.model.common.config import Device
from icon4py.model.common.decomposition.definitions import (
    get_processor_properties,
    get_runtype,
)
from icon4py.model.common.settings import device, xp
from icon4py.model.driver.dycore_driver import initialize
from icon4py.model.driver.initialization_utils import (
    WAVETYPE,
    DivWave,
    ExperimentType,
    SerializationType,
    configure_logging,
)


log = logging.getLogger(__name__)


def determine_divergence_in_div_converge_experiment(
    lat: xp.ndarray, lon: xp.ndarray, div_wave: DivWave
):
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
    else:
        raise NotImplementedError(f"Wave type {div_wave.wave_type} not implemented")
    return analytic_divergence


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
    triangle_vector[1] = np.array([0.5 * math.sqrt(3.0), 0.5], dtype=float)
    triangle_vector[2] = np.array([-0.5 * math.sqrt(3.0), 0.5], dtype=float)
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

    for k in range(number_of_layers):
        if len(data.shape) == 1:
            cell_data = np.asarray(data[mask])
        else:
            cell_data = np.asarray(data[mask, k])
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


def plot_triedgedata(
    grid_filename: str,
    data: xp.ndarray,
    title: str,
    output_filename: str,
    div_wave: DivWave,
    eigen_vector: np.ndarray = None,
    plot_analytic=False,
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

    for number, item in enumerate(mask):
        f, ax = plt.subplots()
        x_elat = grid.elat[item].values
        argsort = x_elat.argsort()
        x_elat = x_elat[argsort[::-1]]
        plot_data = data[item, 0]
        plot_data = plot_data[argsort[::-1]]
        normal_y = grid.edge_primal_normal_cartesian_y[item].values
        normal_y = normal_y[argsort[::-1]]
        plot_data = np.where(normal_y < 1.0e-10, -plot_data, plot_data)
        ax.plot(x_elat, plot_data, linestyle="solid", color="black", label="numerical")
        if plot_analytic:
            v_scale = 180.0 / 7.5
            v_edge = (
                xp.cos(v_scale * x_elat * div_wave.y_wavenumber_factor)
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
            plot_data_next = data[mask[number + 1], 0]
            plot_data_next = plot_data_next[argsort_next[::-1]]
            log.info(f"Number of layers {plot_data_next.shape[0]} ----- {plot_data.shape[0]}")
            for k in range(np.minimum(plot_data_next.shape[0], plot_data.shape[0])):
                log.info(f"{k} == {plot_data_next[k]} ----- {plot_data[k]}")
        ax.set_ylabel("VN (m s-1)")
        ax.set_xlabel("Latitude")
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

    for k in range(number_of_layers):
        if len(data.shape) == 1:
            cell_data = np.asarray(data[mask])
        else:
            cell_data = np.asarray(data[mask, k])
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


def test_divergence_single_time_step():
    resolutions = (
        "1000m",
        "500m",
        # '250m',
        # '125m',
    )
    dx = np.array(
        (
            1000.0,
            500.0,
            250.0,
            125.0,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    # div_wave = DivWave(wave_type=WAVETYPE.SPHERICAL_HARMONICS, x_wavenumber_factor=1.0, y_wavenumber_factor=1.0)
    div_wave = DivWave(wave_type=WAVETYPE.Y_WAVE, y_wavenumber_factor=50.0)
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    print_config(div_wave)

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=dtime_seconds)
    mean_error1 = np.zeros(len(resolutions), dtype=float)
    mean_error2 = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + res + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        (
            timeloop,
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            output_state,
            prep_adv,
            inital_divdamp_fac_o2,
        ) = initialize(
            Path(input_path),
            parallel_props,
            serialization_type,
            experiment_type,
            grid_root,
            grid_level,
            enable_output,
            enable_debug_message,
            dtime_seconds,
            end_date,
            output_interval,
            div_wave,
        )

        mask = create_mask(grid_filename)
        plot_tridata(
            grid_filename,
            diagnostic_state.u.asnumpy(),
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        plot_tridata(
            grid_filename,
            diagnostic_state.v.asnumpy(),
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

        timeloop.time_integration(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv=False,
            output_state=output_state,
            profile=profile,
        )

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        # analytic_divergence = -0.5 / xp.sqrt(2.0 * math.pi) * (xp.sqrt(105.0) * xp.sin(2.0 * cell_lon) * xp.cos(cell_lat * v_scale) * xp.cos(cell_lat * v_scale) * xp.sin(cell_lat * v_scale) + xp.sqrt(15.0) * xp.cos(cell_lon) * xp.cos(2.0 * cell_lat * v_scale))
        analytic_divergence = determine_divergence_in_div_converge_experiment(
            cell_lat, cell_lon, div_wave
        )

        mean_error1[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)
        mean_error2[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error1 = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.ndarray[
                :, 0
            ]
            - analytic_divergence
        )
        divergence_error2 = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.ndarray[
                :, 0
            ]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error1 = divergence_error1.get()
            divergence_error2 = divergence_error2.get()

        plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv1_vn.asnumpy(),
            mask,
            f"Computed 1st order divergence at {res}",
            "plot_computed_1st_order_divergence_" + res,
        )
        plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv2_vn.asnumpy(),
            mask,
            f"Computed 2nd order divergence at {res}",
            "plot_computed_2nd_order_divergence_" + res,
        )
        plot_tridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        plot_tridata(
            grid_filename,
            divergence_error1,
            mask,
            f"First order divergence error at DX = {res}",
            "plot_error_1st_order_divergence_" + res,
        )
        plot_tridata(
            grid_filename,
            divergence_error2,
            mask,
            f"Second order divergence error at DX = {res}",
            "plot_error_2nd_order_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error1[i]} {mean_error2[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error1, dx[0 : len(resolutions)], resolutions, 1)
    plot_error(mean_error2, dx[0 : len(resolutions)], resolutions, 2)


def test_divergence_multiple_time_step():
    resolutions = (
        "1000m",
        # "500m",
        # '250m',
        # '125m',
    )
    dx = np.array(
        (
            1000.0,
            500.0,
            250.0,
            125.0,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/div_converge_res"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.DIVCONVERGE
    serialization_type = SerializationType.SB
    div_wave = DivWave(wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=50.0, y_wavenumber_factor=0.0)
    log.critical(f"Experiment: {ExperimentType.DIVCONVERGE}")
    print_config(div_wave)

    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = 1000 * dtime_seconds
    initial_date = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_time = 1000 * dtime_seconds
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + "Torus_Triangles_100km_x_100km_res" + res + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        end_date = initial_date + datetime.timedelta(seconds=end_time)

        (
            timeloop,
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            output_state,
            prep_adv,
            inital_divdamp_fac_o2,
        ) = initialize(
            Path(input_path),
            parallel_props,
            serialization_type,
            experiment_type,
            grid_root,
            grid_level,
            enable_output,
            enable_debug_message,
            dtime_seconds,
            end_date,
            output_interval,
            div_wave,
        )

        eigen_value, eigen_vector, eigen_vector_cartesian = eigen_divergence(
            divergence_factor=0.0,
            order=timeloop.solve_nonhydro.config.divergence_order,
            grid_space=1000.0,
            div_wave=div_wave,
        )
        damping_eigen_vector = eigen_vector[np.argmax(np.abs(eigen_value))]

        mask = create_mask(grid_filename)

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

        plot_triedgedata(
            grid_filename,
            initial_vn,
            f"VN at {res}",
            "plot_initial_vn" + res,
            div_wave,
            eigen_vector=damping_eigen_vector,
            plot_analytic=True,
        )

        timeloop.time_integration(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv=False,
            output_state=output_state,
            profile=profile,
        )

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        v_scale = 90.0 / 7.5
        sphere_radius = 6371229.0
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        analytic_divergence = determine_divergence_in_div_converge_experiment(
            cell_lat, cell_lon, div_wave
        )

        diff1_divergence = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.ndarray[:, 0]
            - analytic_divergence
        )
        diff2_divergence = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.ndarray[:, 0]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            diff1_divergence = diff1_divergence.get()
            diff2_divergence = diff2_divergence.get()

        plot_tridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        plot_triedgedata(
            grid_filename,
            prognostic_state_list[0].vn.asnumpy(),
            f"VN at {res}",
            "plot_final_vn" + res,
            div_wave,
            eigen_vector=damping_eigen_vector,
            plot_analytic=False,
        )
        plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.asnumpy(),
            mask,
            f"Computed divergence order 1 at {res}",
            "plot_computed_divergence1_" + res,
        )
        plot_tridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.asnumpy(),
            mask,
            f"Computed divergence order 2 at {res}",
            "plot_computed_divergence2_" + res,
        )
        plot_tridata(
            grid_filename,
            diff1_divergence,
            mask,
            f"Difference in divergence with order 1 at {res}",
            "plot_diff1_divergence_" + res,
        )
        plot_tridata(
            grid_filename,
            diff2_divergence,
            mask,
            f"Difference in divergence with order 2 at {res}",
            "plot_diff2_divergence_" + res,
        )

        log.info(
            f"Sum of divergence1: {xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv1_vn.asnumpy()[:,0]))}"
        )
        log.info(
            f"Sum of divergence2: {xp.sum(xp.abs(timeloop.solve_nonhydro.output_intermediate_fields.output_after_flxdiv2_vn.asnumpy()[:,0]))}"
        )
        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

        end_time = end_time * 2


def test_divergence_single_time_step_on_globe():
    resolutions = (
        "r2b4",
        "r2b5",
        "r2b6",
        # 'r2b7',
    )
    grid_name = (
        "icon_grid_0010_R02B04_G",
        "icon_grid_0008_R02B05_G",
        "icon_grid_0002_R02B06_G",
        "icon_grid_0004_R02B07_G",
    )
    dx = np.array(
        (
            220,
            110,
            55,
            27.5,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/flat_earth_"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.GLOBEDIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=dtime_seconds)
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + grid_name[i] + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        (
            timeloop,
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            output_state,
            prep_adv,
            inital_divdamp_fac_o2,
        ) = initialize(
            Path(input_path),
            parallel_props,
            serialization_type,
            experiment_type,
            grid_root,
            grid_level,
            enable_output,
            enable_debug_message,
            dtime_seconds,
            end_date,
            output_interval,
        )

        mask = create_globe_mask(grid_filename)
        plot_globetridata(
            grid_filename,
            diagnostic_state.u.asnumpy(),
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        plot_globetridata(
            grid_filename,
            diagnostic_state.v.asnumpy(),
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        z_ifc = diagnostic_metric_state.z_ifc.asnumpy()

        log.info(
            f"Surface max min:  {z_ifc[:,timeloop.grid.num_levels].max()}, {z_ifc[:,timeloop.grid.num_levels].min()}"
        )
        log.info(
            f"First level max min:  {z_ifc[:,timeloop.grid.num_levels-1].max()}, {z_ifc[:,timeloop.grid.num_levels-1].min()}"
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

        timeloop.time_integration(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv=False,
            output_state=output_state,
            profile=profile,
        )

        os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        sphere_radius = 6371229.0
        theta_shift = 0.5 * math.pi
        lon_shift = math.pi
        cell_lon = cell_lon + lon_shift
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        # u_cell = u_factor * xp.cos(2.0 * cell_lon) * xp.cos(theta_shift - cell_lat) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        # v_cell = v_factor * xp.cos(cell_lon) * xp.cos(theta_shift - cell_lat) * xp.sin(theta_shift - cell_lat)
        analytic_divergence = (
            -u_factor * xp.sin(2.0 * cell_lon) * xp.sin(2.0 * (theta_shift - cell_lat))
            + 0.25
            * v_factor
            * xp.cos(cell_lon)
            * (3.0 * xp.sin(3.0 * (theta_shift - cell_lat)) / xp.sin(theta_shift - cell_lat) - 1.0)
        ) / sphere_radius

        mean_error[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:, 0]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        plot_globetridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.asnumpy(),
            mask,
            f"Computed divergence at {res}",
            "plot_computed_divergence_" + res,
        )
        plot_globetridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        plot_globetridata(
            grid_filename,
            divergence_error,
            mask,
            f"Divergence error at {res}",
            "plot_error_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        os.system(f"mv plot_* dummy* data_output* data_{res}")

    plot_error(mean_error, dx[0 : len(resolutions)], resolutions)


def test_divergence_multiple_time_step_on_globe():
    resolutions = (
        "r2b4",
        # 'r2b5',
        # 'r2b6',
        # 'r2b7',
    )
    grid_name = (
        "icon_grid_0010_R02B04_G",
        "icon_grid_0008_R02B05_G",
        "icon_grid_0002_R02B06_G",
        "icon_grid_0004_R02B07_G",
    )
    dx = np.array(
        (
            220,
            110,
            55,
            27.5,
        ),
        dtype=float,
    )

    base_input_path = "/scratch/mch/cong/data/flat_earth_"
    grid_file_folder = "/scratch/mch/cong/grid-generator/grids/"
    run_path = "./"
    experiment_type = ExperimentType.GLOBEDIVCONVERGE
    serialization_type = SerializationType.SB
    disable_logging = False
    mpi = False
    profile = False
    grid_root = 0
    grid_level = 2
    enable_output = True
    enable_debug_message = False

    dtime_seconds = 1.0
    output_interval = 100 * dtime_seconds
    end_date = datetime.datetime(1, 1, 1, 0, 0, 0) + datetime.timedelta(
        seconds=1000 * dtime_seconds
    )
    mean_error = np.zeros(len(resolutions), dtype=float)
    for i, res in enumerate(resolutions):
        input_path = base_input_path + res + "/ser_data/"
        grid_filename = grid_file_folder + grid_name[i] + ".nc"

        parallel_props = get_processor_properties(get_runtype(with_mpi=mpi))
        configure_logging(run_path, experiment_type, parallel_props, disable_logging)

        (
            timeloop,
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            output_state,
            prep_adv,
            inital_divdamp_fac_o2,
        ) = initialize(
            Path(input_path),
            parallel_props,
            serialization_type,
            experiment_type,
            grid_root,
            grid_level,
            enable_output,
            enable_debug_message,
            dtime_seconds,
            end_date,
            output_interval,
        )

        mask = create_globe_mask(grid_filename)
        plot_globetridata(
            grid_filename,
            diagnostic_state.u.asnumpy(),
            mask,
            f"Initial u at {res}",
            "plot_initial_u_" + res,
        )
        plot_globetridata(
            grid_filename,
            diagnostic_state.v.asnumpy(),
            mask,
            f"Initial v at {res}",
            "plot_initial_v_" + res,
        )

        initial_vn = xp.array(prognostic_state_list[0].vn.ndarray, copy=True)

        timeloop.time_integration(
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            diagnostic_metric_state,
            diagnostic_state,
            prognostic_state_list,
            prep_adv,
            inital_divdamp_fac_o2,
            do_prep_adv=False,
            output_state=output_state,
            profile=profile,
        )

        # os.system(f"mkdir -p data_{res}")

        cell_lat = timeloop.solve_nonhydro.cell_params.cell_center_lat.ndarray
        cell_lon = timeloop.solve_nonhydro.cell_params.cell_center_lon.ndarray
        cell_area = timeloop.solve_nonhydro.cell_params.area.ndarray

        theta_shift = 0.5 * math.pi
        u_factor = 0.25 * xp.sqrt(0.5 * 105.0 / math.pi)
        v_factor = -0.5 * xp.sqrt(0.5 * 15.0 / math.pi)
        u_cell = (
            u_factor
            * xp.cos(2.0 * cell_lon)
            * xp.cos(theta_shift - cell_lat)
            * xp.cos(theta_shift - cell_lat)
            * xp.sin(theta_shift - cell_lat)
        )
        v_cell = (
            v_factor
            * xp.cos(cell_lon)
            * xp.cos(theta_shift - cell_lat)
            * xp.sin(theta_shift - cell_lat)
        )
        analytic_divergence = (
            -0.5
            / xp.sqrt(2.0 * math.pi)
            * (
                1.0e-5
                * 2.0
                * math.pi
                * xp.sqrt(105.0)
                * xp.sin(2.0 * cell_lon)
                * xp.cos(theta_shift - cell_lat)
                * xp.cos(theta_shift - cell_lat)
                * xp.sin(theta_shift - cell_lat)
                + 1.0e-5
                * math.pi
                * xp.sqrt(15.0)
                * xp.cos(cell_lon)
                * xp.cos(2.0 * (theta_shift - cell_lat))
            )
        )

        mean_error[i] = float(
            xp.sum(
                xp.abs(
                    timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[
                        :, 0
                    ]
                    - analytic_divergence
                )
                * cell_area
            )
        )  # / xp.sum(cell_area)

        divergence_error = (
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.ndarray[:, 0]
            - analytic_divergence
        )

        if device == Device.GPU:
            analytic_divergence = analytic_divergence.get()
            divergence_error = divergence_error.get()

        plot_globetridata(
            grid_filename,
            timeloop.solve_nonhydro.output_intermediate_fields.output_before_flxdiv_vn.asnumpy(),
            mask,
            f"Computed divergence at {res}",
            "plot_computed_divergence_" + res,
        )
        plot_globetridata(
            grid_filename,
            analytic_divergence,
            mask,
            f"Analytic divergence at {res}",
            "plot_analytic_divergence_" + res,
        )
        plot_globetridata(
            grid_filename,
            divergence_error,
            mask,
            f"Difference in divergence at {res}",
            "plot_diff_divergence_" + res,
        )

        log.info(f"Mean error at resolution of {res} : {mean_error[i]}")
        diff_vn = prognostic_state_list[0].vn.ndarray - initial_vn
        log.info(f"Max diff vn at resolution of {res} : {xp.abs(diff_vn).max()}")

        # os.system(f"mv plot_* dummy* data_output* data_{res}")


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


def test_eigen_divergence():
    for x in range(1):
        for y in range(10):
            print("WAVE NUMBERS (X Y):", x, y)
            div_wave = DivWave(
                wave_type=WAVETYPE.Y_WAVE,
                x_wavenumber_factor=float(x),
                y_wavenumber_factor=float(y),
            )
            eigen_value, eigen_vector, eigen_vector_cartesian = eigen_divergence(
                divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave, debug=True
            )
            eigen_value, eigen_vector, eigen_vector_cartesian = eigen_divergence(
                divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave, debug=True
            )
            print()
            print()
    # eigen_vorticity(divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave)


def plot_eigen_divergence():
    x_grid = np.arange(0.1, 99.9, 0.1)
    y_grid = np.arange(0.1, 99.9, 0.1)
    x_wave = 2.0 * math.pi / 10000.0 * x_grid * 100.0
    y_wave = 2.0 * math.pi / 10000.0 * y_grid * 100.0
    xy_xwave, xy_ywave = np.meshgrid(x_wave, y_wave, indexing="ij")

    xdamp1_max = np.zeros(x_grid.shape, dtype=float)
    xdamp1_actual = np.zeros(x_grid.shape, dtype=float)
    xdamp2 = np.zeros(x_grid.shape, dtype=float)
    ydamp1_max = np.zeros(y_grid.shape, dtype=float)
    ydamp1_actual = np.zeros(y_grid.shape, dtype=float)
    ydamp2 = np.zeros(y_grid.shape, dtype=float)
    xydamp1_max = np.zeros(xy_xwave.shape, dtype=float)
    xydamp1_actual = np.zeros(xy_xwave.shape, dtype=float)
    xydamp2 = np.zeros(xy_xwave.shape, dtype=float)

    for i, x_wavenumber in enumerate(x_grid):
        div_wave = DivWave(
            wave_type=WAVETYPE.X_WAVE, x_wavenumber_factor=x_wavenumber, y_wavenumber_factor=0.0
        )
        eigen_value1, eigen_vector1, eigen_vector_cartesian1 = eigen_divergence(
            divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
        )
        eigen_value2, eigen_vector2, eigen_vector_cartesian2 = eigen_divergence(
            divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
        )
        xdamp1_max[i] = eigen_value1[np.argmax(np.abs(eigen_value1))]
        eigen_value1 = np.sort(eigen_value1)
        xdamp1_actual[i] = eigen_value1[1]
        xdamp2[i] = eigen_value2[np.argmax(np.abs(eigen_value2))]
        # is_found1, is_found2 = False, False
        # threshold = 1.e-6
        # for k in range(3):
        #     if np.abs(np.abs(eigen_vector_cartesian1[k, 0]) - 1.0) < threshold:
        #         assert np.abs(np.abs(eigen_vector_cartesian1[k, 1]) - 0.0) < threshold, f"{x_wavenumber} -- {eigen_vector_cartesian1[k, 1]}"
        #         assert is_found1 == True, f"Two overlapped eigen vectors 1 for x!!"
        #         xdamp1[i] = eigen_value1[k]
        #         is_found1 = True
        #     if np.abs(np.abs(eigen_vector_cartesian2[k, 0]) - 1.0) < threshold:
        #         assert np.abs(np.abs(eigen_vector_cartesian2[k, 1]) - 0.0) < threshold, f"{x_wavenumber} -- {eigen_vector_cartesian2[k, 1]}"
        #         assert is_found2 == True, f"Two overlapped eigen vectors 2 for x!!"
        #         xdamp2[i] = eigen_value2[k]
        #         is_found2 = True
    is_found1, is_found2 = False, False
    for i, y_wavenumber in enumerate(y_grid):
        div_wave = DivWave(
            wave_type=WAVETYPE.Y_WAVE, x_wavenumber_factor=0.0, y_wavenumber_factor=y_wavenumber
        )
        eigen_value1, eigen_vector1, eigen_vector_cartesian1 = eigen_divergence(
            divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
        )
        eigen_value2, eigen_vector2, eigen_vector_cartesian2 = eigen_divergence(
            divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
        )
        ydamp1_max[i] = eigen_value1[np.argmax(np.abs(eigen_value1))]
        eigen_value1 = np.sort(eigen_value1)
        ydamp1_actual[i] = eigen_value1[1]
        ydamp2[i] = eigen_value2[np.argmax(np.abs(eigen_value2))]
    #     is_found1, is_found2 = False, False
    #     for k in range(3):
    #         if np.abs(np.abs(eigen_vector_cartesian1[k, 1]) - 1.0) < threshold:
    #             assert np.abs(np.abs(eigen_vector_cartesian1[k, 0]) - 0.0) < threshold, f"{y_wavenumber} -- {eigen_vector_cartesian1[k, 0]}"
    #             assert is_found1 == True, f"Two overlapped eigen vectors 1 for y!!"
    #             ydamp1[i] = eigen_value1[k]
    #             is_found1 = True
    #         if np.abs(np.abs(eigen_vector_cartesian2[k, 1]) - 1.0) < threshold:
    #             assert np.abs(np.abs(eigen_vector_cartesian2[k, 0]) - 0.0) < threshold, f"{y_wavenumber} -- {eigen_vector_cartesian2[k, 0]}"
    #             assert is_found2 == True, f"Two overlapped eigen vectors 2 for y!!"
    #             ydamp2[i] = eigen_value2[k]
    #             is_found2 = True

    plt.close()

    f, ax = plt.subplots()

    ax.plot(x_wave, xdamp1_max, label="order 1 (eigen-1)")
    ax.plot(x_wave, xdamp1_actual, label="order 1 (eigen-2)")
    ax.plot(x_wave, xdamp2, label="order 2")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylabel("eigenvalue")
    ax.set_xlabel("wavenumber")
    xtick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, math.pi * 2.0]
    ax.set_xticks(xtick)
    # ax.set_title("Mean error in divergence")
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_xticklabels(labels)
    plt.legend()
    plt.savefig("plot_xdamp.pdf", format="pdf", dpi=500)

    plt.clf()

    f, ax = plt.subplots()

    ax.plot(y_wave, ydamp1_max, label="order 1 (eigen-1)")
    ax.plot(y_wave, ydamp1_actual, label="order 1 (eigen-2)")
    ax.plot(y_wave, ydamp2, label="order 2")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylabel("eigenvalue")
    ax.set_xlabel("wavenumber")
    xtick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, math.pi * 2.0]
    ax.set_xticks(xtick)
    # ax.set_title("Mean error in divergence")
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_xticklabels(labels)
    plt.legend()
    plt.savefig("plot_ydamp.pdf", format="pdf", dpi=500)

    for i, x_wavenumber in enumerate(x_grid):
        for j, y_wavenumber in enumerate(y_grid):
            div_wave = DivWave(
                wave_type=WAVETYPE.X_AND_Y_WAVE,
                x_wavenumber_factor=x_wavenumber,
                y_wavenumber_factor=y_wavenumber,
            )
            eigen_value1, eigen_vector1, eigen_vector_cartesian1 = eigen_divergence(
                divergence_factor=0.0, order=1, grid_space=1000.0, div_wave=div_wave
            )
            eigen_value2, eigen_vector2, eigen_vector_cartesian2 = eigen_divergence(
                divergence_factor=0.0, order=2, grid_space=1000.0, div_wave=div_wave
            )
            xydamp1_max[i, j] = eigen_value1[np.argmax(np.abs(eigen_value1))]
            eigen_value1 = np.sort(eigen_value1)
            xydamp1_actual[i, j] = eigen_value1[1]
            xydamp2[i, j] = eigen_value2[np.argmax(np.abs(eigen_value2))]

    plt.clf()
    f, ax = plt.subplots()
    # cmap = plt.get_cmap('plasma')
    cmap = plt.get_cmap("gist_rainbow")
    cmap = cmap.reversed()
    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax.contourf(xy_xwave, xy_ywave, xydamp1_max, cmap=cmap, levels=boundaries, norm=lnorm)
    cb1 = f.colorbar(cp, location="right")

    ax.set_xlabel("x wavenumber")
    ax.set_ylabel("y wavenumber")
    tick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, math.pi * 2.0]
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_yticklabels(labels)
    plt.savefig("fig_xydamp1_max.pdf", format="pdf", dpi=400)

    plt.clf()
    f, ax = plt.subplots()
    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax.contourf(xy_xwave, xy_ywave, xydamp1_actual, cmap=cmap, levels=boundaries, norm=lnorm)
    cb1 = f.colorbar(cp, location="right")

    ax.set_xlabel("x wavenumber")
    ax.set_ylabel("y wavenumber")
    tick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, math.pi * 2.0]
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_yticklabels(labels)
    plt.savefig("fig_xydamp1_actual.pdf", format="pdf", dpi=400)

    plt.clf()
    f, ax = plt.subplots()
    boundaries = np.linspace(-5.0, 0.0, 101)
    lnorm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    cp = ax.contourf(xy_xwave, xy_ywave, xydamp2, cmap=cmap, levels=boundaries, norm=lnorm)
    cb1 = f.colorbar(cp, location="right")

    ax.set_xlabel("x wavenumber")
    ax.set_ylabel("y wavenumber")
    tick = [0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0, math.pi * 2.0]
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1] = "$\\pi/2$"
    labels[2] = "$\\pi$"
    labels[3] = "$3\\pi$/2"
    labels[4] = "$2\\pi$"
    ax.set_yticklabels(labels)
    plt.savefig("fig_xydamp2.pdf", format="pdf", dpi=400)

    print("MEAN VALUE OF DIV1: ", np.mean(xydamp1_max), np.mean(xydamp1_actual))
    print("MEAN VALUE OF DIV2: ", np.mean(xydamp2))


if __name__ == "__main__":
    # test_divergence_single_time_step()
    test_divergence_multiple_time_step()

    # test_divergence_single_time_step_on_globe()
    # test_divergence_multiple_time_step_on_globe()

    # test_eigen_divergence()
    # plot_eigen_divergence()
