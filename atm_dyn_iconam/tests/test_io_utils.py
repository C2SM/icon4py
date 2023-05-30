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

import pathlib

import pytest

from icon4py.diffusion.horizontal import CellParams, EdgeParams
from icon4py.diffusion.icon_grid import VerticalModelParams
from icon4py.driver.io_utils import (
    SerializationType,
    read_geometry_fields,
    read_icon_grid,
    read_static_fields,
)
from icon4py.driver.parallel_setup import get_processor_properties


test_data_path = pathlib.Path(__file__).parent.joinpath(
    "./ser_icondata/mch_ch_r04b09_dsl/ser_data"
)


@pytest.mark.parametrize(
    "read_fun", (read_geometry_fields, read_static_fields, read_icon_grid)
)
def test_read_geometry_fields_not_implemented_type(read_fun):
    with pytest.raises(NotImplementedError):
        read_fun(path=test_data_path, ser_type=SerializationType.NC)


def assert_grid_size_and_connectivities(grid):
    assert grid.num_edges() > 0
    assert grid.num_cells() > 0
    assert grid.num_vertices() > 0
    assert grid.get_e2v_connectivity()
    assert grid.get_v2e_connectivity()
    assert grid.get_c2e_connectivity()
    assert grid.get_e2c_connectivity()
    assert grid.get_e2c2v_connectivity()
    assert grid.get_c2e2c_connectivity()
    assert grid.get_e2ecv_connectivity()


@pytest.mark.datatest
def test_read_icon_grid_for_type_sb():
    grid = read_icon_grid(test_data_path, ser_type=SerializationType.SB)
    assert_grid_size_and_connectivities(grid)


@pytest.mark.datatest
def test_read_static_fields_for_type_sb():
    metric_state, interpolation_state = read_static_fields(
        test_data_path, ser_type=SerializationType.SB
    )
    assert_metric_state_fields(metric_state)
    assert_interpolation_state_fields(interpolation_state)


@pytest.mark.datatest
def test_read_geometry_fields_for_type_sb():
    edge_geometry, cell_geometry, vertical_geometry = read_geometry_fields(
        test_data_path, ser_type=SerializationType.SB
    )
    assert_edge_geometry_fields(edge_geometry)
    assert_cell_geometry_fields(cell_geometry)
    assert_vertical_params(vertical_geometry)


def assert_vertical_params(vertical_geometry: VerticalModelParams):
    assert vertical_geometry.physical_heights
    assert vertical_geometry.index_of_damping_layer > 0
    assert vertical_geometry.rayleigh_damping_height > 0


def assert_cell_geometry_fields(cell_geometry: CellParams):
    assert cell_geometry.area


def assert_edge_geometry_fields(edge_geometry: EdgeParams):
    assert edge_geometry.edge_areas
    assert edge_geometry.primal_normal_vert
    assert edge_geometry.inverse_primal_edge_lengths
    assert edge_geometry.tangent_orientation
    assert edge_geometry.inverse_dual_edge_lengths
    assert edge_geometry.dual_normal_vert


def assert_metric_state_fields(metric_state):
    assert metric_state.wgtfac_c
    assert metric_state.zd_intcoef
    assert metric_state.zd_diffcoef
    assert metric_state.theta_ref_mc
    assert metric_state.mask_hdiff
    assert metric_state.zd_vertidx


def assert_interpolation_state_fields(interpolation_state):
    assert interpolation_state.geofac_n2s
    assert interpolation_state.e_bln_c_s
    assert interpolation_state.nudgecoeff_e
    assert interpolation_state.geofac_n2s_nbh
    assert interpolation_state.geofac_div
    assert interpolation_state.geofac_grg_y
    assert interpolation_state.geofac_grg_x
    assert interpolation_state.rbf_coeff_2
    assert interpolation_state.rbf_coeff_1
    assert interpolation_state.geofac_n2s_c
