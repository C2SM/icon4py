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


import pytest

from icon4py.model.common.grid.horizontal import CellParams, EdgeParams
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.driver.io_utils import (
    SerializationType,
    read_geometry_fields,
    read_icon_grid,
    read_static_fields,
)


@pytest.mark.datatest
@pytest.mark.parametrize("read_fun", (read_geometry_fields, read_static_fields, read_icon_grid))
def test_read_geometry_fields_not_implemented_type(read_fun, datapath):
    with pytest.raises(NotImplementedError, match=r"Only ser_type='sb'"):
        read_fun(path=datapath, ser_type=SerializationType.NC)


def assert_grid_size_and_connectivities(grid):
    assert grid.num_edges == 31558
    assert grid.num_cells == 20896
    assert grid.num_vertices == 10663
    assert grid.get_offset_provider("E2V")
    assert grid.get_offset_provider("V2E")
    assert grid.get_offset_provider("C2E")
    assert grid.get_offset_provider("E2C")
    assert grid.get_offset_provider("E2C2V")
    assert grid.get_offset_provider("C2E2C")
    assert grid.get_offset_provider("E2ECV")


@pytest.mark.datatest
def test_read_icon_grid_for_type_sb(datapath):
    grid = read_icon_grid(datapath, ser_type=SerializationType.SB)
    assert_grid_size_and_connectivities(grid)


@pytest.mark.datatest
def test_read_static_fields_for_type_sb(datapath):
    (
        diffusion_metric_state,
        diffusion_interpolation_state,
        solve_nonhydro_metric_state,
        solve_nonhydro_interpolation_state,
    ) = read_static_fields(datapath, ser_type=SerializationType.SB)
    assert_diffusion_metric_state_fields(diffusion_metric_state)
    assert_diffusion_interpolation_state_fields(diffusion_interpolation_state)
    assert_nonhydro_metric_state_fields(solve_nonhydro_metric_state)
    assert_nonhydro_interpolation_state_fields(solve_nonhydro_interpolation_state)


@pytest.mark.datatest
def test_read_geometry_fields_for_type_sb(datapath):
    edge_geometry, cell_geometry, vertical_geometry, c_owner_mask = read_geometry_fields(
        datapath, ser_type=SerializationType.SB
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


def assert_diffusion_metric_state_fields(metric_state):
    assert metric_state.wgtfac_c
    assert metric_state.zd_intcoef
    assert metric_state.zd_diffcoef
    assert metric_state.theta_ref_mc
    assert metric_state.mask_hdiff
    assert metric_state.zd_vertoffset


def assert_nonhydro_metric_state_fields(metric_state):
    assert metric_state.bdy_halo_c
    assert metric_state.mask_prog_halo_c
    assert metric_state.rayleigh_w
    assert metric_state.wgtfac_c
    assert metric_state.wgtfacq_c_dsl
    assert metric_state.wgtfac_e
    assert metric_state.wgtfacq_e_dsl
    assert metric_state.exner_exfac
    assert metric_state.exner_ref_mc
    assert metric_state.rho_ref_mc
    assert metric_state.theta_ref_mc
    assert metric_state.rho_ref_me
    assert metric_state.theta_ref_me
    assert metric_state.theta_ref_ic
    assert metric_state.d_exner_dz_ref_ic
    assert metric_state.ddqz_z_half
    assert metric_state.d2dexdz2_fac1_mc
    assert metric_state.d2dexdz2_fac2_mc
    assert metric_state.ddxn_z_full
    assert metric_state.ddqz_z_full_e
    assert metric_state.ddxt_z_full
    assert metric_state.inv_ddqz_z_full
    assert metric_state.vertoffset_gradp
    assert metric_state.zdiff_gradp
    assert metric_state.ipeidx_dsl
    assert metric_state.pg_exdist
    assert metric_state.vwind_expl_wgt
    assert metric_state.vwind_impl_wgt
    assert metric_state.hmask_dd3d
    assert metric_state.scalfac_dd3d
    assert metric_state.coeff1_dwdz
    assert metric_state.coeff2_dwdz
    assert metric_state.coeff_gradekin


def assert_diffusion_interpolation_state_fields(interpolation_state):
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


def assert_nonhydro_interpolation_state_fields(interpolation_state):
    assert interpolation_state.e_bln_c_s
    assert interpolation_state.rbf_coeff_1
    assert interpolation_state.rbf_coeff_2
    assert interpolation_state.geofac_div
    assert interpolation_state.geofac_n2s
    assert interpolation_state.geofac_grg_x
    assert interpolation_state.geofac_grg_y
    assert interpolation_state.nudgecoeff_e
    assert interpolation_state.c_lin_e
    assert interpolation_state.geofac_grdiv
    assert interpolation_state.rbf_vec_coeff_e
    assert interpolation_state.c_intp
    assert interpolation_state.geofac_rot
    assert interpolation_state.pos_on_tplane_e_1
    assert interpolation_state.pos_on_tplane_e_2
    assert interpolation_state.e_flx_avg

    assert interpolation_state.geofac_n2s_c
    assert interpolation_state.geofac_n2s_nbh
