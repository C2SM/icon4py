# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os

import gt4py.next as gtx
import xarray as xr
from gt4py.next import backend as gtx_backend
from gt4py.next.ffront.fbuiltins import broadcast

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import base, horizontal as h_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import utils as testcases_utils
from icon4py.model.testing import serialbox as sb


log = logging.getLogger(__name__)

# sponge length on outflow
SPONGE_LENGTH = float(os.environ.get("ICON4PY_CHANNEL_SPONGE_LENGTH", "50.0"))
# perturbation magnitude for vn profile
RANDOM_PERTURBATION_MAGNITUDE = float(os.environ.get("ICON4PY_CHANNEL_PERTURBATION", "0.001"))


@gtx.field_operator
def _set_boundary_conditions_cell(
    field: fa.CellKField[ta.wpfloat],
    channel_field: fa.CellKField[ta.wpfloat],
    sponge: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    field = (broadcast(1.0, (dims.CellDim, dims.KDim)) - sponge) * field + sponge * channel_field
    return field


@gtx.field_operator
def _set_boundary_conditions_edge(
    field: fa.EdgeKField[ta.wpfloat],
    channel_field: fa.EdgeKField[ta.wpfloat],
    sponge: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    field = (broadcast(1.0, (dims.EdgeDim, dims.KDim)) - sponge) * field + sponge * channel_field
    return field


class ChannelFlow:
    """
    Main class for channel flow
    """

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        savepoint_path: str,
        grid_file_path: str,
        backend: gtx_backend.Backend,
        do_channel: bool = True,
    ):
        """
        Initialize the channel
        """

        self.backend = backend
        self.do_channel = do_channel
        xp = data_alloc.import_array_ns(self.backend)

        self.num_cells = grid.num_cells
        self.num_edges = grid.num_edges
        self.num_levels = grid.num_levels

        self.random_perturbation_magnitude = (
            RANDOM_PERTURBATION_MAGNITUDE  # perturbation magnitude for vn profile
        )

        # Allocate here so it's not re-allocated every BC call
        self.random_field_full_edge_np = xp.random.normal(
            loc=0, scale=self.random_perturbation_magnitude, size=(self.num_edges, self.num_levels)
        )
        self.random_field_full_edge = gtx.as_field(
            (dims.EdgeDim, dims.KDim), self.random_field_full_edge_np, allocator=self.backend
        )

        data_provider = sb.IconSerialDataProvider(
            backend=self.backend,
            fname_prefix="icon_pydycore",
            path=savepoint_path,
        )
        grid_savepoint = data_provider.from_savepoint_grid(
            grid_id="some_grid_uuid",
            grid_shape=icon_grid.GridShape(geometry_type=base.GeometryType.TORUS),
        )

        self.channel_y, self.channel_U = self.load_channel_data()

        self.full_cell_sponge, self.half_cell_sponge, self.full_edge_sponge = self.make_sponges(
            grid_file_path=grid_file_path,
        )

        (
            self.vn_ndarray,
            self.w_ndarray,
            self.rho_ndarray,
            self.exner_ndarray,
            self.theta_v_ndarray,
            self.vn,
            self.w,
            self.rho,
            self.exner,
            self.theta_v,
        ) = self.compute_channel_fields(
            channel_y=self.channel_y,
            channel_U=self.channel_U,
            grid=grid,
            data_provider=data_provider,
            grid_savepoint=grid_savepoint,
        )

        log.info("Channel flow initialized")

    def load_channel_data(
        self,
    ) -> tuple[
        data_alloc.NDArray,
        data_alloc.NDArray,
    ]:
        xp = data_alloc.import_array_ns(self.backend)

        # Lee & Moser data: Re_tau = 5200
        data = xp.loadtxt("../python-scripts/data/LeeMoser_chan5200.mean", skiprows=72)

        channel_y = data[:, 0]
        channel_U = data[:, 2] * 4.14872e-02  # <U> * u_tau (that's how it's normalized in the file)

        return channel_y, channel_U

    def make_sponges(
        self,
        grid_file_path: str,
    ) -> tuple[
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.EdgeKField[ta.wpfloat],
    ]:
        xp = data_alloc.import_array_ns(self.backend)

        grid_file = xr.open_dataset(grid_file_path)
        domain_length = grid_file.domain_length
        cell_x = xp.asarray(grid_file.cell_circumcenter_cartesian_x.values)
        edge_x = xp.asarray(grid_file.edge_middle_cartesian_x.values)

        # Cleanup the grid
        # Adjust x values to coincide with the periodic boundary
        cell_x = xp.where(xp.abs(cell_x - 0.0) < 1e-9, 0.0, cell_x)
        cell_x = xp.where(xp.abs(cell_x - domain_length) < 1e-9, 0.0, cell_x)

        half_cell_sponge_np = xp.zeros((self.num_cells, self.num_levels + 1), dtype=float)
        full_cell_sponge_np = xp.zeros((self.num_cells, self.num_levels), dtype=float)
        full_edge_sponge_np = xp.zeros((self.num_edges, self.num_levels), dtype=float)

        x_inflow = xp.unique(cell_x)[1]  # second cell centre from left
        x_outflow = xp.unique(cell_x)[-1]
        for k in range(self.num_levels):
            # outflow: tanh sponge + outflow
            full_cell_sponge_np[:, k] = (
                1 + xp.tanh((cell_x + SPONGE_LENGTH / 2 - x_outflow) * 2 * xp.pi / SPONGE_LENGTH)
            ) / 2
            full_edge_sponge_np[:, k] = (
                1 + xp.tanh((edge_x + SPONGE_LENGTH / 2 - x_outflow) * 2 * xp.pi / SPONGE_LENGTH)
            ) / 2
            full_cell_sponge_np[:, k] = xp.where(
                cell_x >= x_outflow, 1.0, full_cell_sponge_np[:, k]
            )
            full_edge_sponge_np[:, k] = xp.where(
                edge_x >= x_outflow, 1.0, full_edge_sponge_np[:, k]
            )
            # inflow: Dirichlet on first cell(s?)
            full_cell_sponge_np[:, k] = xp.where(cell_x <= x_inflow, 1.0, full_cell_sponge_np[:, k])
            full_edge_sponge_np[:, k] = xp.where(edge_x <= x_inflow, 1.0, full_edge_sponge_np[:, k])

        half_cell_sponge_np[:, :-1] = full_cell_sponge_np
        half_cell_sponge_np[:, -1] = half_cell_sponge_np[:, -2]

        full_cell_sponge = gtx.as_field((CellDim, KDim), full_cell_sponge_np)
        half_cell_sponge = gtx.as_field((CellDim, KDim), half_cell_sponge_np)
        full_edge_sponge = gtx.as_field((EdgeDim, KDim), full_edge_sponge_np)

        return full_cell_sponge, half_cell_sponge, full_edge_sponge

    def compute_channel_fields(
        self,
        channel_y: data_alloc.NDArray,
        channel_U: data_alloc.NDArray,
        grid: icon_grid.IconGrid,
        data_provider: sb.IconSerialDataProvider,
        grid_savepoint: sb.IconGridSavepoint,
    ) -> tuple[
        data_alloc.NDArray,
        data_alloc.NDArray,
        data_alloc.NDArray,
        data_alloc.NDArray,
        data_alloc.NDArray,
        fa.EdgeKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ]:
        nh_t0 = 300.0
        nh_brunt_vais = 0.0

        xp = data_alloc.import_array_ns(self.backend)

        edge_domain = h_grid.domain(dims.EdgeDim)
        end_edge_lateral_boundary_level_2 = grid.end_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

        wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().ndarray
        ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().ndarray
        theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().ndarray
        theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().ndarray
        exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().ndarray
        d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().ndarray
        geopot = data_provider.from_metrics_savepoint().geopot().ndarray
        full_level_heights = data_provider.from_metrics_savepoint().z_mc().ndarray
        half_level_heights = data_provider.from_metrics_savepoint().z_ifc().ndarray

        edge_geometry = grid_savepoint.construct_edge_geometry()
        primal_normal_x = edge_geometry.primal_normal[0].ndarray
        primal_normal_x = xp.repeat(
            xp.expand_dims(primal_normal_x, axis=-1), self.num_levels, axis=1
        )
        sponge_array_edge_start_plus1_to_edge_end = xp.ones(self.num_edges, dtype=bool)
        sponge_array_edge_start_plus1_to_edge_end[0:end_edge_lateral_boundary_level_2] = False
        u0_sponge = xp.repeat(
            xp.expand_dims(sponge_array_edge_start_plus1_to_edge_end, axis=-1),
            self.num_levels,
            axis=1,
        )

        # Rescale the channel data to the ICON grid and mirror y to full channel height
        model_top = half_level_heights[0, 0]
        channel_y = channel_y * model_top / 2
        channel_y = xp.concatenate((channel_y, model_top - channel_y[::-1]), axis=0)
        channel_U = xp.concatenate((channel_U, channel_U[::-1]), axis=0)

        # Interpolate LM_u onto the ICON grid
        nh_u0 = xp.zeros((self.num_edges, self.num_levels), dtype=float)
        for j in range(self.num_levels):
            LM_j = xp.argmin(
                xp.abs(channel_y - full_level_heights[0, j])
            )  # NOTE: full_level_heights should be identical for all edges because the channel is flat
            nh_u0[:, j] = channel_U[LM_j] + xp.random.normal(
                loc=0, scale=self.random_perturbation_magnitude, size=self.num_edges
            )

        u = xp.where(u0_sponge, nh_u0, 0.0)
        vn_ndarray = u * primal_normal_x

        w_ndarray = xp.zeros((self.num_cells, self.num_levels + 1), dtype=float)

        # ---------------------------------------------------------------------------
        # The following is from the Gauss3D experiment

        theta_v_ndarray = xp.zeros((self.num_cells, self.num_levels), dtype=float)
        exner_ndarray = xp.zeros((self.num_cells, self.num_levels), dtype=float)
        rho_ndarray = xp.zeros((self.num_cells, self.num_levels), dtype=float)

        # Vertical temperature profile
        for k_index in range(self.num_levels - 1, -1, -1):
            z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
            # profile of theta is explicitly given
            theta_v_ndarray[:, k_index] = nh_t0 * xp.exp(z_help)

        # Lower boundary condition for exner pressure
        if nh_brunt_vais != 0.0:
            z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, self.num_levels - 1]
            exner_ndarray[:, self.num_levels - 1] = (
                phy_const.GRAV / nh_brunt_vais
            ) ** 2 / nh_t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
        else:
            exner_ndarray[:, self.num_levels - 1] = (
                1.0 - geopot[:, self.num_levels - 1] / phy_const.CPD / nh_t0
            )

        # Compute hydrostatically balanced exner, by integrating the (discretized!)
        # 3rd equation of motion under the assumption thetav=const.
        rho_ndarray, exner_ndarray = testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
            wgtfac_c,
            ddqz_z_half,
            exner_ref_mc,
            d_exner_dz_ref_ic,
            theta_ref_mc,
            theta_ref_ic,
            rho_ndarray,
            exner_ndarray,
            theta_v_ndarray,
            self.num_levels,
        )

        vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=self.backend)
        w = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=self.backend)
        exner = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=self.backend)
        rho = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=self.backend)
        theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=self.backend)

        return (
            vn_ndarray,
            w_ndarray,
            rho_ndarray,
            exner_ndarray,
            theta_v_ndarray,
            vn,
            w,
            rho,
            exner,
            theta_v,
        )

    def set_initial_conditions(
        self,
        vn: fa.EdgeKField[ta.wpfloat],
        w: fa.CellKField[ta.wpfloat],
        rho: fa.CellKField[ta.wpfloat],
        exner: fa.CellKField[ta.wpfloat],
        theta_v: fa.CellKField[ta.wpfloat],
    ) -> tuple[
        fa.EdgeKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ]:
        """
        Set the initial conditions for the channel flow
        """
        if self.do_channel:
            # overwrite variables with channel fields, else return untouched
            vn = gtx.as_field((dims.EdgeDim, dims.KDim), self.vn_ndarray, allocator=self.backend)
            w = gtx.as_field((dims.CellDim, dims.KDim), self.w_ndarray, allocator=self.backend)
            exner = gtx.as_field((dims.CellDim, dims.KDim), self.exner_ndarray, allocator=self.backend)
            rho = gtx.as_field((dims.CellDim, dims.KDim), self.rho_ndarray, allocator=self.backend)
            theta_v = gtx.as_field(
                (dims.CellDim, dims.KDim), self.theta_v_ndarray, allocator=self.backend
            )
        return vn, w, rho, exner, theta_v

    def set_boundary_conditions(
        self,
        vn: fa.EdgeKField[ta.wpfloat],
        w: fa.CellKField[ta.wpfloat],
        rho: fa.CellKField[ta.wpfloat],
        exner: fa.CellKField[ta.wpfloat],
        theta_v: fa.CellKField[ta.wpfloat],
    ) -> tuple[
        fa.EdgeKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ]:
        """
        Set the boundary conditions for the channel flow
        """

        if self.do_channel:
            # modify variables with channel fields, else return untouched
            xp = data_alloc.import_array_ns(self.backend)
            self.random_field_full_edge_np = xp.random.normal(
                loc=0, scale=self.random_perturbation_magnitude, size=(self.num_edges, self.num_levels)
            )
            self.random_field_full_edge = gtx.as_field(
                (dims.EdgeDim, dims.KDim), self.random_field_full_edge_np, allocator=self.backend
            )

            _set_boundary_conditions_edge(
                field=vn,
                channel_field=self.vn + self.random_field_full_edge,
                sponge=self.full_edge_sponge,
                out=vn,
                offset_provider={},
            )
            _set_boundary_conditions_cell(
                field=w,
                channel_field=self.w,
                sponge=self.half_cell_sponge,
                out=w,
                offset_provider={},
            )
            _set_boundary_conditions_cell(
                field=rho,
                channel_field=self.rho,
                sponge=self.full_cell_sponge,
                out=rho,
                offset_provider={},
            )
            _set_boundary_conditions_cell(
                field=exner,
                channel_field=self.exner,
                sponge=self.full_cell_sponge,
                out=exner,
                offset_provider={},
            )
            _set_boundary_conditions_cell(
                field=theta_v,
                channel_field=self.theta_v,
                sponge=self.full_cell_sponge,
                out=theta_v,
                offset_provider={},
            )

        return vn, w, rho, exner, theta_v
