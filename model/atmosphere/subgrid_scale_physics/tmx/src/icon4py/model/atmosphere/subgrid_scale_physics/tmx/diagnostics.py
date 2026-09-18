# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The Smagorinsky diagnostics component of tmx.

Port of ``Compute_diagnostics`` in ICON's ``mo_vdf_atmo.f90``, together with
``Smagorinsky_init`` in ``mo_tmx_smagorinsky.f90``, which runs at construction.
"""

from __future__ import annotations

import functools
import logging
import typing

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import diagnostics as diag_stencils
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.interpolation.stencils.interpolate_cell_vector_to_edge_normal import (
    interpolate_cell_vector_to_edge_normal,
)
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


if typing.TYPE_CHECKING:
    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.common import field_type_aliases as fa, type_alias as ta


log = logging.getLogger(__name__)


class Diagnostics:
    """The Smagorinsky diagnostics stage of tmx."""

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
        turb_prandtl: float,
        smag_constant: float,
        max_turb_scale: float,
        km_min: float,
        km_const: float,
        use_km_const: bool,
        louis_constant_b: float,
        use_louis: bool,
        use_louis_land: bool,
        use_louis_ice: bool,
    ) -> None:
        self._allocator = model_backends.get_allocator(backend)
        self._exchange = exchange
        self._grid = grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        assert self._cell_params.area is not None

        self._smag_constant = smag_constant
        self._max_turb_scale = max_turb_scale
        self._km_min = km_min
        self._km_const = km_const
        self._use_km_const = use_km_const
        self._louis_constant_b = louis_constant_b
        self._use_louis = use_louis
        self._use_louis_land = use_louis_land
        self._use_louis_ice = use_louis_ice
        # reciprocal turbulent Prandtl number (``rturb_prandtl`` in
        # mo_turb_vdiff_config.f90)
        self._rturb_prandtl = 1.0 / turb_prandtl

        if not (use_louis_land and use_louis_ice):
            log.warning(
                "'use_louis_land' / 'use_louis_ice' make the Louis stability correction "
                "depend on the land and sea-ice fractions, which are not part of "
                "TmxInputState yet and are allocated as zero fields (aqua planet)."
            )

        num_levels = self._grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        self._cell_start_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._cell_start_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._cell_start_lateral_boundary_level_4 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._cell_start_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._cell_end_halo_level_2 = self._grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))

        self._edge_start_lateral_boundary_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._edge_start_lateral_boundary_level_3 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._edge_start_lateral_boundary_level_4 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._edge_start_nudging = self._grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        self._edge_end_halo_level_3 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_3))

        self._vertex_start_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_start_nudging = self._grid.start_index(vertex_domain(h_grid.Zone.NUDGING))
        self._vertex_end_local = self._grid.end_index(vertex_domain(h_grid.Zone.LOCAL))
        self._vertex_end_halo = self._grid.end_index(vertex_domain(h_grid.Zone.HALO))

        self._initialize_static_fields(backend)
        self._setup_programs(backend, num_levels)

    def _initialize_static_fields(self, backend: model_backends.BackendLike) -> None:
        """Allocate the fields that only depend on the grid and run ``Smagorinsky_init``
        (mo_tmx_smagorinsky.f90) into them."""
        zero_field = functools.partial(data_alloc.zero_field, self._grid, allocator=self._allocator)

        # squared Smagorinsky mixing length at half-level cell centers [m^2]
        self.mixing_length_sq: fa.CellKHalfField[ta.wpfloat] = zero_field(
            dims.CellDim, dims.KHalfDim
        )
        # compute_mixing_length (mo_tmx_smagorinsky.f90): cells rl 3..min_rlcell_int,
        # all half levels
        diag_stencils.compute_smagorinsky_mixing_length.with_backend(backend)(
            ddqz_z_half=self._metric_state.ddqz_z_half,
            geopot_agl_ifc=self._metric_state.geopot_agl_ifc,
            cell_area=self._cell_params.area,
            mixing_length_sq=self.mixing_length_sq,
            smag_constant=self._smag_constant,
            max_turb_scale=self._max_turb_scale,
            grav=constants.GRAV,
            horizontal_start=self._cell_start_lateral_boundary_level_3,
            horizontal_end=self._cell_end_local,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self._grid.num_levels + 1),
            offset_provider={},
        )
        # cell-area scaling factor of the Louis constant b; the viscosity stencil takes it
        # in either branch, so it is allocated (and stays zero) with Louis switched off
        self.scaling_factor_louis: fa.CellField[ta.wpfloat] = zero_field(dims.CellDim)
        if self._use_louis:
            # compute_scaling_factor_louis (mo_tmx_smagorinsky.f90): cells rl
            # 3..min_rlcell_int
            diag_stencils.compute_scaling_factor_louis.with_backend(backend)(
                cell_area=self._cell_params.area,
                scaling_factor_louis=self.scaling_factor_louis,
                horizontal_start=self._cell_start_lateral_boundary_level_3,
                horizontal_end=self._cell_end_local,
                offset_provider={},
            )

        # land and sea-ice fractions: the atmosphere-only port has no source for them,
        # so they stay zero (see the warning in __init__)
        self.fract_land: fa.CellField[ta.wpfloat] = zero_field(dims.CellDim)
        self.fract_ice: fa.CellField[ta.wpfloat] = zero_field(dims.CellDim)

    def _setup_programs(
        self,
        backend: model_backends.BackendLike,
        num_levels: int,
    ) -> None:
        """Bind the diagnostics step programs (``Compute_diagnostics`` l. 343-482 in
        mo_vdf_atmo.f90)."""
        # ---------------------------------------------------------------------
        # In the Fortran call order of Compute_diagnostics
        # (mo_vdf_atmo.f90 l. 343-482). One program per halo-exchange interval
        # and horizontal dimension.
        # ---------------------------------------------------------------------
        # compute_static_energy, get_virtual_potential_temperature,
        # vert_intp_full2half_cell_3d (rho -> rho_ic) and brunt_vaisala_freq
        self.compute_thermodynamic_diagnostics = setup_program(
            backend=backend,
            program=diag_stencils.compute_thermodynamic_diagnostics,
            constant_args={
                "height_above_ground": self._metric_state.height_above_ground,
                "wgtfac_c": self._metric_state.wgtfac_c,
                "inv_ddqz_z_half": self._metric_state.inv_ddqz_z_half,
                "wgtfacq1_c": self._metric_state.wgtfacq1_c,
                "wgtfacq_c": self._metric_state.wgtfacq_c,
                "grav": constants.GRAV,
            },
            horizontal_sizes={
                "cell_start_nudging": self._cell_start_nudging,
                "cell_start_lateral_boundary_level_2": self._cell_start_lateral_boundary_level_2,
                "cell_start_lateral_boundary_level_3": self._cell_start_lateral_boundary_level_3,
                "cell_end_local": self._cell_end_local,
                "cell_end_halo_level_2": self._cell_end_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_start_interior": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
                "vertical_end_half": gtx.int32(num_levels + 1),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider={},
        )
        # compute_normal_velocity_edge: edges rl grf_bdywidth_e+1..min_rledge_int,
        # all full levels
        self.interpolate_cell_vector_to_edge_normal = setup_program(
            backend=backend,
            program=interpolate_cell_vector_to_edge_normal,
            constant_args={
                "primal_normal_cell_x": self._edge_params.primal_normal_cell[0],
                "primal_normal_cell_y": self._edge_params.primal_normal_cell[1],
                "c_lin_e": self._interpolation_state.c_lin_e,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging_level_2,
                "horizontal_end": self._edge_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # cells2verts_scalar (w -> w_vert) and rbf_vec_interpol_vertex
        # (vn -> u_vert, v_vert), the three fields synced afterwards
        self.interpolate_wind_to_vertices = setup_program(
            backend=backend,
            program=diag_stencils.interpolate_wind_to_vertices,
            constant_args={
                "cells_aw_verts": self._interpolation_state.cells_aw_verts,
                "rbf_coeff_v1": self._interpolation_state.rbf_coeff_v1,
                "rbf_coeff_v2": self._interpolation_state.rbf_coeff_v2,
            },
            # vertices rl 2..min_rlvert_int
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
                "vertical_end_half": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )
        # cells2edges_scalar (w -> w_ie),
        # interpolate_normal_velocity_edge_interface (vn -> vn_ie),
        # rbf_vec_interpol_edge (vn_ie -> vt_ie),
        # compute_velocity_gradient_tensor + compute_shear,
        # get_horizontal_divergence_strain_rate_cell (div_of_stress -> div_c),
        # interpolate_rate_of_strain_full2half_edge2cell (shear -> mech_prod) and
        # Smagorinsky_model / Assign_constant_eddy_viscosity (-> km_ic, kh_ic)
        self.compute_shear_and_viscosity_diagnostics = setup_program(
            backend=backend,
            program=diag_stencils.compute_shear_and_viscosity_diagnostics,
            constant_args={
                "c_lin_e": self._interpolation_state.c_lin_e,
                "wgtfac_e": self._metric_state.wgtfac_e,
                "wgtfacq1_e": self._metric_state.wgtfacq1_e,
                "wgtfacq_e": self._metric_state.wgtfacq_e,
                "rbf_vec_coeff_e": self._interpolation_state.rbf_coeff_e,
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "inv_ddqz_z_full_e": self._metric_state.inv_ddqz_z_full_e,
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "wgtfac_c": self._metric_state.wgtfac_c,
                "mixing_length_sq": self.mixing_length_sq,
                "scaling_factor_louis": self.scaling_factor_louis,
                "fract_land": self.fract_land,
                "fract_ice": self.fract_ice,
                "rturb_prandtl": self._rturb_prandtl,
                "louis_constant_b": self._louis_constant_b,
                "km_const": self._km_const,
                "use_km_const": self._use_km_const,
                "use_louis": self._use_louis,
                "use_louis_land": self._use_louis_land,
                "use_louis_ice": self._use_louis_ice,
            },
            horizontal_sizes={
                "edge_start_lateral_boundary_level_2": self._edge_start_lateral_boundary_level_2,
                "edge_start_lateral_boundary_level_3": self._edge_start_lateral_boundary_level_3,
                "edge_start_lateral_boundary_level_4": self._edge_start_lateral_boundary_level_4,
                "edge_end_halo_level_2": self._edge_end_halo_level_2,
                "edge_end_halo_level_3": self._edge_end_halo_level_3,
                "cell_start_nudging": self._cell_start_nudging,
                "cell_start_lateral_boundary_level_3": self._cell_start_lateral_boundary_level_3,
                "cell_end_local": self._cell_end_local,
                "cell_end_halo": self._cell_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_start_interior": gtx.int32(1),
                "vertical_end": gtx.int32(num_levels),
                "vertical_end_half": gtx.int32(num_levels + 1),
                "nlev": gtx.int32(num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        # the km/kh loops that follow the kh_ic/km_ic exchange
        # ('interpolate_eddy_viscosity2cell' / '2vertex' / '2edge' in
        # mo_vdf_atmo.f90): one program, three entities. Halo rows are computed
        # on purpose, they are read by the diffusion later.
        self.interpolate_km = setup_program(
            backend=backend,
            program=diag_stencils.interpolate_km,
            constant_args={
                "cells_aw_verts": self._interpolation_state.cells_aw_verts,
                "c_lin_e": self._interpolation_state.c_lin_e,
                "km_min": self._km_min,
            },
            horizontal_sizes={
                # cells rl 4..min_rlcell_int-1
                "cell_start": self._cell_start_lateral_boundary_level_4,
                "cell_end": self._cell_end_halo,
                # vertices rl 5..min_rlvert_int-1
                "vertex_start": self._vertex_start_nudging,
                "vertex_end": self._vertex_end_halo,
                # edges rl grf_bdywidth_e..min_rledge_int-1
                "edge_start": self._edge_start_nudging,
                "edge_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(num_levels),
                "vertical_end_half": gtx.int32(num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

    def run(
        self,
        input_state: tmx_states.TmxInputState,
        diagnostic_state: tmx_states.TmxDiagnosticState,
    ) -> None:
        """
        Compute the Smagorinsky diagnostics.

        Port of ``Compute_diagnostics`` in mo_vdf_atmo.f90 (l. 343-482), with
        the halo exchanges at the Fortran sync points and one program per
        exchange interval and horizontal dimension.
        """
        log.debug("tmx diagnostics (Compute_diagnostics): start")

        self.compute_thermodynamic_diagnostics(
            temperature=input_state.temperature,
            virtual_temperature=input_state.virtual_temperature,
            pressure=input_state.pressure,
            rho=input_state.rho,
            dry_static_energy=diagnostic_state.cptgz,
            theta_v=diagnostic_state.theta_v,
            rho_ic=diagnostic_state.rho_ic,
            bruvais=diagnostic_state.bruvais,
        )

        log.debug("communication of input u, v (cells): start")
        self._exchange.exchange(dims.CellDim, input_state.u, input_state.v)
        log.debug("communication of input u, v (cells): end")

        self.interpolate_cell_vector_to_edge_normal(
            vector_x=input_state.u,
            vector_y=input_state.v,
            normal_component=diagnostic_state.vn,
        )

        # TODO(havogt): this halo_exchange can probably be skipped if we overcompute
        # in `interpolate_cell_vector_to_edge_normal`.
        log.debug("communication of vn (edges): start")
        self._exchange.exchange(dims.EdgeDim, diagnostic_state.vn)
        log.debug("communication of vn (edges): end")

        self.interpolate_wind_to_vertices(
            w=input_state.w,
            vn=diagnostic_state.vn,
            w_vert=diagnostic_state.w_vert,
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
        )

        log.debug("communication of w_vert, u_vert, v_vert (vertices): start")
        self._exchange.exchange(
            dims.VertexDim,
            diagnostic_state.w_vert,
            diagnostic_state.u_vert,
            diagnostic_state.v_vert,
        )
        log.debug("communication of w_vert, u_vert, v_vert (vertices): end")

        self.compute_shear_and_viscosity_diagnostics(
            w=input_state.w,
            vn=diagnostic_state.vn,
            u_vert=diagnostic_state.u_vert,
            v_vert=diagnostic_state.v_vert,
            w_vert=diagnostic_state.w_vert,
            bruvais=diagnostic_state.bruvais,
            rho_ic=diagnostic_state.rho_ic,
            w_ie=diagnostic_state.w_ie,
            vn_ie=diagnostic_state.vn_ie,
            vt_ie=diagnostic_state.vt_ie,
            shear=diagnostic_state.shear,
            div_of_stress=diagnostic_state.div_of_stress,
            div_c=diagnostic_state.div_c,
            mech_prod=diagnostic_state.mech_prod,
            km_ic=diagnostic_state.km_ic,
            kh_ic=diagnostic_state.kh_ic,
        )

        # unconditional, unlike the Fortran, which skips it in the constant-viscosity branch:
        # the viscosity program writes cells rl 3..min_rlcell_int only, and 'interpolate_km'
        # gathers from halo cells
        log.debug("communication of kh_ic, km_ic (cells): start")
        self._exchange.exchange(dims.CellDim, diagnostic_state.kh_ic, diagnostic_state.km_ic)
        log.debug("communication of kh_ic, km_ic (cells): end")

        self.interpolate_km(
            km_ic=diagnostic_state.km_ic,
            km_c=diagnostic_state.km_c,
            km_iv=diagnostic_state.km_iv,
            km_ie=diagnostic_state.km_ie,
        )

        log.debug("tmx diagnostics (Compute_diagnostics): end")
