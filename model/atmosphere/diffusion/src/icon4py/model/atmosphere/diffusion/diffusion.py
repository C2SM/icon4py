# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import logging
import math
import sys
from typing import Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import allocators as gtx_allocators

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import (
    config as diffusion_config,
    diffusion_states,
    diffusion_utils,
)
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    copy_field,
    init_diffusion_local_fields_for_regular_timestep,
    scale_k,
    setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    apply_diffusion_to_theta_and_exner,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.orchestration import decorator as dace_orchestration
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    config: dataclasses.InitVar[diffusion_config.DiffusionConfig]
    K2: Final[float] = dataclasses.field(init=False)
    K4: Final[float] = dataclasses.field(init=False)
    K6: Final[float] = dataclasses.field(init=False)
    K4W: Final[float] = dataclasses.field(init=False)
    smagorinski_factor: Final[float] = dataclasses.field(init=False)
    smagorinski_height: Final[float] = dataclasses.field(init=False)

    def __post_init__(self, config):
        object.__setattr__(
            self,
            "K2",
            (1.0 / (config.hdiff_efdt_ratio * 8.0) if config.hdiff_efdt_ratio > 0.0 else 0.0),
        )
        object.__setattr__(self, "K4", self.K2 / 8.0)
        object.__setattr__(self, "K6", self.K2 / 64.0)
        object.__setattr__(
            self,
            "K4W",
            (1.0 / (config.hdiff_w_efdt_ratio * 36.0) if config.hdiff_w_efdt_ratio > 0 else 0.0),
        )

        (
            smagorinski_factor,
            smagorinski_height,
        ) = self._determine_smagorinski_factor(config)
        object.__setattr__(self, "smagorinski_factor", smagorinski_factor)
        object.__setattr__(self, "smagorinski_height", smagorinski_height)

    def _determine_smagorinski_factor(self, config: diffusion_config.DiffusionConfig):
        """Enhanced Smagorinsky diffusion factor.

        Smagorinsky diffusion factor is defined as a profile in height
        above sea level with 4 height sections.

        It is calculated/used only in the case of diffusion_type 3 or 5
        """
        match config.diffusion_type:
            case 5:
                (
                    smagorinski_factor,
                    smagorinski_height,
                ) = diffusion_type_5_smagorinski_factor(config)
            case 4:
                # according to mo_nh_diffusion.f90 this isn't used anywhere the factor is only
                # used for diffusion_type (3,5) but the defaults are only defined for iequations=3
                smagorinski_factor = (
                    config.smagorinski_scaling_factor
                    if config.smagorinski_scaling_factor
                    else 0.15,
                )
                smagorinski_height = None
            case _:
                raise NotImplementedError("Only implemented for diffusion type 4 and 5")
                smagorinski_factor = None
                smagorinski_height = None
                pass
        return smagorinski_factor, smagorinski_height


def diffusion_type_5_smagorinski_factor(config: diffusion_config.DiffusionConfig):
    """
    Initialize Smagorinski factors used in diffusion type 5.

    The calculation and magic numbers are taken from mo_diffusion_nml.f90
    """
    magic_sqrt = math.sqrt(1600.0 * (1600 + 50000.0))
    magic_fac2_value = 2e-6 * (1600.0 + 25000.0 + magic_sqrt)
    magic_z2 = 1600.0 + 50000.0 + magic_sqrt
    factor = (config.smagorinski_scaling_factor, magic_fac2_value, 0.0, 1.0)
    heights = (32500.0, magic_z2, 50000.0, 90000.0)
    return factor, heights


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: diffusion_config.DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        orchestration: bool = False,
        exchange: decomposition.ExchangeRuntime | None = None,
    ):
        self._allocator = model_backends.get_allocator(backend)
        self._orchestration = orchestration
        self._exchange = exchange or decomposition.SingleNodeExchange()
        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        self.halo_exchange_wait = decomposition.create_halo_exchange_wait(
            self._exchange
        )  # wait on a communication handle
        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points hat activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self._horizontal_start_index_w_diffusion: gtx.int32 = gtx.int32(0)

        self.nudgezone_diff: float = 0.04 / (
            config.max_nudging_coefficient + sys.float_info.epsilon
        )
        self.bdy_diff: float = 0.015 / (config.max_nudging_coefficient + sys.float_info.epsilon)
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float) / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float
        self.diff_multfac_w: float = min(1.0 / 48.0, params.K4W * config.substep_as_float)
        self._determine_horizontal_domains()

        self.mo_intp_rbf_rbf_vec_interpol_vertex = setup_program(
            backend=backend,
            program=mo_intp_rbf_rbf_vec_interpol_vertex,
            constant_args={
                "ptr_coeff_1": self._interpolation_state.rbf_coeff_1,
                "ptr_coeff_2": self._interpolation_state.rbf_coeff_2,
            },
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )

        self.calculate_nabla2_and_smag_coefficients_for_vn = setup_program(
            backend=backend,
            program=calculate_nabla2_and_smag_coefficients_for_vn,
            constant_args={
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_5,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )

        self.calculate_diagnostic_quantities_for_turbulence = setup_program(
            backend=backend,
            program=calculate_diagnostic_quantities_for_turbulence,
            constant_args={
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "geofac_div": self._interpolation_state.geofac_div,
                "wgtfac_c": self._metric_state.wgtfac_c,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={"vertical_start": 1, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_vn = setup_program(
            backend=backend,
            program=apply_diffusion_to_vn,
            constant_args={
                "primal_normal_vert_v1": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_v2": self._edge_params.primal_normal_vert[1],
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "area_edge": self._edge_params.edge_areas,
                "nudgecoeff_e": self._interpolation_state.nudgecoeff_e,
                "nudgezone_diff": self.nudgezone_diff,
                "fac_bdydiff_v": self.fac_bdydiff_v,
                "limited_area": self._grid.limited_area,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_5,
                "horizontal_end": self._edge_end_local,
                "start_2nd_nudge_line_idx_e": self._edge_start_nudging_level_2,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence = setup_program(
            backend=backend,
            program=apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
            constant_args={
                "geofac_n2s": self._interpolation_state.geofac_n2s,
                "geofac_grg_x": self._interpolation_state.geofac_grg_x,
                "geofac_grg_y": self._interpolation_state.geofac_grg_y,
                "area": self._cell_params.area,
                "diff_multfac_w": self.diff_multfac_w,
                "type_shear": gtx.int32(
                    self.config.shear_type.value
                ),  # DaCe parser peculiarity (does not work as gtx.int32)
            },
            horizontal_sizes={
                "horizontal_start": self._horizontal_start_index_w_diffusion,
                "horizontal_end": self._cell_end_halo,
                "halo_idx": self._cell_end_local,
                "interior_idx": self._cell_start_interior,
            },
            vertical_sizes={
                "vertical_start": 0,
                "vertical_end": self._grid.num_levels,
                "nrdmax": gtx.int32(  # DaCe parser peculiarity (does not work as gtx.int32)
                    self._vertical_grid.end_index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
            },
            offset_provider=self._grid.connectivities,
        )
        self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools = setup_program(
            backend=backend,
            program=calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
            constant_args={
                "theta_ref_mc": self._metric_state.theta_ref_mc,
                "thresh_tdiff": self.thresh_tdiff,
                "smallest_vpfloat": constants.DBL_EPS,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": self._grid.num_levels - 2,
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_theta_and_exner = setup_program(
            backend=backend,
            program=apply_diffusion_to_theta_and_exner,
            constant_args={
                "geofac_div": self._interpolation_state.geofac_div,
                "mask": self._metric_state.mask_hdiff,
                "zd_vertoffset": self._metric_state.zd_vertoffset,
                "zd_diffcoef": self._metric_state.zd_diffcoef,
                "vcoef": self._metric_state.zd_intcoef,
                "geofac_n2s_c": self._interpolation_state.geofac_n2s_c,
                "geofac_n2s_nbh": self._interpolation_state.geofac_n2s_nbh,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "area": self._cell_params.area,
                "apply_zdiffusion_t": self.config.apply_zdiffusion_t,
                "rd_o_cvd": self.rd_o_cvd,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": 0,
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self.copy_field = setup_program(backend=backend, program=copy_field)
        self.scale_k = setup_program(backend=backend, program=scale_k)
        self.setup_fields_for_initial_step = setup_program(
            backend=backend, program=setup_fields_for_initial_step
        )

        self.init_diffusion_local_fields_for_regular_timestep = setup_program(
            backend=backend,
            program=init_diffusion_local_fields_for_regular_timestep,
            offset_provider={"Koff": dims.KDim},
        )

        self._allocate_local_fields(model_backends.get_allocator(backend))

        self.init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            config.substep_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
            offset_provider={"Koff": dims.KDim},
        )
        setup_program(
            backend=backend,
            program=diffusion_utils.init_nabla2_factor_in_upper_damping_zone,
            constant_args={
                "physical_heights": self._vertical_grid.interface_physical_height,
                "nshift": 0,
            },
            vertical_sizes={
                "vertical_start": 1,
                "vertical_end": gtx.int32(self._vertical_grid.end_index_of_damping_layer + 1),
                "end_index_of_damping_layer": self._vertical_grid.end_index_of_damping_layer,
                "heights_1": self._vertical_grid.interface_physical_height.ndarray[1].item(),
                "heights_nrd_shift": self._vertical_grid.interface_physical_height.ndarray[
                    self._vertical_grid.end_index_of_damping_layer + 1
                ].item(),
            },
        )(diff_multfac_n2w=self.diff_multfac_n2w)

        # TODO(edopao): we should call gtx.common.offset_provider_to_type()
        #   but this requires some changes in gt4py domain inference.
        self.compile_time_connectivities = self._grid.connectivities

    def _allocate_local_fields(self, allocator: gtx_allocators.FieldBufferAllocationUtil | None):
        self.diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.diff_multfac_n2w = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.enh_smag_fac = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.u_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.v_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_ec = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.z_nabla2_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.diff_multfac_smag = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        # TODO(halungge): this is KHalfDim
        self.vertical_index = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )
        self.horizontal_cell_index = data_alloc.index_field(
            self._grid, dims.CellDim, allocator=allocator
        )
        self.horizontal_edge_index = data_alloc.index_field(
            self._grid, dims.EdgeDim, allocator=allocator
        )
        self.w_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )
        self.theta_v_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        def _get_start_index_for_w_diffusion() -> gtx.int32:
            return (
                self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if self._grid.limited_area
                else self._grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        self._cell_start_interior = self._grid.start_index(cell_domain(h_grid.Zone.INTERIOR))
        self._cell_start_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        self._edge_start_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._edge_start_nudging = self._grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._vertex_start_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_end_local = self._grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        self._horizontal_start_index_w_diffusion = _get_start_index_for_w_diffusion()

    def initial_run(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
    ):
        """
        Calculate initial diffusion step.

        In ICON at the start of the simulation diffusion is run with a parameter linit = True:

        'For real-data runs, perform an extra diffusion call before the first time
        step because no other filtering of the interpolated velocity field is done'

        This run uses special values for diff_multfac_vn, smag_limit and smag_offset

        """
        diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, allocator=self._allocator)
        smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=self._allocator)

        self.setup_fields_for_initial_step(
            self._params.K4,
            self.config.hdiff_efdt_ratio,
            diff_multfac_vn,
            smag_limit,
        )
        self._do_diffusion_step(
            diagnostic_state, prognostic_state, dtime, diff_multfac_vn, smag_limit, 0.0
        )
        self._sync_cell_fields(prognostic_state)

    def run(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
    ):
        """
        Do one diffusion step within regular time loop.

        runs a diffusion step for the parameter linit=False, within regular time loop.
        """

        self._do_diffusion_step(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
            diff_multfac_vn=self.diff_multfac_vn,
            smag_limit=self.smag_limit,
            smag_offset=self.smag_offset,
        )

    def _sync_cell_fields(self, prognostic_state):
        """
        Communicate theta_v, exner and w.

        communication only done in original code if the following condition applies:
        IF ( linit .OR. (iforcing /= inwp .AND. iforcing /= iaes) ) THEN
        """
        log.debug("communication of prognostic cell fields: theta, w, exner - start")
        self._exchange.exchange_and_wait(
            dims.CellDim,
            prognostic_state.w,
            prognostic_state.theta_v,
            prognostic_state.exner,
        )
        log.debug("communication of prognostic cell fields: theta, w, exner - done")

    @dace_orchestration.orchestrate
    def _do_diffusion_step(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
        diff_multfac_vn: fa.KField[float],
        smag_limit: fa.KField[float],
        smag_offset: float,
    ):
        """
        Run a diffusion step.

        Args:
            diagnostic_state: output argument, data class that contains diagnostic variables
            prognostic_state: output argument, data class that contains prognostic variables
            dtime: the time step,
            diff_multfac_vn:
            smag_limit:
            smag_offset:

        """
        self.scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag)

        log.debug("rbf interpolation 1: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=prognostic_state.vn,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
        )
        log.debug("rbf interpolation 1: end")

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
        log.debug("communication rbf extrapolation of vn - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            wait=True,
        )
        log.debug("communication rbf extrapolation of vn - end")

        log.debug("running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start")
        self.calculate_nabla2_and_smag_coefficients_for_vn(
            diff_multfac_smag=self.diff_multfac_smag,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            vn=prognostic_state.vn,
            smag_limit=smag_limit,
            kh_smag_e=self.kh_smag_e,
            kh_smag_ec=self.kh_smag_ec,
            z_nabla2_e=self.z_nabla2_e,
            smag_offset=smag_offset,
        )
        log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")
        if (
            self.config.shear_type
            >= diffusion_config.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
            or self.config.ltkeshs
        ):
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): start"
            )
            self.calculate_diagnostic_quantities_for_turbulence(
                kh_smag_ec=self.kh_smag_ec,
                vn=prognostic_state.vn,
                diff_multfac_smag=self.diff_multfac_smag,
                div_ic=diagnostic_state.div_ic,
                hdef_ic=diagnostic_state.hdef_ic,
            )
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
            )

        # HALO EXCHANGE  IF (discr_vn > 1) THEN CALL sync_patch_array
        # TODO(halungge): move this up and do asynchronous exchange
        if self.config.type_vn_diffu > 1:
            log.debug("communication rbf extrapolation of z_nable2_e - start")
            self._exchange(self.z_nabla2_e, dim=dims.EdgeDim, wait=True)
            log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("2nd rbf interpolation: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=self.z_nabla2_e,
            ptr_coeff_1=self._interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self._interpolation_state.rbf_coeff_2,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
            horizontal_start=self._vertex_start_lateral_boundary_level_2,
            horizontal_end=self._vertex_end_local,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.connectivities,
        )
        log.debug("2nd rbf interpolation: end")

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
        log.debug("communication rbf extrapolation of z_nable2_e - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            wait=True,
        )
        log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): start")
        self.apply_diffusion_to_vn(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            z_nabla2_e=self.z_nabla2_e,
            kh_smag_e=self.kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            vn=prognostic_state.vn,
        )
        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): end")

        log.debug("communication of prognistic.vn : start")
        handle_edge_comm = self._exchange(prognostic_state.vn, dim=dims.EdgeDim, wait=False)

        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): start"
        )
        # TODO(halungge): get rid of this copying. So far passing an empty buffer instead did not verify?
        self.copy_field(prognostic_state.w, self.w_tmp)

        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
            w_old=self.w_tmp,
            w=prognostic_state.w,
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            diff_multfac_w=self.diff_multfac_w,
            diff_multfac_n2w=self.diff_multfac_n2w,
        )
        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): end"
        )

        if self.config.apply_to_temperature:
            log.debug(
                "running fused stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): start"
            )

            self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
                theta_v=prognostic_state.theta_v,
                kh_smag_e=self.kh_smag_e,
            )
            log.debug(
                "running stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): end"
            )
            log.debug("running stencil 13 to 16 (apply_diffusion_to_theta_and_exner): start")
            self.copy_field(
                prognostic_state.theta_v, self.theta_v_tmp
            )  # TODO(): write in a way that we can avoid the copy

            self.apply_diffusion_to_theta_and_exner(
                kh_smag_e=self.kh_smag_e,
                theta_v_in=self.theta_v_tmp,
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
            )
            log.debug("running stencil 13 to 16 apply_diffusion_to_theta_and_exner: end")

        self.halo_exchange_wait(
            handle_edge_comm
        )  # need to do this here, since we currently only use 1 communication object.
        log.debug("communication of prognogistic.vn - end")

    # TODO(kotsaloscv): It is unsafe to set it as cached property -demands more testing-
    def orchestration_uid(self) -> str:
        """Unique id based on the runtime state of the Diffusion object. It is used for caching in DaCe Orchestration."""
        members_to_disregard = [
            "_allocator",
            "_exchange",
            "_grid",
            "compile_time_connectivities",
            *[
                name
                for name in self.__dict__
                if isinstance(self.__dict__[name], gtx_typing.Program)
            ],
        ]
        return dace_orchestration.generate_orchestration_uid(
            self, members_to_disregard=members_to_disregard
        )
