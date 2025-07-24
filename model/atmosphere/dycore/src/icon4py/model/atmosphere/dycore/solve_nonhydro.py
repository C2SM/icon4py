# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: ERA001, B008

import logging
import dataclasses
from typing import Final, Optional

import gt4py.next as gtx
from gt4py.next import backend as gtx_backend

import icon4py.model.atmosphere.dycore.solve_nonhydro_stencils as nhsolve_stencils
import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.common.utils import data_allocation as data_alloc

from icon4py.model.common import constants
from icon4py.model.atmosphere.dycore.stencils import (
    compute_cell_diagnostics_for_dycore,
    compute_edge_diagnostics_for_dycore_and_update_vn,
    vertically_implicit_dycore_solver,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn import compute_avg_vn
from icon4py.model.atmosphere.dycore.stencils.compute_avg_vn_and_graddiv_vn_and_vt import (
    compute_avg_vn_and_graddiv_vn_and_vt,
)
from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_exner_from_rhotheta import (
    compute_exner_from_rhotheta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_mass_flux import compute_mass_flux
from icon4py.model.atmosphere.dycore.stencils.compute_theta_and_exner import (
    compute_theta_and_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_vn_on_lateral_boundary import (
    compute_vn_on_lateral_boundary,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.stencils.init_two_edge_kdim_fields_with_zero_wp import (
    init_two_edge_kdim_fields_with_zero_wp,
)
from icon4py.model.atmosphere.dycore import (
    dycore_states,
    dycore_utils,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_flux_weighted import (
    update_mass_flux_weighted,
)
from icon4py.model.atmosphere.dycore.stencils.update_theta_v import update_theta_v
from icon4py.model.atmosphere.dycore.velocity_advection import (
    VelocityAdvection,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common import dimension as dims, model_options
from icon4py.model.common.grid import (
    base as grid_def,
    horizontal as h_grid,
    vertical as v_grid,
    icon as icon_grid,
)
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common import field_type_aliases as fa, type_alias as ta


# flake8: noqa
log = logging.getLogger(__name__)


@dataclasses.dataclass
class IntermediateFields:
    """
    Encapsulate internal fields of SolveNonHydro that contain shared state over predictor and corrector step.

    Encapsulates internal fields used in SolveNonHydro. Fields (and the class!)
    follow the naming convention of ICON to prepend local fields of a module with z_. Contrary to
    other such z_ fields inside SolveNonHydro the fields in this dataclass
    contain state that is built up over the predictor and corrector part in a timestep.
    """

    horizontal_pressure_gradient: fa.EdgeKField[float]
    """
    Declared as z_gradh_exner in ICON.
    """
    tridiagonal_alpha_coeff_at_cells_on_half_levels: fa.CellKField[
        ta.vpfloat
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as z_alpha in ICON.
    """
    tridiagonal_beta_coeff_at_cells_on_model_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as z_beta in ICON.
    """
    exner_explicit_term: fa.CellKField[ta.wpfloat]
    """
    Declared as z_exner_expl in ICON.
    """
    vertical_mass_flux_at_cells_on_half_levels: fa.EdgeKField[
        ta.wpfloat
    ]  # TODO: change this back to KHalfDim, but how do we treat it wrt to field_operators and domain?
    """
    Declared as z_contr_w_fl_l in ICON.
    """
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat]
    """
    Declared as z_rho_e in ICON.
    """
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat]
    """
    Declared as z_theta_v_e in ICON.
    """
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat]
    """
    Declared as z_kin_hor_e in ICON.
    """
    tangential_wind_on_half_levels: fa.EdgeKField[ta.vpfloat]
    """
    Declared as z_vt_ie in ICON. Tangential wind at edge on k-half levels. NOTE THAT IT ONLY HAS nlev LEVELS because it is only used for computing horizontal advection of w and thus level nlevp1 is not needed because w[nlevp1-1] is diagnostic.
    """
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat]
    """
    Declared as z_graddiv_vn in ICON.
    """
    rho_explicit_term: fa.CellKField[ta.wpfloat]
    """
    Declared as z_rho_expl in ICON.
    """
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat]
    """
    Declared as z_dwdz_dd in ICON.
    """

    @classmethod
    def allocate(
        cls,
        grid: grid_def.BaseGrid,
        backend: Optional[gtx_backend.Backend] = None,
    ):
        return IntermediateFields(
            horizontal_pressure_gradient=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
            tridiagonal_alpha_coeff_at_cells_on_half_levels=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
            ),
            tridiagonal_beta_coeff_at_cells_on_model_levels=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, backend=backend
            ),
            exner_explicit_term=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, backend=backend
            ),
            vertical_mass_flux_at_cells_on_half_levels=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
            ),
            rho_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
            theta_v_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
            horizontal_gradient_of_normal_wind_divergence=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
            rho_explicit_term=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, backend=backend),
            dwdz_at_cells_on_model_levels=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, backend=backend
            ),
            horizontal_kinetic_energy_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
            tangential_wind_on_half_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, backend=backend
            ),
        )


class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    TODO: (magdalena) values should be read from a configuration file.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    def __init__(
        self,
        itime_scheme: dycore_states.TimeSteppingScheme = dycore_states.TimeSteppingScheme.MOST_EFFICIENT,
        iadv_rhotheta: dycore_states.RhoThetaAdvectionType = dycore_states.RhoThetaAdvectionType.MIURA,
        igradp_method: dycore_states.HorizontalPressureDiscretizationType = dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO,
        ndyn_substeps_var: float = 5.0,
        rayleigh_type: model_options.RayleighType = model_options.RayleighType.KLEMP,
        rayleigh_coeff: float = 0.05,
        divdamp_order: dycore_states.DivergenceDampingOrder = dycore_states.DivergenceDampingOrder.COMBINED,  # the ICON default is 4,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
        divdamp_type: dycore_states.DivergenceDampingType = dycore_states.DivergenceDampingType.THREE_DIMENSIONAL,
        divdamp_trans_start: float = 12500.0,
        divdamp_trans_end: float = 17500.0,
        l_vert_nested: bool = False,
        rhotheta_offctr: float = -0.1,
        veladv_offctr: float = 0.25,
        _nudge_max_coeff: float = None,  # default is set in __init__
        max_nudging_coefficient: float = None,  # default is set in __init__
        fourth_order_divdamp_factor: float = 0.0025,
        fourth_order_divdamp_factor2: float = 0.004,
        fourth_order_divdamp_factor3: float = 0.004,
        fourth_order_divdamp_factor4: float = 0.004,
        fourth_order_divdamp_z: float = 32500.0,
        fourth_order_divdamp_z2: float = 40000.0,
        fourth_order_divdamp_z3: float = 60000.0,
        fourth_order_divdamp_z4: float = 80000.0,
    ):
        # parameters from namelist diffusion_nml
        self.itime_scheme: int = itime_scheme

        #: Miura scheme for advection of rho and theta
        self.iadv_rhotheta: dycore_states.RhoThetaAdvectionType = iadv_rhotheta
        #: Use truly horizontal pressure-gradient computation to ensure numerical
        #: stability without heavy orography smoothing
        self.igradp_method: dycore_states.HorizontalPressureDiscretizationType = igradp_method

        #: number of dynamics substeps per fast-physics timestep
        self.ndyn_substeps_var: float = ndyn_substeps_var

        #: type of Rayleigh damping
        self.rayleigh_type: constants.RayleighType = rayleigh_type
        # used for calculation of rayleigh_w, rayleigh_vn in mo_vertical_grid.f90
        self.rayleigh_coeff: float = rayleigh_coeff

        #: order of divergence damping
        self.divdamp_order: dycore_states.DivergenceDampingOrder = divdamp_order

        #: type of divergence damping
        self.divdamp_type: dycore_states.DivergenceDampingType = divdamp_type
        #: Lower and upper bound of transition zone between 2D and 3D divergence damping in case of divdamp_type = 32 [m]
        self.divdamp_trans_start: float = divdamp_trans_start
        self.divdamp_trans_end: float = divdamp_trans_end

        #: off-centering for density and potential temperature at interface levels.
        #: Specifying a negative value here reduces the amount of vertical
        #: wind off-centering needed for stability of sound waves.
        self.rhotheta_offctr: float = rhotheta_offctr

        #: off-centering of velocity advection in corrector step
        self.veladv_offctr: float = veladv_offctr

        #: scaling factor for divergence damping
        self.fourth_order_divdamp_factor: float = fourth_order_divdamp_factor
        """
        Declared as divdamp_fac in ICON. It is a scaling factor for fourth order divergence damping between
        heights of fourth_order_divdamp_z and fourth_order_divdamp_z2.
        """
        self.fourth_order_divdamp_factor2: float = fourth_order_divdamp_factor2
        """
        Declared as divdamp_fac2 in ICON. It is a scaling factor for fourth order divergence damping between
        heights of fourth_order_divdamp_z and fourth_order_divdamp_z2. Divergence damping factor reaches
        fourth_order_divdamp_factor2 at fourth_order_divdamp_z2.
        """
        self.fourth_order_divdamp_factor3: float = fourth_order_divdamp_factor3
        """
        Declared as divdamp_fac3 in ICON. It is a scaling factor to determine the quadratic vertical
        profile of fourth order divergence damping factor between heights of fourth_order_divdamp_z2
        and fourth_order_divdamp_z4.
        """
        self.fourth_order_divdamp_factor4: float = fourth_order_divdamp_factor4
        """
        Declared as divdamp_fac4 in ICON. It is a scaling factor to determine the quadratic vertical
        profile of fourth order divergence damping factor between heights of fourth_order_divdamp_z2
        and fourth_order_divdamp_z4. Divergence damping factor reaches fourth_order_divdamp_factor4
        at fourth_order_divdamp_z4.
        """
        self.fourth_order_divdamp_z: float = fourth_order_divdamp_z
        """
        Declared as divdamp_z in ICON. The upper limit in height where divergence damping factor is a constant.
        """
        self.fourth_order_divdamp_z2: float = fourth_order_divdamp_z2
        """
        Declared as divdamp_z2 in ICON. The upper limit in height above fourth_order_divdamp_z where divergence
        damping factor decreases as a linear function of height.
        """
        self.fourth_order_divdamp_z3: float = fourth_order_divdamp_z3
        """
        Declared as divdamp_z3 in ICON. Am intermediate height between fourth_order_divdamp_z2 and
        fourth_order_divdamp_z4 where divergence damping factor decreases quadratically with height.
        """
        self.fourth_order_divdamp_z4: float = fourth_order_divdamp_z4
        """
        Declared as divdamp_z4 in ICON. The upper limit in height where divergence damping factor decreases
        quadratically with height.
        """

        #: parameters from other namelists:

        #: from mo_interpol_nml.f90

        #: Parameter describing the lateral boundary nudging in limited area mode.
        #:
        #: Maximal value of the nudging coefficients used cell row bordering the boundary interpolation zone,
        #: from there nudging coefficients decay exponentially with `nudge_efold_width` in units of cell rows.
        #: Called 'nudge_max_coeff' in mo_interpol_nml.f90.
        #: Note: The user can pass the ICON namelist paramter `nudge_max_coeff` as `_nudge_max_coeff` or
        #: the properly scaled one as `max_nudging_coefficient`,
        #: see the comment in mo_interpol_nml.f90
        #: TODO: This code is duplicated in `diffusion.py`, clean this up when implementing proper configuration handling.
        if _nudge_max_coeff is not None and max_nudging_coefficient is not None:
            raise ValueError(
                "Cannot set both '_max_nudging_coefficient' and 'scaled_max_nudging_coefficient'."
            )
        elif max_nudging_coefficient is not None:
            self.max_nudging_coefficient: float = max_nudging_coefficient
        elif _nudge_max_coeff is not None:
            self.max_nudging_coefficient: float = (
                constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * _nudge_max_coeff
            )
        else:  # default value in ICON
            self.max_nudging_coefficient: float = (
                constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * 0.02
            )

        #: from mo_run_nml.f90
        #: use vertical nesting
        self.l_vert_nested: bool = l_vert_nested

        #: from mo_initicon_nml.f90/ mo_initicon_config.f90
        #: whether IAU is active at current time
        self.is_iau_active: bool = is_iau_active
        #: IAU weight for dynamics fields
        self.iau_wgt_dyn: float = iau_wgt_dyn

        self._validate()

    def _validate(self):
        """Apply consistency checks and validation on configuration parameters."""

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.igradp_method != dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != dycore_states.TimeSteppingScheme.MOST_EFFICIENT:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.divdamp_order != dycore_states.DivergenceDampingOrder.COMBINED:
            raise NotImplementedError("divdamp_order can only be 24")

        if self.divdamp_type == dycore_states.DivergenceDampingType.TWO_DIMENSIONAL:
            raise NotImplementedError(
                "`DivergenceDampingType.TWO_DIMENSIONAL` (2) is not yet implemented"
            )


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):
        #: Weighting coefficients for velocity advection if tendency averaging is used
        #: The off-centering specified here turned out to be beneficial to numerical
        #: stability in extreme situations
        self.advection_explicit_weight_parameter: Final[float] = 0.5 - config.veladv_offctr
        """
        Declared as wgt_nnow_vel in ICON.
        """
        self.advection_implicit_weight_parameter: Final[float] = 0.5 + config.veladv_offctr
        """
        Declared as wgt_nnew_vel in ICON.
        """

        #: Weighting coefficients for rho and theta at interface levels in the corrector step
        #: This empirically determined weighting minimizes the vertical wind off-centering
        #: needed for numerical stability of vertical sound wave propagation
        self.rhotheta_implicit_weight_parameter: Final[float] = 0.5 + config.rhotheta_offctr
        """
        Declared as wgt_nnew_rth in ICON.
        """
        self.rhotheta_explicit_weight_parameter: Final[float] = (
            1.0 - self.rhotheta_implicit_weight_parameter
        )
        """
        Declared as wgt_nnow_rth in ICON.
        """


class SolveNonhydro:
    def __init__(
        self,
        grid: icon_grid.IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_geometry: grid_states.EdgeParams,
        cell_geometry: grid_states.CellParams,
        owner_mask: fa.CellField[bool],
        backend: Optional[gtx_backend.Backend],
        exchange: decomposition.ExchangeRuntime = decomposition.SingleNodeExchange(),
    ):
        self._exchange = exchange
        self._backend = backend

        self._grid = grid
        self._config = config
        self._params = params
        self._metric_state_nonhydro = metric_state_nonhydro
        self._interpolation_state = interpolation_state
        self._vertical_params = vertical_params
        self._edge_geometry = edge_geometry
        self._cell_params = cell_geometry

        self._compute_theta_and_exner = compute_theta_and_exner.with_backend(self._backend).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._compute_exner_from_rhotheta = compute_exner_from_rhotheta.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._update_theta_v = update_theta_v.with_backend(self._backend).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl = (
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(
                self._backend
            ).compile(
                enable_jit=False,
                vertical_start=[gtx.int32(0)],
                vertical_end=[gtx.int32(self._grid.num_levels)],
                offset_provider=self._grid.connectivities,
            )
        )
        self._init_two_edge_kdim_fields_with_zero_wp = (
            init_two_edge_kdim_fields_with_zero_wp.with_backend(self._backend).compile(
                enable_jit=False,
                vertical_start=[gtx.int32(0)],
                vertical_end=[gtx.int32(self._grid.num_levels)],
                offset_provider={},
            )
        )
        self._compute_hydrostatic_correction_term = (
            compute_hydrostatic_correction_term.with_backend(self._backend).compile(
                enable_jit=False,
                vertical_start=[gtx.int32(self._grid.num_levels - 1)],
                vertical_end=[gtx.int32(self._grid.num_levels)],
                offset_provider=self._grid.connectivities,
            )
        )
        self._compute_theta_rho_face_values_and_pressure_gradient_and_update_vn = compute_edge_diagnostics_for_dycore_and_update_vn.compute_theta_rho_face_values_and_pressure_gradient_and_update_vn.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            iau_wgt_dyn=[self._config.iau_wgt_dyn],
            is_iau_active=[self._config.is_iau_active],
            limited_area=[self._grid.limited_area],
            iadv_rhotheta=[self._config.iadv_rhotheta],
            igradp_method=[self._config.igradp_method],
            nflatlev=[self._vertical_params.nflatlev],
            nflat_gradp=[self._vertical_params.nflat_gradp],
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._apply_divergence_damping_and_update_vn = compute_edge_diagnostics_for_dycore_and_update_vn.apply_divergence_damping_and_update_vn.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            iau_wgt_dyn=[self._config.iau_wgt_dyn],
            is_iau_active=[self._config.is_iau_active],
            limited_area=[self._grid.limited_area],
            divdamp_order=[self._config.divdamp_order],
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._compute_vn_on_lateral_boundary = compute_vn_on_lateral_boundary.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._compute_avg_vn_and_graddiv_vn_and_vt = (
            compute_avg_vn_and_graddiv_vn_and_vt.with_backend(self._backend)
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._compute_mass_flux = compute_mass_flux.with_backend(self._backend).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._vertically_implicit_solver_at_predictor_step = vertically_implicit_dycore_solver.vertically_implicit_solver_at_predictor_step.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            iau_wgt_dyn=[self._config.iau_wgt_dyn],
            is_iau_active=[self._config.is_iau_active],
            rayleigh_type=[self._config.rayleigh_type],
            divdamp_type=[self._config.divdamp_type],
            end_index_of_damping_layer=[self._vertical_params.end_index_of_damping_layer],
            kstart_moist=[self._vertical_params.kstart_moist],
            flat_level_index_plus1=[gtx.int32(self._vertical_params.nflatlev + 1)],
            vertical_start_index_model_top=[gtx.int32(0)],
            vertical_end_index_model_surface=[gtx.int32(self._grid.num_levels + 1)],
            offset_provider=[self._grid.connectivities],
        )
        self._vertically_implicit_solver_at_corrector_step = vertically_implicit_dycore_solver.vertically_implicit_solver_at_corrector_step.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            ndyn_substeps_var=[float(self._config.ndyn_substeps_var)],
            iau_wgt_dyn=[self._config.iau_wgt_dyn],
            is_iau_active=[self._config.is_iau_active],
            rayleigh_type=[self._config.rayleigh_type],
            end_index_of_damping_layer=[self._vertical_params.end_index_of_damping_layer],
            kstart_moist=[self._vertical_params.kstart_moist],
            vertical_start_index_model_top=[gtx.int32(0)],
            vertical_end_index_model_surface=[gtx.int32(self._grid.num_levels + 1)],
            offset_provider=[self._grid.connectivities],
        )
        self._compute_dwdz_for_divergence_damping = (
            compute_dwdz_for_divergence_damping.with_backend(self._backend)
        ).compile(
            enable_jit=False,
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._compute_avg_vn = compute_avg_vn.with_backend(self._backend).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._accumulate_prep_adv_fields = accumulate_prep_adv_fields.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )
        self._init_cell_kdim_field_with_zero_wp = init_cell_kdim_field_with_zero_wp.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels + 1)],
            offset_provider={},
        )
        self._update_mass_flux_weighted = update_mass_flux_weighted.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider={},
        )

        self._compute_perturbed_quantities_and_interpolation = compute_cell_diagnostics_for_dycore.compute_perturbed_quantities_and_interpolation.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            nflatlev=[self._vertical_params.nflatlev],
            nflat_gradp=[self._vertical_params.nflat_gradp],
            limited_area=[self._grid.limited_area],
            igradp_method=[self._config.igradp_method],
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels + 1)],
            offset_provider=self._grid.connectivities,
        )

        self._interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration = compute_cell_diagnostics_for_dycore.interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(1)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._predictor_stencils_35_36 = nhsolve_stencils.predictor_stencils_35_36.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            nflatlev_startindex=[self._vertical_params.nflatlev],
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels)],
            offset_provider=self._grid.connectivities,
        )
        self._predictor_stencils_37_38 = nhsolve_stencils.predictor_stencils_37_38.with_backend(
            self._backend
        ).compile(
            enable_jit=False,
            vertical_start=[0],
            vertical_end=[self._grid.num_levels + 1],
            offset_provider=self._grid.connectivities,
        )
        self._stencils_61_62 = nhsolve_stencils.stencils_61_62.with_backend(self._backend).compile(
            enable_jit=False,
            vertical_start=[gtx.int32(0)],
            vertical_end=[gtx.int32(self._grid.num_levels + 1)],
            offset_provider={},
        )
        self._en_smag_fac_for_zero_nshift = smagorinsky.en_smag_fac_for_zero_nshift.with_backend(
            self._backend
        ).compile(enable_jit=False, offset_provider={"Koff": dims.KDim})
        self._init_test_fields = nhsolve_stencils.init_test_fields.with_backend(
            self._backend
        ).compile(enable_jit=False, offset_provider={})

        self.velocity_advection = VelocityAdvection(
            grid,
            metric_state_nonhydro,
            interpolation_state,
            vertical_params,
            edge_geometry,
            owner_mask,
            backend=self._backend,
        )
        self._allocate_local_fields()
        self._determine_local_domains()

        self._en_smag_fac_for_zero_nshift(
            self._vertical_params.interface_physical_height,
            self._config.fourth_order_divdamp_factor,
            self._config.fourth_order_divdamp_factor2,
            self._config.fourth_order_divdamp_factor3,
            self._config.fourth_order_divdamp_factor4,
            self._config.fourth_order_divdamp_z,
            self._config.fourth_order_divdamp_z2,
            self._config.fourth_order_divdamp_z3,
            self._config.fourth_order_divdamp_z4,
            self.interpolated_fourth_order_divdamp_factor,
            offset_provider={"Koff": dims.KDim},
        )

        self.p_test_run = True

    def _allocate_local_fields(self):
        self.temporal_extrapolation_of_perturbed_exner = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            backend=self._backend,
        )
        """
        Declared as z_exner_ex_pr in ICON.
        """
        self.exner_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            backend=self._backend,
        )
        """
        Declared as z_exner_ic in ICON.
        """
        self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
            data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
            )
        )
        """
        Declared as z_dexner_dz_c_1 in ICON.
        """
        self.perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            backend=self._backend,
        )

        """
        Declared as z_theta_v_pr_ic in ICON.
        """
        self.pressure_buoyancy_acceleration_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
        )
        """
        Declared as z_th_ddz_exner_c in ICON. theta' dpi0/dz + theta (1 - eta_impl) dpi'/dz.
        It represents the vertical pressure gradient and buoyancy acceleration.
        Note that it only has nlev because it is only used in computation of the explicit
        term for updating w, and w at model top/bottom is diagnosed.
        """
        self.perturbed_rho_at_cells_on_model_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
        )
        """
        Declared as z_rth_pr_1 in ICON.
        """
        self.perturbed_theta_v_at_cells_on_model_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
        )
        """
        Declared as z_rth_pr_2 in ICON.
        """
        self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
            data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
            )
        )
        """
        Declared as z_dexner_dz_c_2 in ICON.
        """
        self.z_vn_avg = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.theta_v_flux_at_edges_on_model_levels = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        """
        Declared as z_theta_v_fl_e in ICON.
        """
        self.z_rho_v = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.z_theta_v_v = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        self.k_field = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, backend=self._backend
        )
        self.edge_field = data_alloc.index_field(self._grid, dims.EdgeDim, backend=self._backend)
        self._contravariant_correction_at_edges_on_model_levels = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
        )
        """
        Declared as z_w_concorr_me in ICON. vn dz/dn + vt dz/dt, z is topography height
        """
        self.hydrostatic_correction = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, backend=self._backend
        )
        """
        Declared as z_hydro_corr in ICON. Used for computation of horizontal pressure gradient over steep slope.
        """
        self.rayleigh_damping_factor = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        """
        Declared as z_raylfac in ICON.
        """
        self.interpolated_fourth_order_divdamp_factor = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        """
        Declared as enh_divdamp_fac in ICON.
        """
        self.reduced_fourth_order_divdamp_coeff_at_nest_boundary = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        """
        Declared as bdy_divdamp in ICON.
        """
        self.fourth_order_divdamp_scaling_coeff = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, backend=self._backend
        )
        """
        Declared as scal_divdamp in ICON.
        """
        self.intermediate_fields = IntermediateFields.allocate(
            grid=self._grid, backend=self._backend
        )

    def _determine_local_domains(self):
        vertex_domain = h_grid.domain(dims.VertexDim)
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        edge_halo_level_2 = edge_domain(h_grid.Zone.HALO_LEVEL_2)

        self._start_cell_lateral_boundary = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._start_cell_local = self._grid.start_index(cell_domain(h_grid.Zone.LOCAL))
        self._start_cell_halo = self._grid.start_index(cell_domain(h_grid.Zone.HALO))
        self._start_cell_halo_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.HALO_LEVEL_2)
        )

        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._end_cell_halo_level_2 = self._grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        self._start_edge_lateral_boundary = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._start_edge_halo_level_2 = self._grid.start_index(edge_halo_level_2)

        self._end_edge_nudging = self._grid.end_index(edge_domain(h_grid.Zone.NUDGING))
        self._end_edge_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self._grid.end_index(edge_halo_level_2)
        self._end_edge_end = self._grid.end_index(edge_domain(h_grid.Zone.END))

        self._start_vertex_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self._grid.end_index(vertex_domain(h_grid.Zone.HALO))

    def time_step(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        dtime: float,
        at_initial_timestep: bool,
        lprep_adv: bool,
        at_first_substep: bool,
        at_last_substep: bool,
    ):
        """
        Update prognostic variables (prognostic_states.next) after the dynamical process over one substep.
        Args:
            diagnostic_state_nh: diagnostic variables used for solving the governing equations. It includes local variables and the physics tendency term that comes from physics
            prognostic_states: prognostic variables
            prep_adv: variables for tracer advection
            second_order_divdamp_factor: Originally declared as divdamp_fac_o2 in ICON. Second order (nabla2) divergence damping coefficient.
            dtime: time step
            at_initial_timestep: initial time step of the model run
            lprep_adv: Preparation for tracer advection
            at_first_substep: first substep
            at_last_substep: last substep
        """
        log.info(
            f"running timestep: dtime = {dtime}, initial_timestep = {at_initial_timestep}, first_substep = {at_first_substep}, last_substep = {at_last_substep}, prep_adv = {lprep_adv}"
        )

        if self.p_test_run:
            self._init_test_fields(
                self.intermediate_fields.rho_at_edges_on_model_levels,
                self.intermediate_fields.theta_v_at_edges_on_model_levels,
                self.intermediate_fields.dwdz_at_cells_on_model_levels,
                self.intermediate_fields.horizontal_gradient_of_normal_wind_divergence,
                self._start_edge_lateral_boundary,
                self._end_edge_local,
                self._start_cell_lateral_boundary,
                self._end_cell_end,
                vertical_start=gtx.int32(0),
                vertical_end=self._grid.num_levels,
            )

        self.run_predictor_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            z_fields=self.intermediate_fields,
            dtime=dtime,
            at_initial_timestep=at_initial_timestep,
            at_first_substep=at_first_substep,
        )

        self.run_corrector_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            z_fields=self.intermediate_fields,
            prep_adv=prep_adv,
            second_order_divdamp_factor=second_order_divdamp_factor,
            dtime=dtime,
            lprep_adv=lprep_adv,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
        )

        if self._grid.limited_area:
            self._compute_theta_and_exner(
                bdy_halo_c=self._metric_state_nonhydro.bdy_halo_c,
                rho=prognostic_states.next.rho,
                theta_v=prognostic_states.next.theta_v,
                exner=prognostic_states.next.exner,
                rd_o_cvd=constants.RD_O_CVD,
                rd_o_p0ref=constants.RD_O_P0REF,
                horizontal_start=self._start_cell_local,
                horizontal_end=self._end_cell_end,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
            )

            self._compute_exner_from_rhotheta(
                rho=prognostic_states.next.rho,
                theta_v=prognostic_states.next.theta_v,
                exner=prognostic_states.next.exner,
                rd_o_cvd=constants.RD_O_CVD,
                rd_o_p0ref=constants.RD_O_P0REF,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
            )

        self._update_theta_v(
            mask_prog_halo_c=self._metric_state_nonhydro.mask_prog_halo_c,
            rho_now=prognostic_states.current.rho,
            theta_v_now=prognostic_states.current.theta_v,
            exner_new=prognostic_states.next.exner,
            exner_now=prognostic_states.current.exner,
            rho_new=prognostic_states.next.rho,
            theta_v_new=prognostic_states.next.theta_v,
            horizontal_start=self._start_cell_halo,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
        )

    # flake8: noqa: C901
    def run_predictor_step(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        dtime: float,
        at_initial_timestep: bool,
        at_first_substep: bool,
    ):
        """
        Runs the predictor step of the non-hydrostatic solver.
        """

        log.info(
            f"running predictor step: dtime = {dtime}, initial_timestep = {at_initial_timestep} at_first_substep = {at_first_substep}"
        )

        if at_first_substep:
            # Recompute only vn tendency
            skip_compute_predictor_vertical_advection: bool = (
                self._config.itime_scheme == dycore_states.TimeSteppingScheme.MOST_EFFICIENT
                and not (at_initial_timestep and at_first_substep)
            )

            self.velocity_advection.run_predictor_step(
                skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
                diagnostic_state=diagnostic_state_nh,
                prognostic_state=prognostic_states.current,
                contravariant_correction_at_edges_on_model_levels=self._contravariant_correction_at_edges_on_model_levels,
                horizontal_kinetic_energy_at_edges_on_model_levels=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
                tangential_wind_on_half_levels=z_fields.tangential_wind_on_half_levels,
                dtime=dtime,
                cell_areas=self._cell_params.area,
            )

        #  Precompute Rayleigh damping factor
        dycore_utils._compute_rayleigh_damping_factor(
            rayleigh_w=self._metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            out=self.rayleigh_damping_factor,
        )

        self._compute_perturbed_quantities_and_interpolation(
            temporal_extrapolation_of_perturbed_exner=self.temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels=self.exner_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels=self.perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=self.perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=self.perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            current_rho=prognostic_states.current.rho,
            reference_rho_at_cells_on_model_levels=self._metric_state_nonhydro.reference_rho_at_cells_on_model_levels,
            current_theta_v=prognostic_states.current.theta_v,
            reference_theta_at_cells_on_model_levels=self._metric_state_nonhydro.reference_theta_at_cells_on_model_levels,
            reference_theta_at_cells_on_half_levels=self._metric_state_nonhydro.reference_theta_at_cells_on_half_levels,
            wgtfacq_c=self._metric_state_nonhydro.wgtfacq_c,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            exner_w_explicit_weight_parameter=self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
            ddz_of_reference_exner_at_cells_on_half_levels=self._metric_state_nonhydro.ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            pressure_buoyancy_acceleration_at_cells_on_half_levels=self.pressure_buoyancy_acceleration_at_cells_on_half_levels,
            time_extrapolation_parameter_for_exner=self._metric_state_nonhydro.time_extrapolation_parameter_for_exner,
            current_exner=prognostic_states.current.exner,
            reference_exner_at_cells_on_model_levels=self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            d2dexdz2_fac1_mc=self._metric_state_nonhydro.d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=self._metric_state_nonhydro.d2dexdz2_fac2_mc,
            limited_area=self._grid.limited_area,
            igradp_method=self._config.igradp_method,
            nflatlev=self._vertical_params.nflatlev,
            nflat_gradp=self._vertical_params.nflat_gradp,
            start_cell_lateral_boundary=self._start_cell_lateral_boundary,
            start_cell_lateral_boundary_level_3=self._start_cell_lateral_boundary_level_3,
            start_cell_halo_level_2=self._start_cell_halo_level_2,
            end_cell_end=self._end_cell_end,
            end_cell_halo=self._end_cell_halo,
            end_cell_halo_level_2=self._end_cell_halo_level_2,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(self._grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self._grid.num_levels + 1),
            offset_provider=self._grid.connectivities,
        )

        # Compute rho and theta at edges for horizontal flux divergence term
        if self._config.iadv_rhotheta == dycore_states.RhoThetaAdvectionType.SIMPLE:
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_states.current.rho,
                c_intp=self._interpolation_state.c_intp,
                p_vert_out=self.z_rho_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self._grid.num_levels,  # UBOUND(p_cell_in,2)
                offset_provider=self._grid.connectivities,
            )
            self._mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
                p_cell_in=prognostic_states.current.theta_v,
                c_intp=self._interpolation_state.c_intp,
                p_vert_out=self.z_theta_v_v,
                horizontal_start=self._start_vertex_lateral_boundary_level_2,
                horizontal_end=self._end_vertex_halo,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.connectivities,
            )

        log.debug(
            f"predictor: start stencil compute_theta_rho_face_values_and_pressure_gradient_and_update_vn"
        )
        if (
            self._config.igradp_method
            == dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
        ):
            self._compute_hydrostatic_correction_term(
                theta_v=prognostic_states.current.theta_v,
                ikoffset=self._metric_state_nonhydro.vertoffset_gradp,
                zdiff_gradp=self._metric_state_nonhydro.zdiff_gradp,
                theta_v_ic=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
                z_hydro_corr=self.hydrostatic_correction,
                grav_o_cpd=constants.GRAV_O_CPD,
                horizontal_start=self._start_edge_nudging_level_2,
                horizontal_end=self._end_edge_local,
                vertical_start=self._grid.num_levels - 1,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.connectivities,
            )
            lowest_level = self._grid.num_levels - 1
            hydrostatic_correction_on_lowest_level = gtx.as_field(
                (dims.EdgeDim,),
                self.hydrostatic_correction.ndarray[:, lowest_level],
                allocator=self._backend.allocator,
            )
        self._compute_theta_rho_face_values_and_pressure_gradient_and_update_vn(
            rho_at_edges_on_model_levels=z_fields.rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=z_fields.horizontal_pressure_gradient,
            next_vn=prognostic_states.next.vn,
            current_vn=prognostic_states.current.vn,
            tangential_wind=diagnostic_state_nh.tangential_wind,
            reference_rho_at_edges_on_model_levels=self._metric_state_nonhydro.reference_rho_at_edges_on_model_levels,
            reference_theta_at_edges_on_model_levels=self._metric_state_nonhydro.reference_theta_at_edges_on_model_levels,
            perturbed_rho_at_cells_on_model_levels=self.perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=self.perturbed_theta_v_at_cells_on_model_levels,
            temporal_extrapolation_of_perturbed_exner=self.temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
            predictor_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.predictor,
            normal_wind_tendency_due_to_slow_physics_process=diagnostic_state_nh.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diagnostic_state_nh.normal_wind_iau_increment,
            geofac_grg_x=self._interpolation_state.geofac_grg_x,
            geofac_grg_y=self._interpolation_state.geofac_grg_y,
            pos_on_tplane_e_x=self._interpolation_state.pos_on_tplane_e_1,
            pos_on_tplane_e_y=self._interpolation_state.pos_on_tplane_e_2,
            primal_normal_cell_x=self._edge_geometry.primal_normal_cell[0],
            dual_normal_cell_x=self._edge_geometry.dual_normal_cell[0],
            primal_normal_cell_y=self._edge_geometry.primal_normal_cell[1],
            dual_normal_cell_y=self._edge_geometry.dual_normal_cell[1],
            ddxn_z_full=self._metric_state_nonhydro.ddxn_z_full,
            c_lin_e=self._interpolation_state.c_lin_e,
            ikoffset=self._metric_state_nonhydro.vertoffset_gradp,
            zdiff_gradp=self._metric_state_nonhydro.zdiff_gradp,
            ipeidx_dsl=self._metric_state_nonhydro.pg_edgeidx_dsl,
            pg_exdist=self._metric_state_nonhydro.pg_exdist,
            inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
            dtime=dtime,
            iau_wgt_dyn=self._config.iau_wgt_dyn,
            is_iau_active=self._config.is_iau_active,
            limited_area=self._grid.limited_area,
            iadv_rhotheta=self._config.iadv_rhotheta,
            igradp_method=self._config.igradp_method,
            nflatlev=self._vertical_params.nflatlev,
            nflat_gradp=self._vertical_params.nflat_gradp,
            start_edge_halo_level_2=self._start_edge_halo_level_2,
            end_edge_halo_level_2=self._end_edge_halo_level_2,
            start_edge_lateral_boundary=self._start_edge_lateral_boundary,
            end_edge_halo=self._end_edge_halo,
            start_edge_lateral_boundary_level_7=self._start_edge_lateral_boundary_level_7,
            start_edge_nudging_level_2=self._start_edge_nudging_level_2,
            end_edge_local=self._end_edge_local,
            end_edge_end=self._end_edge_end,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(self._grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self._grid.num_levels),
            offset_provider=self._grid.connectivities,
        )

        if self._grid.limited_area:
            self._compute_vn_on_lateral_boundary(
                grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
                vn_now=prognostic_states.current.vn,
                vn_new=prognostic_states.next.vn,
                dtime=dtime,
                horizontal_start=self._start_edge_lateral_boundary,
                horizontal_end=self._end_edge_nudging,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
            )
        log.debug("exchanging prognostic field 'vn' and local field 'rho_at_edges_on_model_levels'")
        self._exchange.exchange_and_wait(
            dims.EdgeDim, prognostic_states.next.vn, z_fields.rho_at_edges_on_model_levels
        )

        self._compute_avg_vn_and_graddiv_vn_and_vt(
            e_flx_avg=self._interpolation_state.e_flx_avg,
            vn=prognostic_states.next.vn,
            geofac_grdiv=self._interpolation_state.geofac_grdiv,
            rbf_vec_coeff_e=self._interpolation_state.rbf_vec_coeff_e,
            z_vn_avg=self.z_vn_avg,
            z_graddiv_vn=z_fields.horizontal_gradient_of_normal_wind_divergence,
            vt=diagnostic_state_nh.tangential_wind,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.connectivities,
        )

        self._compute_mass_flux(
            z_rho_e=z_fields.rho_at_edges_on_model_levels,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self._metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.theta_v_at_edges_on_model_levels,
            mass_fl_e=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            z_theta_v_fl_e=self.theta_v_flux_at_edges_on_model_levels,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
        )

        self._predictor_stencils_35_36(
            vn=prognostic_states.next.vn,
            ddxn_z_full=self._metric_state_nonhydro.ddxn_z_full,
            ddxt_z_full=self._metric_state_nonhydro.ddxt_z_full,
            vt=diagnostic_state_nh.tangential_wind,
            z_w_concorr_me=self._contravariant_correction_at_edges_on_model_levels,
            wgtfac_e=self._metric_state_nonhydro.wgtfac_e,
            vn_ie=diagnostic_state_nh.vn_on_half_levels,
            z_vt_ie=z_fields.tangential_wind_on_half_levels,
            z_kin_hor_e=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
            k_field=self.k_field,
            nflatlev_startindex=self._vertical_params.nflatlev,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.connectivities,
        )

        self._predictor_stencils_37_38(
            vn=prognostic_states.next.vn,
            vt=diagnostic_state_nh.tangential_wind,
            vn_ie=diagnostic_state_nh.vn_on_half_levels,
            z_vt_ie=z_fields.tangential_wind_on_half_levels,
            z_kin_hor_e=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
            wgtfacq_e_dsl=self._metric_state_nonhydro.wgtfacq_e,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.connectivities,
        )

        self._vertically_implicit_solver_at_predictor_step(
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            vertical_mass_flux_at_cells_on_half_levels=z_fields.vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=z_fields.tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=z_fields.tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w=prognostic_states.next.w,
            rho_explicit_term=z_fields.rho_explicit_term,
            exner_explicit_term=z_fields.exner_explicit_term,
            next_rho=prognostic_states.next.rho,
            next_exner=prognostic_states.next.exner,
            next_theta_v=prognostic_states.next.theta_v,
            dwdz_at_cells_on_model_levels=z_fields.dwdz_at_cells_on_model_levels,
            exner_dynamical_increment=diagnostic_state_nh.exner_dynamical_increment,
            geofac_div=self._interpolation_state.geofac_div,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.predictor,
            pressure_buoyancy_acceleration_at_cells_on_half_levels=self.pressure_buoyancy_acceleration_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=self._contravariant_correction_at_edges_on_model_levels,
            exner_w_explicit_weight_parameter=self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
            current_exner=prognostic_states.current.exner,
            current_rho=prognostic_states.current.rho,
            current_theta_v=prognostic_states.current.theta_v,
            current_w=prognostic_states.current.w,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=diagnostic_state_nh.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diagnostic_state_nh.rho_iau_increment,
            exner_iau_increment=diagnostic_state_nh.exner_iau_increment,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            rayleigh_damping_factor=self.rayleigh_damping_factor,
            reference_exner_at_cells_on_model_levels=self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
            e_bln_c_s=self._interpolation_state.e_bln_c_s,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            wgtfacq_c=self._metric_state_nonhydro.wgtfacq_c,
            iau_wgt_dyn=self._config.iau_wgt_dyn,
            dtime=dtime,
            is_iau_active=self._config.is_iau_active,
            rayleigh_type=self._config.rayleigh_type,
            divdamp_type=self._config.divdamp_type,
            at_first_substep=at_first_substep,
            kstart_moist=self._vertical_params.kstart_moist,
            end_index_of_damping_layer=self._vertical_params.end_index_of_damping_layer,
            flat_level_index_plus1=gtx.int32(self._vertical_params.nflatlev + 1),
            start_cell_index_nudging=self._start_cell_nudging,
            end_cell_index_local=self._end_cell_local,
            start_cell_index_lateral_lvl3=self._start_cell_lateral_boundary_level_3,
            end_cell_index_halo_lvl1=self._end_cell_halo,
            vertical_start_index_model_top=gtx.int32(0),
            vertical_end_index_model_surface=gtx.int32(self._grid.num_levels + 1),
            offset_provider=self._grid.connectivities,
        )

        if self._grid.limited_area:
            self._stencils_61_62(
                rho_now=prognostic_states.current.rho,
                grf_tend_rho=diagnostic_state_nh.grf_tend_rho,
                theta_v_now=prognostic_states.current.theta_v,
                grf_tend_thv=diagnostic_state_nh.grf_tend_thv,
                w_now=prognostic_states.current.w,
                grf_tend_w=diagnostic_state_nh.grf_tend_w,
                rho_new=prognostic_states.next.rho,
                exner_new=prognostic_states.next.exner,
                w_new=prognostic_states.next.w,
                dtime=dtime,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=gtx.int32(self._grid.num_levels + 1),
            )

        if self._config.divdamp_type >= 3:
            self._compute_dwdz_for_divergence_damping(
                inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
                w=prognostic_states.next.w,
                w_concorr_c=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
                z_dwdz_dd=z_fields.dwdz_at_cells_on_model_levels,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
                offset_provider=self._grid.connectivities,
            )
            log.debug(
                "exchanging prognostic field 'w' and local field 'dwdz_at_cells_on_model_levels'"
            )
            self._exchange.exchange_and_wait(
                dims.CellDim, prognostic_states.next.w, z_fields.dwdz_at_cells_on_model_levels
            )
        else:
            log.debug("exchanging prognostic field 'w'")
            self._exchange.exchange_and_wait(dims.CellDim, prognostic_states.next.w)

    def run_corrector_step(
        self,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        second_order_divdamp_factor: float,
        prep_adv: dycore_states.PrepAdvection,
        dtime: float,
        lprep_adv: bool,
        at_first_substep: bool,
        at_last_substep: bool,
    ):
        log.info(
            f"running corrector step: dtime = {dtime}, prep_adv = {lprep_adv},  "
            f"second_order_divdamp_factor = {second_order_divdamp_factor}, at_first_substep = {at_first_substep}, at_last_substep = {at_last_substep}  "
        )

        # TODO (magdalena) is it correct to to use a config parameter here? the actual number of substeps can vary dynmically...
        #                  should this config parameter exist at all in SolveNonHydro?
        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / self._config.ndyn_substeps_var

        # scaling factor for second-order divergence damping: second_order_divdamp_factor_from_sfc_to_divdamp_z*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        # Coefficient for reduced fourth-order divergence d
        second_order_divdamp_scaling_coeff = (
            second_order_divdamp_factor * self._grid.global_properties.mean_cell_area
        )

        dycore_utils._calculate_divdamp_fields(
            self.interpolated_fourth_order_divdamp_factor,
            gtx.int32(self._config.divdamp_order),
            self._grid.global_properties.mean_cell_area,
            second_order_divdamp_factor,
            self._config.max_nudging_coefficient,
            constants.DBL_EPS,
            out=(
                self.fourth_order_divdamp_scaling_coeff,
                self.reduced_fourth_order_divdamp_coeff_at_nest_boundary,
            ),
        )

        log.debug(f"corrector run velocity advection")
        self.velocity_advection.run_corrector_step(
            diagnostic_state=diagnostic_state_nh,
            prognostic_state=prognostic_states.next,
            horizontal_kinetic_energy_at_edges_on_model_levels=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind_on_half_levels=z_fields.tangential_wind_on_half_levels,
            dtime=dtime,
            cell_areas=self._cell_params.area,
        )

        dycore_utils._compute_rayleigh_damping_factor(
            rayleigh_w=self._metric_state_nonhydro.rayleigh_w,
            dtime=dtime,
            out=self.rayleigh_damping_factor,
        )
        log.debug(f"corrector: start stencil 10")

        self._interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration(
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=self.perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            pressure_buoyancy_acceleration_at_cells_on_half_levels=self.pressure_buoyancy_acceleration_at_cells_on_half_levels,
            w=prognostic_states.next.w,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            current_rho=prognostic_states.current.rho,
            next_rho=prognostic_states.next.rho,
            current_theta_v=prognostic_states.current.theta_v,
            next_theta_v=prognostic_states.next.theta_v,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            reference_theta_at_cells_on_model_levels=self._metric_state_nonhydro.reference_theta_at_cells_on_model_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=self._metric_state_nonhydro.ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            wgtfac_c=self._metric_state_nonhydro.wgtfac_c,
            exner_w_explicit_weight_parameter=self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
            dtime=dtime,
            rhotheta_explicit_weight_parameter=self._params.rhotheta_explicit_weight_parameter,
            rhotheta_implicit_weight_parameter=self._params.rhotheta_implicit_weight_parameter,
            horizontal_start=self._start_cell_lateral_boundary_level_3,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(self._grid.num_levels),
            offset_provider=self._grid.connectivities,
        )

        log.debug(f"corrector: start stencil apply_divergence_damping_and_update_vn")
        self._apply_divergence_damping_and_update_vn(
            horizontal_gradient_of_normal_wind_divergence=z_fields.horizontal_gradient_of_normal_wind_divergence,
            next_vn=prognostic_states.next.vn,
            current_vn=prognostic_states.current.vn,
            dwdz_at_cells_on_model_levels=z_fields.dwdz_at_cells_on_model_levels,
            predictor_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.predictor,
            corrector_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.corrector,
            normal_wind_tendency_due_to_slow_physics_process=diagnostic_state_nh.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diagnostic_state_nh.normal_wind_iau_increment,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=z_fields.horizontal_pressure_gradient,
            reduced_fourth_order_divdamp_coeff_at_nest_boundary=self.reduced_fourth_order_divdamp_coeff_at_nest_boundary,
            fourth_order_divdamp_scaling_coeff=self.fourth_order_divdamp_scaling_coeff,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            horizontal_mask_for_3d_divdamp=self._metric_state_nonhydro.horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=self._metric_state_nonhydro.scaling_factor_for_3d_divdamp,
            inv_dual_edge_length=self._edge_geometry.inverse_dual_edge_lengths,
            nudgecoeff_e=self._interpolation_state.nudgecoeff_e,
            geofac_grdiv=self._interpolation_state.geofac_grdiv,
            fourth_order_divdamp_factor=self._config.fourth_order_divdamp_factor,
            second_order_divdamp_factor=second_order_divdamp_factor,
            advection_explicit_weight_parameter=self._params.advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=self._params.advection_implicit_weight_parameter,
            dtime=dtime,
            iau_wgt_dyn=self._config.iau_wgt_dyn,
            is_iau_active=self._config.is_iau_active,
            limited_area=self._grid.limited_area,
            divdamp_order=self._config.divdamp_order,
            end_edge_halo_level_2=self._end_edge_halo_level_2,
            start_edge_lateral_boundary_level_7=self._start_edge_lateral_boundary_level_7,
            start_edge_nudging_level_2=self._start_edge_nudging_level_2,
            end_edge_local=self._end_edge_local,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(self._grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(self._grid.num_levels),
            offset_provider=self._grid.connectivities,
        )

        log.debug("exchanging prognostic field 'vn'")
        self._exchange.exchange_and_wait(dims.EdgeDim, (prognostic_states.next.vn))
        log.debug("corrector: start stencil 31")
        self._compute_avg_vn(
            e_flx_avg=self._interpolation_state.e_flx_avg,
            vn=prognostic_states.next.vn,
            z_vn_avg=self.z_vn_avg,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
            offset_provider=self._grid.connectivities,
        )

        log.debug("corrector: start stencil 32")
        self._compute_mass_flux(
            z_rho_e=z_fields.rho_at_edges_on_model_levels,
            z_vn_avg=self.z_vn_avg,
            ddqz_z_full_e=self._metric_state_nonhydro.ddqz_z_full_e,
            z_theta_v_e=z_fields.theta_v_at_edges_on_model_levels,
            mass_fl_e=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            z_theta_v_fl_e=self.theta_v_flux_at_edges_on_model_levels,
            horizontal_start=self._start_edge_lateral_boundary_level_5,
            horizontal_end=self._end_edge_halo_level_2,
            vertical_start=0,
            vertical_end=self._grid.num_levels,
        )

        if lprep_adv:  # Preparations for tracer advection
            log.debug("corrector: doing prep advection")
            if at_first_substep:
                log.debug("corrector: start stencil 33")
                self._init_two_edge_kdim_fields_with_zero_wp(
                    edge_kdim_field_with_zero_wp_1=prep_adv.vn_traj,
                    edge_kdim_field_with_zero_wp_2=prep_adv.mass_flx_me,
                    horizontal_start=self._start_edge_lateral_boundary,
                    horizontal_end=self._end_edge_end,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels,
                )
            log.debug(f"corrector: start stencil 34")
            self._accumulate_prep_adv_fields(
                z_vn_avg=self.z_vn_avg,
                mass_fl_e=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
                vn_traj=prep_adv.vn_traj,
                mass_flx_me=prep_adv.mass_flx_me,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=self._start_edge_lateral_boundary_level_5,
                horizontal_end=self._end_edge_halo_level_2,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
            )

        self._vertically_implicit_solver_at_corrector_step(
            vertical_mass_flux_at_cells_on_half_levels=z_fields.vertical_mass_flux_at_cells_on_half_levels,
            tridiagonal_beta_coeff_at_cells_on_model_levels=z_fields.tridiagonal_beta_coeff_at_cells_on_model_levels,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=z_fields.tridiagonal_alpha_coeff_at_cells_on_half_levels,
            next_w=prognostic_states.next.w,
            rho_explicit_term=z_fields.rho_explicit_term,
            exner_explicit_term=z_fields.exner_explicit_term,
            next_rho=prognostic_states.next.rho,
            next_exner=prognostic_states.next.exner,
            next_theta_v=prognostic_states.next.theta_v,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            exner_dynamical_increment=diagnostic_state_nh.exner_dynamical_increment,
            geofac_div=self._interpolation_state.geofac_div,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.predictor,
            corrector_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.corrector,
            pressure_buoyancy_acceleration_at_cells_on_half_levels=self.pressure_buoyancy_acceleration_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
            current_exner=prognostic_states.current.exner,
            current_rho=prognostic_states.current.rho,
            current_theta_v=prognostic_states.current.theta_v,
            current_w=prognostic_states.current.w,
            inv_ddqz_z_full=self._metric_state_nonhydro.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=diagnostic_state_nh.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diagnostic_state_nh.rho_iau_increment,
            exner_iau_increment=diagnostic_state_nh.exner_iau_increment,
            ddqz_z_half=self._metric_state_nonhydro.ddqz_z_half,
            rayleigh_damping_factor=self.rayleigh_damping_factor,
            reference_exner_at_cells_on_model_levels=self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
            advection_explicit_weight_parameter=self._params.advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=self._params.advection_implicit_weight_parameter,
            lprep_adv=lprep_adv,
            r_nsubsteps=r_nsubsteps,
            ndyn_substeps_var=float(self._config.ndyn_substeps_var),
            iau_wgt_dyn=self._config.iau_wgt_dyn,
            dtime=dtime,
            is_iau_active=self._config.is_iau_active,
            rayleigh_type=self._config.rayleigh_type,
            end_index_of_damping_layer=self._vertical_params.end_index_of_damping_layer,
            kstart_moist=self._vertical_params.kstart_moist,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            start_cell_index_nudging=self._start_cell_nudging,
            end_cell_index_local=self._end_cell_local,
            vertical_start_index_model_top=gtx.int32(0),
            vertical_end_index_model_surface=gtx.int32(self._grid.num_levels + 1),
            offset_provider=self._grid.connectivities,
        )

        if lprep_adv:
            if at_first_substep:
                log.debug(
                    f"corrector step sets prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels to zero"
                )
                self._init_cell_kdim_field_with_zero_wp(
                    field_with_zero_wp=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
                    horizontal_start=self._start_cell_lateral_boundary,
                    horizontal_end=self._end_cell_lateral_boundary_level_4,
                    vertical_start=0,
                    vertical_end=self._grid.num_levels + 1,
                )
            log.debug(f" corrector: start stencil 65")
            self._update_mass_flux_weighted(
                rho_ic=diagnostic_state_nh.rho_at_cells_on_half_levels,
                vwind_expl_wgt=self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                vwind_impl_wgt=self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
                w_now=prognostic_states.current.w,
                w_new=prognostic_states.next.w,
                w_concorr_c=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
                mass_flx_ic=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
                r_nsubsteps=r_nsubsteps,
                horizontal_start=self._start_cell_lateral_boundary,
                horizontal_end=self._end_cell_lateral_boundary_level_4,
                vertical_start=0,
                vertical_end=self._grid.num_levels,
            )
            log.debug("exchange prognostic fields 'rho' , 'exner', 'w'")
            self._exchange.exchange_and_wait(
                dims.CellDim,
                prognostic_states.next.rho,
                prognostic_states.next.exner,
                prognostic_states.next.w,
            )
