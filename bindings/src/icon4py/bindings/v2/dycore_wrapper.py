# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
v2 solve-nonhydro (dycore) bindings: minimal Fortran interface, factory-derived fields.

`solve_nh_init_v2` receives only the namelist configuration scalars; all interpolation
and metric fields (and nflat_gradp, the owner mask, the backend) come from the grid state
built by `grid_init_v2` (see diffusion_wrapper). `solve_nh_run_v2` is the v1 run logic,
operating on the v2 granule.
"""

import dataclasses
import logging
from collections.abc import Callable
from typing import Annotated, TypeAlias

import gt4py.next as gtx
import numpy as np
from gt4py.next import config as gtx_config
from gt4py.next.instrumentation import metrics as gtx_metrics
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings import common as wrapper_common, config as wrapper_config, icon4py_export
from icon4py.bindings.v2 import diffusion_wrapper, dycore_setup
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.tools import py2fgen


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SolveNonhydroGranuleV2:
    solve_nh: solve_nonhydro.SolveNonhydro
    dummy_field_factory: Callable


granule: SolveNonhydroGranuleV2 | None = None


@icon4py_export.export
def solve_nh_init_v2(  # noqa: PLR0917 [too-many-positional-arguments]
    itime_scheme: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    rayleigh_type: gtx.int32,
    divdamp_order: gtx.int32,
    divdamp_type: gtx.int32,
    l_vert_nested: bool,
    ldeepatmo: bool,
    iau_init: bool,
    extra_diffu: bool,
    rhotheta_offctr: gtx.float64,
    veladv_offctr: gtx.float64,
    nudge_max_coeff: gtx.float64,  # scaled ICON value (not the raw namelist value)
    divdamp_fac: gtx.float64,
    divdamp_fac2: gtx.float64,
    divdamp_fac3: gtx.float64,
    divdamp_fac4: gtx.float64,
    divdamp_z: gtx.float64,
    divdamp_z2: gtx.float64,
    divdamp_z3: gtx.float64,
    divdamp_z4: gtx.float64,
) -> None:
    grid_state = diffusion_wrapper.grid_state
    if grid_state is None:
        raise RuntimeError(
            "Need to initialise grid using 'grid_init_v2' before running 'solve_nh_init_v2'."
        )

    config = solve_nonhydro.NonHydrostaticConfig(
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        rayleigh_type=rayleigh_type,
        divdamp_order=divdamp_order,
        divdamp_type=divdamp_type,
        l_vert_nested=l_vert_nested,
        deepatmos_mode=ldeepatmo,
        iau_init=iau_init,
        extra_diffu=extra_diffu,
        rhotheta_offctr=rhotheta_offctr,
        veladv_offctr=veladv_offctr,
        max_nudging_coefficient=nudge_max_coeff,
        fourth_order_divdamp_factor=divdamp_fac,
        fourth_order_divdamp_factor2=divdamp_fac2,
        fourth_order_divdamp_factor3=divdamp_fac3,
        fourth_order_divdamp_factor4=divdamp_fac4,
        fourth_order_divdamp_z=divdamp_z,
        fourth_order_divdamp_z2=divdamp_z2,
        fourth_order_divdamp_z3=divdamp_z3,
        fourth_order_divdamp_z4=divdamp_z4,
    )
    nonhydro_params = solve_nonhydro.NonHydrostaticParams(config)

    interpolation_state, metric_state = dycore_setup.assemble_dycore_states(grid_state.sources)

    global granule  # noqa: PLW0603 [global-statement]
    granule = SolveNonhydroGranuleV2(
        solve_nh=solve_nonhydro.SolveNonhydro(
            grid=grid_state.grid,
            config=config,
            params=nonhydro_params,
            metric_state_nonhydro=metric_state,
            interpolation_state=interpolation_state,
            vertical_params=grid_state.vertical_grid,
            edge_geometry=grid_state.edge_geometry,
            cell_geometry=grid_state.cell_geometry,
            owner_mask=grid_state.owner_mask,
            backend=grid_state.backend,
            exchange=grid_state.exchange_runtime,
        ),
        dummy_field_factory=wrapper_common.cached_dummy_field_factory(grid_state.allocator),
    )
    if wrapper_config.WAIT_FOR_COMPILATION:
        gtx.wait_for_compilation()


NumpyFloatArray1D: TypeAlias = Annotated[
    np.ndarray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.HOST,
        is_optional=False,
    ),
]


@icon4py_export.export
def solve_nh_run_v2(  # noqa: PLR0917 [too-many-positional-arguments]
    rho_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    w_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    w_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn_now: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_new: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    w_concorr_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_vn_apc_ntl1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_vn_apc_ntl2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_w_adv_ntl1: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_w_adv_ntl2: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_pr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_dyn_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_exner_phy: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_thv: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_w: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    mass_fl_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_vn_phy: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    grf_tend_vn: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_ie: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vt: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_incr: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64] | None,
    rho_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    exner_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    mass_flx_me: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    mass_flx_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vol_flx_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn_traj: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    dtime: gtx.float64,
    max_vcfl_size1_array: NumpyFloatArray1D,  # receive from Fortran as a single-element array
    lprep_adv: bool,
    at_initial_timestep: bool,
    divdamp_fac_o2: gtx.float64,
    ndyn_substeps_var: gtx.int32,
    idyn_timestep: gtx.int32,
    is_iau_active: bool,
    iau_wgt_dyn: gtx.float64,
) -> None:
    if granule is None:
        raise RuntimeError("SolveNonhydro granule not initialized. Call 'solve_nh_init_v2' first.")

    xp = rho_now.array_ns

    if vn_incr is None:
        vn_incr = granule.dummy_field_factory("vn_incr", domain=vn_now.domain, dtype=vn_now.dtype)

    if rho_incr is None:
        rho_incr = granule.dummy_field_factory(
            "rho_incr", domain=rho_now.domain, dtype=rho_now.dtype
        )

    if exner_incr is None:
        exner_incr = granule.dummy_field_factory(
            "exner_incr", domain=exner_now.domain, dtype=exner_now.dtype
        )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=mass_flx_ic,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=vol_flx_ic,
    )

    # Make `max_vcfl` a 0-d array to avoid cupy synchronization, see `velocity_advection.py`.
    # Note, `max_vcfl` needs to be passed back to Fortran after the timestep.
    max_vcfl = data_alloc.scalar_like_array(max_vcfl_size1_array[0], xp)

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=max_vcfl,
        theta_v_at_cells_on_half_levels=theta_v_ic,
        perturbed_exner_at_cells_on_model_levels=exner_pr,
        rho_at_cells_on_half_levels=rho_ic,
        exner_tendency_due_to_slow_physics=ddt_exner_phy,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_flux_at_edges_on_model_levels=mass_fl_e,
        normal_wind_tendency_due_to_slow_physics_process=ddt_vn_phy,
        grf_tend_vn=grf_tend_vn,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            ddt_vn_apc_ntl1, ddt_vn_apc_ntl2
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            ddt_w_adv_ntl1, ddt_w_adv_ntl2
        ),
        tangential_wind=vt,
        vn_on_half_levels=vn_ie,
        contravariant_correction_at_cells_on_half_levels=w_concorr_c,
        rho_iau_increment=rho_incr,
        normal_wind_iau_increment=vn_incr,
        exner_iau_increment=exner_incr,
        exner_dynamical_increment=exner_dyn_incr,
    )

    prognostic_state_nnow = PrognosticState(
        w=w_now,
        vn=vn_now,
        theta_v=theta_v_now,
        rho=rho_now,
        exner=exner_now,
    )
    prognostic_state_nnew = PrognosticState(
        w=w_new,
        vn=vn_new,
        theta_v=theta_v_new,
        rho=rho_new,
        exner=exner_new,
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)

    # adjust for Fortran indexes
    idyn_timestep = idyn_timestep - 1

    granule.solve_nh.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=divdamp_fac_o2,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps_var,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=idyn_timestep == 0,
        at_last_substep=idyn_timestep == (ndyn_substeps_var - 1),
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
    )

    # TODO(havogt): create separate bindings for writing the timers
    if gtx_config.COLLECT_METRICS_LEVEL > 0:
        gtx_metrics.dump_json("gt4py_timers.json")

    max_vcfl_size1_array[0] = diagnostic_state_nh.max_vertical_cfl[()]  # pass back to Fortran
