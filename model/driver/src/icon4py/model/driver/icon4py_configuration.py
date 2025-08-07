# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import datetime
import functools
import logging

from gt4py.next import backend as gtx_backend, metrics as gtx_metrics

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import constants
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.driver import initialization_utils as driver_init


log = logging.getLogger(__name__)

n_substeps_reduced = 2


@dataclasses.dataclass(frozen=True)
class Icon4pyRunConfig:
    backend: gtx_backend.Backend
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)  # length of a time step
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)

    # TODO (Chia Rui): ndyn_substeps in timeloop may change in runtime
    n_substeps: int = 5
    """ndyn_substeps in ICON"""

    apply_initial_stabilization: bool = True
    """
    ltestcase in ICON
        ltestcase has been renamed as apply_initial_stabilization because it is only used for extra damping for
        initial steps in timeloop.
    """

    restart_mode: bool = False

    @functools.cached_property
    def backend(self):
        return self.backend


@dataclasses.dataclass
class Icon4pyConfig:
    run_config: Icon4pyRunConfig
    vertical_grid_config: v_grid.VerticalGridConfig
    diffusion_config: diffusion.DiffusionConfig
    solve_nonhydro_config: solve_nh.NonHydrostaticConfig


def read_config(
    experiment_type: driver_init.ExperimentType,
    backend: gtx_backend.Backend,
) -> Icon4pyConfig:
    def _mch_ch_r04b09_vertical_config():
        return v_grid.VerticalGridConfig(
            num_levels=65,
            lowest_layer_thickness=20.0,
            model_top_height=23000.0,
            stretch_factor=0.65,
            rayleigh_damping_height=12500.0,
        )

    def _mch_ch_r04b09_diffusion_config():
        return diffusion.DiffusionConfig(
            diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
            hdiff_w=True,
            n_substeps=n_substeps_reduced,
            hdiff_vn=True,
            type_t_diffu=2,
            type_vn_diffu=1,
            hdiff_efdt_ratio=24.0,
            hdiff_w_efdt_ratio=15.0,
            smagorinski_scaling_factor=0.025,
            zdiffu_t=True,
            velocity_boundary_diffusion_denom=150.0,
            max_nudging_coefficient=0.075 * constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO,
        )

    def _mch_ch_r04b09_nonhydro_config():
        return solve_nh.NonHydrostaticConfig(
            ndyn_substeps_var=n_substeps_reduced,
            max_nudging_coefficient=0.075 * constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO,
        )

    def _jabw_vertical_config():
        return v_grid.VerticalGridConfig(
            num_levels=35,
            rayleigh_damping_height=45000.0,
        )

    def _jabw_diffusion_config(n_substeps: int):
        return diffusion.DiffusionConfig(
            diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
            hdiff_w=True,
            hdiff_vn=True,
            hdiff_temp=False,
            n_substeps=n_substeps,
            type_t_diffu=2,
            type_vn_diffu=1,
            hdiff_efdt_ratio=10.0,
            hdiff_w_efdt_ratio=15.0,
            smagorinski_scaling_factor=0.025,
            zdiffu_t=True,
            velocity_boundary_diffusion_denom=200.0,
        )

    def _jabw_nonhydro_config():
        return solve_nh.NonHydrostaticConfig(
            # original igradp_method is 2
            # original divdamp_order is 4
            fourth_order_divdamp_factor=0.0025,
        )

    def _mch_ch_r04b09_config():
        return (
            Icon4pyRunConfig(
                dtime=datetime.timedelta(seconds=10.0),
                start_date=datetime.datetime(2021, 6, 20, 12, 0, 0),
                end_date=datetime.datetime(2021, 6, 20, 12, 0, 10),
                n_substeps=n_substeps_reduced,
                apply_initial_stabilization=True,
                backend=backend,
            ),
            _mch_ch_r04b09_vertical_config(),
            _mch_ch_r04b09_diffusion_config(),
            _mch_ch_r04b09_nonhydro_config(),
        )

    def _jablownoski_Williamson_config():
        icon_run_config = Icon4pyRunConfig(
            dtime=datetime.timedelta(seconds=300.0),
            end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
            apply_initial_stabilization=False,
            n_substeps=5,
            backend=backend,
        )
        jabw_vertical_config = _jabw_vertical_config()
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.n_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config()
        return (
            icon_run_config,
            jabw_vertical_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )

    def _gauss3d_vertical_config():
        return v_grid.VerticalGridConfig(
            #num_levels=35,
            #rayleigh_damping_height=45000.0,
            num_levels=20,
            rayleigh_damping_height=100.0,
            model_top_height=100.0,
            flat_height=100.0,
            lowest_layer_thickness=0.0,
            stretch_factor=1.0,
        )

    def _gauss3d_diffusion_config(n_substeps: int):
        return diffusion.DiffusionConfig(
            n_substeps=n_substeps,
        )

    def _gauss3d_nonhydro_config():
        return solve_nh.NonHydrostaticConfig(
            igradp_method=3,
        )

    def _gauss3d_config():
        icon_run_config = Icon4pyRunConfig(
            dtime=datetime.timedelta(seconds=0.04),
            end_date=datetime.datetime(1, 1, 1, 0, 1, 0),
            apply_initial_stabilization=False,
            n_substeps=5,
            backend=backend,
        )
        vertical_config = _gauss3d_vertical_config()
        diffusion_config = _gauss3d_diffusion_config(icon_run_config.n_substeps)
        nonhydro_config = _gauss3d_nonhydro_config()
        return (
            icon_run_config,
            vertical_config,
            diffusion_config,
            nonhydro_config,
        )

    if experiment_type == driver_init.ExperimentType.JABW:
        (
            model_run_config,
            vertical_grid_config,
            diffusion_config,
            nonhydro_config,
        ) = _jablownoski_Williamson_config()
    elif experiment_type == driver_init.ExperimentType.GAUSS3D:
        (
            model_run_config,
            vertical_grid_config,
            diffusion_config,
            nonhydro_config,
        ) = _gauss3d_config()
    else:
        log.warning(
            "Experiment name is not specified, default configuration for mch_ch_r04b09_dsl is used."
        )
        (
            model_run_config,
            vertical_grid_config,
            diffusion_config,
            nonhydro_config,
        ) = _mch_ch_r04b09_config()
    return Icon4pyConfig(
        run_config=model_run_config,
        vertical_grid_config=vertical_grid_config,
        diffusion_config=diffusion_config,
        solve_nonhydro_config=nonhydro_config,
    )


@dataclasses.dataclass
class ProfilingConfig:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True
