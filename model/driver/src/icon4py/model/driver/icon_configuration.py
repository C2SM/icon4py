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
import dataclasses
import datetime
import logging

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import NonHydrostaticConfig
from icon4py.model.common.grid.vertical import VerticalGridConfig
from icon4py.model.driver.initialization_utils import ExperimentType


log = logging.getLogger(__name__)

n_substeps_reduced = 2


@dataclasses.dataclass(frozen=True)
class IconRunConfig:
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)  # length of a time step
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)

    # TODO (Chia Rui): check ICON code if we need to define extra ndyn_substeps in timeloop that changes in runtime
    n_substeps: int = 5
    """ndyn_substeps in ICON"""

    apply_initial_stabilization: bool = True
    """
    ltestcase in ICON
        ltestcase has been renamed as apply_initial_stabilization because it is only used for extra damping for
        initial steps in timeloop.
    """

    restart_mode: bool = False


@dataclasses.dataclass
class IconConfig:
    run_config: IconRunConfig
    vertical_grid_config: VerticalGridConfig
    diffusion_config: DiffusionConfig
    solve_nonhydro_config: NonHydrostaticConfig


def read_config(experiment_type: ExperimentType = ExperimentType.ANY) -> IconConfig:
    def _mch_ch_r04b09_vertical_config():
        return VerticalGridConfig(
            num_levels=65,
            lowest_layer_thickness=20.0,
            model_top_height=23000.0,
            stretch_factor=0.65,
            rayleigh_damping_height=12500.0,
        )

    def _mch_ch_r04b09_diffusion_config():
        return DiffusionConfig(
            diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
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
            max_nudging_coeff=0.075,
        )

    def _mch_ch_r04b09_nonhydro_config():
        return NonHydrostaticConfig(
            ndyn_substeps_var=n_substeps_reduced,
        )

    def _jabw_vertical_config():
        return VerticalGridConfig(
            num_levels=35,
            rayleigh_damping_height=45000.0,
        )

    def _jabw_diffusion_config(n_substeps: int):
        return DiffusionConfig(
            diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
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
            max_nudging_coeff=0.075,
        )

    def _jabw_nonhydro_config(n_substeps: int):
        return NonHydrostaticConfig(
            # original igradp_method is 2
            # original divdamp_order is 4
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            divdamp_fac=0.0025,
        )

    def _mch_ch_r04b09_config():
        return (
            IconRunConfig(
                dtime=datetime.timedelta(seconds=10.0),
                start_date=datetime.datetime(2021, 6, 20, 12, 0, 0),
                end_date=datetime.datetime(2021, 6, 20, 12, 0, 10),
                n_substeps=n_substeps_reduced,
                apply_initial_stabilization=True,
            ),
            _mch_ch_r04b09_vertical_config(),
            _mch_ch_r04b09_diffusion_config(),
            _mch_ch_r04b09_nonhydro_config(),
        )

    def _jablownoski_Williamson_config():
        icon_run_config = IconRunConfig(
            dtime=datetime.timedelta(seconds=300.0),
            end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        jabw_vertical_config = _jabw_vertical_config()
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.n_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            jabw_vertical_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )

    def _gauss3d_diffusion_config(n_substeps: int):
        return DiffusionConfig()

    def _gauss3d_nonhydro_config(n_substeps: int):
        return NonHydrostaticConfig(
            igradp_method=3,
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            divdamp_fac=0.0025,
        )

    def _gauss3d_config():
        icon_run_config = IconRunConfig(
            dtime=datetime.timedelta(seconds=4.0),
            end_date=datetime(1, 1, 1, 0, 0, 4),
            damping_height=45000.0,
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        gauss3d_diffusion_config = _gauss3d_diffusion_config(icon_run_config.n_substeps)
        gauss3d_nonhydro_config = _gauss3d_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            gauss3d_diffusion_config,
            gauss3d_nonhydro_config,
        )

    if experiment_type == ExperimentType.JABW:
        (
            model_run_config,
            vertical_grid_config,
            diffusion_config,
            nonhydro_config,
        ) = _jablownoski_Williamson_config()
    elif experiment_type == ExperimentType.GAUSS3D:
        (
            model_run_config,
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
    return IconConfig(
        run_config=model_run_config,
        vertical_grid_config=vertical_grid_config,
        diffusion_config=diffusion_config,
        solve_nonhydro_config=nonhydro_config,
    )
