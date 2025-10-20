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

import gt4py.next.typing as gtx_typing
from gt4py.next import metrics as gtx_metrics

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import type_alias as ta
from icon4py.model.common.grid import vertical as v_grid


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Icon4pyRunConfig:  # TODO (Yilu) think of a better name
    backend: gtx_typing.Backend
    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)  # length of a time step
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)

    vertical_cfl_threshold: ta.wpfloat = 0.85

    ndyn_substeps: int = 5

    apply_initial_stabilization: bool = True  # TODO (Yilu) remove
    """
    ltestcase in ICON
        ltestcase has been renamed as apply_initial_stabilization because it is only used for extra damping for
        initial steps in timeloop.
    """

    restart_mode: bool = False  # TODO (Yilu) remove

    # TODO (Yilu)
    # TODO (Yilu) add three paths: grid_path, configuration_file_path, output_path

    @functools.cached_property
    def backend(self):
        return self.backend


# TODO (Yilu) remove Icon4pyConfig
@dataclasses.dataclass
class Icon4pyConfig:
    run_config: Icon4pyRunConfig
    vertical_grid_config: v_grid.VerticalGridConfig
    diffusion_config: diffusion.DiffusionConfig
    solve_nonhydro_config: solve_nh.NonHydrostaticConfig


# TODO (Yilu) the read_config should only return the configure for jabw
def read_config(
    backend: gtx_typing.Backend,
) -> Icon4pyConfig:
    def _jabw_vertical_config():
        return v_grid.VerticalGridConfig(
            num_levels=35,
            rayleigh_damping_height=45000.0,
        )

    def _jabw_diffusion_config(ndyn_substeps: int):
        return diffusion.DiffusionConfig(
            diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
            hdiff_w=True,
            hdiff_vn=True,
            hdiff_temp=False,
            n_substeps=ndyn_substeps,
            type_t_diffu=2,
            type_vn_diffu=1,
            hdiff_efdt_ratio=10.0,
            hdiff_w_efdt_ratio=15.0,
            smagorinski_scaling_factor=0.025,
            zdiffu_t=True,
            velocity_boundary_diffusion_denom=200.0,
        )

    def _jablownoski_Williamson_config():
        icon_run_config = Icon4pyRunConfig(
            dtime=datetime.timedelta(seconds=300.0),
            end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
            apply_initial_stabilization=False,
            ndyn_substeps=5,
            backend=backend,
        )
        jabw_vertical_config = _jabw_vertical_config()
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.ndyn_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config()
        return (
            icon_run_config,
            jabw_vertical_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )


@dataclasses.dataclass
class ProfilingConfig:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True
