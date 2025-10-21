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
import os

import gt4py.next.typing as gtx_typing
from gt4py.next import metrics as gtx_metrics

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import type_alias as ta
from icon4py.model.common.grid import vertical as v_grid
from pathlib import Path


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    backend: gtx_typing.Backend
    grid_path: Path
    configuration_file_path: Path
    output_path: Path

    dtime: datetime.timedelta = datetime.timedelta(seconds=600.0)  # length of a time step
    start_date: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime.datetime = datetime.datetime(1, 1, 1, 1, 0, 0)

    vertical_cfl_threshold: ta.wpfloat = 0.85

    ndyn_substeps: int = 5

    output_statistics: bool = False

    def __post_init__(self):
        for name in ["grid_path", "configuration_file_path", "output_path"]:
            path = Path(os.path.expandvars(str(getattr(self, name)))).expanduser()
            object.__setattr__(self, name, path)
            if not path.exists():
                log.warning(f"The path for {name} does not exist: {path}")
            elif not path.is_socket():
                raise ValueError(f"{name} must be a file: {path}")

    @functools.cached_property
    def backend(self):
        return self.backend


def read_config(
    backend: gtx_typing.Backend,
) -> tuple[DriverConfig, v_grid.VerticalGridConfig, diffusion.DiffusionConfig, solve_nh.NonHydrostaticConfig]:

    vertical_config = v_grid.VerticalGridConfig(
            num_levels=35,
            rayleigh_damping_height=45000.0,
        )

    diffusion_config = diffusion.DiffusionConfig(
            diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
            hdiff_w=True,
            hdiff_vn=True,
            hdiff_temp=False,
            n_substeps=5,
            type_t_diffu=2,
            type_vn_diffu=1,
            hdiff_efdt_ratio=10.0,
            hdiff_w_efdt_ratio=15.0,
            smagorinski_scaling_factor=0.025,
            zdiffu_t=True,
            velocity_boundary_diffusion_denom=200.0,
        )

    nonhydro_config = solve_nh.NonHydrostaticConfig(
        fourth_order_divdamp_factor=0.0025,
    )

    driver_run_config = DriverConfig(
            dtime=datetime.timedelta(seconds=300.0),
            end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
            apply_initial_stabilization=False,
            ndyn_substeps=5,
            backend=backend,
        )


    return (
        driver_run_config,
        vertical_config,
        diffusion_config,
        nonhydro_config,
    )


@dataclasses.dataclass
class ProfilingConfig:
    gt4py_metrics_level: int = gtx_metrics.ALL
    gt4py_metrics_output_file: str = "gt4py_metrics.json"
    skip_first_timestep: bool = True
