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

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonydro import NonHydrostaticConfig

n_substeps_reduced = 2

@dataclass
class IconRunConfig:

    dtime: float = 60.0,
    start_date: datetime = datetime(1, 1, 1, 0, 0),
    end_date: datetime = datetime(1,1,1,1,0)
    is_testcase: bool = True

@dataclass
class AtmoNonHydroConfig:

    # TODO ndyn_substeps is not a constant in ICON, it may be modified in restart runs.
    ndyn_substeps: int = 5
    apply_horizontal_diff_at_large_dt: bool = True
    apply_extra_diffusion_for_largeCFL: bool = True


@dataclass
class IconConfig:
    run_config: IconRunConfig
    diffusion_config: DiffusionConfig
    nonhydro_solver_config: NonHydrostaticConfig



def read_config(experiment: Optional[str]) -> IconConfig:
    #def _default_run_config(n_steps: int):
    #    if n_steps > 5:
    #        raise NotImplementedError("only five dummy timesteps available")
    #    return IconRunConfig(n_time_steps=n_steps)

    def mch_ch_r04b09_diffusion_config():
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

    def _default_diffusion_config():
        return DiffusionConfig()

    #def _default_config(n_steps):
    #    run_config = _default_run_config(n_steps)
    #    return run_config, _default_diffusion_config(), AtmoNonHydroConfig()

    def _mch_ch_r04b09_config():
        return (
            IconRunConfig(dtime=10.0),
            mch_ch_r04b09_diffusion_config(),
            AtmoNonHydroConfig(),
        )

    if experiment == "mch_ch_r04b09_dsl":
        (model_run_config, diffusion_config, dycore_config) = _mch_ch_r04b09_config()
    else:
        raise NotImplementedError("Please specify experiment name for icon run config")
        #(model_run_config, diffusion_config, dycore_config) = _default_config()
    return IconConfig(
        run_config=model_run_config,
        diffusion_config=diffusion_config,
        dycore_config=dycore_config,
    )
