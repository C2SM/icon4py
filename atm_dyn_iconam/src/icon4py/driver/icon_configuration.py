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

from dataclasses import dataclass
from typing import Optional

from icon4py.diffusion.diffusion import DiffusionConfig


@dataclass
class IconRunConfig:
    n_time_steps: int = 5
    dtime: float = 600.0


@dataclass
class AtmoNonHydroConfig:
    n_substeps: int = 5


@dataclass
class IconConfig:
    run_config: IconRunConfig
    diffusion_config: DiffusionConfig
    dycore_config: AtmoNonHydroConfig



# TODO @magdalena move to io_utils?
def read_config(experiment: Optional[str], n_time_steps: int) -> IconConfig:
    def _default_run_config(n_steps: int):
        if n_steps > 5:
            raise NotImplementedError("only five dummy timesteps available")
        return IconRunConfig(n_time_steps=n_steps)

    def mch_ch_r04b09_diffusion_config():
        return DiffusionConfig(
            diffusion_type=5,
            hdiff_w=True,
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

    def _default_config(n_steps):
        run_config = _default_run_config(n_steps)
        return run_config, _default_diffusion_config(), AtmoNonHydroConfig()

    def _mch_ch_r04b09_config(n_steps):
        return (
            IconRunConfig(n_time_steps=n_steps, dtime=10.0),
            mch_ch_r04b09_diffusion_config(),
            AtmoNonHydroConfig(),
        )

    if experiment == "mch_ch_r04b09_dsl":
        (model_run_config, diffusion_config, dycore_config) = _mch_ch_r04b09_config(
            n_time_steps
        )
    else:
        (model_run_config, diffusion_config, dycore_config) = _default_config(
            n_time_steps
        )
    return IconConfig(
        run_config=model_run_config,
        diffusion_config=diffusion_config,
        dycore_config=dycore_config,
    )
