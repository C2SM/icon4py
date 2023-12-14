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
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import NonHydrostaticConfig


log = logging.getLogger(__name__)

n_substeps_reduced = 2


@dataclass(frozen=True)
class IconRunConfig:

    dtime: float = 600.0  # length of a time step [s]
    start_date: datetime = datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime = datetime(1, 1, 1, 1, 0, 0)

    """ndyn_substeps in ICON"""
    # TODO (Chia Rui): check ICON code if we need to define extra ndyn_substeps in timeloop that changes in runtime
    n_substeps: int = 5

    """linit_dyn in ICON"""
    apply_initial_stabilization: bool = True  # False if in restart mode

    """veladv_offctr"""
    time_discretization_veladv_offctr: int = 0.25

    """rhotheta_offctr"""
    time_discretization_rhotheta_offctr: int = -0.1

    output_path="./"


@dataclass
class IconConfig:
    run_config: IconRunConfig
    diffusion_config: DiffusionConfig
    solve_nonhydro_config: NonHydrostaticConfig


def read_config(experiment: Optional[str]) -> IconConfig:
    def _default_run_config():
        return IconRunConfig()

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

    def _default_config():
        run_config = _default_run_config()
        return run_config, _default_diffusion_config(), NonHydrostaticConfig()

    def _mch_ch_r04b09_config():
        return (
            IconRunConfig(
                dtime=10.0,
                start_date=datetime(2021, 6, 20, 12, 0, 0),
                end_date=datetime(2021, 6, 20, 12, 0, 10),
                n_substeps=2,
                apply_initial_stabilization=True,
            ),
            mch_ch_r04b09_diffusion_config(),
            NonHydrostaticConfig(),
        )

    def _Jablownoski_Williamson_config():
        return (
            IconRunConfig(
                dtime=300.0,
                end_date=datetime(1, 1, 1, 0, 5, 0),
                apply_initial_stabilization=False,
            ),
            _default_diffusion_config(),
            NonHydrostaticConfig(),
        )

    if experiment == "mch_ch_r04b09_dsl":
        (model_run_config, diffusion_config, nonhydro_config) = _mch_ch_r04b09_config()
    elif experiment == "jabw":
        (model_run_config, diffusion_config, nonhydro_config) = _Jablownoski_Williamson_config()
    else:
        log.warning("Experiment name is not specified, default configuration is used.")
        (model_run_config, diffusion_config, nonhydro_config) = _default_config()
    return IconConfig(
        run_config=model_run_config,
        diffusion_config=diffusion_config,
        solve_nonhydro_config=nonhydro_config,
    )
