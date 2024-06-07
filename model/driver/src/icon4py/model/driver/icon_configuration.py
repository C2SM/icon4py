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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionType
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import NonHydrostaticConfig
from icon4py.model.driver.initialization_utils import ExperimentType


log = logging.getLogger(__name__)

n_substeps_reduced = 2


@dataclass(frozen=True)
class IconRunConfig:
    dtime: timedelta = timedelta(seconds=600.0)  # length of a time step
    start_date: datetime = datetime(1, 1, 1, 0, 0, 0)
    end_date: datetime = datetime(1, 1, 1, 1, 0, 0)

    damping_height: float = 12500.0

    """ndyn_substeps in ICON"""
    # TODO (Chia Rui): check ICON code if we need to define extra ndyn_substeps in timeloop that changes in runtime
    n_substeps: int = 5

    """
    ltestcase in ICON
        ltestcase has been renamed as apply_initial_stabilization because it is only used for extra damping for
        initial steps in timeloop.
    """
    apply_initial_stabilization: bool = True

    restart_mode: bool = False

    run_testcase: bool = False


@dataclass(frozen=True)
class VariableAttributes:
    units: str = ' '
    standard_name: str = ' '
    long_name: str = ' '
    CDI_grid_type: str = ' '
    param: str = ' '
    number_of_grid_in_reference: str = ' '
    coordinates: str = ' '
    scope: str = ' '

from abc import ABC
from icon4py.model.common.utils import builder
class OutputVariableList(ABC):
    variable_name = (
        'vn',
        'w',
        'rho',
        'theta_v',
        'exner'
    )
    variable_attribute = {
        'vn': VariableAttributes(
            units='m s-1',
            standard_name='normal velocity',
            long_name='normal velocity at edge center',
            CDI_grid_type='unstructured',
            param='0.0.0',
            number_of_grid_in_reference='1',
            coordinates='elat elon',
            scope= 'prognostic',
        ),
        'w': VariableAttributes(
            units='m s-1',
            standard_name='normal velocity',
            long_name='normal velocity at edge center',
            CDI_grid_type='unstructured',
            param='0.0.0',
            number_of_grid_in_reference='1',
            coordinates='elat elon',
            scope='prognostic',
        ),
        'rho': VariableAttributes(
            units='m s-1',
            standard_name='normal velocity',
            long_name='normal velocity at edge center',
            CDI_grid_type='unstructured',
            param='0.0.0',
            number_of_grid_in_reference='1',
            coordinates='elat elon',
            scope='prognostic',
        ),
        'theta_v': VariableAttributes(
            units='m s-1',
            standard_name='normal velocity',
            long_name='normal velocity at edge center',
            CDI_grid_type='unstructured',
            param='0.0.0',
            number_of_grid_in_reference='1',
            coordinates='elat elon',
            scope='prognostic',
        ),
        'exner': VariableAttributes(
            units='m s-1',
            standard_name='normal velocity',
            long_name='normal velocity at edge center',
            CDI_grid_type='unstructured',
            param='0.0.0',
            number_of_grid_in_reference='1',
            coordinates='elat elon',
            scope='prognostic',
        ),
    }

    def add_new_variable(
        self, variable_name: str, variable_attribute: VariableAttributes
    ) -> None:
        if variable_name in self.variable_name:
            return
        self.variable_name = self.variable_name + (variable_name,)
        self.variable_attribute[variable_name] = variable_attribute

@dataclass(frozen=True)
class IconOutputConfig:
    output_time_interval: timedelta = timedelta(minutes=1)
    output_file_time_interval: timedelta = timedelta(minutes=1)
    output_path: Path = Path("./")
    output_initial_condition_as_a_separate_file: bool = False
    output_variable_list: OutputVariableList = OutputVariableList()

@dataclass
class IconConfig:
    run_config: IconRunConfig
    output_config: IconOutputConfig
    diffusion_config: DiffusionConfig
    solve_nonhydro_config: NonHydrostaticConfig


def read_config(experiment_type: ExperimentType = ExperimentType.ANY) -> IconConfig:
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
                dtime=timedelta(seconds=10.0),
                start_date=datetime(2021, 6, 20, 12, 0, 0),
                end_date=datetime(2021, 6, 20, 12, 0, 10),
                damping_height=12500.0,
                n_substeps=n_substeps_reduced,
                apply_initial_stabilization=True,
            ),
            _mch_ch_r04b09_diffusion_config(),
            _mch_ch_r04b09_nonhydro_config(),
        )

    def _jablownoski_Williamson_config():
        output_variable_list = OutputVariableList()
        output_variable_list.add_new_variable(
            'temperature',
            VariableAttributes(
                units='K',
                standard_name='temperauture',
                long_name='air temperauture',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope='diagnostic',
            )
        )
        output_variable_list.add_new_variable(
            'pressure_sfc',
            VariableAttributes(
                units='Pa',
                standard_name='surface pressure',
                long_name='surface pressure',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope='diagnostic',
            )
        )
        output_variable_list.add_new_variable(
            'u',
            VariableAttributes(
                units='m s-1',
                standard_name='zonal wind',
                long_name='zonal wind speed',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope='diagnostic',
            )
        )
        output_variable_list.add_new_variable(
            'v',
            VariableAttributes(
                units='m s-1',
                standard_name='meridional wind',
                long_name='meridional wind speed',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope='diagnostic',
            )
        )

        output_config = IconOutputConfig(
            output_time_interval=timedelta(seconds=60),
            output_file_time_interval=timedelta(seconds=60),
            output_path=Path("./"),
            output_initial_condition_as_a_separate_file=True,
        )
        icon_run_config = IconRunConfig(
            dtime=timedelta(seconds=300.0),
            end_date=datetime(1, 1, 1, 0, 30, 0),
            damping_height=45000.0,
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.n_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            output_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )

    if experiment_type == ExperimentType.JABW:
        (
            model_run_config,
            output_config,
            diffusion_config,
            nonhydro_config,
        ) = _jablownoski_Williamson_config()
    else:
        log.warning(
            "Experiment name is not specified, default configuration for mch_ch_r04b09_dsl is used."
        )
        (
            model_run_config,
            diffusion_config,
            nonhydro_config,
        ) = _mch_ch_r04b09_config()
        output_config = None

    return IconConfig(
        run_config=model_run_config,
        output_config=output_config,
        diffusion_config=diffusion_config,
        solve_nonhydro_config=nonhydro_config,
    )
