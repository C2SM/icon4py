# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import logging
import pathlib
import enum
from typing import Optional

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro as solve_nh
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.driver import initialization_utils as driver_init


log = logging.getLogger(__name__)

n_substeps_reduced = 2


@dataclasses.dataclass(frozen=True)
class Icon4pyRunConfig:
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


@dataclasses.dataclass(frozen=True)
class VariableAttributes:
    units: str = ' '
    standard_name: str = ' '
    long_name: str = ' '
    CDI_grid_type: str = ' '
    param: str = ' '
    number_of_grid_in_reference: str = ' '
    coordinates: str = ' '
    scope: str = ' '


class OutputDimension(str, enum.Enum):
    CELL_DIM = 'ncells'
    EDGE_DIM = 'ncells_2'
    VERTEX_DIM = 'ncells_3'
    FULL_LEVEL = 'height'
    HALF_LEVEL = 'height_2'
    TIME = 'time'


@dataclasses.dataclass(frozen=True)
class VariableDimension:
    horizon_dimension: str = None
    vertical_dimension: str = None
    time_dimension: str = None


class OutputScope(str, enum.Enum):
    prognostic = 'prognostic'
    diagnostic = 'diagnostic'
    diffusion = 'diffusion'
    solve_nonhydro = 'solve_nonhydro'


class OutputVariableList:
    def __init__(self):
        self._variable_name = (
            'vn',
            'w',
            'rho',
            'theta_v',
            'exner'
        )
        self._variable_dimension = {
            'vn': VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            'w': VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            'rho': VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            'theta_v': VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            'exner': VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
        }
        self._variable_attribute = {
            'vn': VariableAttributes(
                units='m s-1',
                standard_name='normal velocity',
                long_name='normal wind speed at edge center',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='elat elon',
                scope= OutputScope.prognostic,
            ),
            'w': VariableAttributes(
                units='m s-1',
                standard_name='vertical velocity',
                long_name='vertical wind speed at half levels',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.prognostic,
            ),
            'rho': VariableAttributes(
                units='kg m-3',
                standard_name='density',
                long_name='air density',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.prognostic,
            ),
            'theta_v': VariableAttributes(
                units='K',
                standard_name='virtual temperature',
                long_name='virtual temperature',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.prognostic,
            ),
            'exner': VariableAttributes(
                units='',
                standard_name='exner function',
                long_name='exner function',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.prognostic,
            ),
        }

    @property
    def variable_name_list(self):
        return self._variable_name

    @property
    def variable_dim_list(self):
        return self._variable_dimension

    @property
    def variable_attr_list(self):
        return self._variable_attribute

    def add_new_variable(
        self, variable_name: str, variable_dimenson: VariableDimension, variable_attribute: VariableAttributes
    ) -> None:
        if variable_name in self._variable_name:
            log.warning(f"Output variable name {variable_name} is already in variable list {self._variable_name}. Nothing to do.")
            return
        self._variable_name = self._variable_name + (variable_name,)
        self._variable_attribute[variable_name] = variable_attribute
        self._variable_dimension[variable_name] = variable_dimenson


@dataclasses.dataclass(frozen=True)
class IconOutputConfig:
    output_time_interval: datetime.timedelta = datetime.timedelta(minutes=1)
    output_file_time_interval: datetime.timedelta = datetime.timedelta(minutes=1)
    output_path: pathlib.Path = pathlib.Path("./")
    output_initial_condition_as_a_separate_file: bool = False
    output_variable_list: OutputVariableList = OutputVariableList()

@dataclasses.dataclass
class Icon4pyConfig:
    run_config: Icon4pyRunConfig
    output_config: IconOutputConfig
    vertical_grid_config: v_grid.VerticalGridConfig
    diffusion_config: diffusion.DiffusionConfig
    solve_nonhydro_config: solve_nh.NonHydrostaticConfig


def read_config(
    experiment_type: driver_init.ExperimentType = driver_init.ExperimentType.ANY,
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
            max_nudging_coeff=0.075,
        )

    def _mch_ch_r04b09_nonhydro_config():
        return solve_nh.NonHydrostaticConfig(
            ndyn_substeps_var=n_substeps_reduced,
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
            zdiffu_t=False,
            velocity_boundary_diffusion_denom=200.0,
            max_nudging_coeff=0.075,
        )

    def _jabw_nonhydro_config(n_substeps: int):
        return solve_nh.NonHydrostaticConfig(
            # original igradp_method is 2
            # original divdamp_order is 4
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            divdamp_fac=0.0025,
        )

    def _mch_ch_r04b09_config():
        return (
            Icon4pyRunConfig(
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
        output_variable_list = OutputVariableList()
        output_variable_list.add_new_variable(
            'temperature',
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units='K',
                standard_name='temperauture',
                long_name='air temperauture',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.diagnostic,
            )
        )
        output_variable_list.add_new_variable(
            'pressure',
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units='Pa',
                standard_name='pressure',
                long_name='air pressure',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.diagnostic,
            )
        )
        output_variable_list.add_new_variable(
            'pressure_sfc',
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units='Pa',
                standard_name='surface pressure',
                long_name='surface pressure',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.diagnostic,
            )
        )
        output_variable_list.add_new_variable(
            'u',
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units='m s-1',
                standard_name='zonal wind',
                long_name='zonal wind speed',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.diagnostic,
            )
        )
        output_variable_list.add_new_variable(
            'v',
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units='m s-1',
                standard_name='meridional wind',
                long_name='meridional wind speed',
                CDI_grid_type='unstructured',
                param='0.0.0',
                number_of_grid_in_reference='1',
                coordinates='clat clon',
                scope=OutputScope.diagnostic,
            )
        )
        icon_run_config = Icon4pyRunConfig(
            dtime=datetime.timedelta(seconds=300.0),
            end_date=datetime.datetime(1, 1, 1, 0, 30, 0),
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        jabw_output_config = IconOutputConfig(
            output_time_interval=datetime.timedelta(seconds=60),
            output_file_time_interval=datetime.timedelta(seconds=60),
            output_path=pathlib.Path("./"),
            output_initial_condition_as_a_separate_file=True,
            output_variable_list=output_variable_list,
        )
        jabw_vertical_config = _jabw_vertical_config()
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.n_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            jabw_output_config,
            jabw_vertical_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )

    def _gauss3d_vertical_config():
        return v_grid.VerticalGridConfig(
            num_levels=35,
            rayleigh_damping_height=45000.0,
        )

    def _gauss3d_diffusion_config(n_substeps: int):
        return diffusion.DiffusionConfig()

    def _gauss3d_nonhydro_config(n_substeps: int):
        return solve_nh.NonHydrostaticConfig(
            igradp_method=3,
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            divdamp_fac=0.0025,
        )

    def _gauss3d_config():
        icon_run_config = Icon4pyRunConfig(
            dtime=datetime.timedelta(seconds=4.0),
            end_date=datetime.datetime(1, 1, 1, 0, 0, 4),
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        vertical_config = _gauss3d_vertical_config()
        diffusion_config = _gauss3d_diffusion_config(icon_run_config.n_substeps)
        nonhydro_config = _gauss3d_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            vertical_config,
            diffusion_config,
            nonhydro_config,
        )

    if experiment_type == driver_init.ExperimentType.JABW:
        (
            model_run_config,
            output_config,
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
            model_output_config,
            vertical_grid_config,
            diffusion_config,
            nonhydro_config,
        ) = _mch_ch_r04b09_config()
    return Icon4pyConfig(
        run_config=model_run_config,
        output_config=output_config,
        vertical_grid_config=vertical_grid_config,
        diffusion_config=diffusion_config,
        solve_nonhydro_config=nonhydro_config,
    )
