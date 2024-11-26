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
from enum import Enum
from pathlib import Path

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
    units: str = " "
    standard_name: str = " "
    long_name: str = " "
    CDI_grid_type: str = " "
    param: str = " "
    number_of_grid_in_reference: str = " "
    coordinates: str = " "
    scope: str = " "


class OutputDimension(str, Enum):
    CELL_DIM = "ncells"
    EDGE_DIM = "ncells_2"
    VERTEX_DIM = "ncells_3"
    FULL_LEVEL = "height"
    HALF_LEVEL = "height_2"
    TIME = "time"


@dataclass(frozen=True)
class VariableDimension:
    horizon_dimension: str = None
    vertical_dimension: str = None
    time_dimension: str = None


class OutputScope(str, Enum):
    prognostic = "prognostic"
    diagnostic = "diagnostic"
    diffusion = "diffusion"
    solve_nonhydro = "solve_nonhydro"


class OutputVariableList:
    def __init__(self):
        self._variable_name = ("vn", "w", "rho", "theta_v", "exner")
        self._variable_dimension = {
            "vn": VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            "w": VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            "rho": VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            "theta_v": VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            "exner": VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
        }
        self._variable_attribute = {
            "vn": VariableAttributes(
                units="m s-1",
                standard_name="normal velocity",
                long_name="normal wind speed at edge center",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.prognostic,
            ),
            "w": VariableAttributes(
                units="m s-1",
                standard_name="vertical velocity",
                long_name="vertical wind speed at half levels",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.prognostic,
            ),
            "rho": VariableAttributes(
                units="kg m-3",
                standard_name="density",
                long_name="air density",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.prognostic,
            ),
            "theta_v": VariableAttributes(
                units="K",
                standard_name="virtual temperature",
                long_name="virtual temperature",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.prognostic,
            ),
            "exner": VariableAttributes(
                units="",
                standard_name="exner function",
                long_name="exner function",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
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
        self,
        variable_name: str,
        variable_dimenson: VariableDimension,
        variable_attribute: VariableAttributes,
    ) -> None:
        if variable_name in self._variable_name:
            log.warning(
                f"Output variable name {variable_name} is already in variable list {self._variable_name}. Nothing to do."
            )
            return
        self._variable_name = self._variable_name + (variable_name,)
        self._variable_attribute[variable_name] = variable_attribute
        self._variable_dimension[variable_name] = variable_dimenson


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
            IconOutputConfig(),
            _mch_ch_r04b09_diffusion_config(),
            _mch_ch_r04b09_nonhydro_config(),
        )

    def _output_config():
        output_variable_list = OutputVariableList()
        output_variable_list.add_new_variable(
            "temperature",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="K",
                standard_name="temperauture",
                long_name="air temperauture",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "pressure",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="Pa",
                standard_name="pressure",
                long_name="air pressure",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "pressure_sfc",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="Pa",
                standard_name="surface pressure",
                long_name="surface pressure",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "u",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="zonal wind",
                long_name="zonal wind speed",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "v",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="meridional wind",
                long_name="meridional wind speed",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )

        ######################################################################################
        output_variable_list.add_new_variable(
            "predictor_theta_v_e",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="K",
                standard_name="virtual temperature at edge",
                long_name="virtual temperature at edge center",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_pressure_grad",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="pressure gradient",
                long_name="horizontal gradient of pressure in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_advection",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="advection",
                long_name="advection term in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        ######################################################################################
        output_variable_list.add_new_variable(
            "corrector_theta_v_e",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="K",
                standard_name="virtual temperature at edge",
                long_name="virtual temperature at edge center",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_pressure_grad",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="pressure gradient",
                long_name="horizontal gradient of pressure in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_advection",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="advection",
                long_name="advection term in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        ######################################################################################
        output_variable_list.add_new_variable(
            "predictor_hgrad_kinetic",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="hgrad_kinetic",
                long_name="horizontal gradient of kinetic energy in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_tangent_wind",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="tangential wind speed",
                long_name="tangential wind speed in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_total_vorticity",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="total vorticity",
                long_name="total vorticity in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_vertical_wind",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="vertical wind speed at edge",
                long_name="vertical wind speed in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "predictor_vgrad_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="vertical gradient of normal wind at edge",
                long_name="vertical gradient of normal wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        ######################################################################################
        output_variable_list.add_new_variable(
            "corrector_hgrad_kinetic",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-2",
                standard_name="hgrad_kinetic",
                long_name="horizontal gradient of kinetic energy in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_tangent_wind",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="tangential wind speed",
                long_name="tangential wind speed in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_total_vorticity",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="total vorticity",
                long_name="total vorticity in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_vertical_wind",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="vertical wind speed at edge",
                long_name="vertical wind speed in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "corrector_vgrad_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="vertical gradient of normal wind at edge",
                long_name="vertical gradient of normal wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        ######################################################################################
        output_variable_list.add_new_variable(
            "graddiv_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="laplacian of normal wind",
                long_name="laplacian of normal wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "graddiv2_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m-1 s-1",
                standard_name="double laplacian of normal wind",
                long_name="double laplacian of normal wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "graddiv2_normal",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="horizontal gradient of double laplacian",
                long_name="horizontal gradient of double laplacian of wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "graddiv2_vertical",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="vertical fradient of double laplacian of wind",
                long_name="vertical fradient of double laplacian of wind in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "scal_divdamp",
            VariableDimension(
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m2",
                standard_name="4th order divdamp scaling factor",
                long_name="fourth order divergence damping scaling factor in horizontal momentum equation",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "before_flxdiv_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="initial wind divergence",
                long_name="divergence of wind before divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "after_flxdiv_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="s-1",
                standard_name="final wind divergence",
                long_name="divergence of wind after divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "before_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="initial normal wind",
                long_name="normal wind before divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "after_vn",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="final normal wind",
                long_name="normal wind after divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "before_w",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="initial vertical wind",
                long_name="vertical wind before divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "after_w",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="final vertical wind",
                long_name="vertical wind after divergence damping",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )
        # output_variable_list.add_new_variable(
        #     "corrector_flxdiv_vn",
        #     VariableDimension(
        #         horizon_dimension=OutputDimension.CELL_DIM,
        #         vertical_dimension=OutputDimension.FULL_LEVEL,
        #         time_dimension=OutputDimension.TIME,
        #     ),
        #     VariableAttributes(
        #         units="s-1",
        #         standard_name="corrector wind divergence",
        #         long_name="divergence of wind after w update in the corrector step",
        #         CDI_grid_type="unstructured",
        #         param="0.0.0",
        #         number_of_grid_in_reference="1",
        #         coordinates="clat clon",
        #         scope=OutputScope.diagnostic,
        #     ),
        # )
        output_variable_list.add_new_variable(
            "nabla2_diff",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="2nd order normal wind diffusion",
                long_name="second order diffusion of normal wind",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "nabla4_diff",
            VariableDimension(
                horizon_dimension=OutputDimension.EDGE_DIM,
                vertical_dimension=OutputDimension.FULL_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="4th order normal wind diffusion",
                long_name="fourth order diffusion of normal wind",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="elat elon",
                scope=OutputScope.diagnostic,
            ),
        )
        output_variable_list.add_new_variable(
            "w_nabla4_diff",
            VariableDimension(
                horizon_dimension=OutputDimension.CELL_DIM,
                vertical_dimension=OutputDimension.HALF_LEVEL,
                time_dimension=OutputDimension.TIME,
            ),
            VariableAttributes(
                units="m s-1",
                standard_name="4th order normal wind diffusion",
                long_name="fourth order diffusion of vertical wind",
                CDI_grid_type="unstructured",
                param="0.0.0",
                number_of_grid_in_reference="1",
                coordinates="clat clon",
                scope=OutputScope.diagnostic,
            ),
        )

        return IconOutputConfig(
            # output_time_interval=timedelta(seconds=14400),
            # output_file_time_interval=timedelta(seconds=14400),
            output_time_interval=timedelta(seconds=180),
            output_file_time_interval=timedelta(seconds=180),
            # output_time_interval=timedelta(seconds=12),
            # output_file_time_interval=timedelta(seconds=12),
            output_path=Path("./"),
            output_initial_condition_as_a_separate_file=True,
            output_variable_list=output_variable_list,
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
            # hdiff_efdt_ratio=1000.0,
            # hdiff_w_efdt_ratio=1000.0,
            # hdiff_efdt_ratio=1.e15,
            # hdiff_w_efdt_ratio=1.e15,
            # smagorinski_scaling_factor=0.025,
            smagorinski_scaling_factor=0.015,
            # smagorinski_scaling_factor=0.0,
            # smagorinski_scaling_factor=0.0000025,
            zdiffu_t=False,
            velocity_boundary_diffusion_denom=200.0,
            max_nudging_coeff=0.075,
            call_frequency=1,
        )

    def _jabw_nonhydro_config(n_substeps: int):
        return NonHydrostaticConfig(
            # original igradp_method is 2
            # original divdamp_order is 4
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            divdamp_fac=0.0025,
            # divdamp_fac=0.00025,
            # divdamp_fac=0.0,
            do_o2_divdamp=False,
            do_3d_divergence_damping=False,
            do_second_order_3d_divergence_damping=True,
            do_multiple_divdamp=True,
            number_of_divdamp_step=10,
        )
    
    def _gauss3d_diffusion_config(n_substeps: int):
        return DiffusionConfig(
            diffusion_type=DiffusionType.SMAGORINSKY_4TH_ORDER,
            hdiff_w=True,
            hdiff_vn=True,
            hdiff_temp=False,
            n_substeps=n_substeps,
            type_t_diffu=2,
            type_vn_diffu=1,
            hdiff_efdt_ratio=36.0,
            hdiff_w_efdt_ratio=15.0,
            # hdiff_efdt_ratio=1.e15,
            # hdiff_w_efdt_ratio=1.e15,
            # hdiff_efdt_ratio=1000.0,
            # hdiff_w_efdt_ratio=1000.0,
            smagorinski_scaling_factor=0.015,
            # smagorinski_scaling_factor=0.025,
            # smagorinski_scaling_factor=0.0,
            # smagorinski_scaling_factor=0.0000025,
            zdiffu_t=False,
            velocity_boundary_diffusion_denom=200.0,
            max_nudging_coeff=0.02,
            call_frequency=1,
        )

    def _gauss3d_nonhydro_config(n_substeps: int):
        return NonHydrostaticConfig(
            igradp_method=3,
            ndyn_substeps_var=n_substeps,
            max_nudging_coeff=0.02,
            rayleigh_coeff=0.1,
            divdamp_fac=0.0025,
            divdamp_z=40000.0,
            divdamp_z2=50000.0,
            scal_divsign=1.0,
            do_o2_divdamp=False,
            do_3d_divergence_damping=True,
            do_second_order_3d_divergence_damping=False,
            do_multiple_divdamp=True,
            number_of_divdamp_step=100,
        )
    
    def _jablownoski_Williamson_config():
        icon_run_config = IconRunConfig(
            dtime=timedelta(seconds=600.0),
            end_date=datetime(1, 1, 30, 0, 0, 0),
            # end_date=datetime(1, 1, 1, 0, 30, 0),
            damping_height=45000.0,
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        jabw_output_config = _output_config()
        jabw_diffusion_config = _jabw_diffusion_config(icon_run_config.n_substeps)
        jabw_nonhydro_config = _jabw_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            jabw_output_config,
            jabw_diffusion_config,
            jabw_nonhydro_config,
        )
    
    def _gauss3d_config():
        icon_run_config = IconRunConfig(
            dtime=timedelta(seconds=1.5),
            end_date=datetime(1, 1, 1, 0, 42, 0),
            # end_date=datetime(1, 1, 1, 0, 0, 6),
            damping_height=25000.0,
            apply_initial_stabilization=False,
            n_substeps=5,
        )
        gauss3d_output_config = _output_config()
        gauss3d_diffusion_config = _gauss3d_diffusion_config(icon_run_config.n_substeps)
        gauss3d_nonhydro_config = _gauss3d_nonhydro_config(icon_run_config.n_substeps)
        return (
            icon_run_config,
            gauss3d_output_config,
            gauss3d_diffusion_config,
            gauss3d_nonhydro_config,
        )

    if experiment_type == ExperimentType.JABW:
        (
            model_run_config,
            output_config,
            diffusion_config,
            nonhydro_config,
        ) = _jablownoski_Williamson_config()
    elif experiment_type == ExperimentType.GAUSS3D:
        (
            model_run_config,
            output_config,
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
