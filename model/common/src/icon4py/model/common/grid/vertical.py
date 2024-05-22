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
import math
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from gt4py.next import as_field
from gt4py.next.common import Field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.common.dimension import KDim
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerticalGridConfig:
    """
    Contains necessary parameter to configure vertical grid.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # Number of full levels.
    num_lev: Final[int]
    # Defined as max_lay_thckn in ICON namelist mo_sleve_nml. Maximum thickness of grid cells below top_height_limit_for_maximal_layer_thickness.
    maximal_layer_thickness: Final[float] = 25000.0
    # Defined as htop_thcknlimit in ICON namelist mo_sleve_nml. Height below which thickness of grid cells must not exceed maximal_layer_thickness.
    top_height_limit_for_maximal_layer_thickness: Final[float] = 15000.0
    # Defined as min_lay_thckn in ICON namelist mo_sleve_nml. Thickness of lowest level grid cells.
    lowest_layer_thickness: Final[float] = 50.0
    # Model top height.
    model_top_height: Final[float] = 23500.0
    # Defined in ICON namelist mo_sleve_nml. Height above which coordinate surfaces are flat
    flat_height: Final[float] = 16000.0
    # Defined as stretch_fac in ICON namelist mo_sleve_nml. Scaling factor for stretching/squeezing the model layer distribution.
    stretch_factor: Final[float] = 1.0
    # Defined as damp_height in ICON namelist nonhydrostatic_nml. Height [m] at which Rayleigh damping of vertical wind starts.
    rayleigh_damping_height: Final[float] = 45000.0
    # Defined in ICON namelist nonhydrostatic_nml. Height [m] above which moist physics and advection of cloud and precipitation variables are turned off.
    htop_moist_proc: Final[float] = 22500.0
    # file name containing vct_a and vct_b table
    file_path: Path = None


@dataclass(frozen=True)
class VerticalModelParams:
    """
    Contains vertical physical parameters defined on the vertical grid derived from vertical grid configuration.

    """

    vertical_config: InitVar[VerticalGridConfig]
    vct_a: Field[[KDim], float]
    vct_b: Field[[KDim], float]
    index_of_damping_layer: Final[int32] = field(init=False)
    _start_index_for_moist_physics: Final[int32] = field(init=False)
    _start_index_of_flat_layer: Final[int32] = field(init=False)
    # TODO: @nfarabullini: check this value # according to mo_init_vgrid.f90 line 329
    nflatlev: Final[int32] = None
    nflat_gradp: Final[int32] = None

    def __post_init__(self, vertical_config):
        vct_a_array = self.vct_a.ndarray
        object.__setattr__(
            self,
            "index_of_damping_layer",
            self._determine_damping_height_index(
                vct_a_array, vertical_config.rayleigh_damping_height
            ),
        )
        object.__setattr__(
            self,
            "_start_index_for_moist_physics",
            self._determine_kstart_moist(vct_a_array, vertical_config.htop_moist_proc),
        )
        object.__setattr__(
            self,
            "_start_index_of_flat_layer",
            self._determine_kstart_flat(vct_a_array, vertical_config.flat_height),
        )
        log.info(f"computation of moist physics start on layer: {self.kstart_moist}")
        log.info(f"end index of Rayleigh damping layer for w: {self.nrdmax} ")

    def __str__(self):
        vct_a_array = self.vct_a.ndarray
        nlev = vct_a_array.shape[0]
        vertical_params_description = (
            "Nominal heights of coordinate half levels and layer thicknesses (m):\n"
        )
        for k_lev in range(nlev - 1):
            vertical_params_description.join(
                f"k, vct_a, dvct: {k_lev:4d} {vct_a_array[k_lev]:12.3f} {vct_a_array[k_lev] - vct_a_array[k_lev+1]:12.3f}\n"
            )
        vertical_params_description.join(
            f"k, vct_a, dvct: {nlev-1:4d} {vct_a_array[nlev-1]:12.3f} {0.0:12.3f}"
        )
        return vertical_params_description

    @property
    def kstart_moist(self):
        """Vertical index for start level of moist physics."""
        return self._start_index_for_moist_physics

    @property
    def kstart_flat(self):
        """Vertical index for start level at which coordinate surfaces are flat."""
        return self._start_index_of_flat_layer

    @property
    def nrdmax(self):
        """Vertical index where damping starts."""
        return self.index_of_damping_layer

    @classmethod
    def _determine_kstart_moist(
        cls, vct_a: np.ndarray, top_moist_threshold: float, nshift_total: int = 0
    ) -> int32:
        n_levels = vct_a.shape[0]
        interface_height = 0.5 * (vct_a[: n_levels - 1 - nshift_total] + vct_a[1 + nshift_total :])
        return int32(xp.min(xp.where(interface_height < top_moist_threshold)[0]).item())

    @classmethod
    def _determine_damping_height_index(cls, vct_a: np.ndarray, damping_height: float):
        assert damping_height >= 0.0, "Damping height must be positive."
        return (
            0
            if damping_height > vct_a[0]
            else int32(xp.argmax(xp.where(vct_a >= damping_height)[0]).item())
        )

    @classmethod
    def _determine_kstart_flat(cls, vct_a: np.ndarray, flat_height: float) -> int32:
        return int32(xp.max(xp.where(vct_a >= flat_height)[0]).item())

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self.vct_a


def read_vct_a_and_vct_b_from_file(file_path: Path, num_lev: int):
    """
    Read vct_a and vct_b from a file.
    The file format should be as follows:
        #    k     vct_a(k) [Pa]   vct_b(k) []
        1  12000.0000000000  0.0000000000
        2  11800.0000000000  0.0000000000
        3  11600.0000000000  0.0000000000
        4  11400.0000000000  0.0000000000
        5  11200.0000000000  0.0000000000
        ...

    Args:
        file_path: Path to the vertical grid file
        num_lev: number of cell levels
    Returns:  one dimensional vct_a and vct_b arrays.
    """
    num_lev_plus_one = num_lev + 1
    vct_a = xp.zeros(num_lev_plus_one, dtype=float)
    vct_b = xp.zeros(num_lev_plus_one, dtype=float)
    with open(file_path, "r") as vertical_grid_file:
        # skip the first line that contains titles
        vertical_grid_file.readline()
        for k in range(num_lev_plus_one):
            grid_content = vertical_grid_file.readline().split()
            vct_a[k] = float(grid_content[1])
            vct_b[k] = float(grid_content[2])
    return vct_a, vct_b


def compute_vct_a_and_vct_b(vertical_config: VerticalGridConfig) -> tuple[Field[KDim], Field[KDim]]:
    """
    Compute vct_a and vct_b.
    When file_name is given in vertical_config, it will read both vct_a and vct_b from that file. Otherwise, they are analytically derived based on vertical configuration.
    When the thickness of lowest level grid cells is smaller than 0.01, a uniform vertical grid is generated based on model top height and number of levels.

    vct_a is an array that contains the height of grid interfaces (or half levels) from model surface to model top, before terrain-following coordinates are applied.

    vct_b is an array that is used to initialize vertical wind speed above surface by a prescribed vertical profile when the surface vertical wind is given.
    It is also used to modify the initial vertical wind speed above surface according to a prescribed vertical profile by linearly merging the surface vertica wind with the existing vertical wind.
    See init_w and adjust_w in mo_nh_init_utils.f90.

    Args:
        vertical_config: Vertical grid configuration
    Returns:  one dimensional (KDim) vct_a and vct_b fields.
    """

    if vertical_config.file_path is not None:
        read_vct_a_and_vct_b_from_file(vertical_config.file_path, vertical_config.num_lev)
    else:
        num_lev_plus_one = vertical_config.num_lev + 1
        if vertical_config.lowest_layer_thickness > 0.01:
            vct_a = xp.zeros(num_lev_plus_one, dtype=float)
            vct_a_exponential_factor = math.log(
                vertical_config.lowest_layer_thickness / vertical_config.model_top_height
            ) / math.log(
                2.0
                / math.acos(
                    float(vertical_config.num_lev - 1) ** vertical_config.stretch_factor
                    / float(vertical_config.num_lev) ** vertical_config.stretch_factor
                )
            )
            for k in range(num_lev_plus_one):
                vct_a[k] = (
                    vertical_config.model_top_height
                    * (
                        2.0
                        / math.pi
                        * math.acos(
                            float(vertical_config.num_lev - 1) ** vertical_config.stretch_factor
                            / float(vertical_config.num_lev) ** vertical_config.stretch_factor
                        )
                    )
                    ** vct_a_exponential_factor
                )

            if (
                vertical_config.maximal_layer_thickness
                > 2.0 * vertical_config.lowest_layer_thickness
                and vertical_config.maximal_layer_thickness
                < 0.5 * vertical_config.top_height_limit_for_maximal_layer_thickness
            ):
                layer_thickness = vct_a[: num_lev_plus_one - 1] - vct_a[1:]
                lowest_level_exceeding_limit = xp.max(
                    xp.where(layer_thickness > vertical_config.maximal_layer_thickness)
                )
                modified_vct_a = xp.zeros(num_lev_plus_one, dtype=float)
                lowest_level_unmodified_thickness = 0
                shifted_levels = 0
                for k in range(vertical_config.num_lev, -1, -1):
                    if (
                        modified_vct_a[k + 1]
                        < vertical_config.top_height_limit_for_maximal_layer_thickness
                    ):
                        modified_vct_a[k] = modified_vct_a[k + 1] + xp.min(
                            vertical_config.maximal_layer_thickness, layer_thickness[k]
                        )
                    elif lowest_level_unmodified_thickness == 0:
                        lowest_level_unmodified_thickness = k + 1
                        shifted_levels = max(
                            0, lowest_level_exceeding_limit - lowest_level_unmodified_thickness
                        )
                        modified_vct_a[k] = (
                            modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]
                        )
                    else:
                        modified_vct_a[k] = (
                            modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]
                        )

                stretchfac = (
                    1.0
                    if shifted_levels == 0
                    else (
                        vct_a[0]
                        - modified_vct_a[lowest_level_unmodified_thickness]
                        - float(lowest_level_unmodified_thickness - 1)
                        * vertical_config.maximal_layer_thickness
                    )
                    / (
                        modified_vct_a[0]
                        - modified_vct_a[lowest_level_unmodified_thickness]
                        - float(lowest_level_unmodified_thickness - 1)
                        * vertical_config.maximal_layer_thickness
                    )
                )

                for k in range(vertical_config.num_lev, -1, -1):
                    if vct_a[k + 1] < vertical_config.top_height_limit_for_maximal_layer_thickness:
                        vct_a[k] = vct_a[k + 1] + xp.min(
                            vertical_config.maximal_layer_thickness, layer_thickness[k]
                        )
                    else:
                        vct_a[k] = (
                            vct_a[k + 1]
                            + vertical_config.maximal_layer_thickness
                            + (
                                layer_thickness[k + shifted_levels]
                                - vertical_config.maximal_layer_thickness
                            )
                            * stretchfac
                        )

                # Try to apply additional smoothing on the stretching factor above the constant-thickness layer
                if stretchfac != 1.0 and lowest_level_exceeding_limit < vertical_config.num_lev - 4:
                    for k in range(vertical_config.num_lev, -1, -1):
                        if (
                            modified_vct_a[k + 1]
                            < vertical_config.top_height_limit_for_maximal_layer_thickness
                        ):
                            modified_vct_a[k] = vct_a[k]
                        else:
                            modified_layer_thickness = xp.min(
                                1.025 * (vct_a[k] - vct_a[k + 1]),
                                1.025
                                * (
                                    modified_vct_a[lowest_level_exceeding_limit + 1]
                                    - modified_vct_a[lowest_level_exceeding_limit + 2]
                                )
                                / (
                                    modified_vct_a[lowest_level_exceeding_limit + 2]
                                    - modified_vct_a[lowest_level_exceeding_limit + 3]
                                )
                                * (modified_vct_a[k + 1] - modified_vct_a[k + 2]),
                            )
                            modified_vct_a[k] = xp.min(
                                vct_a[k], modified_vct_a[k + 1] + modified_layer_thickness
                            )
                    if modified_vct_a[0] == vct_a[0]:
                        vct_a[0:2] = modified_vct_a[0:2]
                        vct_a[
                            lowest_level_unmodified_thickness + 1 : vertical_config.num_lev + 1
                        ] = modified_vct_a[
                            lowest_level_unmodified_thickness + 1 : vertical_config.num_lev + 1
                        ]
                        vct_a[3 : lowest_level_unmodified_thickness + 1] = 0.5 * (
                            modified_vct_a[2:lowest_level_unmodified_thickness]
                            + modified_vct_a[4 : lowest_level_unmodified_thickness + 2]
                        )
        else:
            vct_a = (
                vertical_config.model_top_height
                * (
                    float(vertical_config.num_lev)
                    - xp.arange(vertical_config.num_lev + 1, dtype=float)
                )
                / float(vertical_config.num_lev)
            )
        vct_b = math.exp(-vct_a / 5000.0)

    assert xp.allclose(
        vct_a[0], vertical_config.model_top_height
    ), "vct_a[0] is not equal to model top height in vertical configuration. Please check again."

    return as_field((KDim,), vct_a), as_field((KDim,), vct_b)
