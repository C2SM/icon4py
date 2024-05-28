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
    num_lev: int
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

    vct_a and vct_b: see docstring of compute_vct_a_and_vct_b.
    index_of_damping_layer: height index above which Rayleigh damping of vertical wind is applied.
    _start_index_for_moist_physics: Height index above which moist physics and advection of cloud and precipitation variables are turned off.
    _start_index_of_flat_layer:
    nflatlev: height index: height index above which coordinate surfaces are flat.
    """

    vertical_config: InitVar[VerticalGridConfig]
    vct_a: Field[[KDim], float]
    vct_b: Field[[KDim], float]
    index_of_damping_layer: Final[int32] = field(init=False)
    _start_index_for_moist_physics: Final[int32] = field(init=False)
    _end_index_of_flat_layer: Final[int32] = field(init=False)
    # TODO: @nfarabullini: check this value # according to mo_init_vgrid.f90 line 329
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
            "_end_index_of_flat_layer",
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
            vertical_params_description += f"k, vct_a, dvct: {k_lev:4d} {vct_a_array[k_lev]:12.3f} {vct_a_array[k_lev] - vct_a_array[k_lev+1]:12.3f}\n"
        vertical_params_description += (
            f"k, vct_a, dvct: {nlev-1:4d} {vct_a_array[nlev-1]:12.3f} {0.0:12.3f}"
        )
        return vertical_params_description

    @property
    def kstart_moist(self):
        """Vertical index for start level of moist physics."""
        return self._start_index_for_moist_physics

    @property
    def nflatlev(self):
        """Vertical index for bottommost level at which coordinate surfaces are flat."""
        return self._end_index_of_flat_layer

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
        assert flat_height >= 0.0, "Flat surface height must be positive."
        return (
            0 if flat_height > vct_a[0] else int32(xp.max(xp.where(vct_a >= flat_height)[0]).item())
        )

    @property
    def physical_heights(self) -> Field[[KDim], float]:
        return self.vct_a


def read_vct_a_and_vct_b_from_file(file_path: Path, num_lev: int):
    """
    Read vct_a and vct_b from a file.
    The file format should be as follows (the same format used for icon):
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
    try:
        with open(file_path, "r") as vertical_grid_file:
            # skip the first line that contains titles
            vertical_grid_file.readline()
            for k in range(num_lev_plus_one):
                grid_content = vertical_grid_file.readline().split()
                vct_a[k] = float(grid_content[1])
                vct_b[k] = float(grid_content[2])
    except OSError as err:
        raise FileNotFoundError(
            f"Vertical coord table file {file_path} could not be read."
        ) from err
    except IndexError as err:
        raise IndexError(
            f"The number of levels in the vertical coord table file {file_path} is possibly not the same as {num_lev_plus_one} or data is missing at {k}-th line."
        ) from err
    except ValueError as err:
        raise ValueError(f"data is not float at {k}-th line.") from err
    return vct_a, vct_b


def get_vct_a_and_vct_b(vertical_config: VerticalGridConfig):
    """
    Compute vct_a and vct_b.
    vct_a is an array that contains the height of grid interfaces (or half levels) from model surface to model top, before terrain-following coordinates are applied.
    vct_b is an array that is used to initialize vertical wind speed above surface by a prescribed vertical profile when the surface vertical wind is given.
    It is also used to modify the initial vertical wind speed above surface according to a prescribed vertical profile by linearly merging the surface vertica wind with the existing vertical wind.
    See init_w and adjust_w in mo_nh_init_utils.f90.

    When file_name is given in vertical_config, it will read both vct_a and vct_b from that file. Otherwise, they are analytically derived based on vertical configuration.

    When the thickness of lowest level grid cells is larger than 0.01:
        vct_a[k] = H (2/pi arccos((k/N)^s) )^d      N = num_lev      s = stretch_factor in vertical configuration
        d is such that the lowest level grid cell thickness is equal to lowest_layer_thickness in vertical configuration.
        d = ln(lowest_layer_thickness/H) / ln(2/pi arccos(((N-1)/N)^s) )
        When top_height_limit_for_maximal_layer_thicknessis larger than model_top_height, the final model top height will be adjusted if there are layers thicker than maximal_layer_thickness.
        Otherwise, layers above top_height_limit_for_maximal_layer_thickness will be modified such that h_0 is equal to model_top_height.
         ------- 0     model top, z_0 = Z, h_0 = H, Z is the model top height after layer thickness > m is reduced to m, and H is model_top_height.
        ...
         ------- k-2
        k-2
         ------- k-1
        k-1
         ------- k
        k            ------ top_height_limit_for_maximal_layer_thickness
         ------- k+1
        k+1
         ------- k+2
        k+2
         ------- k+3
        h_k..N+1 = z_k..N+1,  h and z are the same at levels k to N+1.
        h_k-1 = h_k + m + (dh_k-1 - m) * a, the stretch factor aims to stretch layer thickness different from m. dh_k-1 = dh_k-1+shift, shift is equal to (k + s) - k, where k+s is the lowest layer index that needs to be modified, this is to prevent sudden jump.
        h_0 = z_k + a * sum_i=0^k-1 (dh_i) + k * (1 - a) * m = H, h is vct_a before layer thickness > m is reduced to m, z is modified_vct_a after layer thickness > m is reduced, m = maximal_layer_thickness. dh_i is the layer thickness of vct_a.
        z_0 = z_k + sum_i=0^k-1 (dh_i) = Z.
        Hence, the stretch factor a = (H - z_k - m * k) / (Z - z_k - m * k).

        An additional smoothing is performed for levels 2, 3, 4, ..., k if modified_vct_a[0] after additional smoothing is still the same as model_top_height.

        THERE IS A WARNING MESSAGE IF vct_a[0] AND model_top_height ARE NOT EQUAL.

    When the thickness of lowest level grid cells is smaller than or equal to 0.01:
        a uniform vertical grid is generated based on model top height and number of levels.

    Args:
        vertical_config: Vertical grid configuration
    Returns:  one dimensional (KDim) vct_a and vct_b gt4py fields.
    """

    if vertical_config.file_path is not None:
        vct_a, vct_b = read_vct_a_and_vct_b_from_file(
            vertical_config.file_path, vertical_config.num_lev
        )
    else:
        num_lev_plus_one = vertical_config.num_lev + 1
        if vertical_config.lowest_layer_thickness > 0.01:
            vct_a_exponential_factor = xp.log(
                vertical_config.lowest_layer_thickness / vertical_config.model_top_height
            ) / xp.log(
                2.0
                / math.pi
                * xp.arccos(
                    float(vertical_config.num_lev - 1) ** vertical_config.stretch_factor
                    / float(vertical_config.num_lev) ** vertical_config.stretch_factor
                )
            )

            vct_a = (
                vertical_config.model_top_height
                * (
                    2.0
                    / math.pi
                    * xp.arccos(
                        xp.arange(vertical_config.num_lev + 1, dtype=float)
                        ** vertical_config.stretch_factor
                        / float(vertical_config.num_lev) ** vertical_config.stretch_factor
                    )
                )
                ** vct_a_exponential_factor
            )

            if (
                2.0 * vertical_config.lowest_layer_thickness
                < vertical_config.maximal_layer_thickness
                < 0.5 * vertical_config.top_height_limit_for_maximal_layer_thickness
            ):
                layer_thickness = vct_a[: num_lev_plus_one - 1] - vct_a[1:]
                lowest_level_exceeding_limit = xp.max(
                    xp.where(layer_thickness > vertical_config.maximal_layer_thickness)
                )
                modified_vct_a = xp.zeros(num_lev_plus_one, dtype=float)
                lowest_level_unmodified_thickness = 0
                shifted_levels = 0
                for k in range(vertical_config.num_lev - 1, -1, -1):
                    if (
                        modified_vct_a[k + 1]
                        < vertical_config.top_height_limit_for_maximal_layer_thickness
                    ):
                        modified_vct_a[k] = modified_vct_a[k + 1] + xp.minimum(
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
                        - float(lowest_level_unmodified_thickness)
                        * vertical_config.maximal_layer_thickness
                    )
                    / (
                        modified_vct_a[0]
                        - modified_vct_a[lowest_level_unmodified_thickness]
                        - float(lowest_level_unmodified_thickness)
                        * vertical_config.maximal_layer_thickness
                    )
                )

                for k in range(vertical_config.num_lev - 1, -1, -1):
                    if vct_a[k + 1] < vertical_config.top_height_limit_for_maximal_layer_thickness:
                        vct_a[k] = vct_a[k + 1] + xp.minimum(
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
                    for k in range(vertical_config.num_lev - 1, -1, -1):
                        if (
                            modified_vct_a[k + 1]
                            < vertical_config.top_height_limit_for_maximal_layer_thickness
                        ):
                            modified_vct_a[k] = vct_a[k]
                        else:
                            modified_layer_thickness = xp.minimum(
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
                            modified_vct_a[k] = xp.minimum(
                                vct_a[k], modified_vct_a[k + 1] + modified_layer_thickness
                            )
                    if modified_vct_a[0] == vct_a[0]:
                        vct_a[0:2] = modified_vct_a[0:2]
                        vct_a[
                            lowest_level_unmodified_thickness + 1 : vertical_config.num_lev
                        ] = modified_vct_a[
                            lowest_level_unmodified_thickness + 1 : vertical_config.num_lev
                        ]
                        vct_a[2 : lowest_level_unmodified_thickness + 1] = 0.5 * (
                            modified_vct_a[1:lowest_level_unmodified_thickness]
                            + modified_vct_a[3 : lowest_level_unmodified_thickness + 2]
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
        vct_b = xp.exp(-vct_a / 5000.0)

    if not xp.allclose(vct_a[0], vertical_config.model_top_height):
        log.warning(
            f" Warning. vct_a[0], {vct_a[0]}, is not equal to model top height, {vertical_config.model_top_height}, of vertical configuration. Please consider changing the vertical setting."
        )

    return as_field((KDim,), vct_a), as_field((KDim,), vct_b)
