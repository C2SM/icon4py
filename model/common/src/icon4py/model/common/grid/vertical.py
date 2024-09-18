# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import functools
import logging
import math
import pathlib
from typing import Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, exceptions, field_type_aliases as fa
from icon4py.model.common.settings import xp


log = logging.getLogger(__name__)


class Zone(enum.Enum):
    """
    Vertical zone markers to be used to select vertical domain indices.

    The values here are taken from the special indices computed in `VerticalGridParams`.
    """

    TOP = 0
    BOTTOM = 1
    DAMPING = 2
    MOIST = 3
    FLAT = 4


@dataclasses.dataclass(frozen=True)
class Domain:
    """
    Simple data class used to specify a vertical domain such that index lookup and domain specification can be separated.
    """

    dim: dims.VerticalDim
    marker: Zone
    offset: int = 0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert self.dim.kind == gtx.DimensionKind.VERTICAL
        if self.marker == Zone.TOP:
            assert (
                self.offset >= 0
            ), f"{self.marker} needs to be combined with positive offest, but offset = {self.offset}"


@dataclasses.dataclass(frozen=True)
class VerticalGridConfig:
    """
    Contains necessary parameter to configure vertical grid.

    Encapsulates namelist parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    #: Number of full levels.
    num_levels: int
    #: Defined as max_lay_thckn in ICON namelist mo_sleve_nml. Maximum thickness of grid cells below top_height_limit_for_maximal_layer_thickness.
    maximal_layer_thickness: Final[float] = 25000.0
    #: Defined as htop_thcknlimit in ICON namelist mo_sleve_nml. Height below which thickness of grid cells must not exceed maximal_layer_thickness.
    top_height_limit_for_maximal_layer_thickness: Final[float] = 15000.0
    #: Defined as min_lay_thckn in ICON namelist mo_sleve_nml. Thickness of lowest level grid cells.
    lowest_layer_thickness: Final[float] = 50.0
    #: Model top height.
    model_top_height: Final[float] = 23500.0
    #: Defined in ICON namelist mo_sleve_nml. Height above which coordinate surfaces are flat
    flat_height: Final[float] = 16000.0
    #: Defined as stretch_fac in ICON namelist mo_sleve_nml. Scaling factor for stretching/squeezing the model layer distribution.
    stretch_factor: Final[float] = 1.0
    #: Defined as damp_height in ICON namelist nonhydrostatic_nml. Height [m] at which Rayleigh damping of vertical wind starts.
    rayleigh_damping_height: Final[float] = 45000.0
    #: Defined in ICON namelist nonhydrostatic_nml. Height [m] above which moist physics and advection of cloud and precipitation variables are turned off.
    htop_moist_proc: Final[float] = 22500.0
    #: file name containing vct_a and vct_b table
    file_path: pathlib.Path = None


@dataclasses.dataclass(frozen=True)
class VerticalGrid:
    """
    Contains vertical physical parameters defined on the vertical grid derived from vertical grid configuration.

    _vct_a and _vct_b: See docstring of get_vct_a_and_vct_b. Note that the height index starts from the model top.
    _end_index_of_damping_layer: Height index above which Rayleigh damping of vertical wind is applied.
    _start_index_for_moist_physics: Height index above which moist physics and advection of cloud and precipitation variables are turned off.
    _end_index_of_flat_layer: Height index above which coordinate surfaces are flat.
    _min_index_flat_horizontal_grad_pressure: The minimum height index at which the height of the center of an edge lies within two neighboring cells so that horizontal pressure gradient can be computed by first order discretization scheme.
    """

    config: VerticalGridConfig
    vct_a: dataclasses.InitVar[fa.KField[float]]
    vct_b: dataclasses.InitVar[fa.KField[float]]
    _vct_a: fa.KField[float] = dataclasses.field(init=False)
    _vct_b: fa.KField[float] = dataclasses.field(init=False)
    _end_index_of_damping_layer: Final[gtx.int32] = dataclasses.field(init=False)
    _start_index_for_moist_physics: Final[gtx.int32] = dataclasses.field(init=False)
    _end_index_of_flat_layer: Final[gtx.int32] = dataclasses.field(init=False)
    _min_index_flat_horizontal_grad_pressure: Final[gtx.int32] = None

    def __post_init__(self, vct_a, vct_b):
        object.__setattr__(
            self,
            "_vct_a",
            vct_a,
        )
        object.__setattr__(
            self,
            "_vct_b",
            vct_b,
        )
        vct_a_array = self._vct_a.ndarray
        object.__setattr__(
            self,
            "_end_index_of_damping_layer",
            self._determine_damping_height_index(vct_a_array, self.config.rayleigh_damping_height),
        )
        object.__setattr__(
            self,
            "_start_index_for_moist_physics",
            self._determine_start_level_of_moist_physics(vct_a_array, self.config.htop_moist_proc),
        )
        object.__setattr__(
            self,
            "_end_index_of_flat_layer",
            self._determine_end_index_of_flat_layers(vct_a_array, self.config.flat_height),
        )
        log.info(f"computation of moist physics start on layer: {self.kstart_moist}")
        log.info(f"end index of Rayleigh damping layer for w: {self.nrdmax} ")

    def __str__(self) -> str:
        vertical_params_properties = ["Model interface height properties:"]
        for key, value in self.metadata_interface_physical_height.items():
            vertical_params_properties.append(f"    {key}: {value}")
        vertical_params_properties.append("Level    Coordinate    Thickness:")
        vct_a_array = self._vct_a.ndarray
        dvct = vct_a_array[:-1] - vct_a_array[1:]
        array_value = [f"   0   {vct_a_array[0]:12.3f}"]
        for k in range(vct_a_array.shape[0] - 1):
            array_value.append(f"{k+1:4d}   {vct_a_array[k+1]:12.3f} {dvct[k]:12.3f}")
        array_value[self._end_index_of_flat_layer] += " End of flat layer "
        array_value[self._end_index_of_damping_layer] += " End of damping layer "
        array_value[self._start_index_for_moist_physics] += " Start of moist physics"
        vertical_params_properties.extend(array_value)
        return "\n".join(vertical_params_properties)

    @property
    def metadata_interface_physical_height(self) -> dict:
        return dict(
            standard_name="model_interface_height",
            long_name="height value of half levels without topography",
            units="m",
            positive="up",
            icon_var_name="vct_a",
        )

    def index(self, domain: Domain) -> gtx.int32:
        match domain.marker:
            case Zone.TOP:
                index = gtx.int32(0)
            case Zone.BOTTOM:
                index = self._bottom_level(domain)
            case Zone.MOIST:
                index = self._start_index_for_moist_physics
            case Zone.FLAT:
                index = self._end_index_of_flat_layer
            case Zone.DAMPING:
                index = self._end_index_of_damping_layer
            case _:
                raise exceptions.IconGridError(f"not a valid vertical zone: {domain.marker}")

        index += domain.offset
        assert (
            0 <= index <= self._bottom_level(domain)
        ), f"vertical index {index} outside of grid levels for {domain.dim}"
        return index

    def _bottom_level(self, domain: Domain) -> gtx.int32:
        return gtx.int32(self.size(domain.dim))

    @property
    def interface_physical_height(self) -> fa.KField[float]:
        return self._vct_a

    @functools.cached_property
    def kstart_moist(self) -> gtx.int32:
        """Vertical index for start level of moist physics."""
        return self.index(Domain(dims.KDim, Zone.MOIST))

    @functools.cached_property
    def nflatlev(self) -> gtx.int32:
        """Vertical index for bottom most level at which coordinate surfaces are flat."""
        return self.index(Domain(dims.KDim, Zone.FLAT))

    @functools.cached_property
    def nrdmax(self) -> gtx.int32:
        """Vertical index where damping starts."""
        return self.end_index_of_damping_layer

    @functools.cached_property
    def end_index_of_damping_layer(self) -> gtx.int32:
        """Vertical index where damping starts."""
        return self.index(Domain(dims.KDim, Zone.DAMPING))

    @property
    def nflat_gradp(self) -> gtx.int32:
        return self._min_index_flat_horizontal_grad_pressure

    def size(self, dim: dims.VerticalDim) -> int:
        assert dim.kind == gtx.DimensionKind.VERTICAL, "Only vertical dimensions are supported"
        match dim:
            case dims.KDim:
                return self.config.num_levels
            case dims.KHalfDim:
                return self.config.num_levels + 1
            case _:
                raise ValueError(f"Unkown dimension {dim}.")

    @classmethod
    def _determine_start_level_of_moist_physics(
        cls, vct_a: xp.ndarray, top_moist_threshold: float, nshift_total: int = 0
    ) -> gtx.int32:
        n_levels = vct_a.shape[0]
        interface_height = 0.5 * (vct_a[: n_levels - 1 - nshift_total] + vct_a[1 + nshift_total :])
        return gtx.int32(xp.min(xp.where(interface_height < top_moist_threshold)[0]).item())

    @classmethod
    def _determine_damping_height_index(cls, vct_a: xp.ndarray, damping_height: float) -> gtx.int32:
        assert damping_height >= 0.0, "Damping height must be positive."
        return (
            0
            if damping_height > vct_a[0]
            else gtx.int32(xp.argmax(xp.where(vct_a >= damping_height)[0]).item())
        )

    @classmethod
    def _determine_end_index_of_flat_layers(
        cls, vct_a: xp.ndarray, flat_height: float
    ) -> gtx.int32:
        assert flat_height >= 0.0, "Flat surface height must be positive."
        return (
            0
            if flat_height > vct_a[0]
            else gtx.int32(xp.max(xp.where(vct_a >= flat_height)[0]).item())
        )


def _read_vct_a_and_vct_b_from_file(
    file_path: pathlib.Path, num_levels: int
) -> tuple[fa.KField, fa.KField]:
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
        num_levels: number of cell levels
    Returns:  one dimensional vct_a and vct_b arrays.
    """
    num_levels_plus_one = num_levels + 1
    vct_a = xp.zeros(num_levels_plus_one, dtype=float)
    vct_b = xp.zeros(num_levels_plus_one, dtype=float)
    try:
        with open(file_path, "r") as vertical_grid_file:
            # skip the first line that contains titles
            vertical_grid_file.readline()
            for k in range(num_levels_plus_one):
                grid_content = vertical_grid_file.readline().split()
                vct_a[k] = float(grid_content[1])
                vct_b[k] = float(grid_content[2])
    except OSError as err:
        raise FileNotFoundError(
            f"Vertical coord table file {file_path} could not be read."
        ) from err
    except IndexError as err:
        raise IndexError(
            f"The number of levels in the vertical coord table file {file_path} is possibly not the same as {num_levels_plus_one} or data is missing at {k}-th line."
        ) from err
    except ValueError as err:
        raise ValueError(f"data is not float at {k}-th line.") from err
    return gtx.as_field((dims.KDim,), vct_a), gtx.as_field((dims.KDim,), vct_b)


def _compute_vct_a_and_vct_b(vertical_config: VerticalGridConfig) -> tuple[fa.KField, fa.KField]:
    """
    Compute vct_a and vct_b.

    When the thickness of lowest level grid cells is larger than 0.01:
        vct_a[k] = H (2/pi arccos((k/N)^s) )^d      N = num_levels      s = stretch_factor in vertical configuration
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
    Returns:  one dimensional (dims.KDim) vct_a and vct_b gt4py fields.
    """
    num_levels_plus_one = vertical_config.num_levels + 1
    if vertical_config.lowest_layer_thickness > 0.01:
        vct_a_exponential_factor = xp.log(
            vertical_config.lowest_layer_thickness / vertical_config.model_top_height
        ) / xp.log(
            2.0
            / math.pi
            * xp.arccos(
                float(vertical_config.num_levels - 1) ** vertical_config.stretch_factor
                / float(vertical_config.num_levels) ** vertical_config.stretch_factor
            )
        )

        vct_a = (
            vertical_config.model_top_height
            * (
                2.0
                / math.pi
                * xp.arccos(
                    xp.arange(vertical_config.num_levels + 1, dtype=float)
                    ** vertical_config.stretch_factor
                    / float(vertical_config.num_levels) ** vertical_config.stretch_factor
                )
            )
            ** vct_a_exponential_factor
        )

        if (
            2.0 * vertical_config.lowest_layer_thickness
            < vertical_config.maximal_layer_thickness
            < 0.5 * vertical_config.top_height_limit_for_maximal_layer_thickness
        ):
            layer_thickness = vct_a[: vertical_config.num_levels] - vct_a[1:]
            lowest_level_exceeding_limit = xp.max(
                xp.where(layer_thickness > vertical_config.maximal_layer_thickness)
            )
            modified_vct_a = xp.zeros(num_levels_plus_one, dtype=float)
            lowest_level_unmodified_thickness = 0
            shifted_levels = 0
            for k in range(vertical_config.num_levels - 1, -1, -1):
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
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]
                else:
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]

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

            for k in range(vertical_config.num_levels - 1, -1, -1):
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
            if stretchfac != 1.0 and lowest_level_exceeding_limit < vertical_config.num_levels - 4:
                for k in range(vertical_config.num_levels - 1, -1, -1):
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
                        lowest_level_unmodified_thickness + 1 : vertical_config.num_levels
                    ] = modified_vct_a[
                        lowest_level_unmodified_thickness + 1 : vertical_config.num_levels
                    ]
                    vct_a[2 : lowest_level_unmodified_thickness + 1] = 0.5 * (
                        modified_vct_a[1:lowest_level_unmodified_thickness]
                        + modified_vct_a[3 : lowest_level_unmodified_thickness + 2]
                    )
    else:
        vct_a = (
            vertical_config.model_top_height
            * (float(vertical_config.num_levels) - xp.arange(num_levels_plus_one, dtype=float))
            / float(vertical_config.num_levels)
        )
    vct_b = xp.exp(-vct_a / 5000.0)

    if not xp.allclose(vct_a[0], vertical_config.model_top_height):
        log.warning(
            f" Warning. vct_a[0], {vct_a[0]}, is not equal to model top height, {vertical_config.model_top_height}, of vertical configuration. Please consider changing the vertical setting."
        )

    return gtx.as_field((dims.KDim,), vct_a), gtx.as_field((dims.KDim,), vct_b)


def get_vct_a_and_vct_b(vertical_config: VerticalGridConfig) -> tuple[fa.KField, fa.KField]:
    """
    get vct_a and vct_b.
    vct_a is an array that contains the height of grid interfaces (or half levels) from model surface to model top, before terrain-following coordinates are applied.
    vct_b is an array that is used to initialize vertical wind speed above surface by a prescribed vertical profile when the surface vertical wind is given.
    It is also used to modify the initial vertical wind speed above surface according to a prescribed vertical profile by linearly merging the surface vertica wind with the existing vertical wind.
    See init_w and adjust_w in mo_nh_init_utils.f90.

    When file_name is given in vertical_config, it will read both vct_a and vct_b from that file. Otherwise, they are analytically derived based on vertical configuration.

    Args:
        vertical_config: Vertical grid configuration
    Returns:  one dimensional (dims.KDim) vct_a and vct_b gt4py fields.
    """

    return (
        _read_vct_a_and_vct_b_from_file(vertical_config.file_path, vertical_config.num_levels)
        if vertical_config.file_path
        else _compute_vct_a_and_vct_b(vertical_config)
    )
