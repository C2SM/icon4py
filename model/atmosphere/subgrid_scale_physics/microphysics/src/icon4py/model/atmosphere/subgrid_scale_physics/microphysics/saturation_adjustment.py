# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import Final

import gt4py.next as gtx
from gt4py.eve import utils as eve_utils
from gt4py.next import backend as gtx_backend
from gt4py.next.ffront.fbuiltins import abs, exp, maximum, where  # noqa: A004

import icon4py.model.common.dimension as dims
import icon4py.model.common.utils as common_utils
from icon4py.model.common import constants as phy_const, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.states import model
from icon4py.model.common.utils import data_allocation as data_alloc


physics_constants: Final = phy_const.PhysicsConstants()


# TODO(OngChia): Refactor this class when direct import is enabled for gt4py stencils
class MicrophysicsConstants(eve_utils.FrozenNamespace[ta.wpfloat]):
    """
    Constants used for the computation of saturated pressure in saturation adjustment and microphysics.
    It was originally in mo_lookup_tables_constants.f90.
    """

    #: Latent heat of vaporisation for water [J kg-1]. Originally expressed as alv in ICON.
    vaporisation_latent_heat = 2.5008e6
    #: Melting temperature of ice/snow [K]
    melting_temperature = 273.15

    #: Specific heat capacity of liquid water [J K-1 kg-1]. Originally expressed as clw in ICON. 4.1733 is dry air heat capacity at constant pressure / water heat capacity at constant pressure.
    specific_heat_capacity_for_water = 4.1733 * phy_const.CPD

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in COSMO documentation. Originally expressed as c1es in ICON.
    tetens_p0 = 610.78
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    tetens_aw = 17.269
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    tetens_bw = 35.86
    #: numerator in temperature partial derivative of Tetens formula for saturation water pressure (psat tetens_der / (t - tetens_bw)^2). Originally expressed as c5les in ICON.
    tetens_der = tetens_aw * (melting_temperature - tetens_bw)


# Instantiate the class
microphy_const: Final = MicrophysicsConstants()


@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    #: in ICON, 10 is always used for max iteration when subroutine satad_v_3D is called.
    max_iter: int = 10
    #: in ICON, 1.e-3 is always used for the tolerance when subroutine satad_v_3D is called.
    tolerance: ta.wpfloat = 1.0e-3


@dataclasses.dataclass
class MetricStateSaturationAdjustment:
    ddqz_z_full: fa.CellKField[ta.wpfloat]


class ConvergenceError(Exception):
    pass


#: CF attributes of saturation adjustment input variables
_SATURATION_ADJUST_INPUT_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    air_density=dict(
        standard_name="air_density",
        long_name="density",
        units="kg m-3",
        icon_var_name="rho",
    ),
    temperature=dict(
        standard_name="air_temperature",
        long_name="air temperature",
        units="K",
        icon_var_name="temp",
    ),
    specific_humidity=dict(
        standard_name="specific_humidity",
        long_name="ratio of water vapor mass to total moist air parcel mass",
        units="1",
        icon_var_name="qv",
    ),
    specific_cloud=dict(
        standard_name="specific_cloud_content",
        long_name="ratio of cloud water mass to total moist air parcel mass",
        units="1",
        icon_var_name="qc",
    ),
)


#: CF attributes of saturation adjustment output variables
_SATURATION_ADJUST_OUTPUT_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    tend_temperature_due_to_satad=dict(
        standard_name="tendency_of_air_temperature_due_to_saturation_adjustment",
        long_name="tendency of air temperature due to saturation adjustment",
        units="K s-1",
    ),
    tend_specific_humidity_due_to_satad=dict(
        standard_name="tendency_of_specific_humidity_due_to_saturation_adjustment",
        long_name="tendency of ratio of water vapor mass to total moist air parcel mass due to saturation adjustment",
        units="s-1",
    ),
    tend_specific_cloud_due_to_satad=dict(
        standard_name="tendency_of_specific_cloud_content_due_to_saturation_adjustment",
        long_name="tendency of ratio of cloud water mass to total moist air parcel mass due to saturation adjustment",
        units="s-1",
    ),
)


class SaturationAdjustment:
    def __init__(
        self,
        config: SaturationAdjustmentConfig,
        grid: icon_grid.IconGrid,
        vertical_params: v_grid.VerticalGrid,
        metric_state: MetricStateSaturationAdjustment,
        backend: gtx_backend.Backend | None,
    ):
        self._backend = backend
        self.config = config
        self.grid = grid
        self.vertical_params: v_grid.VerticalGrid = vertical_params
        self.metric_state: MetricStateSaturationAdjustment = metric_state
        self._allocate_local_variables()
        self._determine_horizontal_domains()

        self._xp = data_alloc.import_array_ns(self._backend)

        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.start_index(cell_domain(h_grid.Zone.END))

    # TODO(OngChia): add in input and output data properties, and refactor this component to follow the physics component protocol.
    def input_properties(self) -> dict[str, model.FieldMetaData]:
        raise NotImplementedError

    def output_properties(self) -> dict[str, model.FieldMetaData]:
        raise NotImplementedError

    def _allocate_local_variables(self):
        #: it was originally named as tworkold in ICON. Old temperature before iteration.
        self._temperature1 = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        #: it was originally named as twork in ICON. New temperature before iteration.
        self._temperature2 = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        #: A mask that indicates whether the grid cell is subsaturated or not.
        self._subsaturated_mask = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=bool, backend=self._backend
        )
        #: A mask that indicates whether next Newton iteration is required.
        self._newton_iteration_mask = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, dtype=bool, backend=self._backend
        )
        #: latent heat vaporization / dry air heat capacity at constant volume
        self._lwdocvd = data_alloc.zero_field(
            self.grid, dims.CellDim, dims.KDim, backend=self._backend
        )
        self.compute_subsaturated_case_and_initialize_newton_iterations = (
            compute_subsaturated_case_and_initialize_newton_iterations.with_backend(self._backend)
        )
        self.update_temperature_by_newton_iteration = (
            update_temperature_by_newton_iteration.with_backend(self._backend)
        )
        self.compute_newton_iteration_mask_and_copy_temperature_on_converged_cells = (
            compute_newton_iteration_mask_and_copy_temperature_on_converged_cells.with_backend(
                self._backend
            )
        )
        self.update_temperature_qv_qc_tendencies = update_temperature_qv_qc_tendencies.with_backend(
            self._backend
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self.grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self.grid.start_index(cell_domain(h_grid.Zone.END))

    def _not_converged(self) -> bool:
        return self._xp.any(
            self._newton_iteration_mask.ndarray[
                self._start_cell_nudging : self._end_cell_local, 0 : self.grid.num_levels
            ]
        )

    def run(
        self,
        dtime: ta.wpfloat,
        rho: fa.CellKField[ta.wpfloat],
        temperature: fa.CellKField[ta.wpfloat],
        qv: fa.CellKField[ta.wpfloat],
        qc: fa.CellKField[ta.wpfloat],
        temperature_tendency: fa.CellKField[ta.wpfloat],
        qv_tendency: fa.CellKField[ta.wpfloat],
        qc_tendency: fa.CellKField[ta.wpfloat],
    ):
        """
        Adjust saturation at each grid point.
        Saturation adjustment condenses/evaporates specific humidity (qv) into/from
        cloud water content (qc) such that a gridpoint is just saturated. Temperature (t)
        is adapted accordingly and pressure adapts itself in ICON.

        Saturation adjustment at constant total density (adjustment of T and p accordingly)
        assuming chemical equilibrium of water and vapor. For the heat capacity of
        of the total system (dry air, vapor, and hydrometeors) the value of dry air
        is taken, which is a common approximation and introduces only a small error.

        Originally inspired from satad_v_3D of ICON.

        Args:
            dtime: time step [s]
            rho: air density [kg m-3]
            temperature: air temperature [K]
            qv: specific humidity [kg kg-1]
            qc: specific cloud water content [kg kg-1]
            temperature_tendency: air temperature tendency [K s-1]
            qv_tendency: specific humidity tendency [s-1]
            qc_tendency: specific cloud water content tendency [s-1]
        """

        self.compute_subsaturated_case_and_initialize_newton_iterations(
            self.config.tolerance,
            temperature,
            qv,
            qc,
            rho,
            self._subsaturated_mask,
            self._lwdocvd,
            self._temperature1,
            self._temperature2,
            self._newton_iteration_mask,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        temperature_pair = common_utils.TimeStepPair(self._temperature1, self._temperature2)

        # TODO(OngChia): this is inspired by the cpu version of the original ICON saturation_adjustment code. Consider to refactor this code when break and for loop features are ready in gt4py.
        num_iter = 0
        while self._not_converged():
            if num_iter > self.config.max_iter:
                raise ConvergenceError(
                    f"Maximum iteration of saturation adjustment ({self.config.max_iter}) is not enough. The max absolute error is {self._xp.abs(self._temperature1.ndarray - self._temperature2.ndarray).max()} . Please raise max_iter"
                )

            self.update_temperature_by_newton_iteration(
                temperature,
                qv,
                rho,
                self._newton_iteration_mask,
                self._lwdocvd,
                temperature_pair.next,
                temperature_pair.current,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            self.compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
                self.config.tolerance,
                temperature_pair.current,
                temperature_pair.next,
                self._newton_iteration_mask,
                horizontal_start=self._start_cell_nudging,
                horizontal_end=self._end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            temperature_pair.swap()
            num_iter = num_iter + 1

        self.update_temperature_qv_qc_tendencies(
            dtime,
            temperature,
            temperature_pair.current,
            qv,
            qc,
            rho,
            self._subsaturated_mask,
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            horizontal_start=self._start_cell_nudging,
            horizontal_end=self._end_cell_local,
            vertical_start=gtx.int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )


@gtx.field_operator
def _latent_heat_vaporization(
    t: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the latent heat of vaporisation with Kirchoff's relations (users can refer to Pruppacher and Klett textbook).
        dL/dT ~= cpv - cpw + v dp/dT
        L ~= (cpv - cpw) (T - T0) - Rv T

    Args:
        t: temperature [K]
    Returns:
        latent heat of vaporization.
    """
    return (
        microphy_const.vaporisation_latent_heat
        + (1850.0 - microphy_const.specific_heat_capacity_for_water)
        * (t - microphy_const.melting_temperature)
        - physics_constants.rv * t
    )


@gtx.field_operator
def _sat_pres_water(t: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """
    Compute saturation water vapour pressure by the Tetens formula.
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        t: temperature [K]
    Returns:
        saturation water vapour pressure.
    """
    return microphy_const.tetens_p0 * exp(
        microphy_const.tetens_aw
        * (t - microphy_const.melting_temperature)
        / (t - microphy_const.tetens_bw)
    )


@gtx.field_operator
def _qsat_rho(
    t: fa.CellKField[ta.wpfloat], rho: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute specific humidity at water saturation (with respect to flat surface).
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        t: temperature [K]
        rho: total air density (including hydrometeors) [kg m-3]
    Returns:
        specific humidity at water saturation.
    """
    return _sat_pres_water(t) / (rho * physics_constants.rv * t)


@gtx.field_operator
def _dqsatdT_rho(
    t: fa.CellKField[ta.wpfloat], zqsat: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the partical derivative of the specific humidity at water saturation (qsat) with respect to the temperature at
    constant total density. qsat is approximated as
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]
    FInally, the derivative with respect to temperature is
        dpsat/dT = psat (T0-bw)/(T-bw)^2
        dqsat/dT = 1/Rv psat/(rho T) (T0-bw)/(T-bw)^2 - 1/Rv psat/(rho T^2) = qsat ((T0-bw)/(T-bw)^2 - 1/T)

    Args:
        t: temperature [K]
        zqsat: saturated water mixing ratio
    Returns:
        partial derivative of the specific humidity at water saturation.
    """
    beta = microphy_const.tetens_der / (t - microphy_const.tetens_bw) ** 2 - 1.0 / t
    return beta * zqsat


@gtx.field_operator
def _new_temperature_in_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    lwdocvd: fa.CellKField[ta.wpfloat],
    new_temperature2: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update the temperature in saturation adjustment by Newton iteration. Moist enthalpy and mass are conserved.
    The latent heat is assumed to be constant with its value computed from the initial temperature.
        T + Lv / cvd qv = TH, qv + qc = QTOT
        T = TH - Lv / cvd qsat(T), which is a transcendental function. Newton method is applied to solve it for T.
        f(T) = Lv / cvd qsat(T) + T - TH
        f'(T) = Lv / cvd dqsat(T)/dT + 1
        T_new = T - f(T)/f'(T) = ( TH - Lv / cvd (qsat(T) - T dqsat(T)/dT) ) / (Lv / cvd dqsat(T)/dT + 1)

    Args:
        temperature: initial temperature [K]
        qv: specific humidity [kg kg-1]
        rho: total air density [kg m-3]
        lwdocvd: Lv / cvd [K]
        new_temperature2: temperature at previous iteration [K]
    Returns:
        updated temperature [K]
    """
    ft = new_temperature2 - temperature + lwdocvd * (_qsat_rho(new_temperature2, rho) - qv)
    dft = 1.0 + lwdocvd * _dqsatdT_rho(new_temperature2, _qsat_rho(new_temperature2, rho))

    return new_temperature2 - ft / dft


@gtx.field_operator
def _update_temperature_by_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    new_temperature2: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    new_temperature1 = where(
        newton_iteration_mask,
        _new_temperature_in_newton_iteration(temperature, qv, rho, lwdocvd, new_temperature2),
        new_temperature2,
    )
    return new_temperature1


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_by_newton_iteration(
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    new_temperature2: fa.CellKField[ta.wpfloat],
    new_temperature1: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_by_newton_iteration(
        temperature,
        qv,
        rho,
        newton_iteration_mask,
        lwdocvd,
        new_temperature2,
        out=new_temperature1,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _update_temperature_qv_qc_tendencies(
    dtime: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """
    Compute temperature, qv, and qc tendencies from the saturation adjustment.

    Args:
        dtime: time step
        temperature: initial temperature [K]
        temperature_next: temperature updated by saturation adjustment [K]
        qv: initial specific humidity [kg kg-1]
        qc: initial cloud mixing ratio [kg kg-1]
        rho: total air density [kg m-3]
        subsaturated_mask: a mask where the air is subsaturated even if all cloud particles evaporate
    Returns:
        (updated temperature - initial temperautre) / dtime [K s-1],
        (saturated specific humidity - initial specific humidity) / dtime [s-1],
        (total specific mixing ratio - saturated specific humidity - initial cloud specific mixing ratio) / dtime [s-1],
    """
    zqwmin = 1e-20
    qv_tendency, qc_tendency = where(
        subsaturated_mask,
        (qc / dtime, -qc / dtime),
        (
            (_qsat_rho(temperature_next, rho) - qv) / dtime,
            (maximum(qv + qc - _qsat_rho(temperature_next, rho), zqwmin) - qc) / dtime,
        ),
    )
    return (temperature_next - temperature) / dtime, qv_tendency, qc_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature_qv_qc_tendencies(
    dtime: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
    temperature_tendency: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature_qv_qc_tendencies(
        dtime,
        temperature,
        temperature_next,
        qv,
        qc,
        rho,
        subsaturated_mask,
        out=(temperature_tendency, qv_tendency, qc_tendency),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[bool],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[bool],
]:
    """
    Preparation for saturation adjustment.
    First obtain the subsaturated case, where the saturation specific humidity is larger than qv + qc. This can be
    derived by computing saturation specific humidity at the temperature after all cloud particles are evaporated.
    If this is the case, the new temperature is simply the temperature after all cloud particles are evaporated, and
    qv_new = qv + qc, qc = 0.
    All the remaining grid cells are marked as newton_iteration_mask for which Newton iteration is required to solve
    for the new temperature.

    Args:
        tolerance: tolerance for convergence in Newton iteration
        temperature: initial temperature [K]
        qv: initial specific humidity [kg kg-1]
        qc: initial cloud mixing ratio [kg kg-1]
        rho: total air density [kg m-3]
    Returns:
        mask for subsaturated case,
        Lv / cvd,
        current temperature for starting the Newton iteration,
        next temperature for starting the Newton iteration,
        mask for Newton iteration case
    """
    temperature_after_all_qc_evaporated = (
        temperature - _latent_heat_vaporization(temperature) / physics_constants.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= _qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = _latent_heat_vaporization(temperature) / physics_constants.cvd

    new_temperature1 = where(
        subsaturated_mask,
        temperature_after_all_qc_evaporated,
        temperature - 2.0 * tolerance,
    )
    new_temperature2 = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature)
    newton_iteration_mask = where(subsaturated_mask, False, True)

    return subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_subsaturated_case_and_initialize_newton_iterations(
    tolerance: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    subsaturated_mask: fa.CellKField[bool],
    lwdocvd: fa.CellKField[ta.wpfloat],
    new_temperature1: fa.CellKField[ta.wpfloat],
    new_temperature2: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_subsaturated_case_and_initialize_newton_iterations(
        tolerance,
        temperature,
        qv,
        qc,
        rho,
        out=(subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
    tolerance: ta.wpfloat,
    temperature_current: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[bool], fa.CellKField[ta.wpfloat]]:
    """
    Compute a mask for the next Newton iteration when the difference between new and old temperature is larger
    than the tolerance.
    Then, copy temperature from the current to the new temperature field where the convergence criterion is already met.
    Otherwise, it is zero (the value does not matter because it will be updated in next iteration).

    Args:
        newton_iteration_mask: mask for the next Newton iteration to be executed
        temperature_current: temperature at previous Newtion iteration [K]
        temperature_next: temperature  at current Newtion iteration [K]
    Returns:
        new temperature [K]
    """
    newton_iteration_mask = where(
        abs(temperature_current - temperature_next) > tolerance, True, False
    )
    new_temperature = where(newton_iteration_mask, 0.0, temperature_current)
    return newton_iteration_mask, new_temperature


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
    tolerance: ta.wpfloat,
    temperature_current: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
        tolerance,
        temperature_current,
        temperature_next,
        out=(newton_iteration_mask, temperature_next),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
