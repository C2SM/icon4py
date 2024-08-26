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
from gt4py.eve.utils import FrozenNamespace
from gt4py.next import broadcast
from gt4py.next.ffront.fbuiltins import (
    abs,
    exp,
    maximum,
    where,
)

from icon4py.model.common import (
    constants as phy_const,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.diagnostic_calculations.stencils import (
    diagnose_pressure as pressure,
    diagnose_surface_pressure as surface_pressure,
)
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.settings import backend, xp
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    tracer_state as tracers,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc


# TODO (Chia Rui): Refactor this class when direct import is enabled for gt4py stencils
class SaturatedPressureConstants(FrozenNamespace):
    """
    Constants used for the computation of saturated pressure in saturation adjustment and microphysics.
    It was originally in mo_lookup_tables_constants.f90.
    """

    #: Latent heat of vaporisation for water [J/kg]. Originally expressed as alv in ICON.
    vaporisation_latent_heat: ta.wpfloat = 2.5008e6
    #: Melting temperature of ice/snow [K]
    tmelt: ta.wpfloat = 273.15

    #: See docstring in common/constanst.py
    rd: ta.wpfloat = phy_const.GAS_CONSTANT_DRY_AIR
    #: See docstring in common/constanst.py
    rv: ta.wpfloat = phy_const.GAS_CONSTANT_WATER_VAPOR
    #: See docstring in common/constanst.py
    cvd: ta.wpfloat = phy_const.SPECIFIC_HEAT_CONSTANT_VOLUME
    #: See docstring in common/constanst.py
    cpd: ta.wpfloat = phy_const.SPECIFIC_HEAT_CONSTANT_PRESSURE

    rv_o_rd_minus_1: ta.wpfloat = rv / rd - 1.0
    rd_o_cpd: ta.wpfloat = phy_const.RD_O_CPD

    #: Dry air heat capacity at constant pressure / water heat capacity at constant pressure - 1
    rcpl: ta.wpfloat = 3.1733
    #: Specific heat capacity of liquid water. Originally expressed as clw in ICON.
    spec_heat_cap_water: ta.wpfloat = (rcpl + 1.0) * cpd

    #: p0 in Tetens formula for saturation water pressure, see eq. 5.33 in COSMO documentation. Originally expressed as c1es in ICON.
    tetens_p0: ta.wpfloat = 610.78
    #: aw in Tetens formula for saturation water pressure. Originally expressed as c3les in ICON.
    tetens_aw: ta.wpfloat = 17.269
    #: bw in Tetens formula for saturation water pressure. Originally expressed as c4les in ICON.
    tetens_bw: ta.wpfloat = 35.86
    #: numerator in temperature partial derivative of Tetens formula for saturation water pressure (psat tetens_der / (t - tetens_bw)^2). Originally expressed as c5les in ICON.
    tetens_der: ta.wpfloat = tetens_aw * (tmelt - tetens_bw)


# Instantiate the class
satpres_const: Final = SaturatedPressureConstants()


@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    #: in ICON, 10 is always used for max iteration when subroutine satad_v_3D is called.
    max_iter: int = 10
    #: in ICON, 1.e-3 is always used for the tolerance when subroutine satad_v_3D is called.
    tolerance: ta.wpfloat = 1.0e-3
    #: An extra step of updating the variables from new temperature is done in ICON after satad is called in at the beginning of mo_nh_interface_nwp.f90. This is a new option to update those variables.
    diagnose_variables_from_new_temperature: bool = True


@dataclasses.dataclass
class MetricStateSaturationAdjustment:
    ddqz_z_full: fa.CellKField[ta.wpfloat]


class ConvergenceError(Exception):
    pass


class SaturationAdjustment:
    def __init__(
        self,
        config: SaturationAdjustmentConfig,
        grid: icon_grid.IconGrid,
        vertical_params: v_grid.VerticalGridParams,
        metric_state: MetricStateSaturationAdjustment,
    ):
        self.config = config
        self.grid = grid
        self.vertical_params: v_grid.VerticalGridParams = vertical_params
        self.metric_state: MetricStateSaturationAdjustment = metric_state
        self._allocate_tendencies()

    # TODO (Chia Rui): add in input and output data properties when physics interface protocal is ready.

    def _allocate_tendencies(self):
        #: it was originally named as tworkold in ICON. Old temperature before iteration.
        self._temperature1 = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        #: it was originally named as twork in ICON. New temperature before iteration.
        self._temperature2 = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        #: A mask that indicates whether the grid cell is subsaturated or not.
        self._subsaturated_mask = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=bool
        )
        #: A mask that indicates whether next Newton iteration is required.
        self._newton_iteration_mask = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=bool
        )
        #: latent heat vaporization / dry air heat capacity at constant volume
        self._lwdocvd = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self._k_field = field_alloc.allocate_indices(
            KDim, grid=self.grid, is_halfdim=True, dtype=gtx.int32
        )
        # TODO (Chia Rui): remove local pressure and pressire_ifc when scan operator can be called along with pressure tendency computation
        self._pressure = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self._pressure_ifc = field_alloc.allocate_zero_field(
            CellDim,
            KDim,
            grid=self.grid,
            is_halfdim=True,
            dtype=ta.wpfloat,
        )
        self._pressure_ifc = field_alloc.allocate_zero_field(
            CellDim,
            KDim,
            grid=self.grid,
            is_halfdim=True,
            dtype=ta.wpfloat,
        )
        self._new_exner = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self._new_virtual_temperature = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        # TODO (Chia Rui): remove the tendency terms below when architecture of the entire phyiscs component is ready to use.
        self.temperature_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.qv_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.qc_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.virtual_temperature_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.exner_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.pressure_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, dtype=ta.wpfloat
        )
        self.pressure_ifc_tendency = field_alloc.allocate_zero_field(
            CellDim, KDim, grid=self.grid, is_halfdim=True, dtype=ta.wpfloat
        )

    def run(
        self,
        dtime: ta.wpfloat,
        prognostic_state: prognostics.PrognosticState,
        diagnostic_state: diagnostics.DiagnosticState,
        tracer_state: tracers.TracerState,
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
        """

        start_cell_nudging = self.grid.get_start_index(
            CellDim, h_grid.HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(
            CellDim, h_grid.HorizontalMarkerIndex.local(CellDim)
        )

        compute_subsaturated_case_and_initialize_newton_iterations(
            self.config.tolerance,
            diagnostic_state.temperature,
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self._subsaturated_mask,
            self._lwdocvd,
            self._temperature1,
            self._temperature2,
            self._newton_iteration_mask,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=gtx.int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        # TODO (Chia Rui): this is inspired by the cpu version of the original ICON saturation_adjustment code. Consider to refactor this code when break and for loop features are ready in gt4py.
        temperature_list = [self._temperature1, self._temperature2]
        ncurrent, nnext = 0, 1
        for _ in range(self.config.max_iter):
            if xp.any(
                self._newton_iteration_mask.ndarray[
                    start_cell_nudging:end_cell_local, 0 : self.grid.num_levels
                ]
            ):
                update_temperature_by_newton_iteration(
                    diagnostic_state.temperature,
                    tracer_state.qv,
                    prognostic_state.rho,
                    self._newton_iteration_mask,
                    self._lwdocvd,
                    temperature_list[nnext],
                    temperature_list[ncurrent],
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=gtx.int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

                compute_newton_iteration_mask(
                    self.config.tolerance,
                    temperature_list[ncurrent],
                    temperature_list[nnext],
                    self._newton_iteration_mask,
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=gtx.int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )

                copy_temperature(
                    self._newton_iteration_mask,
                    temperature_list[ncurrent],
                    temperature_list[nnext],
                    horizontal_start=start_cell_nudging,
                    horizontal_end=end_cell_local,
                    vertical_start=gtx.int32(0),
                    vertical_end=self.grid.num_levels,
                    offset_provider={},
                )
                ncurrent = (ncurrent + 1) % 2
                nnext = (nnext + 1) % 2
            else:
                break
        if xp.any(
            self._newton_iteration_mask.ndarray[
                start_cell_nudging:end_cell_local, 0 : self.grid.num_levels
            ]
        ):
            raise ConvergenceError(
                f"Maximum iteration of saturation adjustment ({self.config.max_iter}) is not enough. The max absolute error is {xp.abs(self.new_temperature1.ndarray - self.new_temperature2.ndarray).max()} . Please raise max_iter"
            )
        update_temperature_qv_qc_tendencies(
            dtime,
            diagnostic_state.temperature,
            temperature_list[ncurrent],
            tracer_state.qv,
            tracer_state.qc,
            prognostic_state.rho,
            self._subsaturated_mask,
            self.temperature_tendency,
            self.qv_tendency,
            self.qc_tendency,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=gtx.int32(0),
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        if self.config.diagnose_variables_from_new_temperature:
            # TODO (Chia Rui): Consider to merge the gt4py stencils below if scan operator can be called in the intermediate step
            compute_temperature_and_exner_tendencies_after_saturation_adjustment(
                dtime,
                tracer_state.qv,
                tracer_state.qc,
                tracer_state.qi,
                tracer_state.qr,
                tracer_state.qs,
                tracer_state.qg,
                self.qv_tendency,
                self.qc_tendency,
                temperature_list[ncurrent],
                diagnostic_state.virtual_temperature,
                prognostic_state.exner,
                self.virtual_temperature_tendency,
                self.exner_tendency,
                self._new_exner,
                self._new_virtual_temperature,
                self._k_field,
                self.vertical_params.kstart_moist,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            surface_pressure.diagnose_surface_pressure(
                self._new_exner,
                self._new_virtual_temperature,
                self.metric_state.ddqz_z_full,
                self._pressure_ifc,
                phy_const.CPD_O_RD,
                phy_const.P0REF,
                phy_const.GRAV_O_RD,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=self.grid.num_levels,
                vertical_end=self.grid.num_levels + gtx.int32(1),
                offset_provider={"Koff": KDim},
            )

            pressure.diagnose_pressure(
                self.metric_state.ddqz_z_full,
                self._new_virtual_temperature,
                gtx.as_field((CellDim,), self._pressure_ifc.ndarray[:, -1]),
                self._pressure,
                self._pressure_ifc,
                phy_const.GRAV_O_RD,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            compute_pressure_tendency_after_saturation_adjustment(
                dtime,
                diagnostic_state.pressure,
                self._pressure,
                self.pressure_tendency,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels,
                offset_provider={},
            )

            compute_pressure_ifc_tendency_after_saturation_adjustment(
                dtime,
                diagnostic_state.pressure_ifc,
                self._pressure_ifc,
                self.pressure_ifc_tendency,
                horizontal_start=start_cell_nudging,
                horizontal_end=end_cell_local,
                vertical_start=gtx.int32(0),
                vertical_end=self.grid.num_levels + gtx.int32(1),
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
        satpres_const.vaporisation_latent_heat
        + (1850.0 - satpres_const.spec_heat_cap_water) * (t - satpres_const.tmelt)
        - satpres_const.rv * t
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
    return satpres_const.tetens_p0 * exp(
        satpres_const.tetens_aw * (t - satpres_const.tmelt) / (t - satpres_const.tetens_bw)
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
    return _sat_pres_water(t) / (rho * satpres_const.rv * t)


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
    beta = satpres_const.tetens_der / (t - satpres_const.tetens_bw) ** 2 - 1.0 / t
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
        qv: specific humidity
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
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
        qv: initial specific humidity
        qc: initial cloud specific mixing ratio
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


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
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
        qv: initial specific humidity
        qc: initial cloud specific mixing ratio
        rho: total air density [kg m-3]
    Returns:
        mask for subsaturated case,
        Lv / cvd,
        current temperature for starting the Newton iteration,
        next temperature for starting the Newton iteration,
        mask for Newton iteration case
    """
    temperature_after_all_qc_evaporated = (
        temperature - _latent_heat_vaporization(temperature) / satpres_const.cvd * qc
    )

    # Check, which points will still be subsaturated even after evaporating all cloud water.
    subsaturated_mask = qv + qc <= _qsat_rho(temperature_after_all_qc_evaporated, rho)

    # Remains const. during iteration
    lwdocvd = _latent_heat_vaporization(temperature) / satpres_const.cvd

    new_temperature1 = where(
        subsaturated_mask,
        temperature_after_all_qc_evaporated,
        temperature - 2.0 * tolerance,
    )
    new_temperature2 = where(subsaturated_mask, temperature_after_all_qc_evaporated, temperature)
    newton_iteration_mask = where(subsaturated_mask, False, True)

    return subsaturated_mask, lwdocvd, new_temperature1, new_temperature2, newton_iteration_mask


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_newton_iteration_mask(
    tolerance: ta.wpfloat,
    temperature_current: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[bool]:
    return where(abs(temperature_current - temperature_next) > tolerance, True, False)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_newton_iteration_mask(
    tolerance: ta.wpfloat,
    temperature_current: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
    newton_iteration_mask: fa.CellKField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_newton_iteration_mask(
        tolerance,
        temperature_current,
        temperature_next,
        out=newton_iteration_mask,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _copy_temperature(
    newton_iteration_mask: fa.CellKField[bool],
    temperature_current: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Copy temperature from the current to the new temperature field where the convergence criterion is already met.
    Otherwise, it is zero (the value does not matter because it will be updated in next iteration).

    Args:
        newton_iteration_mask: mask for the next Newton iteration to be executed
        temperature: current temperature [K]
    Returns:
        new temperature [K]
    """
    return where(newton_iteration_mask, 0.0, temperature_current)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def copy_temperature(
    newton_iteration_mask: fa.CellKField[bool],
    temperature_current: fa.CellKField[ta.wpfloat],
    temperature_next: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _copy_temperature(
        newton_iteration_mask,
        temperature_current,
        out=temperature_next,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_temperature_and_exner_tendencies_after_saturation_adjustment(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    k_field: fa.KField[gtx.int32],
    kstart_moist: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """
    Update virtual temperature and exner tendencies after saturation adjustment updates temperature, qv , and qc.

    Args:
        dtime: time step [s]
        qv: specific humidity [kg/kg]
        qc: specific cloud water content [kg/kg]
        qi: specific cloud ice content [kg/kg]
        qr: specific rain water content [kg/kg]
        qs: specific snow content [kg/kg]
        qg: specific graupel content [kg/kg]
        qv_tendency: specific humidity tendency [kg/kg/s]
        qc_tendency: specific cloud water content tendency [kg/kg/s]
        temperature: air temperature [K]
        virtual_temperature: air virtual temperature [K]
        exner: exner function
        k_field: k level indices
        kstart_moist: starting level for all moisture processes
    Returns:
        virtual temperature tendency [K/s], exner tendency [/s], new exner, new virtual temperature [K]
    """
    qsum = where(
        k_field < kstart_moist,
        broadcast(0.0, (CellDim, KDim)),
        qc + qc_tendency * dtime + qi + qr + qs + qg,
    )

    new_virtual_temperature = where(
        k_field < kstart_moist,
        virtual_temperature,
        temperature * (1.0 + satpres_const.rv_o_rd_minus_1 * (qv + qv_tendency * dtime) - qsum),
    )
    new_exner = where(
        k_field < kstart_moist,
        exner,
        exner
        * (1.0 + satpres_const.rd_o_cpd * (new_virtual_temperature / virtual_temperature - 1.0)),
    )

    return (
        (new_virtual_temperature - virtual_temperature) / dtime,
        (new_exner - exner) / dtime,
        new_exner,
        new_virtual_temperature,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_temperature_and_exner_tendencies_after_saturation_adjustment(
    dtime: ta.wpfloat,
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature_tendency: fa.CellKField[ta.wpfloat],
    exner_tendency: fa.CellKField[ta.wpfloat],
    new_exner: fa.CellKField[ta.wpfloat],
    new_virtual_temperature: fa.CellKField[ta.wpfloat],
    k_field: fa.KField[gtx.int32],
    kstart_moist: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_temperature_and_exner_tendencies_after_saturation_adjustment(
        dtime,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qv_tendency,
        qc_tendency,
        temperature,
        virtual_temperature,
        exner,
        k_field,
        kstart_moist,
        out=(
            virtual_temperature_tendency,
            exner_tendency,
            new_exner,
            new_virtual_temperature,
        ),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_pressure_tendency_after_saturation_adjustment(
    dtime: ta.wpfloat,
    old_pressure: fa.CellKField[ta.wpfloat],
    new_pressure: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update pressure tendency at full levels after saturation adjustment updates temperature, qv , and qc.

    Args:
        dtime: time step [s]
        old_pressure: old air pressure at full levels [Pa]
        new_pressure: new air pressure at full levels [Pa]
    Returns:
        pressure tendency [Pa/s]
    """

    return (new_pressure - old_pressure) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_pressure_tendency_after_saturation_adjustment(
    dtime: ta.wpfloat,
    old_pressure: fa.CellKField[ta.wpfloat],
    new_pressure: fa.CellKField[ta.wpfloat],
    pressure_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_pressure_tendency_after_saturation_adjustment(
        dtime,
        old_pressure,
        new_pressure,
        out=pressure_tendency,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )


@gtx.field_operator
def _compute_pressure_ifc_tendency_after_saturation_adjustment(
    dtime: ta.wpfloat,
    old_pressure_ifc: fa.CellKField[ta.wpfloat],
    new_pressure_ifc: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Update pressure tendency at half levels (including surface level) after saturation adjustment updates temperature, qv , and qc.

    Args:
        dtime: time step [s]
        old_pressure_ifc: old air pressure at interface [Pa]
        new_pressure_ifc: new air pressure at interface [Pa]
    Returns:
        interface pressure tendency [Pa/s]
    """
    return (new_pressure_ifc - old_pressure_ifc) / dtime


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=backend)
def compute_pressure_ifc_tendency_after_saturation_adjustment(
    dtime: ta.wpfloat,
    old_pressure_ifc: fa.CellKField[ta.wpfloat],
    new_pressure_ifc: fa.CellKField[ta.wpfloat],
    pressure_ifc_tendency: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_pressure_ifc_tendency_after_saturation_adjustment(
        dtime,
        old_pressure_ifc,
        new_pressure_ifc,
        out=pressure_ifc_tendency,
        domain={CellDim: (horizontal_start, horizontal_end), KDim: (vertical_start, vertical_end)},
    )
