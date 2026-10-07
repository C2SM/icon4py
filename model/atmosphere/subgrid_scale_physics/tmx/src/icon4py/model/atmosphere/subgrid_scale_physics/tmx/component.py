# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The tmx turbulent mixing as a component of the physics driver."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
from icon4py.model.common import (
    dimension as dims,
    model_backends,
    model_options,
    time,
    type_alias as ta,
)
from icon4py.model.common.components import framework as fw, quantities as qty
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import compute_energy
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import icon4py.model.common.grid.states as grid_states
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_state
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface_fluxes import (
        SurfaceFluxProvider,
    )
    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.common.grid import base as base_grid


class TmxComponent(fw.Component):
    """
    The tmx turbulent mixing granule (`Tmx`) as a component.

    `run` derives what tmx needs besides the inputs and owns no state of the model: the air
    mass and the heat capacity of the air per unit area (ICON's `mair` and `cvair`), and the
    surface fluxes, which the provider computes from the step's surface pressure. The granule
    keeps its own state classes (`tmx_states`); `run` builds them over the leaves without
    copies.
    """

    class Input(fw.State):
        temperature: fw.Field[qty.TemperatureOnCellK]
        virtual_temperature: fw.Field[qty.VirtualTemperatureOnCellK]
        pressure: fw.Field[qty.PressureOnCellK]
        pressure_ifc: fw.Field[qty.PressureOnCellKHalf]
        u: fw.Field[qty.UOnCellK]
        v: fw.Field[qty.VOnCellK]
        w: fw.Field[qty.WOnCellKHalf]
        rho: fw.Field[qty.RhoOnCellK]
        qv: fw.Field[qty.QvOnCellK]
        qc: fw.Field[qty.QcOnCellK]
        qi: fw.Field[qty.QiOnCellK]
        qr: fw.Field[qty.QrOnCellK]
        qs: fw.Field[qty.QsOnCellK]
        qg: fw.Field[qty.QgOnCellK]

    class Output(fw.State):
        # the tendencies (tmx_states.TmxTendencyState)
        tend_temperature: fw.Field[qty.TendencyOfTemperatureOnCellK]
        tend_qv: fw.Field[qty.TendencyOfQvOnCellK]
        tend_qc: fw.Field[qty.TendencyOfQcOnCellK]
        tend_qi: fw.Field[qty.TendencyOfQiOnCellK]
        tend_u: fw.Field[qty.TendencyOfUOnCellK]
        tend_v: fw.Field[qty.TendencyOfVOnCellK]
        tend_w: fw.Field[qty.TendencyOfWOnCellKHalf]
        # the diagnostics (tmx_states.TmxDiagnosticState)
        theta_v: fw.Field[qty.ThetaVOnCellK]
        cptgz: fw.Field[qty.DryStaticEnergyOnCellK]
        div_c: fw.Field[qty.DivergenceOnCellK]
        km_c: fw.Field[qty.TurbulentViscosityOnCellK]
        km: fw.Field[qty.ExchangeCoefficientForMomentumOnCellK]
        kh: fw.Field[qty.ExchangeCoefficientForHeatOnCellK]
        dissip_ke: fw.Field[qty.KineticEnergyDissipationOnCellK]
        heating: fw.Field[qty.TurbulentHeatingOnCellK]
        cptgz_vi: fw.Field[qty.VerticallyIntegratedDryStaticEnergyOnCell]
        dissip_ke_vi: fw.Field[qty.VerticallyIntegratedKineticEnergyDissipationOnCell]
        int_energy_vi: fw.Field[qty.VerticallyIntegratedInternalEnergyOnCell]
        tend_int_energy_vi: fw.Field[qty.RateOfChangeOfVerticallyIntegratedInternalEnergyOnCell]
        rho_ic: fw.Field[qty.RhoOnCellKHalf]
        bruvais: fw.Field[qty.BruntVaisalaFrequencySquaredOnCellKHalf]
        mech_prod: fw.Field[qty.MechanicalProductionOfTurbulentKineticEnergyOnCellKHalf]
        km_ic: fw.Field[qty.TurbulentViscosityOnCellKHalf]
        kh_ic: fw.Field[qty.TurbulentDiffusivityOnCellKHalf]
        vn: fw.Field[qty.VnOnEdgeK]
        shear: fw.Field[qty.HorizontalShearProductionOnEdgeK]
        div_of_stress: fw.Field[qty.DivergenceOfStressOnEdgeK]
        vn_ie: fw.Field[qty.VnOnEdgeKHalf]
        vt_ie: fw.Field[qty.TangentialWindOnEdgeKHalf]
        w_ie: fw.Field[qty.WOnEdgeKHalf]
        km_ie: fw.Field[qty.TurbulentViscosityOnEdgeKHalf]
        u_vert: fw.Field[qty.UOnVertexK]
        v_vert: fw.Field[qty.VOnVertexK]
        w_vert: fw.Field[qty.WOnVertexKHalf]
        km_iv: fw.Field[qty.TurbulentViscosityOnVertexKHalf]

    def __init__(
        self,
        *,
        grid: base_grid.Grid,
        config: tmx_config.TmxConfig,
        dtime: time.RelativeTime,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        surface_fluxes: SurfaceFluxProvider,
        backend: model_backends.BackendLike,
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        super().__init__(grid, model_backends.get_allocator(backend))
        self._dt_seconds = dtime.total_seconds()
        self._ddqz_z_full = metric_state.ddqz_z_full
        self._surface_fluxes = surface_fluxes
        self._tmx = tmx.Tmx(
            grid=grid,
            config=config,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=edge_params,
            cell_params=cell_params,
            backend=backend,
            exchange=exchange,
        )

        # mair on all cells, halos included, as ICON; cvair likewise here (ICON's get_cvair
        # stops at the prognostic cells)
        cells = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        levels = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}
        # mass per unit area = density * layer thickness, the shallow-atmosphere airmass of
        # mo_nh_diagnose_pres_temp.f90 (compute_airmass) that ICON binds to mair
        self._compute_air_mass = model_options.setup_program(
            program=generic_math_operations.compute_product_on_cell_k,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_cv_air = model_options.setup_program(
            program=compute_energy.compute_moist_air_heat_capacity_per_area,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )

        on_cells = functools.partial(
            data_alloc.zero_field, grid, dims.CellDim, dtype=ta.wpfloat, allocator=self.allocator
        )
        self._air_mass = on_cells(dims.KDim)
        self._cv_air = on_cells(dims.KDim)
        self._surface_flux_state = tmx_states.TmxSurfaceFluxState(
            evapotranspiration=on_cells(),
            sensible_heat_flux=on_cells(),
            u_stress=on_cells(),
            v_stress=on_cells(),
            q_snocpymlt=on_cells(),
        )
        # the updated fields tmx writes besides the tendencies; nothing reads them
        self._new_state = tmx_states.TmxNewState.allocate(grid, allocator=self.allocator)

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        self._compute_air_mass(
            field_a=inputs.rho.data, field_b=self._ddqz_z_full, output_field=self._air_mass
        )
        self._compute_cv_air(
            qv=inputs.qv.data,
            qc=inputs.qc.data,
            qi=inputs.qi.data,
            qr=inputs.qr.data,
            qs=inputs.qs.data,
            qg=inputs.qg.data,
            air_mass=self._air_mass,
            heat_capacity=self._cv_air,
        )
        self._surface_fluxes.compute(
            pressure_ifc=inputs.pressure_ifc.data, out=self._surface_flux_state
        )
        self._tmx.run(
            input_state=tmx_states.TmxInputState(
                temperature=inputs.temperature.data,
                virtual_temperature=inputs.virtual_temperature.data,
                pressure=inputs.pressure.data,
                u=inputs.u.data,
                v=inputs.v.data,
                w=inputs.w.data,
                qv=inputs.qv.data,
                qc=inputs.qc.data,
                qi=inputs.qi.data,
                qr=inputs.qr.data,
                qs=inputs.qs.data,
                qg=inputs.qg.data,
                rho=inputs.rho.data,
                air_mass=self._air_mass,
                cv_air=self._cv_air,
            ),
            surface_flux_state=self._surface_flux_state,
            diagnostic_state=tmx_states.TmxDiagnosticState(**_data(out, _DIAGNOSTICS)),
            tendency_state=tmx_states.TmxTendencyState(**_data(out, _TENDENCIES)),
            new_state=self._new_state,
            dtime=self._dt_seconds,
        )
        return out


# the Output leaves that make up each tmx state, by name
_TENDENCIES = tuple(field.name for field in dataclasses.fields(tmx_states.TmxTendencyState))
_DIAGNOSTICS = tuple(field.name for field in dataclasses.fields(tmx_states.TmxDiagnosticState))


def _data(out: TmxComponent.Output, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: getattr(out, name).data for name in names}


def collect_input(entry: physics_state.EntryState) -> TmxComponent.Input:
    """The tmx input from the physics entry state (no copies)."""
    return TmxComponent.Input(
        temperature=entry.temperature,
        virtual_temperature=entry.virtual_temperature,
        pressure=entry.pressure,
        pressure_ifc=entry.pressure_ifc,
        u=entry.u,
        v=entry.v,
        w=entry.w,
        rho=entry.rho,
        qv=entry.qv,
        qc=entry.qc,
        qi=entry.qi,
        qr=entry.qr,
        qs=entry.qs,
        qg=entry.qg,
    )
