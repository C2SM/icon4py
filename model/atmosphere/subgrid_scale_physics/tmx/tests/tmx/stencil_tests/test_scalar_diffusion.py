# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.scalar_diffusion import (
    compute_energy_from_temperature,
    diffuse_energy_and_update_temperature,
    diffuse_tracer,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.constants import PhysicsConstants as phy
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests

from .test_vertical_diffusion import (
    diffusion_matrix_numpy,
    implicit_diffusion_tendency_numpy,
    matrix_diagonals_on_rows,
)


_DOMAIN_ARGS = ("horizontal_start", "horizontal_end", "vertical_start", "vertical_end")


def moist_heat_capacity_numpy(
    qv: np.ndarray, q_liquid: np.ndarray, q_solid: np.ndarray
) -> np.ndarray:
    return (
        phy.cvd * (1.0 - qv - q_liquid - q_solid)
        + phy.cvv * qv
        + phy.cpl * q_liquid
        + phy.cpi * q_solid
    )


def energy_from_temperature_numpy(
    *,
    temperature: np.ndarray,
    qv: np.ndarray,
    q_liquid: np.ndarray,
    q_solid: np.ndarray,
    height_above_ground: np.ndarray,
    grav: float,
    use_internal_energy: bool,
) -> np.ndarray:
    if use_internal_energy:
        return (
            moist_heat_capacity_numpy(qv, q_liquid, q_solid) * temperature
            - q_liquid * phy.lvc
            - q_solid * phy.lsc
            + grav * height_above_ground * phy.cvd / phy.cpd
        )
    return phy.cpd * temperature + grav * height_above_ground


def temperature_from_energy_numpy(
    *,
    energy: np.ndarray,
    qv: np.ndarray,
    q_liquid: np.ndarray,
    q_solid: np.ndarray,
    height_above_ground: np.ndarray,
    grav: float,
    use_internal_energy: bool,
) -> np.ndarray:
    if use_internal_energy:
        internal_energy = energy - grav * height_above_ground * phy.cvd / phy.cpd
        return (internal_energy + q_liquid * phy.lvc + q_solid * phy.lsc) / (
            moist_heat_capacity_numpy(qv, q_liquid, q_solid)
        )
    return (energy - grav * height_above_ground) / phy.cpd


def diffuse_scalar_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    *,
    var: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    surface_flux: np.ndarray,
    air_mass: np.ndarray,
    rho: np.ndarray,
    km_ie: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    geofac_div: np.ndarray,
    rturb_prandtl: float,
    prefactor: float,
    dtime: float,
    cells: slice,
    rows: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """New value and tendency on cells x rows; surface_flux is per cell of the bottom row."""
    rhs = np.zeros_like(var)
    bottom = rows.stop - 1
    rhs[:, bottom] = -surface_flux * prefactor / air_mass[:, bottom]
    vertical_tend = implicit_diffusion_tendency_numpy(
        var=var, a=a, b=b, c=c, rhs=rhs, tend=np.zeros_like(var), dtime=dtime, rows=rows
    )

    e2c = connectivities[dims.E2C]
    c2e = connectivities[dims.C2E]
    diffusivity_e = prefactor * rturb_prandtl * 0.5 * (km_ie[:, :-1] + km_ie[:, 1:])
    flux_e = diffusivity_e * inv_dual_edge_length[:, np.newaxis] * (var[e2c[:, 1]] - var[e2c[:, 0]])
    divergence = np.sum(flux_e[c2e] * geofac_div[:, :, np.newaxis], axis=1)

    tend = np.zeros_like(var)
    new_var = np.zeros_like(var)
    tend[cells, rows] = vertical_tend[cells, rows] + divergence[cells, rows] / rho[cells, rows]
    new_var[cells, rows] = var[cells, rows] + tend[cells, rows] * dtime
    return new_var, tend


def _cells(grid: base.Grid) -> tuple[gtx.int32, gtx.int32]:
    cell_domain = h_grid.domain(dims.CellDim)
    # strictly inside the field, so that an output written on the whole field is caught
    return (
        grid.start_index(cell_domain(h_grid.Zone.NUDGING)) + 1,
        grid.end_index(cell_domain(h_grid.Zone.LOCAL)) - 1,
    )


def _diffusion_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, var_name: str
) -> dict:
    horizontal_start, horizontal_end = _cells(grid)
    return {
        var_name: data_alloc.random_field(dims.CellDim, dims.KDim),
        "air_mass": data_alloc.random_field(dims.CellDim, dims.KDim, low=1.0, high=2.0),
        "rho": data_alloc.random_field(dims.CellDim, dims.KDim, low=0.5, high=1.5),
        "km_ie": data_alloc.random_field(dims.EdgeDim, dims.KHalfDim, low=0.0),
        "inv_dual_edge_length": data_alloc.random_field(dims.EdgeDim, low=0.1),
        "geofac_div": data_alloc.random_field(dims.CellDim, dims.C2EDim),
        "rturb_prandtl": wpfloat(3.0),
        "dtime": wpfloat(0.5),
        "horizontal_start": horizontal_start,
        "horizontal_end": horizontal_end,
        "vertical_start": gtx.int32(0),
        "vertical_end": gtx.int32(grid.num_levels),
    }


def _slices(horizontal_start, horizontal_end, vertical_start, vertical_end) -> tuple[slice, slice]:
    return slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end)


class TestDiffuseTracer(stencil_tests.StencilTest):
    PROGRAM = diffuse_tracer
    OUTPUTS = ("new_var", "tend")
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            *_DOMAIN_ARGS,
            "rturb_prandtl",
            "prefactor",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        var: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        surface_flux: np.ndarray,
        air_mass: np.ndarray,
        rho: np.ndarray,
        km_ie: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        geofac_div: np.ndarray,
        rturb_prandtl: float,
        prefactor: float,
        dtime: float,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        cells, rows = _slices(horizontal_start, horizontal_end, vertical_start, vertical_end)
        new_var, tend = diffuse_scalar_numpy(
            stencil_tests.connectivities_asnumpy(grid),
            var=var,
            a=a,
            b=b,
            c=c,
            surface_flux=surface_flux,
            air_mass=air_mass,
            rho=rho,
            km_ie=km_ie,
            inv_dual_edge_length=inv_dual_edge_length,
            geofac_div=geofac_div,
            rturb_prandtl=rturb_prandtl,
            prefactor=prefactor,
            dtime=dtime,
            cells=cells,
            rows=rows,
        )
        return dict(new_var=new_var, tend=tend)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return dict(
            **_diffusion_input_data(data_alloc, grid, "var"),
            a=data_alloc.random_field(dims.CellDim, dims.KDim, low=-1.0, high=0.0),
            b=data_alloc.random_field(dims.CellDim, dims.KDim, low=2.0, high=3.0),
            c=data_alloc.random_field(dims.CellDim, dims.KDim, low=-1.0, high=0.0),
            surface_flux=data_alloc.random_field(dims.CellDim),
            new_var=data_alloc.zero_field(dims.CellDim, dims.KDim),
            tend=data_alloc.zero_field(dims.CellDim, dims.KDim),
            prefactor=wpfloat(1.5),
        )


def _q_liquid_and_solid(qc, qi, qr, qs, qg) -> tuple[np.ndarray, np.ndarray]:
    return qc + qr, qi + qs + qg


def _tracers(data_alloc: stencil_tests.DataAllocationWrapper, prefix: str = "") -> dict:
    return {
        f"{prefix}{name}": data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=0.01)
        for name in ("qv", "qc", "qi")
    } | {
        name: data_alloc.random_field(dims.CellDim, dims.KDim, low=0.0, high=0.01)
        for name in ("qr", "qs", "qg")
    }


class _ComputeEnergyFromTemperature:
    PROGRAM = compute_energy_from_temperature
    OUTPUTS = ("energy",)
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            *_DOMAIN_ARGS,
            "grav",
            "use_internal_energy",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        temperature: np.ndarray,
        qv: np.ndarray,
        qc: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        height_above_ground: np.ndarray,
        grav: float,
        use_internal_energy: bool,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        cells, rows = _slices(horizontal_start, horizontal_end, vertical_start, vertical_end)
        q_liquid, q_solid = _q_liquid_and_solid(qc, qi, qr, qs, qg)
        energy = np.zeros_like(temperature)
        energy[cells, rows] = energy_from_temperature_numpy(
            temperature=temperature,
            qv=qv,
            q_liquid=q_liquid,
            q_solid=q_solid,
            height_above_ground=height_above_ground,
            grav=grav,
            use_internal_energy=use_internal_energy,
        )[cells, rows]
        return dict(energy=energy)


def _compute_energy_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, use_internal_energy: bool
) -> dict:
    horizontal_start, horizontal_end = _cells(grid)
    return dict(
        temperature=data_alloc.random_field(dims.CellDim, dims.KDim, low=200.0, high=300.0),
        **_tracers(data_alloc),
        height_above_ground=data_alloc.random_field(dims.CellDim, dims.KDim, high=1.0e4),
        energy=data_alloc.zero_field(dims.CellDim, dims.KDim),
        grav=constants.GRAV,
        use_internal_energy=use_internal_energy,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(grid.num_levels),
    )


class TestComputeInternalEnergyFromTemperature(
    _ComputeEnergyFromTemperature, stencil_tests.StencilTest
):
    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _compute_energy_input_data(data_alloc, grid, use_internal_energy=True)


class TestComputeDryStaticEnergyFromTemperature(
    _ComputeEnergyFromTemperature, stencil_tests.StencilTest
):
    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _compute_energy_input_data(data_alloc, grid, use_internal_energy=False)


class _DiffuseEnergyAndUpdateTemperature:
    PROGRAM = diffuse_energy_and_update_temperature
    OUTPUTS = ("new_temperature", "tend_temperature")
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            *_DOMAIN_ARGS,
            "rturb_prandtl",
            "prefactor",
            "grav",
            "use_internal_energy",
        ),
    }

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        energy: np.ndarray,
        sensible_heat_flux: np.ndarray,
        evapotranspiration: np.ndarray,
        temperature: np.ndarray,
        new_qv: np.ndarray,
        new_qc: np.ndarray,
        new_qi: np.ndarray,
        qr: np.ndarray,
        qs: np.ndarray,
        qg: np.ndarray,
        height_above_ground: np.ndarray,
        diffusivity: np.ndarray,
        inv_dz: np.ndarray,
        air_mass: np.ndarray,
        prefactor: float,
        grav: float,
        dtime: float,
        use_internal_energy: bool,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: Any,
    ) -> dict:
        cells, rows = _slices(horizontal_start, horizontal_end, vertical_start, vertical_end)
        interfaces = slice(rows.start + 1, rows.stop)
        matrix = diffusion_matrix_numpy(
            prefactor * diffusivity[:, interfaces] * inv_dz[:, interfaces],
            1.0 / air_mass[:, rows],
        )
        a, b, c = matrix_diagonals_on_rows(matrix, air_mass.shape, rows)
        if use_internal_energy:
            temperature_sfc = temperature[:, vertical_end - 1]
            surface_flux = sensible_heat_flux + temperature_sfc * evapotranspiration * (
                phy.cvv - phy.cvd
            )
        else:
            surface_flux = sensible_heat_flux * phy.cpd / phy.cvd
        new_energy, _ = diffuse_scalar_numpy(
            stencil_tests.connectivities_asnumpy(grid),
            var=energy,
            a=a,
            b=b,
            c=c,
            air_mass=air_mass,
            surface_flux=surface_flux,
            prefactor=prefactor,
            dtime=dtime,
            cells=cells,
            rows=rows,
            **{
                name: kwargs[name]
                for name in (
                    "rho",
                    "km_ie",
                    "inv_dual_edge_length",
                    "geofac_div",
                    "rturb_prandtl",
                )
            },
        )
        q_liquid, q_solid = _q_liquid_and_solid(new_qc, new_qi, qr, qs, qg)
        new_temperature = np.zeros_like(temperature)
        tend_temperature = np.zeros_like(temperature)
        new_temperature[cells, rows] = temperature_from_energy_numpy(
            energy=new_energy,
            qv=new_qv,
            q_liquid=q_liquid,
            q_solid=q_solid,
            height_above_ground=height_above_ground,
            grav=grav,
            use_internal_energy=use_internal_energy,
        )[cells, rows]
        tend_temperature[cells, rows] = (
            new_temperature[cells, rows] - temperature[cells, rows]
        ) / dtime
        return dict(new_temperature=new_temperature, tend_temperature=tend_temperature)


def _diffuse_energy_input_data(
    data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid, use_internal_energy: bool
) -> dict:
    return dict(
        **_diffusion_input_data(data_alloc, grid, "energy"),
        diffusivity=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.0),
        inv_dz=data_alloc.random_field(dims.CellDim, dims.KHalfDim, low=0.1),
        **_tracers(data_alloc, prefix="new_"),
        sensible_heat_flux=data_alloc.random_field(dims.CellDim),
        evapotranspiration=data_alloc.random_field(dims.CellDim),
        temperature=data_alloc.random_field(dims.CellDim, dims.KDim, low=200.0, high=300.0),
        height_above_ground=data_alloc.random_field(dims.CellDim, dims.KDim, high=1.0e4),
        new_temperature=data_alloc.zero_field(dims.CellDim, dims.KDim),
        tend_temperature=data_alloc.zero_field(dims.CellDim, dims.KDim),
        prefactor=wpfloat(1.5),
        grav=constants.GRAV,
        use_internal_energy=use_internal_energy,
    )


class TestDiffuseInternalEnergyAndUpdateTemperature(
    _DiffuseEnergyAndUpdateTemperature, stencil_tests.StencilTest
):
    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _diffuse_energy_input_data(data_alloc, grid, use_internal_energy=True)


class TestDiffuseDryStaticEnergyAndUpdateTemperature(
    _DiffuseEnergyAndUpdateTemperature, stencil_tests.StencilTest
):
    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        return _diffuse_energy_input_data(data_alloc, grid, use_internal_energy=False)
