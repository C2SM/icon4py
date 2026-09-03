# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.config import (
    EnergyType,
    SurfaceType,
    TmxConfig,
    TurbulenceSolverType,
)
from icon4py.model.common import model_backends
from icon4py.model.common.config import config_io


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


def test_default_config_matches_fortran_defaults() -> None:
    """Defaults must match ``vdiff_config_init`` in mo_turb_vdiff_config.f90."""
    config = TmxConfig()
    assert config.solver_type == TurbulenceSolverType.IMPLICIT
    assert config.energy_type == EnergyType.INTERNAL
    assert config.dissipation_factor == 1.0
    assert config.use_louis is True
    assert config.use_louis_land is True
    assert config.use_louis_ice is True
    assert config.louis_constant_b == 4.2
    assert config.use_km_const is False
    assert config.km_const == 1.0
    assert config.use_scale_turb_energy_flux is False
    assert config.scale_turb_energy_flux == 1.0
    assert config.smag_constant == 0.23
    # exact Fortran literal, not 1/3
    assert config.turb_prandtl == 0.33333333333
    assert config.km_min == 0.001
    assert config.max_turb_scale == 300.0


@pytest.mark.parametrize("turb_prandtl", [0.0, -1.0])
def test_config_rejects_non_positive_turb_prandtl(turb_prandtl: float) -> None:
    with pytest.raises(ValueError, match="turb_prandtl"):
        TmxConfig(turb_prandtl=turb_prandtl)


def test_config_rejects_negative_km_min() -> None:
    with pytest.raises(ValueError, match="km_min"):
        TmxConfig(km_min=-1.0)


def test_config_coerces_enums_from_ints() -> None:
    config = TmxConfig(solver_type=1, energy_type=1)
    assert config.solver_type is TurbulenceSolverType.EXPLICIT
    assert config.energy_type is EnergyType.DRY_STATIC


def test_config_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValueError):
        TmxConfig(solver_type=3)


def _echoed_vdf_record(**overrides: object) -> list[object]:
    """A positional t_vdiff_config record as echoed in aes_vdf_nml.

    Positions not pinned by a TmxConfig option get a dummy value; the
    overrides are placed at the pinned 'unnamed_index' positions.
    """
    positions = {
        "use_tmx": 22,
        "solver_type": 23,
        "energy_type": 24,
        "dissipation_factor": 25,
        "use_louis": 26,
        "use_louis_land": 27,
        "use_louis_ice": 28,
        "louis_constant_b": 29,
        "use_km_const": 30,
        "km_const": 31,
        "use_scale_turb_energy_flux": 32,
        "scale_turb_energy_flux": 33,
        "smag_constant": 34,
        "turb_prandtl": 35,
        "km_min": 37,
        "max_turb_scale": 38,
    }
    record: list[object] = [0.0] * 42
    record[positions["use_tmx"]] = True
    for name, value in overrides.items():
        record[positions[name]] = value
    return record


def test_config_from_fortran_dict() -> None:
    fortran_dict = {
        "aes_vdf_nml": {
            "aes_vdf_config": _echoed_vdf_record(
                solver_type=1,
                energy_type=1,
                dissipation_factor=0.5,
                use_louis=False,
                use_louis_land=False,
                use_louis_ice=False,
                louis_constant_b=2.1,
                use_km_const=True,
                km_const=2.0,
                use_scale_turb_energy_flux=True,
                scale_turb_energy_flux=0.9,
                smag_constant=0.28,
                turb_prandtl=0.5,
                km_min=0.002,
                max_turb_scale=150.0,
            )
        }
    }
    config = TmxConfig.from_fortran_dict(atm_dict=fortran_dict, input_dict={})
    assert config.solver_type is TurbulenceSolverType.EXPLICIT
    assert config.energy_type is EnergyType.DRY_STATIC
    assert config.dissipation_factor == 0.5
    assert config.use_louis is False
    assert config.use_louis_land is False
    assert config.use_louis_ice is False
    assert config.louis_constant_b == 2.1
    assert config.use_km_const is True
    assert config.km_const == 2.0
    assert config.use_scale_turb_energy_flux is True
    assert config.scale_turb_energy_flux == 0.9
    assert config.smag_constant == 0.28
    assert config.turb_prandtl == 0.5
    assert config.km_min == 0.002
    assert config.max_turb_scale == 150.0


def test_config_from_fortran_dict_rejects_changed_member_count() -> None:
    record = _echoed_vdf_record()
    with pytest.raises(ValueError, match="not a multiple"):
        TmxConfig.from_fortran_dict(
            atm_dict={"aes_vdf_nml": {"aes_vdf_config": [*record, 0.0]}}, input_dict={}
        )


def test_config_from_fortran_dict_rejects_missing_use_tmx() -> None:
    record = _echoed_vdf_record()
    record[22] = False
    with pytest.raises(ValueError, match="use_tmx"):
        TmxConfig.from_fortran_dict(
            atm_dict={"aes_vdf_nml": {"aes_vdf_config": record}}, input_dict={}
        )


def _expected_shapes(
    grid: base_grid.Grid,
) -> dict[str, tuple[int, ...]]:
    nlev = grid.num_levels
    return {
        "cell_full": (grid.num_cells, nlev),
        "cell_half": (grid.num_cells, nlev + 1),
        "edge_full": (grid.num_edges, nlev),
        "edge_half": (grid.num_edges, nlev + 1),
        "vertex_full": (grid.num_vertices, nlev),
        "vertex_half": (grid.num_vertices, nlev + 1),
        "cell_2d": (grid.num_cells,),
    }


SURFACE_FLUX_FIELD_KINDS = {
    "evapotranspiration": "cell_2d",
    "sensible_heat_flux": "cell_2d",
    "u_stress": "cell_2d",
    "v_stress": "cell_2d",
    "q_snocpymlt": "cell_2d",
}


@pytest.mark.parametrize(
    ("state_cls", "field_kinds"),
    [(tmx_states.TmxSurfaceFluxState, SURFACE_FLUX_FIELD_KINDS)],
    ids=["surface_flux"],
)
def test_state_allocation_produces_zero_fields_with_correct_shapes(
    grid: base_grid.Grid,
    backend_like: model_backends.BackendLike,
    state_cls: type,
    field_kinds: dict[str, str],
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    state = state_cls.allocate(grid, allocator=allocator)
    shapes = _expected_shapes(grid)

    state_field_names = {f.name for f in dataclasses.fields(state_cls)}
    assert state_field_names == set(field_kinds.keys())

    for name, kind in field_kinds.items():
        field = getattr(state, name).asnumpy()
        assert field.shape == shapes[kind], f"Wrong shape for field '{name}'."
        assert np.all(field == 0.0), f"Field '{name}' is not zero-initialized."


def test_config_round_trips_through_config_io() -> None:
    """Every enum option is registered, so the config survives (un)structuring."""
    config = TmxConfig()
    unstructured = config_io.CONV.unstructure(config)

    assert unstructured["solver_type"] == "implicit"
    assert unstructured["energy_type"] == "internal"
    assert unstructured["surface_type"] == "interactive"
    assert config_io.CONV.structure(unstructured, TmxConfig) == config


def test_surface_flux_options_come_from_the_input_namelist() -> None:
    """Absent 'nh_testcase_nml' members keep the TmxConfig default."""
    record = _echoed_vdf_record(solver_type=2, energy_type=2, turb_prandtl=0.33333333333)
    config = TmxConfig.from_fortran_dict(
        atm_dict={"aes_vdf_nml": {"aes_vdf_config": record}},
        input_dict={"nh_testcase_nml": {"isrfc_type": 1, "shflx": 0.2}},
    )
    assert config.surface_type is SurfaceType.FIXED_HEAT_FLUXES
    assert config.shflx == 0.2
    assert config.lhflx == 0.0
