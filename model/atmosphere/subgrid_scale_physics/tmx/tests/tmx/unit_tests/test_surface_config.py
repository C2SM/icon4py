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
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.surface import surface, surface_states
from icon4py.model.common import dimension as dims, model_backends, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


def test_default_config_matches_fortran_defaults() -> None:
    """Defaults must match ``vdiff_config_init`` (mo_turb_vdiff_config.f90) and mo_sea_ice_nml.f90."""
    config = surface.TmxSurfaceConfig()
    assert config.fsl == 0.4
    assert config.z0m_min == 1.5e-5
    assert config.z0m_ice == 1.0e-3
    assert config.z0m_oce == 1.0e-3
    assert config.min_sfc_wind == 0.3
    assert config.wind_g == 3.0
    assert config.ice_thermodynamics_type == 1
    assert config.ice_layer_heat_capacity_thickness == 0.10
    assert config.ocean_freezing_temperature == -1.80
    assert config.use_no_flux_gradients is True
    assert config.ice_albedo_scheme == 1
    assert config.prescribed_flux_mode is False


@pytest.mark.parametrize("ice_type", [0, 2, 3, 4])
def test_config_rejects_unsupported_ice_thermodynamics(ice_type: int) -> None:
    with pytest.raises(ValueError, match="ice_thermodynamics_type"):
        surface.TmxSurfaceConfig(ice_thermodynamics_type=ice_type)


def test_config_rejects_unsupported_ice_albedo_scheme() -> None:
    with pytest.raises(ValueError, match="ice_albedo_scheme"):
        surface.TmxSurfaceConfig(ice_albedo_scheme=2)


@pytest.mark.parametrize("min_sfc_wind", [0.0, -1.0])
def test_config_rejects_non_positive_min_sfc_wind(min_sfc_wind: float) -> None:
    with pytest.raises(ValueError, match="min_sfc_wind"):
        surface.TmxSurfaceConfig(min_sfc_wind=min_sfc_wind)


def test_config_rejects_non_positive_z0m_min() -> None:
    with pytest.raises(ValueError, match="z0m_min"):
        surface.TmxSurfaceConfig(z0m_min=0.0)


def _echoed_vdf_record(**overrides: object) -> list[object]:
    """A positional t_vdiff_config record as echoed in aes_vdf_nml.

    Positions not pinned by a TmxSurfaceConfig option get a dummy value; the
    overrides are placed at the 'unnamed_index' positions of the surface options.
    """
    positions = {
        "use_tmx": 22,
        "fsl": 15,
        "z0m_min": 18,
        "z0m_ice": 19,
        "z0m_oce": 20,
        "min_sfc_wind": 39,
        "wind_g": 40,
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
                fsl=0.5,
                z0m_min=2.0e-5,
                z0m_ice=1.1e-3,
                z0m_oce=1.2e-3,
                min_sfc_wind=0.4,
                wind_g=3.5,
            )
        }
    }
    config = surface.TmxSurfaceConfig.from_fortran_dict(fortran_dict)
    assert config.fsl == 0.5
    assert config.z0m_min == 2.0e-5
    assert config.z0m_ice == 1.1e-3
    assert config.z0m_oce == 1.2e-3
    assert config.min_sfc_wind == 0.4
    assert config.wind_g == 3.5
    # sea-ice options are not read from the namelist yet; keep the defaults
    assert config.ice_thermodynamics_type == 1
    assert config.prescribed_flux_mode is False


def test_config_from_fortran_dict_rejects_changed_member_count() -> None:
    record = _echoed_vdf_record()
    with pytest.raises(ValueError, match="not a multiple"):
        surface.TmxSurfaceConfig.from_fortran_dict(
            {"aes_vdf_nml": {"aes_vdf_config": [*record, 0.0]}}
        )


def test_config_from_fortran_dict_rejects_missing_use_tmx() -> None:
    record = _echoed_vdf_record()
    record[22] = False
    with pytest.raises(ValueError, match="use_tmx"):
        surface.TmxSurfaceConfig.from_fortran_dict({"aes_vdf_nml": {"aes_vdf_config": record}})


def test_params_defaults() -> None:
    params = surface.TmxSurfaceParams()
    assert params.von_karman == 0.4
    assert params.charnock == 0.018
    assert params.viscous_coeff == 0.11
    assert params.bsm == 5.0 and params.bsh == 5.0
    assert params.bum == 16.0 and params.buh == 16.0
    assert params.half_pi == pytest.approx(np.pi / 2.0)
    assert params.ln2 == pytest.approx(np.log(2.0))


def test_surface_flux_and_state_allocation_produces_zero_cell_fields(
    grid: base_grid.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    expected_shape = (grid.num_cells,)
    for state in (
        tmx_states.TmxSurfaceFluxState.allocate(grid, allocator=allocator),
        surface_states.SurfaceState.allocate(grid, allocator=allocator),
    ):
        for field in dataclasses.fields(state):
            data = getattr(state, field.name).asnumpy()
            assert data.shape == expected_shape, f"Wrong shape for field '{field.name}'."
            assert np.all(data == 0.0), f"Field '{field.name}' is not zero-initialized."


def _zero_surface_input(
    grid: base_grid.Grid, allocator: model_backends.BackendLike
) -> surface_states.SurfaceInputState:
    fields = {
        f.name: data_alloc.zero_field(grid, dims.CellDim, dtype=ta.wpfloat, allocator=allocator)
        for f in dataclasses.fields(surface_states.SurfaceInputState)
    }
    return surface_states.SurfaceInputState(**fields)


def test_granule_prescribed_flux_mode_is_a_no_op(
    grid: base_grid.Grid,
    backend_like: model_backends.BackendLike,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    granule = surface.TmxSurface(
        grid=grid,
        config=surface.TmxSurfaceConfig(prescribed_flux_mode=True),
        params=surface.TmxSurfaceParams(),
        backend=backend_like,
    )
    granule.run(
        input_state=_zero_surface_input(grid, allocator),
        surface_state=surface_states.SurfaceState.allocate(grid, allocator=allocator),
        flux_state=tmx_states.TmxSurfaceFluxState.allocate(grid, allocator=allocator),
        dtime=1.0,
    )
