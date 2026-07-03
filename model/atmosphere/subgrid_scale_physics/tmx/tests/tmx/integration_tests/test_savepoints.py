# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the TMX serialbox savepoint readers.

Opens every tmx savepoint written by the instrumented ICON run
(exp.exclaim_ape_aesPhys) and verifies field shapes and dtypes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx
from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403
from .utils import TMX_DATES


if TYPE_CHECKING:
    from icon4py.model.testing import serialbox as sb


#: vertical size specifiers: number of full levels, half levels (nlev + 1),
#: 3 extrapolation coefficients, or no vertical dimension (2D surface field)
FULL = "full"
HALF = "half"
COEFF3 = "coeff3"
SURFACE = "surface"

SAVEPOINT_FIELDS: tuple[tuple[str, str, gtx.Dimension, str], ...] = (
    # (factory method on IconSerialDataProvider, field accessor, horizontal dim, vertical size)
    ("from_savepoint_tmx_init", "inv_ddqz_z_half", dims.CellDim, HALF),
    ("from_savepoint_tmx_init", "inv_ddqz_z_half_e", dims.EdgeDim, HALF),
    ("from_savepoint_tmx_init", "inv_ddqz_z_half_v", dims.VertexDim, HALF),
    ("from_savepoint_tmx_init", "inv_ddqz_z_full_e", dims.EdgeDim, FULL),
    ("from_savepoint_tmx_init", "wgtfacq1_c", dims.CellDim, COEFF3),
    ("from_savepoint_tmx_init", "wgtfacq1_e", dims.EdgeDim, COEFF3),
    ("from_savepoint_tmx_init", "geopot_agl_ifc", dims.CellDim, HALF),
    ("from_savepoint_tmx_init", "mix_len_sq", dims.CellDim, HALF),
    ("from_savepoint_tmx_init", "scaling_factor_louis", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_entry", "ta", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "ta_phy", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "ua", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "va", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "wa", dims.CellDim, HALF),
    ("from_savepoint_tmx_entry", "qv", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "qc", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "qi", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "qr", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "qs", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "qg", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "rho", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "tempv", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "pres", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "pres_ifc", dims.CellDim, HALF),
    ("from_savepoint_tmx_entry", "mair", dims.CellDim, FULL),
    ("from_savepoint_tmx_entry", "cvair", dims.CellDim, FULL),
    ("from_savepoint_tmx_surface_fluxes", "evspsbl", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_surface_fluxes", "hfss", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_surface_fluxes", "tauu", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_surface_fluxes", "tauv", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_surface_fluxes", "q_snocpymlt", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_diagnostics_exit", "theta_v", dims.CellDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "cptgz", dims.CellDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "ghf", dims.CellDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "bruvais", dims.CellDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "rho_ic", dims.CellDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "vn", dims.EdgeDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "u_vert", dims.VertexDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "v_vert", dims.VertexDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "w_vert", dims.VertexDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "vn_ie", dims.EdgeDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "vt_ie", dims.EdgeDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "w_ie", dims.EdgeDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "shear", dims.EdgeDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "div_of_stress", dims.EdgeDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "div_c", dims.CellDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "mech_prod", dims.CellDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "km_ic", dims.CellDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "kh_ic", dims.CellDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "km_c", dims.CellDim, FULL),
    ("from_savepoint_tmx_diagnostics_exit", "km_iv", dims.VertexDim, HALF),
    ("from_savepoint_tmx_diagnostics_exit", "km_ie", dims.EdgeDim, HALF),
    ("from_savepoint_tmx_hydro_exit", "tend_qv", dims.CellDim, FULL),
    ("from_savepoint_tmx_hydro_exit", "tend_qc", dims.CellDim, FULL),
    ("from_savepoint_tmx_hydro_exit", "tend_qi", dims.CellDim, FULL),
    ("from_savepoint_tmx_hydro_exit", "qv_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_hydro_exit", "qc_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_hydro_exit", "qi_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_temperature_exit", "energy", dims.CellDim, FULL),
    ("from_savepoint_tmx_temperature_exit", "tend_energy", dims.CellDim, FULL),
    ("from_savepoint_tmx_temperature_exit", "tend_ta", dims.CellDim, FULL),
    ("from_savepoint_tmx_temperature_exit", "ta_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_hor_wind_exit", "tot_tend", dims.EdgeDim, FULL),
    ("from_savepoint_tmx_hor_wind_exit", "tend_ua", dims.CellDim, FULL),
    ("from_savepoint_tmx_hor_wind_exit", "tend_va", dims.CellDim, FULL),
    ("from_savepoint_tmx_hor_wind_exit", "ua_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_hor_wind_exit", "va_new", dims.CellDim, FULL),
    ("from_savepoint_tmx_vert_wind_exit", "tend_wa", dims.CellDim, HALF),
    ("from_savepoint_tmx_vert_wind_exit", "wa_new", dims.CellDim, HALF),
    ("from_savepoint_tmx_exit", "tend_ta", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_qv", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_qc", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_qi", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_ua", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_va", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "tend_wa", dims.CellDim, HALF),
    ("from_savepoint_tmx_exit", "heating", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "dissip_ke", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "cptgzvi", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_exit", "dissip_ke_vi", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_exit", "int_energy_vi", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_exit", "tend_int_energy_vi", dims.CellDim, SURFACE),
    ("from_savepoint_tmx_exit", "km", dims.CellDim, FULL),
    ("from_savepoint_tmx_exit", "kh", dims.CellDim, FULL),
)


def _open_savepoint(data_provider: sb.IconSerialDataProvider, factory_name: str, date: str):
    factory = getattr(data_provider, factory_name)
    if factory_name == "from_savepoint_tmx_init":
        return factory()
    return factory(date=date)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, date",
    # shape/dtype smoke tests: one serialized timestep is enough
    [(definitions.Experiments.EXCLAIM_APE_AES, TMX_DATES[0])],
)
@pytest.mark.parametrize(
    "factory_name, field_name, horizontal_dim, vertical",
    SAVEPOINT_FIELDS,
    ids=[f"{factory[15:]}-{field}" for factory, field, _, _ in SAVEPOINT_FIELDS],
)
def test_tmx_savepoint_field_shapes_and_dtypes(  # noqa: PLR0917 [too-many-positional-arguments]
    data_provider: sb.IconSerialDataProvider,
    factory_name: str,
    field_name: str,
    horizontal_dim: gtx.Dimension,
    vertical: str,
    date: str,
) -> None:
    savepoint = _open_savepoint(data_provider, factory_name, date)
    field = getattr(savepoint, field_name)()

    num_levels = data_provider.grid_size[dims.KDim]
    expected_vertical_size = {
        FULL: num_levels,
        HALF: num_levels + 1,
        COEFF3: 3,
        SURFACE: None,
    }[vertical]

    expected_dims = (
        (horizontal_dim,) if expected_vertical_size is None else (horizontal_dim, dims.KDim)
    )
    assert field.domain.dims == expected_dims

    array = field.asnumpy()
    assert array.dtype == np.float64
    assert array.shape[0] == data_provider.grid_size[horizontal_dim]
    if expected_vertical_size is not None:
        assert array.shape[1] == expected_vertical_size


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_tmx_init_savepoint_scalars(data_provider: sb.IconSerialDataProvider) -> None:
    savepoint = data_provider.from_savepoint_tmx_init()

    for name in (
        "smag_constant",
        "turb_prandtl",
        "rturb_prandtl",
        "km_min",
        "max_turb_scale",
        "dissipation_factor",
        "louis_constant_b",
        "km_const",
        "dtime",
    ):
        value = getattr(savepoint, name)()
        assert np.isscalar(value), f"{name} is not a scalar"
        assert np.issubdtype(type(value), np.floating), f"{name} is not a float"

    for name in ("solver_type", "energy_type"):
        value = getattr(savepoint, name)()
        assert np.issubdtype(type(value), np.integer), f"{name} is not an int"

    for name in ("use_louis", "use_km_const"):
        value = getattr(savepoint, name)()
        assert isinstance(value, bool), f"{name} is not a bool"

    assert savepoint.dtime() > 0.0
    assert savepoint.turb_prandtl() * savepoint.rturb_prandtl() == pytest.approx(1.0)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
def test_tmx_config_matches_serialized_scalars(
    tmx_config: tmx.TmxConfig, data_provider: sb.IconSerialDataProvider
) -> None:
    """Guard against Fortran/Python default drift.

    ``TmxConfig.from_fortran_dict`` reads the converted *input* namelists,
    which only contain the explicitly set ``aes_vdf_config`` members; all
    other options rely on the dataclass defaults matching the Fortran
    initialization. Cross-check every config member against the values the
    instrumented Fortran run actually used (serialized in the tmx-init
    savepoint).
    """
    savepoint = data_provider.from_savepoint_tmx_init()

    assert tmx_config.solver_type == tmx.TurbulenceSolverType(int(savepoint.solver_type()))
    assert tmx_config.energy_type == tmx.EnergyType(int(savepoint.energy_type()))
    assert tmx_config.use_louis == savepoint.use_louis()
    assert tmx_config.use_km_const == savepoint.use_km_const()
    for name in (
        "dissipation_factor",
        "louis_constant_b",
        "km_const",
        "smag_constant",
        "turb_prandtl",
        "km_min",
        "max_turb_scale",
    ):
        assert getattr(tmx_config, name) == getattr(savepoint, name)(), name
