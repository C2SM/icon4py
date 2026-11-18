# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import math
import types

import numpy as np
import pytest

from icon4py.model.common import constants as phy_const
from icon4py.model.common.grid import geometry_attributes as geometry_meta, icon as icon_grid
from icon4py.model.common.initial_condition.analytical import aes_bubble, utils as testcases_utils
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.physics.thermodynamics import compute_moisture, compute_pressure
from icon4py.model.common.states import tracer_states
from icon4py.model.driver import config as driver_config
from icon4py.model.testing import config as test_config

from .. import utils


_NUM_CELLS = 7
_NUM_LEVELS = 40
_MODEL_TOP = 20000.0
_REFERENCE_TEMPERATURE = 250.0
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


@dataclasses.dataclass(frozen=True)
class _Column:
    z_mc: np.ndarray
    wgtfac_c: np.ndarray
    ddqz_z_half: np.ndarray
    exner_ref_mc: np.ndarray
    d_exner_dz_ref_ic: np.ndarray
    theta_ref_mc: np.ndarray


def _column() -> _Column:
    """Flat, vertically stretched levels with an isothermal reference atmosphere.

    The stretching makes the interpolation weights differ from 1/2, so that mixing up
    full and half level indices shows in the results.
    """
    s = np.linspace(0.0, 1.0, _NUM_LEVELS + 1)
    z_ifc = _MODEL_TOP * (1.0 - s) ** 2
    z_mc = 0.5 * (z_ifc[:-1] + z_ifc[1:])

    wgtfac_c = np.zeros(_NUM_LEVELS + 1)
    wgtfac_c[1:-1] = (z_mc[:-1] - z_ifc[1:-1]) / (z_mc[:-1] - z_mc[1:])
    ddqz_z_half = np.empty(_NUM_LEVELS + 1)
    ddqz_z_half[1:-1] = z_mc[:-1] - z_mc[1:]
    ddqz_z_half[0] = 2.0 * (z_ifc[0] - z_mc[0])
    ddqz_z_half[-1] = 2.0 * (z_mc[-1] - z_ifc[-1])

    def exner_ref(z: np.ndarray) -> np.ndarray:
        return np.exp(-phy_const.GRAV * z / (phy_const.CPD * _REFERENCE_TEMPERATURE))

    exner_ref_mc = exner_ref(z_mc)
    d_exner_dz_ref_ic = (
        -phy_const.GRAV / (phy_const.CPD * _REFERENCE_TEMPERATURE) * exner_ref(z_ifc)
    )
    theta_ref_mc = _REFERENCE_TEMPERATURE / exner_ref_mc

    def on_cells(column: np.ndarray) -> np.ndarray:
        return np.tile(column, (_NUM_CELLS, 1))

    return _Column(
        z_mc=on_cells(z_mc),
        wgtfac_c=on_cells(wgtfac_c),
        ddqz_z_half=on_cells(ddqz_z_half),
        exner_ref_mc=on_cells(exner_ref_mc),
        d_exner_dz_ref_ic=on_cells(d_exner_dz_ref_ic),
        theta_ref_mc=on_cells(theta_ref_mc),
    )


def _cell_coordinates() -> tuple[np.ndarray, np.ndarray]:
    cell_x = np.linspace(-30000.0, 30000.0, _NUM_CELLS)
    cell_y = np.linspace(20000.0, -10000.0, _NUM_CELLS)
    return cell_x, cell_y


def _horizontal_weight(config: aes_bubble.AesBubbleConfig) -> np.ndarray:
    cell_x, cell_y = _cell_coordinates()
    sigma = config.bubble_horizontal_width * _FWHM_TO_SIGMA
    weight = np.exp(-0.5 * ((cell_x - config.bubble_center_x) / sigma) ** 2)
    if config.gaussian_in_y:
        weight = weight * np.exp(-0.5 * (cell_y / sigma) ** 2)
    return weight


def _fortran_first_guess(config: aes_bubble.AesBubbleConfig, column: _Column) -> dict:
    cell_x, cell_y = _cell_coordinates()
    return utils.reference_first_guess(
        z_mc=column.z_mc,
        cell_x=cell_x,
        cell_y=cell_y,
        psfc=config.surface_pressure,
        t_am=config.minimum_temperature,
        t0=config.surface_temperature,
        gamma0=config.boundary_layer_lapse_rate,
        z0=config.boundary_layer_height,
        gamma1=config.free_atmosphere_lapse_rate,
        t_perturb=config.bubble_temperature_amplitude,
        relhum_bg=config.background_relative_humidity,
        relhum_mx=config.bubble_relative_humidity,
        hw_x=config.bubble_horizontal_width,
        hw_z=config.bubble_vertical_width,
        x_center=config.bubble_center_x,
        y_center=0.0,
        lgaussxy=config.gaussian_in_y,
    )


# A configuration that reaches the minimum temperature and moves the bubble off centre.
_CONFIGS = [
    aes_bubble.AesBubbleConfig(),
    aes_bubble.AesBubbleConfig(gaussian_in_y=True),
    aes_bubble.AesBubbleConfig(
        free_atmosphere_lapse_rate=0.008,
        minimum_temperature=230.0,
        bubble_center_x=12000.0,
        bubble_vertical_width=2000.0,
        bubble_temperature_amplitude=5.0,
    ),
]


def test_config_defaults_match_fortran() -> None:
    config = aes_bubble.AesBubbleConfig()
    assert {
        "psfc": config.surface_pressure,
        "t_am": config.minimum_temperature,
        "t0": config.surface_temperature,
        "gamma0": config.boundary_layer_lapse_rate,
        "z0": config.boundary_layer_height,
        "gamma1": config.free_atmosphere_lapse_rate,
        "t_perturb": config.bubble_temperature_amplitude,
        "relhum_bg": config.background_relative_humidity,
        "relhum_mx": config.bubble_relative_humidity,
        "hw_x": config.bubble_horizontal_width,
        "hw_z": config.bubble_vertical_width,
        "x_center": config.bubble_center_x,
        "lgaussxy": config.gaussian_in_y,
    } == utils.AES_BUBBLE_DEFAULTS


def test_experiment_config() -> None:
    config = driver_config.read_experiment_config_from_yaml(
        test_config.EXPERIMENT_CONFIG_PATH / "aes_bubble.yaml"
    )
    assert config.initial_condition == aes_bubble.AesBubbleConfig()


@pytest.mark.parametrize(
    "field, value",
    [
        ("minimum_temperature", 0.0),
        ("surface_pressure", -1.0),
        ("boundary_layer_lapse_rate", -0.001),
        ("free_atmosphere_lapse_rate", -0.001),
        ("boundary_layer_height", 0.0),
        ("bubble_temperature_amplitude", -1.0),
        ("background_relative_humidity", 1.1),
        ("bubble_relative_humidity", -0.1),
        ("bubble_horizontal_width", 0.0),
        ("bubble_vertical_width", 0.0),
    ],
)
def test_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        aes_bubble.AesBubbleConfig(**{field: value})


@pytest.mark.parametrize("config", _CONFIGS)
def test_first_guess_profiles_match_fortran(config: aes_bubble.AesBubbleConfig) -> None:
    column = _column()
    computed = aes_bubble.compute_first_guess_profiles(
        ic_config=config, z_mc=column.z_mc, horizontal_weight=_horizontal_weight(config)
    )
    reference = _fortran_first_guess(config, column)

    for name, value in zip(("temp", "rh", "exner", "theta_v", "rho", "qv"), computed, strict=True):
        np.testing.assert_allclose(value, reference[name], rtol=1e-14, atol=0.0, err_msg=name)


@pytest.mark.parametrize("config", _CONFIGS)
def test_hydrostatic_adjustment_iterative_matches_fortran(
    config: aes_bubble.AesBubbleConfig,
) -> None:
    column = _column()
    reference_first_guess = _fortran_first_guess(config, column)
    reference = utils.reference_hydro_adjust_iterative(
        temp=reference_first_guess["temp"],
        rh=reference_first_guess["rh"],
        exner=reference_first_guess["exner"],
        theta_v=reference_first_guess["theta_v"],
        rho=reference_first_guess["rho"],
        qv=reference_first_guess["qv"],
        wgtfac_c=column.wgtfac_c,
        ddqz_z_half=column.ddqz_z_half,
        exner_ref_mc=column.exner_ref_mc,
        d_exner_dz_ref_ic=column.d_exner_dz_ref_ic,
        theta_ref_mc=column.theta_ref_mc,
    )

    temperature, relative_humidity, exner, theta_v, rho, qv = (
        aes_bubble.compute_first_guess_profiles(
            ic_config=config, z_mc=column.z_mc, horizontal_weight=_horizontal_weight(config)
        )
    )
    testcases_utils.hydrostatic_adjustment_iterative_ndarray(
        temperature=temperature,
        relative_humidity=relative_humidity,
        wgtfac_c=column.wgtfac_c,
        ddqz_z_half=column.ddqz_z_half,
        exner_ref_mc=column.exner_ref_mc,
        d_exner_dz_ref_ic=column.d_exner_dz_ref_ic,
        theta_ref_mc=column.theta_ref_mc,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        qv=qv,
        num_levels=_NUM_LEVELS,
    )

    for name, value in (("exner", exner), ("theta_v", theta_v), ("rho", rho), ("qv", qv)):
        np.testing.assert_allclose(value, reference[name], rtol=1e-13, atol=0.0, err_msg=name)


@pytest.mark.parametrize("config", _CONFIGS)
def test_hydrostatic_adjustment_iterative_balances_the_state(
    config: aes_bubble.AesBubbleConfig,
) -> None:
    column = _column()
    temperature, relative_humidity, exner, theta_v, rho, qv = (
        aes_bubble.compute_first_guess_profiles(
            ic_config=config, z_mc=column.z_mc, horizontal_weight=_horizontal_weight(config)
        )
    )
    testcases_utils.hydrostatic_adjustment_iterative_ndarray(
        temperature=temperature,
        relative_humidity=relative_humidity,
        wgtfac_c=column.wgtfac_c,
        ddqz_z_half=column.ddqz_z_half,
        exner_ref_mc=column.exner_ref_mc,
        d_exner_dz_ref_ic=column.d_exner_dz_ref_ic,
        theta_ref_mc=column.theta_ref_mc,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        qv=qv,
        num_levels=_NUM_LEVELS,
    )

    # Converged: theta_v is consistent with the prescribed temperature and the final qv.
    np.testing.assert_allclose(
        theta_v * exner, temperature * (1.0 + phy_const.RV_O_RD_MINUS_1 * qv), rtol=1e-12
    )
    # The prescribed relative humidity is kept.
    pressure = phy_const.P0REF * exner**phy_const.CPD_O_RD
    np.testing.assert_allclose(
        qv,
        compute_moisture.specific_humidity(
            relative_humidity * compute_pressure.sat_pres_water(temperature), pressure
        ),
        rtol=1e-12,
    )
    # Equation of state.
    np.testing.assert_allclose(
        rho, exner**phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v), rtol=1e-14
    )
    # The lowest level is the boundary condition of the upward integration.
    first_guess_exner = aes_bubble.compute_first_guess_profiles(
        ic_config=config, z_mc=column.z_mc, horizontal_weight=_horizontal_weight(config)
    )[2]
    np.testing.assert_array_equal(exner[:, -1], first_guess_exner[:, -1])


def test_first_guess_profiles_shape_of_the_bubble() -> None:
    config = aes_bubble.AesBubbleConfig()
    column = _column()
    weight = _horizontal_weight(config)
    temperature, relative_humidity, *_ = aes_bubble.compute_first_guess_profiles(
        ic_config=config, z_mc=column.z_mc, horizontal_weight=weight
    )
    background, *_ = aes_bubble.compute_first_guess_profiles(
        ic_config=config, z_mc=column.z_mc, horizontal_weight=np.zeros_like(weight)
    )
    sigma_z = config.bubble_vertical_width * _FWHM_TO_SIGMA
    vertical_weight = np.exp(-0.5 * (column.z_mc / sigma_z) ** 2)
    bubble_weight = weight[:, np.newaxis] * vertical_weight

    np.testing.assert_allclose(
        temperature - background,
        config.bubble_temperature_amplitude * bubble_weight,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        relative_humidity,
        config.background_relative_humidity
        + (config.bubble_relative_humidity - config.background_relative_humidity) * bubble_weight,
        rtol=1e-14,
    )


class _FakeFactory:
    def __init__(self, fields: dict[str, np.ndarray]) -> None:
        self._fields = fields

    def get_full_precision(self, name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(ndarray=self._fields[name])


def _fake_field(shape: tuple[int, ...]) -> types.SimpleNamespace:
    return types.SimpleNamespace(ndarray=np.ones(shape))


def test_aes_bubble_on_a_grid_not_centred_at_the_origin() -> None:
    """The bubble is placed relative to the domain centre of the grid, as in ICON."""
    config = aes_bubble.AesBubbleConfig(gaussian_in_y=True, bubble_center_x=12000.0)
    domain_center = (50000.0, 40000.0)
    column = _column()
    cell_x, cell_y = _cell_coordinates()
    cell_x = cell_x + domain_center[0]
    cell_y = cell_y + domain_center[1]

    grid = types.SimpleNamespace(
        geometry_type=icon_grid.GeometryType.TORUS,
        num_levels=_NUM_LEVELS,
        grid_params=icon_grid.GridParams(
            icon_grid.TorusParams(
                domain_length=100000.0, domain_height=80000.0, domain_center=domain_center
            )
        ),
    )
    static_fields = types.SimpleNamespace(
        geometry=_FakeFactory(
            {geometry_meta.CELL_CENTER_X: cell_x, geometry_meta.CELL_CENTER_Y: cell_y}
        ),
        metrics=_FakeFactory(
            {
                metrics_attributes.Z_MC: column.z_mc,
                metrics_attributes.WGTFAC_C: column.wgtfac_c,
                metrics_attributes.DDQZ_Z_HALF: column.ddqz_z_half,
                metrics_attributes.EXNER_REF_MC: column.exner_ref_mc,
                metrics_attributes.D_EXNER_DZ_REF_IC: column.d_exner_dz_ref_ic,
                metrics_attributes.THETA_REF_MC: column.theta_ref_mc,
            }
        ),
    )
    full_level_shape = (_NUM_CELLS, _NUM_LEVELS)
    prognostic_state = types.SimpleNamespace(
        exner=_fake_field(full_level_shape),
        theta_v=_fake_field(full_level_shape),
        rho=_fake_field(full_level_shape),
        vn=_fake_field((3 * _NUM_CELLS, _NUM_LEVELS)),
        w=_fake_field((_NUM_CELLS, _NUM_LEVELS + 1)),
    )
    tracer_state = tracer_states.TracerState(
        qv=_fake_field(full_level_shape),
        qc=_fake_field(full_level_shape),
        qr=_fake_field(full_level_shape),
    )

    aes_bubble.aes_bubble(
        config=types.SimpleNamespace(initial_condition=config),
        grid=grid,
        static_fields=static_fields,
        prognostic_state_now=prognostic_state,
        tracer_state_now=tracer_state,
    )

    first_guess = utils.reference_first_guess(
        z_mc=column.z_mc,
        cell_x=cell_x,
        cell_y=cell_y,
        psfc=config.surface_pressure,
        t_am=config.minimum_temperature,
        t0=config.surface_temperature,
        gamma0=config.boundary_layer_lapse_rate,
        z0=config.boundary_layer_height,
        gamma1=config.free_atmosphere_lapse_rate,
        t_perturb=config.bubble_temperature_amplitude,
        relhum_bg=config.background_relative_humidity,
        relhum_mx=config.bubble_relative_humidity,
        hw_x=config.bubble_horizontal_width,
        hw_z=config.bubble_vertical_width,
        x_center=domain_center[0] + config.bubble_center_x,
        y_center=domain_center[1],
        lgaussxy=config.gaussian_in_y,
    )
    reference = utils.reference_hydro_adjust_iterative(
        temp=first_guess["temp"],
        rh=first_guess["rh"],
        exner=first_guess["exner"],
        theta_v=first_guess["theta_v"],
        rho=first_guess["rho"],
        qv=first_guess["qv"],
        wgtfac_c=column.wgtfac_c,
        ddqz_z_half=column.ddqz_z_half,
        exner_ref_mc=column.exner_ref_mc,
        d_exner_dz_ref_ic=column.d_exner_dz_ref_ic,
        theta_ref_mc=column.theta_ref_mc,
    )
    computed = {
        "exner": prognostic_state.exner.ndarray,
        "theta_v": prognostic_state.theta_v.ndarray,
        "rho": prognostic_state.rho.ndarray,
        "qv": tracer_state.qv.ndarray,
    }
    for name, value in computed.items():
        np.testing.assert_allclose(value, reference[name], rtol=1e-13, atol=0.0, err_msg=name)
    for field in (prognostic_state.vn, prognostic_state.w, tracer_state.qc, tracer_state.qr):
        np.testing.assert_array_equal(field.ndarray, 0.0)
