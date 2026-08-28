# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the turbulence granule configuration and its reject list.

The reject list is the implementation's statement about the granule interface (port spec D5/D6):
every namelist switch of the Fortran scheme is accepted as an argument, but the ones whose
alternative formulations were not ported are refused rather than silently ignored. These tests
pin both halves of that contract -- what is accepted and what is refused -- and the defaults the
refusal is anchored on, which are read from 'mo_turbdiff_config.f90'.
"""

import dataclasses

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_options as options,
)


#: The switches that are frozen at their compiled-in default, with that default and the line of
#: 'icon/src/configure_model/mo_turbdiff_config.f90' it was read from. Spelled out here rather
#: than imported so that a change to 'turbulence.FROZEN_SWITCHES' has to be made twice.
EXPECTED_FROZEN_DEFAULTS = {
    "imode_turb": 1,  # :299
    "imode_tran": 0,  # :298
    "imode_stbcalc": 1,  # :318
    "imode_tkediff": 2,  # :383
    "imode_trancnf": 2,  # :361
    "imode_adshear": 2,  # :386
    "imode_tkemini": 1,  # :372
    "imode_suradap": 0,  # :378
    "imode_vel_min": 2,  # :223
    "imode_tkvmini": 2,  # :129
    "itype_wcld": 2,  # :309
    "ilow_def_cond": 2,  # :322
    "imode_lamdiff": 1,  # :369
    "imode_nsf_wind": 1,  # :168
    "imode_stadlim": 2,  # :357
    "imode_shshear": 2,  # :339
    "lexpcor": False,  # :279
    "ltmpcor": False,  # :276
    "lcpfluc": False,  # :277
    "lcirflx": False,  # :281
    "ltkecon": False,  # :266
    "l3dturb": False,  # not a namelist switch: hardcoded at mo_nwp_turbdiff_interface.f90:584
}


def _off_default(value: int | bool) -> int | bool:
    """A value the frozen switch is not allowed to take."""
    return (not value) if isinstance(value, bool) else value + 1


# --- the reject list ------------------------------------------------------------------------


def test_rejects_unported_switch() -> None:
    with pytest.raises(NotImplementedError, match="imode_turb"):
        turbulence.TurbulenceConfig(imode_turb=2)


def test_frozen_switch_table_covers_exactly_the_unported_switches() -> None:
    assert {switch.name for switch in turbulence.FROZEN_SWITCHES} == set(EXPECTED_FROZEN_DEFAULTS)


@pytest.mark.parametrize("name, expected", sorted(EXPECTED_FROZEN_DEFAULTS.items()))
def test_frozen_switch_default_matches_fortran(name: str, expected: int | bool) -> None:
    """The default of every frozen switch is the compiled-in Fortran default."""
    assert getattr(turbulence.TurbulenceConfig(), name) == expected
    (switch,) = [s for s in turbulence.FROZEN_SWITCHES if s.name == name]
    assert switch.supported_value == expected


@pytest.mark.parametrize("name, default", sorted(EXPECTED_FROZEN_DEFAULTS.items()))
def test_frozen_switch_accepts_its_default(name: str, default: int | bool) -> None:
    config = turbulence.TurbulenceConfig(**{name: default})
    assert getattr(config, name) == default


@pytest.mark.parametrize("name, default", sorted(EXPECTED_FROZEN_DEFAULTS.items()))
def test_frozen_switch_rejects_any_other_value(name: str, default: int | bool) -> None:
    offending = _off_default(default)
    with pytest.raises(NotImplementedError) as excinfo:
        turbulence.TurbulenceConfig(**{name: offending})

    message = str(excinfo.value)
    assert name in message, "the message must name the parameter"
    assert f"{default}" in message, "the message must state the one supported value"
    assert str(offending) in message, "the message must echo what was given"
    assert "Fortran" in message, "the message must offer the Fortran fallback"


@pytest.mark.parametrize("name, default", sorted(EXPECTED_FROZEN_DEFAULTS.items()))
def test_frozen_switch_message_explains_the_physics(name: str, default: int | bool) -> None:
    """Each rejection says what the supported value means, not just what it is."""
    (switch,) = [s for s in turbulence.FROZEN_SWITCHES if s.name == name]
    assert len(switch.meaning) > 15
    with pytest.raises(NotImplementedError, match=switch.meaning[:15]):
        turbulence.TurbulenceConfig(**{name: _off_default(default)})


def test_every_frozen_switch_is_a_config_field() -> None:
    fields = {field.name for field in dataclasses.fields(turbulence.TurbulenceConfig)}
    assert {switch.name for switch in turbulence.FROZEN_SWITCHES} <= fields


# --- the switches that vary operationally ---------------------------------------------------


def test_accepts_operational_shear_types() -> None:
    for v in (1, 2, 3):
        assert turbulence.TurbulenceConfig(itype_sher=v).itype_sher == v


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_accepts_every_shear_type(value: int) -> None:
    """'itype_sher = 0' is the Fortran default and is forced by 'mo_nml_crosscheck.f90:329'."""
    assert turbulence.TurbulenceConfig(itype_sher=value).itype_sher == value


@pytest.mark.parametrize("value", [1, 2])
def test_accepts_operational_cloud_representations(value: int) -> None:
    assert turbulence.TurbulenceConfig(icldm_turb=value).icldm_turb == value


@pytest.mark.parametrize("value", [-1, 0])
def test_rejects_unported_cloud_representations(value: int) -> None:
    with pytest.raises(NotImplementedError, match="icldm_turb"):
        turbulence.TurbulenceConfig(icldm_turb=value)


@pytest.mark.parametrize("value", [1, 2])
def test_accepts_operational_sso_tke_modes(value: int) -> None:
    assert turbulence.TurbulenceConfig(imode_tkesso=value).imode_tkesso == value


@pytest.mark.parametrize("value", [0, 3])
def test_rejects_unported_sso_tke_modes(value: int) -> None:
    with pytest.raises(NotImplementedError, match="imode_tkesso"):
        turbulence.TurbulenceConfig(imode_tkesso=value)


@pytest.mark.parametrize("value", [2, 3])
def test_accepts_operational_charnock_modes(value: int) -> None:
    assert turbulence.TurbulenceConfig(imode_charpar=value).imode_charpar == value


def test_rejects_constant_charnock_parameter() -> None:
    with pytest.raises(NotImplementedError, match="imode_charpar"):
        turbulence.TurbulenceConfig(imode_charpar=1)


@pytest.mark.parametrize("value", [0.0, 0.2])
def test_accepts_operational_tke_forcing_smoothing(value: float) -> None:
    assert turbulence.TurbulenceConfig(frcsmot=value).frcsmot == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_rejects_tke_forcing_smoothing_outside_the_unit_interval(value: float) -> None:
    with pytest.raises(ValueError, match="frcsmot"):
        turbulence.TurbulenceConfig(frcsmot=value)


@pytest.mark.parametrize("value", [1.25, 2.0])
def test_accepts_operational_horizontal_shear_length_scales(value: float) -> None:
    assert turbulence.TurbulenceConfig(a_hshr=value).a_hshr == value


def test_rejects_negative_horizontal_shear_length_scale() -> None:
    with pytest.raises(ValueError, match="a_hshr"):
        turbulence.TurbulenceConfig(a_hshr=-1.0)


def test_horizontal_shear_switch_and_length_scale_must_agree() -> None:
    """Reproduces the ICON crosscheck at 'mo_nml_crosscheck.f90:432'."""
    with pytest.raises(ValueError, match="ltkeshs"):
        turbulence.TurbulenceConfig(ltkeshs=True, a_hshr=0.0)
    with pytest.raises(ValueError, match="ltkeshs"):
        turbulence.TurbulenceConfig(ltkeshs=False, a_hshr=2.0)


def test_options_are_coerced_to_their_enum() -> None:
    config = turbulence.TurbulenceConfig(itype_sher=2, icldm_turb=1)
    assert config.itype_sher is options.ShearProductionType.VERTICAL_AND_VERTICAL_VELOCITY
    assert config.icldm_turb is options.CloudRepresentationType.GRID_SCALE


# --- the operational configurations of icon/run -----------------------------------------------


#: The six switches that vary across the operational setups, as tabulated in port spec section 4.3
#: and re-read from 'icon/run/'. 'def' entries are the compiled-in default.
OPERATIONAL_SETUPS = {
    "mch_icon-ch1": dict(
        itype_sher=2, icldm_turb=2, imode_tkesso=2, imode_charpar=3, frcsmot=0.0, a_hshr=2.0
    ),
    "icon-d2_oper": dict(
        itype_sher=2, icldm_turb=2, imode_tkesso=2, imode_charpar=2, frcsmot=0.2, a_hshr=2.0
    ),
    "glob_oper": dict(
        itype_sher=3, icldm_turb=1, imode_tkesso=1, imode_charpar=2, frcsmot=0.2, a_hshr=2.0
    ),
    "glob_eps": dict(
        itype_sher=1, icldm_turb=1, imode_tkesso=1, imode_charpar=2, frcsmot=0.2, a_hshr=2.0
    ),
    "ruc_fc": dict(
        itype_sher=2, icldm_turb=2, imode_tkesso=2, imode_charpar=3, frcsmot=0.2, a_hshr=1.25
    ),
}


@pytest.mark.parametrize("setup", sorted(OPERATIONAL_SETUPS))
def test_operational_configurations_are_accepted(setup: str) -> None:
    settings = OPERATIONAL_SETUPS[setup]
    config = turbulence.TurbulenceConfig(**settings)
    for name, value in settings.items():
        assert getattr(config, name) == value


# --- from_fortran_dict ------------------------------------------------------------------------


def test_config_from_fortran_dict() -> None:
    """The echoed 'turbdiff_nml' of 'exp.mch_icon-ch2_small'."""
    nml = {
        "turbdiff_nml": {
            "tkhmin": 0.5,
            "tkmmin": 0.75,
            "pat_len": 750.0,
            "tur_len": 300.0,
            "rat_sea": 0.8,
            "ltkesso": True,
            "frcsmot": 0.0,
            "imode_frcsmot": 2,
            "itype_sher": 2,
            "ltkeshs": True,
            "a_hshr": 2.0,
            "icldm_turb": 2,
            "q_crit": 2.0,
            "imode_tkesso": 2,
            "rlam_heat": 10.0,
            "alpha1": 0.125,
            "imode_charpar": 3,
        }
    }

    config = turbulence.TurbulenceConfig.from_fortran_dict(nml)

    assert config.itype_sher == 2
    assert config.icldm_turb == 2
    assert config.imode_tkesso == 2
    assert config.imode_charpar == 3
    assert config.frcsmot == 0.0
    assert config.a_hshr == 2.0
    assert config.tur_len == 300.0
    assert config.pat_len == 750.0
    assert config.alpha1 == 0.125
    # not echoed: keeps the Fortran default
    assert config.tkesmot == turbulence.TurbulenceConfig().tkesmot


def test_config_from_fortran_dict_unwraps_single_element_arrays() -> None:
    config = turbulence.TurbulenceConfig.from_fortran_dict(
        {"turbdiff_nml": {"itype_sher": [2], "tur_len": [300.0]}}
    )
    assert config.itype_sher == 2
    assert config.tur_len == 300.0


def test_config_from_fortran_dict_rejects_domain_specific_arrays() -> None:
    with pytest.raises(ValueError, match="pat_len"):
        turbulence.TurbulenceConfig.from_fortran_dict({"turbdiff_nml": {"pat_len": [750.0, 500.0]}})


def test_config_from_fortran_dict_rejects_unknown_namelist_entries() -> None:
    with pytest.raises(ValueError, match="itype_shear"):
        turbulence.TurbulenceConfig.from_fortran_dict({"turbdiff_nml": {"itype_shear": 2}})


def test_config_from_fortran_dict_requires_the_namelist_group() -> None:
    with pytest.raises(KeyError, match="turbdiff_nml"):
        turbulence.TurbulenceConfig.from_fortran_dict({"nwp_phy_nml": {}})


def test_config_from_fortran_dict_applies_overrides() -> None:
    config = turbulence.TurbulenceConfig.from_fortran_dict(
        {"turbdiff_nml": {"itype_sher": 2}}, itype_sher=1
    )
    assert config.itype_sher == 1


def test_config_from_fortran_dict_enforces_the_reject_list() -> None:
    with pytest.raises(NotImplementedError, match="itype_wcld"):
        turbulence.TurbulenceConfig.from_fortran_dict({"turbdiff_nml": {"itype_wcld": 1}})


# --- the dataclass contract -------------------------------------------------------------------


def test_config_is_frozen() -> None:
    config = turbulence.TurbulenceConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.itype_sher = 1  # type: ignore[misc]


def test_config_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        turbulence.TurbulenceConfig(1)  # type: ignore[misc]


def test_valid_config_round_trips() -> None:
    config = turbulence.TurbulenceConfig(
        itype_sher=2, icldm_turb=1, imode_tkesso=2, imode_charpar=3, frcsmot=0.2, a_hshr=1.25
    )
    assert turbulence.TurbulenceConfig(**dataclasses.asdict(config)) == config
    assert dataclasses.replace(config) == config


# --- derived parameters -----------------------------------------------------------------------


def test_params_reproduce_the_fortran_stability_constants() -> None:
    """Hand-evaluated from 'turb_setup' (turb_utilities.f90:400-435) at the ICON defaults."""
    config = turbulence.TurbulenceConfig()
    params = turbulence.TurbulenceParams(config)

    a_h, a_m, d_h, d_m = 0.74, 0.92, 10.1, 16.6
    c_tke = d_m ** (1.0 / 3.0)
    c_m = 1.0 - 1.0 / (a_m * c_tke) - 6.0 * a_m / d_m
    d_1, d_2, d_3, d_4 = 1.0 / a_h, 1.0 / a_m, 9.0 * a_h, 6.0 * a_m
    d_5, d_6 = 3.0 * (d_h + d_4), d_3 + 3.0 * d_4

    assert params.c_tke == c_tke
    assert params.c_m == c_m
    assert params.c_h == 0.0
    assert params.b_m == 1.0 - c_m
    assert params.b_h == 1.0
    assert (params.d_1, params.d_2, params.d_3, params.d_4) == (d_1, d_2, d_3, d_4)
    assert (params.d_5, params.d_6) == (d_5, d_6)
    assert params.rim == 1.0 / (1.0 + (d_m - d_4) / d_5)
    assert params.a_3 == d_3 / (d_2 * d_m)
    assert params.a_5 == d_5 / (d_1 * d_m)
    assert params.a_6 == d_6 / (d_2 * d_m)
    assert params.sh_0 == (1.0 - d_4 / d_m) / d_1
    assert params.sm_0 == (1.0 - c_m - d_4 / d_m) / d_2


def test_params_drop_the_heat_capacity_fluctuation_terms() -> None:
    """'lcpfluc' is frozen at .FALSE., so 'tur_rcpv' and 'tur_rcpl' are zero."""
    params = turbulence.TurbulenceParams(turbulence.TurbulenceConfig())
    assert params.tur_rcpv == 0.0
    assert params.tur_rcpl == 0.0


def test_params_adiabatic_temperature_gradient() -> None:
    params = turbulence.TurbulenceParams(turbulence.TurbulenceConfig())
    assert params.tet_g == pytest.approx(9.80665 / 1004.64)


def test_params_are_frozen() -> None:
    params = turbulence.TurbulenceParams(turbulence.TurbulenceConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.c_tke = 1.0  # type: ignore[misc]


# --- the option enums ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "enum_type, expected",
    [
        (options.ShearProductionType, {0, 1, 2, 3}),
        (options.CloudRepresentationType, {-1, 0, 1, 2}),
        (options.SsoTkeProductionType, {0, 1, 2, 3}),
        (options.CharnockParameterType, {1, 2, 3}),
    ],
)
def test_enums_carry_the_full_fortran_range(enum_type: type, expected: set[int]) -> None:
    """The interface carries every value the Fortran accepts; '_validate' refuses the rest."""
    assert {int(member) for member in enum_type} == expected
