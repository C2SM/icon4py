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

WHAT IS REFUSED HERE IS NOT ALL OF WHAT IS REFUSED. The configuration is the interface and stays
as wide as ICON's namelist; the granule is narrower and refuses eight further settings at
construction. Those refusals are exercised in 'test_granule.py', which is where a reader looking
for 'icldm_turb = 1', 'rsur_sher', 'a_stab' or 'it_end' should go.

THE ACCOUNTING TEST IS THE POINT OF THIS MODULE. Every one of the 93 fields of
'TurbulenceConfig' has to be classified into one of the five buckets below, and that requirement
is what the 2026-08-31 audit found missing: its predecessor selected the fields
to account for by 'bool' default and by the 'imode_'/'itype_'/'icldm_'/'ilow_' prefixes, so it
inspected no 'float' field at all -- and 'rsur_sher' and 'a_stab' are floats the Fortran branches
on with '> 0', i.e. switches that do not look like one. Both were accepted, unread and unremarked
for the whole port. The five buckets below are exhaustive and, apart from the deliberate overlap
of 'THRESHOLD_SWITCHES', disjoint.
"""

import ast
import dataclasses
import inspect
import pathlib

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
    "imode_frcsmot": 2,  # :140
    "icldm_tran": 2,  # :303
    "itype_2m_diag": 1,  # :353
    "lexpcor": False,  # :279
    "ltmpcor": False,  # :276
    "lcpfluc": False,  # :277
    "lcirflx": False,  # :281
    "ltkecon": False,  # :266
    "ltkenst": True,  # :268
    "loutshs": True,  # :271
    "lsflcnd": True,  # :280
    "lfreeslip": False,  # :284
    "ldiff_qi": False,  # :282
    "ldiff_qs": False,  # :283
    "l3dturb": False,  # not a namelist switch: hardcoded at mo_nwp_turbdiff_interface.f90:584
}


#: The switches no statement of the ported scheme reads, so the granule takes them at any value.
#: Each is consumed by an ICON file that is out of scope (port spec D1) or gates an output
#: argument the interfaces never pass; the doc comment of the field names the line. Listed here
#: so that freezing one later has to be a deliberate edit in two places rather than a silent one.
#:
#: 'ldiff_qi' and 'ldiff_qs' USED TO BE HERE AND ARE NOT ANY MORE. The scheme does not read them
#: -- that much was and stays true -- but the interface that does read them
#: ('mo_nwp_turbdiff_interface.f90:356', ':389') builds the tracer list out of them, and the
#: tracer list is exactly what the granule refuses. "No statement of the scheme reads it" is
#: therefore not sufficient grounds for accepting a switch: what matters is whether the value can
#: be HONOURED, and 'ldiff_qi = True' cannot be. Both are in 'FROZEN_SWITCHES' now.
NO_BEARING_ON_THE_GRANULE = (
    "imode_pat_len",
    "imode_snowsmot",
    "lconst_z0",
    "loutsso",
    "loutnst",
    "loutbms",
)


#: The parameters that are not spelled as switches and behave as switches anyway: the Fortran
#: branches on a threshold, almost always '> 0', so the value selects a formulation rather than
#: scaling one. This is the class the accounting test used to be blind to. Each entry says what
#: the granule does with it and where that is pinned; 'refused' entries are refused by
#: 'Turbulence', not by 'TurbulenceConfig', and are exercised in 'test_granule.py'.
#:
#: Two of them ALSO appear above -- 'frcsmot' and 'a_hshr' vary operationally as well -- which is
#: why this mapping is kept beside the accounting rather than as a sixth disjoint bucket. The
#: other six are classified here and nowhere else.
THRESHOLD_SWITCHES = {
    #: 'lcircterm' (turb_diffusion.f90:945): at zero the circulation term is off. Honoured --
    #: 'Turbulence._determine_derived_switches' selects the pair of programs from it.
    "pat_len": "honoured",
    #: Section 2c) smooths the TKE forcing when it is positive. Honoured, and range-checked to
    #: [0, 1] in '_validate'.
    "frcsmot": "honoured",
    #: Crosschecked against 'ltkeshs', as ICON does at 'mo_nml_crosscheck.f90:432'.
    "a_hshr": "honoured",
    #: At zero 'turbdiff' skips sections 6) and 8) to 10) (turb_diffusion.f90:2541). Refused.
    "c_diff": "refused",
    #: 'lsrfshear' (turb_diffusion.f90:968). Refused; see the field's doc comment.
    "rsur_sher": "refused",
    #: The stability correction of the master length scale (turb_utilities.f90:1339). Refused.
    "a_stab": "refused",
    #: The iteration count, a threshold in the sense that only '1' means "no loop". Refused.
    "it_end": "refused",
    #: The EDR limit (turb_utilities.f90:1846), reachable only under 'lpres_edr'. Argued in the
    #: field's doc comment to be unreachable once 'rsur_sher = 0' is enforced, and NOT refused --
    #: which is why the two have to be revisited together.
    "vel_max": "argued",
}


#: Everything else: values the scheme reads arithmetically, with no Fortran branch on them. A
#: wrong number here is a tuning error, not a missing formulation, so none of them is refused and
#: none needs an argument. Spelled out so that a NEW parameter cannot join them in silence --
#: adding a field to 'TurbulenceConfig' without classifying it fails
#: 'test_every_configuration_parameter_is_accounted_for'.
CONTINUOUS_PARAMETERS = (
    "impl_s",
    "impl_t",
    "tkhmin",
    "tkmmin",
    "tkhmin_strat",
    "tkmmin_strat",
    "ditsmot",
    "tkesmot",
    "frcsecu",
    "tkesecu",
    "stbsecu",
    "prfsecu",
    "epsi",
    "rlam_heat",
    "rlam_mom",
    "rat_lam",
    "rat_sea",
    "rat_glac",
    "rat_can",
    "alpha0",
    "alpha0_max",
    "alpha0_pert",
    "alpha1",
    "c_lnd",
    "c_sea",
    "c_soil",
    "c_stm",
    "e_surf",
    "const_z0",
    "z0m_dia",
    "z0_ice",
    "tur_len",
    "len_min",
    "vel_min",
    "akt",
    "a_heat",
    "a_mom",
    "d_heat",
    "d_mom",
    "clc_diag",
    "q_crit",
    "c_scld",
)


#: The switches supported over a range rather than frozen: the six tabulated in port spec
#: section 4.3, plus 'ltkesso', which 'mo_turbdiff_nml.f90:158' ties to 'imode_tkesso', and
#: 'ltkeshs', which 'mo_nml_crosscheck.f90:432' ties to 'a_hshr'. Three configurations under
#: 'icon/run/' set 'ltkeshs = .false.' with 'a_hshr = 0.', and two set 'ltkesso = .false.'.
VARY_OPERATIONALLY = (
    "itype_sher",
    "icldm_turb",
    "imode_tkesso",
    "imode_charpar",
    "frcsmot",
    "a_hshr",
    "ltkesso",
    "ltkeshs",
)


def _off_default(value: int | bool) -> int | bool:
    """A value the frozen switch is not allowed to take."""
    return (not value) if isinstance(value, bool) else value + 1


def _doc_comment(name: str) -> str:
    """The '#:' block immediately above the declaration of a `TurbulenceConfig` field."""
    lines = inspect.getsource(turbulence.TurbulenceConfig).splitlines()
    (index,) = [i for i, line in enumerate(lines) if line.startswith(f"    {name}:")]
    comment = []
    while index > 0 and lines[index - 1].lstrip().startswith("#:"):
        index -= 1
        comment.insert(0, lines[index].lstrip()[2:].strip())
    return " ".join(comment)


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


@pytest.mark.parametrize("name", NO_BEARING_ON_THE_GRANULE)
def test_switch_without_bearing_is_accepted_at_either_value(name: str) -> None:
    default = getattr(turbulence.TurbulenceConfig(), name)
    other = _off_default(default)
    assert getattr(turbulence.TurbulenceConfig(**{name: other}), name) == other


@pytest.mark.parametrize("name", NO_BEARING_ON_THE_GRANULE)
def test_switch_without_bearing_says_why_in_its_doc_comment(name: str) -> None:
    """D6 forbids silence, so a switch that is neither frozen nor range-checked must be argued.

    The argument has two halves and both have to be there: which Fortran line consumes the switch,
    and the conclusion that the granule cannot see it.
    """
    comment = _doc_comment(name)
    assert ".f90:" in comment, f"{name}: no Fortran citation for the consuming line"
    assert "change what the granule computes" in comment, f"{name}: no conclusion stated"


def test_every_formulation_switch_is_accounted_for() -> None:
    """Every switch of 'turbdiff_nml' is frozen, range-checked or argued to have no bearing.

    The point of the reject list is that a formulation switch may not pass unremarked (port spec
    D6); this is the test that notices when a new one does.

    KEPT, THOUGH 'test_every_configuration_parameter_is_accounted_for' SUBSUMES IT. What this one
    says and the wider one does not is that a field which LOOKS like a switch may not be
    classified as a continuous parameter: the wider test would be satisfied by putting
    'imode_whatever' in 'CONTINUOUS_PARAMETERS', and this one would not.
    """
    frozen = {switch.name for switch in turbulence.FROZEN_SWITCHES}
    accounted = frozen | set(VARY_OPERATIONALLY) | set(NO_BEARING_ON_THE_GRANULE)
    switches = {
        field.name
        for field in dataclasses.fields(turbulence.TurbulenceConfig)
        if isinstance(field.default, bool)
        or field.name.startswith(("imode_", "itype_", "icldm_", "ilow_"))
    }
    assert switches - accounted == set()


def _accounting_buckets() -> dict[str, set[str]]:
    """The five classifications, as sets, in the order a field should be looked for."""
    return {
        "FROZEN_SWITCHES": {switch.name for switch in turbulence.FROZEN_SWITCHES},
        "VARY_OPERATIONALLY": set(VARY_OPERATIONALLY),
        "NO_BEARING_ON_THE_GRANULE": set(NO_BEARING_ON_THE_GRANULE),
        "THRESHOLD_SWITCHES": set(THRESHOLD_SWITCHES),
        "CONTINUOUS_PARAMETERS": set(CONTINUOUS_PARAMETERS),
    }


def test_every_configuration_parameter_is_accounted_for() -> None:
    """EVERY field of 'TurbulenceConfig' is classified, not only the ones that look like switches.

    This is the widened form of the test above, and the hole it closes is a measured one: keying
    on 'bool' defaults and on the integer prefixes inspects no 'float' field, and 'rsur_sher',
    'a_stab' and 'vel_max' are floats the Fortran branches on. All three passed the narrow test
    and none of them was read by the granule.

    Both directions are asserted. An unclassified field is the hole; a classified name that is
    not a field is a rename the classification did not follow, which would silently stop
    covering the parameter it was written for.
    """
    fields = {field.name for field in dataclasses.fields(turbulence.TurbulenceConfig)}
    accounted = set().union(*_accounting_buckets().values())

    assert fields - accounted == set(), "unclassified parameter: add it to one of the buckets"
    assert accounted - fields == set(), "classified name that is not a parameter"


def test_the_continuous_parameters_claim_nothing_that_is_a_switch() -> None:
    """'CONTINUOUS_PARAMETERS' is the residue, so nothing may be in it and in another bucket.

    Without this the accounting could be satisfied by listing a frozen switch twice, and the
    second listing would read as an assertion that it has no Fortran branch behind it.
    """
    buckets = _accounting_buckets()
    residue = buckets.pop("CONTINUOUS_PARAMETERS")
    for name, bucket in buckets.items():
        assert residue & bucket == set(), f"CONTINUOUS_PARAMETERS overlaps {name}"


@pytest.mark.parametrize("name", sorted(THRESHOLD_SWITCHES))
def test_threshold_switch_is_a_parameter_with_a_disposition(name: str) -> None:
    """A float the Fortran branches on is a switch, and D6 forbids it passing unremarked.

    Each has to be refused, honoured or argued -- and 'argued' has to be argued in the field's
    own doc comment, where the next reader of the declaration will find it, rather than only
    here.
    """
    fields = {field.name for field in dataclasses.fields(turbulence.TurbulenceConfig)}
    assert name in fields
    disposition = THRESHOLD_SWITCHES[name]
    assert disposition in ("refused", "honoured", "argued")
    if disposition == "argued":
        comment = _doc_comment(name)
        assert ".f90:" in comment, f"{name}: no Fortran citation for the branch"
        assert "unreachable" in comment, f"{name}: no conclusion stated"


def test_the_two_threshold_switches_the_audit_found_are_refused() -> None:
    """The specific finding, named, so that a bucket rename cannot quietly drop it.

    'rsur_sher' and 'a_stab' are the two parameters the 2026-08-31 audit found accepted and
    silently ignored, and 'a_stab' is the one an operational EPS member really sets, through
    'ensemble_pert_nml' rather than 'turbdiff_nml'.
    """
    assert THRESHOLD_SWITCHES["rsur_sher"] == "refused"
    assert THRESHOLD_SWITCHES["a_stab"] == "refused"


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


def test_switching_the_sso_source_term_off_forces_the_sso_mode_off() -> None:
    """Reproduces the assignment at 'mo_turbdiff_nml.f90:158', which ICON makes silently."""
    config = turbulence.TurbulenceConfig(ltkesso=False)
    assert config.imode_tkesso is options.SsoTkeProductionType.OFF


@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_the_sso_mode_is_irrelevant_once_the_sso_source_term_is_off(value: int) -> None:
    """ICON overwrites whatever was set, so no pair with 'ltkesso = .FALSE.' may be refused."""
    config = turbulence.TurbulenceConfig(ltkesso=False, imode_tkesso=value)
    assert config.imode_tkesso is options.SsoTkeProductionType.OFF


def test_the_configuration_icon_cannot_produce_is_the_one_refused() -> None:
    """'ltkesso = .TRUE.' with no SSO mode is a silent no-op in the Fortran; refuse it instead."""
    with pytest.raises(NotImplementedError, match="imode_tkesso"):
        turbulence.TurbulenceConfig(ltkesso=True, imode_tkesso=0)


def test_the_sso_switch_survives_a_fortran_namelist_that_sets_both() -> None:
    """The echo of 'mo_turbdiff_nml.f90' is written before line 158, so it shows the raw pair."""
    config = turbulence.TurbulenceConfig.from_fortran_dict(
        {"turbdiff_nml": {"ltkesso": False, "imode_tkesso": 2}}
    )
    assert config.ltkesso is False
    assert config.imode_tkesso is options.SsoTkeProductionType.OFF


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


def test_the_configuration_still_accepts_the_cloud_representation_the_granule_refuses() -> None:
    """'icldm_turb = 1' must reach 'TurbulenceConfig' and be stopped by 'Turbulence'.

    The interface is the contract (port spec D5/D6) and 'from_fortran_dict' has to keep reading
    the echoed 'turbdiff_nml' of the DWD global setup, which sets 1. Refusing it here instead
    would refuse the namelist rather than the run, and the two are not the same message. The
    refusal that matters is in 'test_granule.py'.
    """
    config = turbulence.TurbulenceConfig.from_fortran_dict({"turbdiff_nml": {"icldm_turb": 1}})
    assert config.icldm_turb is options.CloudRepresentationType.GRID_SCALE


def _package_modules_that_compute() -> list[pathlib.Path]:
    """Every module of the package that a stencil is built from.

    'turbulence.py' is excluded on purpose: it is where 'icldm_turb' is declared, validated and
    refused, so it is the one module that must mention it.
    """
    root = pathlib.Path(inspect.getfile(turbulence)).parent
    return [
        path
        for path in sorted(root.rglob("*.py"))
        if path.name not in ("turbulence.py", "__init__.py")
    ]


def test_no_stencil_reads_the_cloud_representation_mode() -> None:
    """Why 'icldm_turb = 1' has to be refused rather than dispatched on.

    The mode-2 arm of 'adjust_satur_equil' is carried unguarded: no stencil takes 'icldm_turb'
    as an argument and none branches on it, so at mode 1 the granule would run the sub-grid
    statistical saturation adjustment that the Fortran does not reach at all
    (turb_utilities.f90:869-882 terminates the ELSEIF chain before 'turb_cloud').

    Identifiers only, by way of the AST, so that the doc comments which DISCUSS 'icldm_turb' --
    and several of them do, at length -- do not register as a use.
    """
    for path in _package_modules_that_compute():
        # The encoding is spelled out because 'read_text()' defaults to the LOCALE encoding,
        # which is ASCII in a SLURM job that sets no LANG -- and several modules of this package
        # carry an em dash. Measured: this passed on two runs and failed on the third with a
        # UnicodeDecodeError, same tree, same node, same flags (job 843849).
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
        assert "icldm_turb" not in identifiers, f"{path.name} reads icldm_turb"


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
