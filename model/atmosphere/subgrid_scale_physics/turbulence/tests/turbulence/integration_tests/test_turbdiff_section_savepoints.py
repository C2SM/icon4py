# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for the fifteen 'turbdiff' section savepoint readers.

These pin the contract between 'serialize_turbdiff_section' in ICON's
mo_icon4py_verification.f90 (commit f6803c9fae) and 'IconTurbdiffSectionSavepoint'.

The point of a section savepoint is that a wrong value inside a fused operator can be localised
to one of Raschendorfer's numbered sections instead of to the whole 2600-line routine. That only
works if the reader knows what each storage slot means AT THAT SECTION, because 'turbdiff'
reuses its working arrays for unrelated quantities as it proceeds: 'len_scale' is a mixing
length until section 8) and the effective TKE flux from 9) on, 'rcld' alternates between cloud
cover and the standard deviation of the local super-saturation, 'zaux' carries thermodynamic
factors and then the tridiagonal coefficients. The tests below assert both halves of that: that
the accessor for a role returns the field where the role holds, and that it REFUSES where it
does not.

They also freeze the empirical facts that the accessor docstrings rest on, so that a re-capture
under a different namelist fails loudly instead of quietly invalidating them:
'test_turbdiff_section_fields_written_only_where_expected' is the whole 27-by-14 change matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions, serialbox as sb

from ..fixtures import *  # noqa: F403


#: The four timesteps 'exp.mch_icon-ch2_small' serializes.
TURBDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: The superset every section savepoint writes: 31 names for sections 0)..3).
SECTION_SERIALIZED_FIELDS = frozenset(
    {
        "td_ivstart", "td_ivend", "td_nvor",
        # the working set that carries state between the numbered sections
        "td_dicke", "td_frh", "td_frm", "td_ftm", "td_hlp", "td_shv",
        "td_zaux", "td_zvari", "td_len_scale", "td_edr",
        "td_lays", "td_layr", "td_hor_scale", "td_xri",
        # prognostic and diagnostic turbulence state
        "td_tke", "td_tkvm", "td_tkvh", "td_tprn", "td_rcld", "td_rhon", "td_tketens",
        # transfer-layer scalars
        "td_tfm", "td_tfh", "td_tfv",
        # arguments that are OPTIONAL in 'turbdiff' and present at this call site
        "td_u_tens", "td_v_tens", "td_t_tens", "td_tket_hshr",
    }
)  # fmt: skip

#: 'ldoexpcor' and 'ldocirflx' are assigned at the end of section 4), so the hooks for 0)..3)
#: leave them out rather than pass an undefined INTENT(OUT) dummy on.
SECTIONS_WITH_CORRECTION_FLAGS = ("4", "5", "6", "7", "8", "9", "10")

#: Lower limit of the turbulent velocity 'q = SQRT(2*TKE)' [m/s]: 'vel_min'.
TKE_FLOOR = 0.01

#: Which section last wrote each storage slot, as measured over the archive by comparing every
#: consecutive pair of section savepoints inside 'ivstart:ivend'. This is the table the accessor
#: docstrings are built on; see the module docstring.
FIELDS_WRITTEN_AT = {
    "td_dicke": ("1a",),
    "td_edr": (),  # 'ldiagnose_tke' is off, so 'ediss' targets an uninitialised local
    "td_frh": ("1b", "6", "9"),
    "td_frm": ("1b", "2a", "6"),
    "td_ftm": (),  # neither 'lssintact' nor 'loutbms' nor "rsur_sher > 0"
    "td_hlp": ("1a", "2a", "8"),
    "td_hor_scale": ("2a",),
    "td_layr": ("2a",),
    "td_lays": ("1a",),
    "td_len_scale": ("9",),  # section 0) writes it too, before the first savepoint
    "td_rcld": ("3",),
    "td_rhon": (),  # written by section 0), before the first savepoint
    "td_shv": (),  # 'lcirflx' is off and 'tket_nstc' is not passed
    "td_tfh": (),  # 'lsrfshear' is off
    "td_tfm": (),  # 'lsrfshear' is off
    "td_tfv": (),  # "rsur_sher > 0" is false
    "td_tke": ("3",),
    "td_tket_hshr": ("2a",),
    "td_tketens": ("10",),
    "td_tkvh": ("2c", "3", "4"),
    "td_tkvm": ("2c", "3", "4"),
    "td_tprn": (),  # a (1,1) dummy in this configuration
    "td_u_tens": (),  # only section 2b), which is dead here
    "td_v_tens": (),
    "td_t_tens": (),  # sections 2b) and 5), both inactive
    "td_xri": ("2a",),
    "td_zaux": ("6", "9"),
    "td_zvari": ("1a", "3"),
}

experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


def _masked(serializer, savepoint, name: str) -> np.ndarray:
    """The serialized buffer of 'name', restricted to the computed columns."""
    buffer = np.asarray(serializer.read(name, savepoint.savepoint))
    if buffer.ndim >= 2 and buffer.shape[0] > savepoint.ivend():
        return buffer[savepoint.ivstart() : savepoint.ivend()]
    return buffer


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("section", sb.TURBDIFF_SECTIONS)
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_savepoint_serializes_the_expected_fields(
    section: str,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_section(section=section, date=date)

    serialized = {
        name
        for name in data_provider.serializer.fields_at_savepoint(savepoint.savepoint)
        if name.startswith("td_")
    }
    expected = set(SECTION_SERIALIZED_FIELDS)
    if section in SECTIONS_WITH_CORRECTION_FLAGS:
        expected |= {"td_ldoexpcor", "td_ldocirflx"}
    assert serialized == expected

    assert savepoint.section == section
    assert savepoint.savepoint.metainfo.to_dict()["date"] == date
    assert savepoint.savepoint.metainfo.to_dict()["id"] == 1
    assert savepoint.block() == 1


@pytest.mark.datatest
@experiment_for_turbulence
def test_turbdiff_section_factory_rejects_an_unknown_label(
    *, data_provider: sb.IconSerialDataProvider
) -> None:
    with pytest.raises(ValueError, match="unknown turbdiff section"):
        data_provider.from_savepoint_turbdiff_section(section="11", date=TURBDIFF_DATES[0])
    with pytest.raises(ValueError, match="unknown turbdiff section"):
        data_provider.from_savepoint_turbdiff_section(section="2", date=TURBDIFF_DATES[0])


@pytest.mark.datatest
@experiment_for_turbulence
def test_turbdiff_9_exit_is_present_but_the_reader_does_not_assume_it(
    *, data_provider: sb.IconSerialDataProvider
) -> None:
    """
    Section 9)'s hook is inside 'IF (ldotkedif .OR. lcircterm)' and cannot be lifted out.

    It fires in this capture because 'c_diff = 0.2'; a run with 'c_diff = 0' and no circulation
    term would leave the savepoint out of the archive entirely. What the factory must not do is
    pass a raw serialbox error on, so the membership check is exercised here through a label
    that is well-formed but absent.
    """
    savepoint = data_provider.from_savepoint_turbdiff_section(section="9", date=TURBDIFF_DATES[0])
    assert savepoint.section == "9"

    #: the same code path the missing 'turbdiff-9-exit' would take
    provider_savepoints = {sp.name for sp in data_provider.serializer.savepoint_list()}
    assert "turbdiff-9-exit" in provider_savepoints
    assert "turbdiff-11-exit" not in provider_savepoints
    with pytest.raises(ValueError, match="unknown turbdiff section"):
        data_provider.from_savepoint_turbdiff_section(section="11", date=TURBDIFF_DATES[0])


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("section", sb.TURBDIFF_SECTIONS)
def test_turbdiff_section_bounds_match_the_entry_savepoint(
    section: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    entry = data_provider.from_savepoint_turbdiff_entry(date=TURBDIFF_DATES[0])
    savepoint = data_provider.from_savepoint_turbdiff_section(
        section=section, date=TURBDIFF_DATES[0]
    )

    assert savepoint.ivstart() == entry.ivstart()
    assert savepoint.ivend() == entry.ivend()
    assert 0 <= savepoint.ivstart() < savepoint.ivend() <= entry.nvec()
    assert savepoint.nvor() == entry.nvor()


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("section", sb.TURBDIFF_SECTIONS)
def test_turbdiff_section_field_shapes(
    section: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    entry = data_provider.from_savepoint_turbdiff_entry(date=TURBDIFF_DATES[0])
    savepoint = data_provider.from_savepoint_turbdiff_section(
        section=section, date=TURBDIFF_DATES[0]
    )
    num_cells = int(data_provider.grid_size[dims.CellDim])

    half = (num_cells, entry.ke1())
    full = (num_cells, entry.ke())
    for accessor, expected in (
        ("tke", half),
        ("rhon", half),
        ("tketens", half),
        ("edr", half),
        ("ftm", half),
        ("hlp", half),
        ("u_tens", full),
        ("v_tens", full),
        ("t_tens", full),
        ("tket_hshr", half),
    ):
        field = getattr(savepoint, accessor)()
        assert field.domain.dims == (dims.CellDim, dims.KDim), accessor
        assert field.shape == expected, accessor
        assert field.asnumpy().dtype == np.float64, accessor

    # 'hor_scale' and 'xri' live on MAIN levels, unlike almost everything else here.
    if section in sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("2a") :]:
        assert savepoint.hor_scale().shape == full
        assert savepoint.xri().shape == full
        assert savepoint.layr().shape == (num_cells,)
    if section != "0":
        assert savepoint.lays(0).shape == (num_cells,)
        assert savepoint.lays(1).shape == (num_cells,)
        with pytest.raises(IndexError):
            savepoint.lays(2)

    # 'tprn' is a (1,1) dummy unless the namelist selects a TMod that fills it.
    assert savepoint.tprn().shape == (1, 1)

    for component in range(5):
        assert savepoint.raw_zaux(component).shape == half
    with pytest.raises(IndexError):
        savepoint.raw_zaux(5)
    for component in range(6):
        assert savepoint.raw_zvari(component).shape == half
    with pytest.raises(IndexError):
        savepoint.raw_zvari(6)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_fields_written_only_where_expected(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """
    The measured change matrix, frozen.

    Every accessor guard and every 'not written in this configuration' claim in
    'IconTurbdiffSectionSavepoint' rests on this table. It is a property of the namelist, not of
    the scheme, so a re-capture that turns on 'ldiagnose_tke', 'ltmpcor', 'lsrfshear',
    'l3dturb' or a resolved canopy will fail here first -- which is the point.
    """
    savepoints = [
        data_provider.from_savepoint_turbdiff_section(section=section, date=date)
        for section in sb.TURBDIFF_SECTIONS
    ]

    measured = {}
    for name in FIELDS_WRITTEN_AT:
        changed = []
        previous = _masked(data_provider.serializer, savepoints[0], name)
        for savepoint in savepoints[1:]:
            current = _masked(data_provider.serializer, savepoint, name)
            if not np.array_equal(previous, current):
                changed.append(savepoint.section)
            previous = current
        measured[name] = tuple(changed)

    assert measured == {name: tuple(at) for name, at in FIELDS_WRITTEN_AT.items()}


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_2b_is_dead_in_this_configuration(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """
    Section 2b), the roughness-layer form drag, does nothing here.

    'c_big', 'c_sml' and 'r_air' are not passed by mo_nwp_turbdiff_interface.f90 and 'kcm =
    ke+1', so the guard at turb_diffusion.f90:1628 is false and 'turbdiff-2b-exit' is a copy of
    'turbdiff-2a-exit'. A canary: if this ever fails, either the configuration grew a canopy or
    the instrumentation moved, and the port task for 2b) stops being a no-op.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="2a", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)

    names = sorted(data_provider.serializer.fields_at_savepoint(before.savepoint))
    assert len(names) == 31
    differing = [
        name
        for name in names
        if not np.array_equal(
            np.asarray(data_provider.serializer.read(name, before.savepoint)),
            np.asarray(data_provider.serializer.read(name, after.savepoint)),
        )
    ]
    assert differing == []

    # 'raw_field()' is the escape hatch for a slot the reader refuses to name; it must hand back
    # exactly what the serializer holds, unsqueezed and untruncated.
    assert np.array_equal(
        np.asarray(before.raw_field("td_frm")),
        np.asarray(data_provider.serializer.read("td_frm", before.savepoint)),
    )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_len_scale_stops_being_a_length_at_section_9(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'len_scale' is a mixing length up to section 8) and the effective TKE flux from 9) on.

    The two accessors refuse outside their range, and the numbers show why a single name would
    be a trap: the mixing length is bounded by 'akt*l_scal' at about 118 m here, while the flux
    that replaces it reaches 2165.
    """
    section8 = data_provider.from_savepoint_turbdiff_section(section="8", date=date)
    section9 = data_provider.from_savepoint_turbdiff_section(section="9", date=date)

    mixing_length = section8.mixing_length().asnumpy()[section8.ivstart() : section8.ivend()]
    eff_flux = section9.eff_tke_flux().asnumpy()[section9.ivstart() : section9.ivend()]
    assert mixing_length.max() < 200.0
    assert eff_flux.max() > 1000.0
    assert not np.array_equal(mixing_length, eff_flux)

    with pytest.raises(ValueError, match="eff_tke_flux"):
        section8.eff_tke_flux()
    with pytest.raises(ValueError, match="mixing_length"):
        section9.mixing_length()

    # Nothing restores the length scale, so 'turbdiff-exit' carries the flux as well.
    exit_savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)
    assert np.array_equal(
        exit_savepoint.eff_tke_flux().asnumpy(), section9.eff_tke_flux().asnumpy()
    )
    assert not np.array_equal(
        exit_savepoint.len_scale().asnumpy(), section8.mixing_length().asnumpy()
    )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_role_accessors_refuse_outside_their_sections(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """Every aliased storage slot: the role accessor answers where the role holds, and only there."""
    savepoints = {
        section: data_provider.from_savepoint_turbdiff_section(section=section, date=date)
        for section in sb.TURBDIFF_SECTIONS
    }

    roles = {
        "mixing_length": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8"),
        "eff_tke_flux": ("9", "10"),
        "layer_depth": ("0",),
        "disc_mom": sb.TURBDIFF_SECTIONS[1:],
        "cloud_cover": ("0", "1a", "1b", "1c", "2a", "2b", "2c"),
        "sdss": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("3") :],
        "thermal_forcing": ("1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
        "cke_flux_density": ("6", "7", "8"),
        "invs_fac": ("9", "10"),
        "mech_forcing": ("1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
        "cke_flux_at_main_levels": ("6", "7", "8"),
        "tkvm": tuple(s for s in sb.TURBDIFF_SECTIONS if s != "2c"),
        "tkvh": tuple(s for s in sb.TURBDIFF_SECTIONS if s != "2c"),
        "stab_len_m": ("2c",),
        "stab_len_h": ("2c",),
        "tfm": ("0", "1a", "1b", "1c", "2a", "2b", "2c"),
        "tfh": ("0", "1a", "1b", "1c", "2a", "2b", "2c"),
        "tfv": ("0", "1a", "1b", "1c", "2a", "2b"),
        "exner_factor": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8"),
        "upd_prof": ("9", "10"),
        "r_cpd": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
        "sav_prof": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("6") :],
        "dqsat_dt": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5"),
        "expl_mom": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("6") :],
        "g_tet_l": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8"),
        "impl_mom": ("9", "10"),
        "g_h2o": ("0", "1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5", "6", "7", "8"),
        "invs_mom": ("9", "10"),
        "shv": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("6") :],
        "hor_scale": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("2a") :],
        "xri": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("2a") :],
        "layr": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("2a") :],
    }
    component_roles = {
        "conserved_variable": ("0",),
        "vertical_gradient": ("1a", "1b", "1c", "2a", "2b", "2c"),
        "effective_gradient": sb.TURBDIFF_SECTIONS[sb.TURBDIFF_SECTIONS.index("3") :],
    }

    for accessor, sections in roles.items():
        for section, savepoint in savepoints.items():
            if section in sections:
                assert getattr(savepoint, accessor)() is not None, (accessor, section)
            else:
                with pytest.raises(ValueError):
                    getattr(savepoint, accessor)()
    for accessor, sections in component_roles.items():
        for section, savepoint in savepoints.items():
            if section in sections:
                assert getattr(savepoint, accessor)(3) is not None, (accessor, section)
            else:
                with pytest.raises(ValueError):
                    getattr(savepoint, accessor)(3)

    # 'lays' is not aliased, only undefined before 1a).
    with pytest.raises(ValueError):
        savepoints["0"].lays(0)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_zvari_component_indices_are_the_fortran_ones(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'zvari(:,:,0:5)' is zero-based in 'turbdiff', and the serializer's rebasing to 1:6 by the
    assumed-shape interface changes nothing about the buffer.

    Section 0) copies 'u' into component 1 and 'v' into component 2 at the main levels, so the
    convention is checkable against the entry savepoint rather than asserted from the source.
    An off-by-one here would silently shift every component.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    section0 = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    window = slice(entry.ivstart(), entry.ivend())
    nlev = entry.ke()

    assert np.array_equal(
        section0.conserved_variable(1).asnumpy()[window, :nlev], entry.u().asnumpy()[window]
    )
    assert np.array_equal(
        section0.conserved_variable(2).asnumpy()[window, :nlev], entry.v().asnumpy()[window]
    )

    # Component 0 is the half-level pressure; its model-top level is never written.
    pressure = section0.conserved_variable(0).asnumpy()[window]
    assert pressure[:, 1].min() > 4000.0
    assert pressure[:, -1].max() < 1.1e5

    # And the effective gradients are what 'vertdiff' receives.
    exit_savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)
    vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)
    for component in range(6):
        assert np.array_equal(
            exit_savepoint.zvari(component).asnumpy(), vertdiff.zvari(component).asnumpy()
        )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_tkv_are_length_scales_only_at_section_2c(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """Section 2c) divides 'tkv[m|h]' by 'tke'; section 3) multiplies them back."""
    before = data_provider.from_savepoint_turbdiff_section(section="2b", date=date)
    at2c = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    window = slice(before.ivstart(), before.ivend())

    coefficient = before.tkvh().asnumpy()[window]
    length = at2c.stab_len_h().asnumpy()[window]
    tke = before.tke().asnumpy()[window]
    # 'tkvh(:,k) / tke(:,k,nvor)' for the atmospheric levels 2..ke (Python 1..ke-1).
    assert np.allclose(length[:, 1:-1], coefficient[:, 1:-1] / tke[:, 1:-1], rtol=1.0e-12)
    assert after.tkvh().asnumpy()[window].max() > length.max()


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_correction_flags(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ldoexpcor'/'ldocirflx' exist from section 4) on and are None before."""
    for section in sb.TURBDIFF_SECTIONS:
        savepoint = data_provider.from_savepoint_turbdiff_section(section=section, date=date)
        if section in SECTIONS_WITH_CORRECTION_FLAGS:
            assert savepoint.ldoexpcor() is False
            assert savepoint.ldocirflx() is False
        else:
            assert savepoint.ldoexpcor() is None
            assert savepoint.ldocirflx() is None


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_tke_is_physical_only_between_ivstart_and_ivend(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    The reason 'ivstart' and 'ivend' are serialized at all, checked at the section where 'tke'
    is written.

    Outside the window the slab is untouched memory that looks like data: 'q = SQRT(2*TKE)'
    reaches -0.026 below 'ivstart', a value it cannot take at all.
    """
    savepoint = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    tke = savepoint.tke().asnumpy()
    ivstart, ivend = savepoint.ivstart(), savepoint.ivend()

    computed = tke[ivstart:ivend]
    assert computed.size > 0
    assert np.isfinite(computed).all()
    assert computed.min() >= TKE_FLOOR - 1.0e-12
    assert tke[:ivstart].min() < 0.0
    assert not (tke >= TKE_FLOOR - 1.0e-12).all()


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_rcld_is_cloud_cover_then_sdss_then_main_levels(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'rcld' alternates between the cloud cover and the standard deviation of the local
    super-saturation, and changes staggering once more after the last section savepoint.

    Section 0) leaves the saturation fraction there, section 3) replaces it with SDSS on half
    levels, and section 11) -- which has no savepoint of its own -- interpolates SDSS back to
    main levels, which is what 'turbdiff-exit' carries. Three states, one name.
    """
    section2c = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    section3 = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    section10 = data_provider.from_savepoint_turbdiff_section(section="10", date=date)
    exit_savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)
    window = slice(section3.ivstart(), section3.ivend())

    cloud_cover = section2c.cloud_cover().asnumpy()[window]
    sdss = section3.sdss().asnumpy()[window]
    assert not np.array_equal(cloud_cover, sdss)
    assert cloud_cover.min() >= 0.0
    assert cloud_cover.max() <= 1.0

    # Section 11) is the only thing between the last section savepoint and the exit.
    assert np.array_equal(section10.sdss().asnumpy(), section3.sdss().asnumpy())
    assert not np.array_equal(exit_savepoint.rcld().asnumpy(), section10.sdss().asnumpy())


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_section_edr_is_never_written_in_this_capture(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'td_edr' is uninitialised memory here, not an eddy dissipation rate.

    mo_nwp_turbdiff_interface.f90:309-315 nullifies 'edr_ptr' unless 'ldiagnose_tke', so 'ediss'
    targets the routine-local 'diss_tar' and 'solve_turb_budgets' never fills it
    ('lpres_edr = lsrfshear .OR. ASSOCIATED(edr)', both false). The values are negative, which
    'q**3/(d_m*l)' cannot be. Asserted so that a Python EDR is never compared against them.
    """
    edr = [
        data_provider.from_savepoint_turbdiff_section(section=section, date=date).edr().asnumpy()
        for section in ("0", "3", "10")
    ]
    assert np.array_equal(edr[0], edr[1])
    assert np.array_equal(edr[0], edr[2])

    savepoint = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    inside = edr[0][savepoint.ivstart() : savepoint.ivend()]
    assert inside.min() < 0.0, (
        "td_edr is no longer negative inside the computed window -- if the capture was redone "
        "with ldiagnose_tke = .TRUE., edr() is a real EDR now and its docstring is stale"
    )
