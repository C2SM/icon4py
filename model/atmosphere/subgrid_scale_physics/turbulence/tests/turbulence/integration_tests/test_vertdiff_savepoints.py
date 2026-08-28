# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for the 'vertdiff' savepoint readers.

These pin the contract between 'serialize_vertdiff_entry' / 'serialize_vertdiff_exit' in ICON's
mo_icon4py_verification.f90 (commit f6803c9fae) and 'IconVertdiffEntrySavepoint' /
'IconVertdiffExitSavepoint'.

'vertdiff' shares argument names with 'turbdiff' without sharing their meaning -- its 'zaux'
holds the tridiagonal coefficients where 'turbdiff's holds thermodynamic factors, its
'len_scale' is the diffusion momentum, its 'frh' and 'frm' are the inversion and scaling factors
of the solve -- which is why the serialized names carry a 'vd_' prefix and the exit accessors are
named after the quantity. 'test_vertdiff_exit_workspace_is_not_turbdiffs' asserts that the two
'zaux' arrays really are unrelated, so that nobody compares them.

The other thing worth freezing is the handover: 'vertdiff' is called from SUB 'nwp_turbdiff'
immediately after 'turbdiff' for the same block, so its entry state must be byte-identical to
'turbdiff-exit'. If it ever is not, the two hooks are no longer looking at the same call.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions, serialbox as sb

from ..fixtures import *  # noqa: F403


#: The four timesteps 'exp.mch_icon-ch2_small' serializes.
VERTDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: Every name the entry savepoint writes.
ENTRY_SERIALIZED_FIELDS = frozenset(
    {
        # sizes, control flags and the valid horizontal window
        "vd_nvec", "vd_ke", "vd_ke1", "vd_kcm", "vd_kstart_cloud",
        "vd_ivstart", "vd_ivend", "vd_itndcon", "vd_ndtr", "vd_ntrac", "vd_ndiff", "vd_dt_var",
        "vd_lentire", "vd_lsfluse", "vd_lqvcrst", "vd_lrunscm",
        "vd_ldoexpcor", "vd_ldocirflx", "vd_l3dflxout", "vd_ldogrdcor",
        # grid and surface
        "vd_hhl", "vd_t_g", "vd_qv_s", "vd_ps",
        # atmospheric state
        "vd_u_in", "vd_v_in", "vd_t_in", "vd_qv_in", "vd_qc_in",
        "vd_prs", "vd_rhoh", "vd_rhon_in", "vd_epr",
        # turbulence state produced by 'turbdiff' / 'turbtran'
        "vd_tvm", "vd_tvh", "vd_tkvm_in", "vd_tkvh_in", "vd_zvari_in",
        # tendencies before the diffusion increment is added
        "vd_u_tens_in", "vd_v_tens_in", "vd_t_tens_in", "vd_qv_tens_in", "vd_qc_tens_in",
        # optional arguments present at this call site
        "vd_shfl_s_in", "vd_qvfl_s_in",
    }
)  # fmt: skip

#: Every name the exit savepoint writes.
EXIT_SERIALIZED_FIELDS = frozenset(
    {
        "vd_ivstart", "vd_ivend", "vd_ncorr", "vd_mcorr", "vd_igrdcon", "vd_ivtype",
        "vd_u", "vd_v", "vd_t", "vd_qv", "vd_qc",
        "vd_rhon", "vd_tkvm", "vd_tkvh", "vd_zvari",
        "vd_u_tens", "vd_v_tens", "vd_t_tens", "vd_qv_tens", "vd_qc_tens",
        # 'vertdiff's own !$ACC CREATE working set, under its storage names
        "vd_eprs", "vd_len_scale", "vd_zaux", "vd_frh", "vd_frm", "vd_dicke", "vd_hlp",
        # optional arguments present at this call site
        "vd_shfl_s", "vd_qvfl_s",
    }
)  # fmt: skip

#: Optional dummies that mo_nwp_turbdiff_interface.f90 does not pass. 'dp0' in particular is
#: declared and never supplied; the reader must hand back the (1,) dummy, not raise.
ABSENT_OPTIONALS = {
    "entry": ("dp0", "qv_conv"),
    "exit": ("qv_conv", "umfl_s", "vmfl_s"),
}

experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_savepoints_serialize_the_expected_fields(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    for savepoint, expected in (
        (data_provider.from_savepoint_vertdiff_entry(date=date), ENTRY_SERIALIZED_FIELDS),
        (data_provider.from_savepoint_vertdiff_exit(date=date), EXIT_SERIALIZED_FIELDS),
    ):
        serialized = {
            name
            for name in data_provider.serializer.fields_at_savepoint(savepoint.savepoint)
            if name.startswith("vd_")
        }
        assert serialized == set(expected)

        metadata = savepoint.savepoint.metainfo.to_dict()
        assert metadata["date"] == date
        assert metadata["id"] == 1  # the domain 'jg'
        assert metadata["block"] == 1
        assert savepoint.block() == 1

    with pytest.raises(Exception):  # noqa: B017 [assert-raises-exception]
        # serialbox raises when the metadata does not select exactly one savepoint
        data_provider.from_savepoint_vertdiff_entry(date=VERTDIFF_DATES[0], block=2)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_entry_scalars(date: str, *, data_provider: sb.IconSerialDataProvider) -> None:
    savepoint = data_provider.from_savepoint_vertdiff_entry(date=date)
    num_cells = int(data_provider.grid_size[dims.CellDim])
    nlev = int(data_provider.grid_size[dims.KDim])

    assert savepoint.ke() == nlev
    assert savepoint.ke1() == savepoint.ke() + 1
    assert savepoint.nvec() >= num_cells
    assert 0 <= savepoint.ivstart() < savepoint.ivend() <= savepoint.nvec()
    assert savepoint.ivend() == num_cells

    # The canopy is switched off, so nothing is inside the roughness layer.
    assert savepoint.kcm() == savepoint.ke1()
    assert savepoint.kstart_cloud() == 1

    # nmvar = 5 first-order variables and no passive tracer on top of them, which is what makes
    # the un-serializable 'ptr(:)' fan-out empty here. 'ntrac' is the declared length of the
    # tracer vector, not the number in use.
    assert savepoint.ndtr() == 0
    assert savepoint.ndiff() == 5
    assert savepoint.ntrac() >= savepoint.ndtr()

    assert savepoint.dt_var() > 0.0
    assert savepoint.itndcon() == 0
    assert savepoint.lentire() is True  # hard-coded at mo_nwp_turbdiff_interface.f90:679
    assert savepoint.lsfluse() is True
    assert savepoint.lrunscm() is False
    assert savepoint.ldoexpcor() is False
    assert savepoint.ldocirflx() is False
    assert savepoint.ldogrdcor() is (savepoint.ldoexpcor() or savepoint.ldocirflx())
    assert isinstance(savepoint.lqvcrst(), bool)
    assert isinstance(savepoint.l3dflxout(), bool)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_field_shapes(date: str, *, data_provider: sb.IconSerialDataProvider) -> None:
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_vertdiff_exit(date=date)

    num_cells = int(data_provider.grid_size[dims.CellDim])
    cell = (num_cells,)
    full = (num_cells, entry.ke())
    half = (num_cells, entry.ke1())

    for accessor, expected in (
        ("t_g", cell), ("qv_s", cell), ("ps", cell), ("tvm", cell), ("tvh", cell),
        ("shfl_s", cell), ("qvfl_s", cell),
        ("hhl", half), ("rhon", half), ("tkvm", half), ("tkvh", half),
        ("u", full), ("v", full), ("t", full), ("qv", full), ("qc", full),
        ("prs", full), ("rhoh", full), ("epr", full),
        ("u_tens", full), ("v_tens", full), ("t_tens", full),
        ("qv_tens", full), ("qc_tens", full),
    ):  # fmt: skip
        field = getattr(entry, accessor)()
        assert field.shape == expected, accessor
        assert field.asnumpy().dtype == np.float64, accessor

    for accessor, expected in (
        ("u", full), ("v", full), ("t", full), ("qv", full), ("qc", full),
        ("u_tens", full), ("v_tens", full), ("t_tens", full),
        ("qv_tens", full), ("qc_tens", full),
        ("rhon", half), ("tkvm", half), ("tkvh", half),
        ("diff_mom", half), ("disc_mom", half), ("expl_mom", half), ("impl_mom", half),
        ("invs_mom", half), ("diff_dep", half), ("invs_fac", half), ("scal_fac", half),
        ("dif_tend", half), ("cur_prof", half),
        ("shfl_s", cell), ("qvfl_s", cell),
        # 'eprs' is declared '(nvec, ke1:ke1)', a one-level slab; the level is squeezed away.
        ("eprs", cell),
    ):  # fmt: skip
        field = getattr(exit_savepoint, accessor)()
        assert field.shape == expected, accessor
        assert field.asnumpy().dtype == np.float64, accessor

    for component in range(6):
        assert entry.zvari(component).shape == half
        assert exit_savepoint.zvari(component).shape == half
    for savepoint in (entry, exit_savepoint):
        with pytest.raises(IndexError):
            savepoint.zvari(6)
    for component in range(5):
        assert exit_savepoint.raw_zaux(component).shape == half
    with pytest.raises(IndexError):
        exit_savepoint.raw_zaux(5)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_absent_optionals_come_back_as_dummies(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """'dp0', 'qv_conv', 'umfl_s' and 'vmfl_s' are OPTIONAL and not passed by the NWP interface."""
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_vertdiff_exit(date=date)

    for accessor in ABSENT_OPTIONALS["entry"]:
        field = getattr(entry, accessor)()
        assert field.shape == (1, 1), accessor
    for accessor in ABSENT_OPTIONALS["exit"]:
        field = getattr(exit_savepoint, accessor)()
        assert field.shape in ((1,), (1, 1)), accessor


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_entry_is_the_state_turbdiff_left(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'vertdiff' is called from SUB 'nwp_turbdiff' right after 'turbdiff' for the same block, so
    'vertdiff-entry' must be byte-identical to 'turbdiff-exit' for the fields they share.

    This is also what justifies 'serialize_vertdiff_*' reading the serialization context
    ('ser_turb_jg', 'ser_turb_date', 'ser_turb_linit') set at the 'turbdiff' call site: if the
    two ever drifted apart, this test would say so before any comparison used the wrong date.
    """
    turbdiff = data_provider.from_savepoint_turbdiff_exit(date=date)
    vertdiff = data_provider.from_savepoint_vertdiff_entry(date=date)

    assert vertdiff.ivstart() == turbdiff.ivstart()
    assert vertdiff.ivend() == turbdiff.ivend()

    assert np.array_equal(vertdiff.tkvm().asnumpy(), turbdiff.tkvm().asnumpy())
    assert np.array_equal(vertdiff.tkvh().asnumpy(), turbdiff.tkvh().asnumpy())
    assert np.array_equal(vertdiff.rhon().asnumpy(), turbdiff.rhon().asnumpy())
    assert np.array_equal(vertdiff.tvm().asnumpy(), turbdiff.tvm().asnumpy())
    assert np.array_equal(vertdiff.tvh().asnumpy(), turbdiff.tvh().asnumpy())
    for component in range(6):
        assert np.array_equal(
            vertdiff.zvari(component).asnumpy(), turbdiff.zvari(component).asnumpy()
        )


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_writes_tendencies_and_not_the_prognostic_variables(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'u'..'qc' are 'INTENT(INOUT)' and stay untouched because every tendency field is present.

    'vertdiff' adds the diffusion increment directly to the prognostic variable only when the
    corresponding tendency is absent, which is why both values are serialized. It does rescale
    'rhon' in place, and it does rewrite 'zvari' with the gradients the semi-implicit procedure
    produced.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    exit_savepoint = data_provider.from_savepoint_vertdiff_exit(date=date)

    for accessor in ("u", "v", "t", "qv", "qc"):
        assert np.array_equal(
            getattr(entry, accessor)().asnumpy(), getattr(exit_savepoint, accessor)().asnumpy()
        ), accessor
    for accessor in ("u_tens", "v_tens", "t_tens", "qv_tens", "qc_tens"):
        assert not np.array_equal(
            getattr(entry, accessor)().asnumpy(), getattr(exit_savepoint, accessor)().asnumpy()
        ), accessor

    assert not np.array_equal(entry.rhon().asnumpy(), exit_savepoint.rhon().asnumpy())
    assert np.array_equal(entry.tkvm().asnumpy(), exit_savepoint.tkvm().asnumpy())
    assert any(
        not np.array_equal(
            entry.zvari(component).asnumpy(), exit_savepoint.zvari(component).asnumpy()
        )
        for component in range(6)
    )

    # The surface fluxes are used as the lower boundary condition ('lsfluse'), not recomputed.
    assert np.array_equal(entry.shfl_s().asnumpy(), exit_savepoint.shfl_s().asnumpy())
    assert np.array_equal(entry.qvfl_s().asnumpy(), exit_savepoint.qvfl_s().asnumpy())


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_exit_loop_state(date: str, *, data_provider: sb.IconSerialDataProvider) -> None:
    """
    The workspace belongs to the last variable of the last variable type; these say which.

    'ncorr > mcorr' means the gradient-correction range is empty, which is the same statement as
    'ldogrdcor = F' at the entry: without 'ldoexpcor' or 'ldocirflx' the effective gradients play
    no part in the solve, and 'igrdcon' stays 0.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    savepoint = data_provider.from_savepoint_vertdiff_exit(date=date)

    assert savepoint.ivtype() == 2  # scalars, the second and last variable type
    assert savepoint.igrdcon() == 0
    assert savepoint.ncorr() > savepoint.mcorr()
    assert entry.ldogrdcor() is False


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_exit_workspace_is_not_turbdiffs(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'vertdiff's 'zaux' and 'len_scale' are a different set of aliases from 'turbdiff's.

    Both schemes serialize a '(nvec, ke1, 5)' array called 'zaux' and a '(nvec, ke1)' array
    called 'len_scale', and neither pair holds the same quantity: 'vd_zaux(:,:,1)' is the
    discretisation momentum where 'td_zaux(:,:,1)' is the Exner factor, and 'vd_len_scale' is the
    diffusion momentum where 'td_len_scale' is (by then) the effective TKE flux. The prefixes are
    what keeps them apart in the global field table; this asserts they are worth keeping apart.
    """
    turbdiff = data_provider.from_savepoint_turbdiff_exit(date=date)
    vertdiff = data_provider.from_savepoint_vertdiff_exit(date=date)
    window = slice(vertdiff.ivstart(), vertdiff.ivend())

    for component in range(5):
        assert not np.array_equal(
            turbdiff.zaux(component).asnumpy()[window],
            vertdiff.raw_zaux(component).asnumpy()[window],
        ), component
    assert not np.array_equal(
        turbdiff.len_scale().asnumpy()[window], vertdiff.diff_mom().asnumpy()[window]
    )

    # The diffusion depth is a length in metres and the discretisation momentum a mass flux;
    # both are strictly positive over the computed window, which the aliased 'turbdiff' contents
    # of the same slots are not.
    assert vertdiff.diff_dep().asnumpy()[window][:, 1:].min() > 0.0
    assert vertdiff.disc_mom().asnumpy()[window][:, 1:].min() > 0.0


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", VERTDIFF_DATES)
def test_vertdiff_is_physical_only_between_ivstart_and_ivend(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    The same masking rule as for 'turbdiff', shown on a routine-local array.

    Which fields are garbage outside the window depends on where they come from. 'qv' and the
    other interface fields hold the un-diffused lateral-boundary rows: not the result of this
    call, but physically valid, so they cannot show the trap. 'vertdiff's own '!$ACC CREATE'
    working set can: the surface Exner factor is 0.894..1.002 over the computed columns -- it
    is 'p/p0' to the power kappa at the ground and cannot be far from 1 -- and drops to 0 below
    'ivstart', where the array was never written.
    """
    savepoint = data_provider.from_savepoint_vertdiff_exit(date=date)
    eprs = savepoint.eprs().asnumpy()
    ivstart, ivend = savepoint.ivstart(), savepoint.ivend()

    computed = eprs[ivstart:ivend]
    assert computed.size > 0
    assert np.isfinite(computed).all()
    assert computed.min() > 0.5
    assert computed.max() < 1.5
    assert eprs[:ivstart].min() < 0.5
    assert not (eprs > 0.5).all()

    # The interface fields, by contrast, are valid everywhere: masking is about provenance,
    # not about NaN.
    qv = savepoint.qv().asnumpy()
    assert qv.min() >= 0.0
