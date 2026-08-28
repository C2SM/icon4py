# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for the 'turbdiff' savepoint readers.

These pin the contract between 'serialize_turbdiff_entry' / 'serialize_turbdiff_exit' in ICON's
mo_icon4py_verification.f90 (commit 38e3720277) and 'IconTurbdiffEntrySavepoint' /
'IconTurbdiffExitSavepoint': which fields exist, what shape and dtype they come back with, and
-- the one that matters for every later comparison -- that only the columns 'ivstart:ivend' hold
computed values.

The scheme writes the whole 'nproma' slab but loops over 'ivstart:ivend' only. Outside that
window the memory is untouched rather than NaN, and it holds plausible numbers: 'tke' is
'q = SQRT(2*TKE)', which never falls below the scheme's minimal turbulent velocity scale
'vel_min = 0.01 m/s' (mo_turbdiff_config.f90:226), yet the serialized slab reaches -0.026 in the
lateral boundary rows below 'ivstart' -- a value 'q' cannot take at all. A
comparison that ignores the bounds therefore fails like a physics bug, which is why
'test_turbdiff_entry_tke_is_physical_only_between_ivstart_and_ivend' asserts both halves of the
claim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.testing import definitions

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    from icon4py.model.testing import serialbox as sb


#: The four timesteps 'exp.mch_icon-ch2_small' serializes: the serialization window is the last
#: four of its six steps, and 'turbdiff' is called once per step per block.
TURBDIFF_DATES = (
    "2020-12-10T06:01:00.000",
    "2020-12-10T06:01:20.000",
    "2020-12-10T06:01:40.000",
    "2020-12-10T06:02:00.000",
)

#: Lower limit of the turbulent velocity 'q = SQRT(2*TKE)' [m/s]: 'vel_min', the minimal
#: turbulent velocity scale (mo_turbdiff_config.f90:226). The observed minimum over the computed
#: columns of this capture is exactly this value.
TKE_FLOOR = 0.01

#: Every name the entry savepoint writes. Frozen here so that adding or dropping a field in the
#: Fortran hook fails loudly instead of silently leaving a reader accessor unbacked.
ENTRY_SERIALIZED_FIELDS = frozenset(
    {
        # sizes, control flags and the valid horizontal window
        "td_nvec", "td_ke", "td_ke1", "td_kcm", "td_ivstart", "td_ivend", "td_iini",
        "td_ntur", "td_nprv", "td_ntim", "td_dt_var", "td_dt_tke",
        "td_ltkeinp", "td_l3dturb", "td_lrunsso", "td_lruncnv", "td_lrunscm",
        # output of 'turb_setup'
        "td_lini", "td_it_start", "td_nvor", "td_fr_tke", "td_l_scal", "td_fc_min",
        # grid and surface
        "td_l_hori", "td_hhl", "td_dp0", "td_trop_mask", "td_innertrop_mask",
        "td_gz0", "td_l_pat", "td_t_g", "td_qv_s", "td_ps",
        # atmospheric state
        "td_u", "td_v", "td_t", "td_qv", "td_qc", "td_prs", "td_rhoh", "td_epr",
        # transfer-layer state produced by 'turbtran'
        "td_tvm", "td_tvh", "td_tfm", "td_tfh", "td_tfv",
        # prognostic and diagnostic turbulence state
        "td_tke", "td_tkvm", "td_tkvh", "td_tprn", "td_rcld", "td_zvari", "td_tketens_in",
        # arguments that are OPTIONAL in 'turbdiff' and present at this call site
        "td_w", "td_tkred_sfc", "td_tkred_sfc_h",
        "td_hdef2", "td_hdiv", "td_dwdx", "td_dwdy", "td_tket_conv",
        "td_u_tens_in", "td_v_tens_in", "td_t_tens_in", "td_ut_sso", "td_vt_sso",
    }
)  # fmt: skip

#: Every name the exit savepoint writes.
EXIT_SERIALIZED_FIELDS = frozenset(
    {
        "td_ivstart", "td_ivend",
        "td_gz0", "td_tvm", "td_tvh", "td_tfm", "td_tfh", "td_tfv",
        "td_tke", "td_tkvm", "td_tkvh", "td_tprn", "td_rcld", "td_rhon", "td_zvari",
        "td_tketens",
        # 'turbdiff's own !$ACC CREATE working set: routine locals with no interface
        "td_dicke", "td_frh", "td_frm", "td_ftm", "td_hlp", "td_shv", "td_zaux",
        "td_len_scale", "td_edr",
        # optional arguments present at this call site
        "td_ldoexpcor", "td_ldocirflx", "td_tkred_sfc", "td_tkred_sfc_h",
        "td_u_tens", "td_v_tens", "td_t_tens", "td_tket_hshr",
    }
)  # fmt: skip

#: Entry accessors returning a bare cell field.
ENTRY_CELL_ACCESSORS = (
    "l_scal", "fc_min", "l_hori", "trop_mask", "innertrop_mask", "gz0", "l_pat",
    "t_g", "qv_s", "ps", "tvm", "tvh", "tfm", "tfh", "tfv", "tkred_sfc", "tkred_sfc_h",
)  # fmt: skip

#: Entry accessors on main ('full') levels, so 'ke' of them.
ENTRY_FULL_LEVEL_ACCESSORS = (
    "dp0", "u", "v", "t", "qv", "qc", "prs", "rhoh", "epr",
    "u_tens", "v_tens", "t_tens", "ut_sso", "vt_sso",
)  # fmt: skip

#: Entry accessors on half levels, so 'ke1' of them.
ENTRY_HALF_LEVEL_ACCESSORS = (
    "hhl", "tke", "tkvm", "tkvh", "rcld", "tketens", "w",
    "hdef2", "hdiv", "dwdx", "dwdy", "tket_conv",
)  # fmt: skip

#: Exit accessors, by vertical staggering. 'tprn' and the component arrays are checked apart.
EXIT_CELL_ACCESSORS = ("gz0", "tvm", "tvh", "tfm", "tfh", "tfv", "tkred_sfc", "tkred_sfc_h")
EXIT_FULL_LEVEL_ACCESSORS = ("u_tens", "v_tens", "t_tens")
EXIT_HALF_LEVEL_ACCESSORS = (
    "tke", "tkvm", "tkvh", "rcld", "rhon", "tketens", "dicke", "frh", "frm", "ftm",
    "hlp", "shv", "len_scale", "edr", "tket_hshr",
)  # fmt: skip

experiment_for_turbulence = pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_ICON_CH2_SMALL],
    ids=lambda d: d.name,
)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_entry_savepoint_serializes_the_expected_fields(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)

    serialized = {
        name
        for name in data_provider.serializer.fields_at_savepoint(savepoint.savepoint)
        if name.startswith("td_")
    }
    assert serialized == set(ENTRY_SERIALIZED_FIELDS)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_exit_savepoint_serializes_the_expected_fields(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)

    serialized = {
        name
        for name in data_provider.serializer.fields_at_savepoint(savepoint.savepoint)
        if name.startswith("td_")
    }
    assert serialized == set(EXIT_SERIALIZED_FIELDS)


@pytest.mark.datatest
@experiment_for_turbulence
def test_turbdiff_savepoints_are_selected_by_date_id_and_block(
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """One entry and one exit savepoint per timestep, keyed by ('date', 'id', 'block')."""
    for date in TURBDIFF_DATES:
        for savepoint in (
            data_provider.from_savepoint_turbdiff_entry(date=date),
            data_provider.from_savepoint_turbdiff_exit(date=date),
        ):
            metadata = savepoint.savepoint.metainfo.to_dict()
            assert metadata["date"] == date
            assert metadata["id"] == 1  # the domain 'jg'
            assert metadata["block"] == 1
            assert savepoint.block() == 1

    with pytest.raises(Exception):  # noqa: B017 [assert-raises-exception]
        # serialbox raises when the metadata does not select exactly one savepoint
        data_provider.from_savepoint_turbdiff_entry(date=TURBDIFF_DATES[0], block=2)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_entry_savepoint_scalars(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)

    num_cells = int(data_provider.grid_size[dims.CellDim])
    nlev = int(data_provider.grid_size[dims.KDim])

    assert savepoint.ke() == nlev
    assert savepoint.ke1() == savepoint.ke() + 1

    # 'nproma' covers the whole patch in one block, which is what makes the cell-field
    # truncation in '_reduce_to_dim_size' meaningful (see 'IconTurbdiffSavepoint').
    assert savepoint.nvec() >= num_cells

    # Half-open Python bounds: shifted start, unshifted end.
    assert 0 <= savepoint.ivstart() < savepoint.ivend() <= savepoint.nvec()
    assert savepoint.ivend() == num_cells

    # The NWP interface always calls 'turbdiff' as a regular (not initializing) step with a
    # single TKE time level; 'tke()' relies on 'ntim == 1'.
    assert savepoint.iini() == 0
    assert savepoint.ntim() == 1
    assert savepoint.lini() is False
    assert savepoint.ntur() == 1
    assert savepoint.nprv() == 1
    assert savepoint.nvor() == 1
    assert savepoint.it_start() == 1

    assert savepoint.dt_var() > 0.0
    assert savepoint.dt_tke() > 0.0
    assert savepoint.fr_tke() == pytest.approx(1.0 / savepoint.dt_tke())

    # 3D turbulence is hardcoded off at mo_nwp_turbdiff_interface.f90:584.
    assert savepoint.l3dturb() is False
    assert savepoint.ltkeinp() is False
    assert savepoint.lrunscm() is False
    assert isinstance(savepoint.lrunsso(), bool)
    assert isinstance(savepoint.lruncnv(), bool)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
@pytest.mark.parametrize("accessor", ENTRY_CELL_ACCESSORS)
def test_turbdiff_entry_cell_fields(
    accessor: str,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    field = getattr(savepoint, accessor)()

    assert field.domain.dims == (dims.CellDim,)
    assert field.shape == (int(data_provider.grid_size[dims.CellDim]),)
    assert field.asnumpy().dtype == np.float64


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
@pytest.mark.parametrize("accessor", ENTRY_FULL_LEVEL_ACCESSORS)
def test_turbdiff_entry_full_level_fields(
    accessor: str,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    field = getattr(savepoint, accessor)()

    assert field.domain.dims == (dims.CellDim, dims.KDim)
    assert field.shape == (
        int(data_provider.grid_size[dims.CellDim]),
        savepoint.ke(),
    )
    assert field.asnumpy().dtype == np.float64


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
@pytest.mark.parametrize("accessor", ENTRY_HALF_LEVEL_ACCESSORS)
def test_turbdiff_entry_half_level_fields(
    accessor: str,
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    field = getattr(savepoint, accessor)()

    assert field.domain.dims == (dims.CellDim, dims.KDim)
    assert field.shape == (
        int(data_provider.grid_size[dims.CellDim]),
        savepoint.ke1(),
    )
    assert field.asnumpy().dtype == np.float64


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_entry_zvari_components(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """'zvari(:,:,0:ndim)' has a zero lower bound, so component 0..5 is the Fortran index."""
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    expected_shape = (int(data_provider.grid_size[dims.CellDim]), savepoint.ke1())

    for component in range(6):
        field = savepoint.zvari(component)
        assert field.domain.dims == (dims.CellDim, dims.KDim)
        assert field.shape == expected_shape
        assert field.asnumpy().dtype == np.float64

    with pytest.raises(IndexError):
        savepoint.zvari(6)


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_entry_tprn_is_a_dummy_in_this_configuration(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    'prm_diag%tprn' is allocated '(/1, 1, kblks/)' unless a TMod that fills it is selected
    (mo_nwp_phy_state.f90:4706-4709), and this experiment takes that branch. The reader must
    hand the degenerate shape back rather than pretend it is a cell field.
    """
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    field = savepoint.tprn()

    assert field.domain.dims == (dims.CellDim, dims.KDim)
    assert field.shape == (1, 1)
    assert field.shape[0] != int(data_provider.grid_size[dims.CellDim])


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_entry_tke_is_physical_only_between_ivstart_and_ivend(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    """
    The reason 'ivstart' and 'ivend' are serialized at all.

    'tke' is 'q = SQRT(2*TKE)' in m/s and bounded below by 'vel_min = 0.01', so inside the
    computed window every value is finite and at or above the floor. Outside it the slab is
    untouched
    memory that looks like data -- and here it is not merely below the floor but negative,
    which 'q' can never be. A comparison run over the whole slab would fail as if the physics
    were wrong.
    """
    savepoint = data_provider.from_savepoint_turbdiff_entry(date=date)
    tke = savepoint.tke().asnumpy()
    ivstart, ivend = savepoint.ivstart(), savepoint.ivend()

    computed = tke[ivstart:ivend]
    assert computed.size > 0
    assert np.isfinite(computed).all()
    assert computed.min() >= TKE_FLOOR - 1.0e-12, (
        f"tke below the {TKE_FLOOR} m/s floor inside ivstart:ivend = {ivstart}:{ivend}"
    )

    # And the same claim is false without the mask.
    untouched = tke[:ivstart]
    assert untouched.size > 0
    assert untouched.min() < 0.0, (
        "expected untouched memory below ivstart; if this ever holds the floor, the masking "
        "rule is still required -- re-check the capture rather than dropping the mask"
    )
    assert not (tke >= TKE_FLOOR - 1.0e-12).all()

    # The columns above 'ivend' are gone already: the cell-field accessors truncate 'nproma'
    # to 'num_cells', which for this single-block capture is exactly 'ivend'.
    assert tke.shape[0] == ivend
    assert savepoint.nvec() > ivend


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_exit_fields(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)

    num_cells = int(data_provider.grid_size[dims.CellDim])

    # The exit savepoint carries only 'ivstart'/'ivend'; the sizes come from the entry
    # savepoint of the same (date, id, block).
    assert savepoint.ivstart() == entry.ivstart()
    assert savepoint.ivend() == entry.ivend()

    for accessor, expected_shape in (
        *((name, (num_cells,)) for name in EXIT_CELL_ACCESSORS),
        *((name, (num_cells, entry.ke())) for name in EXIT_FULL_LEVEL_ACCESSORS),
        *((name, (num_cells, entry.ke1())) for name in EXIT_HALF_LEVEL_ACCESSORS),
    ):
        field = getattr(savepoint, accessor)()
        assert field.shape == expected_shape, accessor
        assert field.asnumpy().dtype == np.float64, accessor

    for component in range(5):
        # 'zaux(nvec,ke1,ndim)' is one-based in Fortran; 'component' here is zero-based.
        field = savepoint.zaux(component)
        assert field.shape == (num_cells, entry.ke1())
    with pytest.raises(IndexError):
        savepoint.zaux(5)

    assert savepoint.ldoexpcor() is False
    assert savepoint.ldocirflx() is False


@pytest.mark.datatest
@experiment_for_turbulence
@pytest.mark.parametrize("date", TURBDIFF_DATES)
def test_turbdiff_exit_tke_is_physical_only_between_ivstart_and_ivend(
    date: str,
    *,
    data_provider: sb.IconSerialDataProvider,
) -> None:
    savepoint = data_provider.from_savepoint_turbdiff_exit(date=date)
    tke = savepoint.tke().asnumpy()
    ivstart, ivend = savepoint.ivstart(), savepoint.ivend()

    assert tke[ivstart:ivend].min() >= TKE_FLOOR - 1.0e-12
    assert not (tke >= TKE_FLOOR - 1.0e-12).all()
