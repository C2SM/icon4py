# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 7) of 'turbdiff' -- which is inert in this configuration.

Section 7) (turb_diffusion.f90:2311-2342, between 'turbdiff-6-exit' and 'turbdiff-7-exit') would
add the non-turbulent theta gradient of the circulation heat flux to the effective gradient
'zvari(:,:,tet)'. In 'exp.mch_icon-ch2_small' it does nothing at all: its whole body is under
'IF (ldocirflx)' (:2315) and 'ldocirflx' is false, so the two savepoints are identical over the
computed columns in every one of the 33 serialized fields, at all four timesteps.

THERE ARE THEREFORE NO STENCILS AND NO GATE ENTRIES FOR THIS SECTION. The tests below measure
that state of affairs instead, so that it is a checked fact in the repository rather than a
sentence in a note: if a future capture turns the switch on, the first of them fails and says
what to port. This is the same disposition section 2b) was given, for the same reason.

WHY IT IS OFF, AND WHAT WOULD TURN IT ON
----------------------------------------
'ldocirflx = (tdc%lcirflx .AND. lcircterm)' (:2047). The second conjunct is TRUE here -- the raw
circulation term itself is active, 'lcircterm = (pat_len > 0) .AND. ltkenst' (:945, :959) with
'pat_len = 750.0' in the experiment's 'turbdiff_nml' and 'ltkenst' at its default '.TRUE.', which
is why sections 6) and 8) are live. What is off is 'lcirflx', a separate switch that decides
whether the circulation term also carries a HEAT flux; it defaults to '.FALSE.'
(mo_turbdiff_config.f90:281) and the experiment does not set it.

So this section is not dead because the thermal circulation is absent. It is dead because only
its TKE branch is enabled, and a capture with 'turbdiff_nml/lcirflx = .TRUE.' would exercise it.

THE INPUT DOES NOT EXIST EITHER, WHICH IS THE STRONGER STATEMENT
----------------------------------------------------------------
A flag that suppresses a store still leaves an oracle for what the store would have been. Here
there is not even that. The only quantity section 7) consumes that no other section produces is
'shv', the buoyant heat flux of the non-turbulent circulations, and section 6) fills it under
'IF (tdc%lcirflx .OR. loutthcrc)' (:2255) -- false on both counts, 'loutthcrc' because
'tket_nstc' is not passed (:954). 'shv' is consequently untouched memory throughout, holding
plausible-looking values in 1e-6..7e-3 that are not a heat flux;
'test_the_buoyant_heat_flux_this_section_would_read_is_never_produced' pins that.

Porting the arithmetic anyway would give a stencil with no reference data, no caller and no way
to tell a mistranslation from a correct translation. For the record, it is one program's worth,
pointwise in k despite the name -- no 'Koff', no 'concat_where', the excluded model top and
surface row being a matter for the vertical domain:

    wert = shv(i,k)
    wert = SIGN(z1,wert) * MIN(hlp(i,k), ABS(wert))
    wert = -wert / (zaux(i,k,4) * tkvh(i,k))
    zvari(i,k,tet) = zvari(i,k,tet) + wert   ! or '= wert' when 'ldoexpcor' is false

and 'ldoexpcor' is false here too ('lexpcor' defaults '.FALSE.', mo_turbdiff_config.f90:279), so
even the choice between those two stores is uncovered by this data.
"""

from __future__ import annotations

import numpy as np
import pytest

from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_7_writes_nothing(date: str, *, data_provider: sb.IconSerialDataProvider) -> None:
    """Nothing changes across section 7): this is what says there is nothing to port.

    'utils.fields_that_changed' compares all 33 serialized fields over 'ivstart:ivend', which is
    the same instrument that established the output set of every ported section. An empty result
    is as strong a statement as a non-empty one, and the only reason this module exists.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="7", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == frozenset()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_circulation_heat_flux_is_switched_off(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'ldocirflx' is false, which is the reason the section stores nothing.

    Read from the savepoint rather than argued from the namelist: 'turbdiff' recomputes both
    flags at the end of section 4) from three configuration switches, and the serialized value is
    what the run actually used. 'ldoexpcor' is asserted alongside it because it selects between
    the two stores of the section body, so its being false is a second thing this data cannot
    cover.
    """
    at_entry = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    at_exit = data_provider.from_savepoint_turbdiff_section(section="7", date=date)

    assert at_entry.ldocirflx() is False
    assert at_exit.ldocirflx() is False
    assert at_entry.ldoexpcor() is False


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_buoyant_heat_flux_this_section_would_read_is_never_produced(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'shv' is untouched memory across section 6), so section 7) has no input to be given.

    Section 6) is the only writer of 'shv' in the whole scheme, and its guard
    'tdc%lcirflx .OR. loutthcrc' (turb_diffusion.f90:2255) is false here. The slot is therefore
    the same bytes at 'turbdiff-4-exit' as at 'turbdiff-6-exit' -- which is what makes this
    section unportable rather than merely unexercised, since even a stencil written from the
    Fortran could not be validated against anything.

    'turbdiff-4-exit' is read through 'raw_field', because the reader deliberately refuses to
    name this slot before section 6): 'shv()' raises there, on the grounds that the quantity does
    not exist yet. That refusal is exactly the claim being measured, so the escape hatch is the
    right instrument and not a way around one.
    """
    at_section_4 = data_provider.from_savepoint_turbdiff_section(section="4", date=date)
    at_section_6 = data_provider.from_savepoint_turbdiff_section(section="6", date=date)
    columns = slice(at_section_6.ivstart(), at_section_6.ivend())

    before = utils.copy_of_raw_field(at_section_4, "td_shv", backend).asnumpy()
    after = utils.copy_of_raw_field(at_section_6, "td_shv", backend).asnumpy()

    assert np.array_equal(before[columns], after[columns]), (
        "'shv' changed across section 6): the circulation heat flux is being produced after all, "
        "so section 7) is live and needs porting."
    )
