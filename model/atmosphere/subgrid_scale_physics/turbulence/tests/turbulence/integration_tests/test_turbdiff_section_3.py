# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for section 3) of 'turbdiff': the turbulent budgets.

The oracle is a real ICON run: 'turbdiff-2c-exit' supplies the inputs, 'turbdiff-3-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

Section 3) is the heart of the scheme. It is one call of SUB 'solve_turb_budgets'
(turb_utilities.f90:1052-1861) inside a fixed one-step iteration loop, followed by five short
blocks back in 'turbdiff' (turb_diffusion.f90:1863-1913). What survives of it under the
operational configuration is six programs.

WHAT THIS SECTION WRITES
------------------------
'tke', 'tkvm', 'tkvh', 'rcld' and component 0 of 'zvari', and nothing else. That is measured,
not read off the source: 'test_section_3_writes_only_the_five_expected_fields' compares every
serialized field between the two savepoints and exactly those five differ. The measurement is
what rules out the large parts of 'solve_turb_budgets' this port does not contain -- the
roughness-layer parameter overrides, the eddy dissipation rate, and the whole turbulent flux
conversion, whose absence shows up as components 1..5 of 'zvari' being byte-identical across
the section.

THE K-LOOP IS NOT A SCAN
------------------------
'solve_turb_budgets' declares its main k-loop '!$ACC LOOP SEQ', which is what a scan looks like.
It is not one. Verified mechanically over the whole subroutine: no array reference anywhere in
it subscripts the vertical axis with anything but the bare loop index 'k'. There is no 'k-1',
no 'k+1', and the one indirect index, 'k_tvs', is assigned 'k' on the branch this configuration
takes (turb_utilities.f90:1490) and '1' only under 'lpres_avt', which needs a 'tketadv' the ICON
interface never passes. The loop is sequential because its scratch arrays are one-dimensional
and because 'dd' carries the roughness-layer parameters forward -- and 'lporous' is a '.FALSE.'
PARAMETER, so nothing ever writes 'dd' inside the loop. Every level is independent, and the
port is six wide field operators. 'test_the_k_loop_of_solve_turb_budgets_has_no_vertical_
neighbour_access' re-measures this against the Fortran source on every run, so a future ICON
that introduces a 'k-1' fails here instead of silently producing wrong numbers.

WHAT THE PROGRAMS ARE, AND WHY THEY ARE SPLIT THAT WAY
------------------------------------------------------
    compute_turbulent_velocity_scale         'tke' at half levels 2..kem
    set_turbulent_velocity_scale_at_model_top 'tke' at half level 1, copied from level 2
    compute_stability_lengths                'lsm', 'lsh' at 2..kem -- NOT serialized
    compute_circulation_acceleration         'zvari(:,:,0)' at 2..ke1
    compute_supersaturation_standard_deviation 'rcld' at 2..kem
    compute_diffusion_coefficients_from_stability_lengths 'tkvm', 'tkvh' at 2..kem

The velocity scale and the stability functions come out of the same Fortran k-loop and share the
TKE forcing 'frc', but Raschendorfer gives them separate sub-headings and separate ACC loops,
and each deserves its own gate; they are two programs sharing a private field operator. The
model-top row is its own program because the value it copies is the one just computed, so a
merged program would read the field it writes.

TWO ALIASES THIS PORT DOES NOT REPRODUCE, and both of them turn a Fortran ordering constraint
into nothing at all:

  * 'rcld' arrives as the cloud cover and leaves as the standard deviation of the local
    super-saturation. The raw circulation term is the last reader of the cloud cover, so in the
    Fortran it must run before the SDSS block. Here they are separate fields and the order is
    free.
  * 'tkvm'/'tkvh' arrive as stability lengths (section 2c) divided them by 'tke') and leave as
    diffusion coefficients. The SDSS block reads the updated stability length, so in the Fortran
    it must run before the multiplication back. Here, again, separate fields.

The one ordering that does survive is real: 'compute_stability_lengths' needs the 'tke' that
'compute_turbulent_velocity_scale' produces, and both consumers of the stability lengths need
those.

WHERE THE CONSTANTS COME FROM
-----------------------------
Every closure constant is derived from four length-scale factors that 'turbdiff_nml' of
exp.mch_icon-ch2_small does not set, so 'TurbulenceConfig()' at its compiled-in defaults is the
configuration of the capture. 'test_the_capture_runs_the_compiled_in_closure_constants' states
that dependency as an assertion rather than leaving it to the reader.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final, NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import turbulence
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_circulation_acceleration import (
    compute_circulation_acceleration,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_coefficients_from_stability_lengths import (
    compute_diffusion_coefficients_from_stability_lengths,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_stability_lengths import (
    compute_stability_lengths,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_supersaturation_standard_deviation import (
    compute_supersaturation_standard_deviation,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_velocity_scale import (
    compute_turbulent_velocity_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.set_turbulent_velocity_scale_at_model_top import (
    set_turbulent_velocity_scale_at_model_top,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: Scalar conductivity of dry air, 'con_h' (mo_physical_constants.f90:116) [m2/s]. ICON's
#: 'mo_physical_constants' has it and 'icon4py.model.common.constants' does not; it lives here
#: until a shared home is agreed, because section 4) needs 'con_m' from the same block and a
#: constant should be added once, for both.
MOLECULAR_DIFFUSIVITY_FOR_SCALARS: Final[float] = 2.20e-5

#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), zero-based as the reader takes them.
TET_L, H2O_G = 3, 4

#: The Fortran this section was translated from, as a path relative to any ancestor of this
#: file. Read by 'test_the_k_loop_of_solve_turb_budgets_has_no_vertical_neighbour_access',
#: which searches upwards for it and skips when the ICON checkout is not there.
_TURB_UTILITIES: Final[str] = "icon/src/atm_phy_schemes/turb_utilities.f90"


def _icon_source() -> pathlib.Path | None:
    """The ICON file this section came from, if an ICON checkout sits above the icon4py one."""
    for ancestor in pathlib.Path(__file__).resolve().parents:
        candidate = ancestor / _TURB_UTILITIES
        if candidate.is_file():
            return candidate
    return None


class Section3(NamedTuple):
    """One timestep of section 3): the savepoints, the bounds and every computed output."""

    entry: sb.IconTurbdiffEntrySavepoint
    before: sb.IconTurbdiffSectionSavepoint
    after: sb.IconTurbdiffSectionSavepoint
    ke: int
    ke1: int
    #: Half-open range of columns 'turbdiff' actually computed; every comparison is masked with it.
    columns: slice
    #: Half levels the turbulence model covers, Fortran 'k_st..k_en' = 2..kem.
    model_levels: slice
    #: Those plus the surface half level, Fortran 'k_st..k_sf' = 2..ke1.
    boundary_levels: slice
    turbulent_velocity_scale: gtx.Field
    stability_length_for_momentum: gtx.Field
    stability_length_for_scalars: gtx.Field
    circulation_acceleration: gtx.Field
    supersaturation_standard_deviation: gtx.Field
    diffusion_coefficient_for_momentum: gtx.Field
    diffusion_coefficient_for_scalars: gtx.Field


def _run_section_3(data_provider, date: str, backend) -> Section3:
    """Run all six programs of section 3) on the 'turbdiff-2c-exit' state of one timestep."""
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    ke, ke1 = entry.ke(), entry.ke1()
    columns = slice(before.ivstart(), before.ivend())

    config = turbulence.TurbulenceConfig()
    params = turbulence.TurbulenceParams(config)

    # 'tke' starts as its entry state: the surface half level is what 'turbtran' left there and
    # this section must not touch it. The velocity-scale program reads the same values as
    # 'previous_velocity_scale', which is exactly the Fortran aliasing of 'nvor' and 'ntur'.
    velocity_scale = utils.copy_of(before.tke(), backend)
    # The stability lengths are not serialized -- section 3) destroys them again -- so there is
    # no entry state to copy and NaN is the adversarial choice.
    stability_m = utils.nan_like(before.stab_len_m(), backend)
    stability_h = utils.nan_like(before.stab_len_h(), backend)
    circulation = utils.copy_of(before.raw_zvari(0), backend)
    sdss = utils.copy_of(before.cloud_cover(), backend)
    tkvm = utils.copy_of(before.stab_len_m(), backend)
    tkvh = utils.copy_of(before.stab_len_h(), backend)

    # Fortran 'DO k=k_st,k_en' with 'k_st=2, k_en=kem=ke' (turb_diffusion.f90:1814).
    compute_turbulent_velocity_scale.with_backend(backend)(
        master_length_scale=before.mixing_length(),
        stability_length_for_momentum=before.stab_len_m(),
        stability_length_for_scalars=before.stab_len_h(),
        mechanical_forcing=before.mech_forcing(),
        thermal_forcing=before.thermal_forcing(),
        previous_velocity_scale=before.tke(),
        transport_tendency=before.tketens(),
        d_m=config.d_mom,
        d_4=params.d_4,
        b_m=params.b_m,
        rim=params.rim,
        frcsecu=config.frcsecu,
        tkesecu=config.tkesecu,
        tkesmot=config.tkesmot,
        vel_min=config.vel_min,
        tke_time_step=entry.dt_tke(),
        inverse_tke_time_step=entry.fr_tke(),
        turbulent_velocity_scale=velocity_scale,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    # In place on purpose: the read set is row 1 and the write set is row 0.
    set_turbulent_velocity_scale_at_model_top.with_backend(backend)(
        turbulent_velocity_scale=velocity_scale,
        turbulent_velocity_scale_with_top=velocity_scale,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(1),
        offset_provider={dims.Koff.value: dims.KDim},
    )
    compute_stability_lengths.with_backend(backend)(
        master_length_scale=before.mixing_length(),
        stability_length_for_momentum=before.stab_len_m(),
        stability_length_for_scalars=before.stab_len_h(),
        mechanical_forcing=before.mech_forcing(),
        thermal_forcing=before.thermal_forcing(),
        turbulent_velocity_scale=velocity_scale,
        a_h=config.a_heat,
        a_m=config.a_mom,
        b_h=params.b_h,
        b_m=params.b_m,
        d_m=config.d_mom,
        d_1=params.d_1,
        d_2=params.d_2,
        d_3=params.d_3,
        d_4=params.d_4,
        d_5=params.d_5,
        d_6=params.d_6,
        rim=params.rim,
        frcsecu=config.frcsecu,
        stbsecu=config.stbsecu,
        updated_stability_length_for_momentum=stability_m,
        updated_stability_length_for_scalars=stability_h,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    # Fortran 'DO k=k_st,k_sf', one half level deeper than the rest of the section.
    compute_circulation_acceleration.with_backend(backend)(
        cloud_cover=before.cloud_cover(),
        master_length_scale=before.mixing_length(),
        thermal_forcing=before.thermal_forcing(),
        half_level_pressure=before.raw_zvari(0),
        air_density=before.rhon(),
        pattern_length_scale=entry.l_pat(),
        horizontal_grid_scale=entry.l_hori(),
        gravitational_acceleration=constants.GRAV,
        circulation_acceleration=circulation,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke1),
        offset_provider={},
    )
    compute_supersaturation_standard_deviation.with_backend(backend)(
        master_length_scale=before.mixing_length(),
        stability_length_for_scalars=stability_h,
        exner_factor=before.exner_factor(),
        saturation_humidity_derivative=before.dqsat_dt(),
        gradient_of_liquid_water_potential_temperature=before.vertical_gradient(TET_L),
        gradient_of_total_water=before.vertical_gradient(H2O_G),
        d_h=config.d_heat,
        supersaturation_standard_deviation=sdss,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    compute_diffusion_coefficients_from_stability_lengths.with_backend(backend)(
        stability_length_for_momentum=stability_m,
        stability_length_for_scalars=stability_h,
        turbulent_velocity_scale=velocity_scale,
        molecular_diffusivity_for_scalars=MOLECULAR_DIFFUSIVITY_FOR_SCALARS,
        diffusion_coefficient_for_momentum=tkvm,
        diffusion_coefficient_for_scalars=tkvh,
        horizontal_start=gtx.int32(columns.start),
        horizontal_end=gtx.int32(columns.stop),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(ke),
        offset_provider={},
    )
    return Section3(
        entry=entry,
        before=before,
        after=after,
        ke=ke,
        ke1=ke1,
        columns=columns,
        model_levels=slice(1, ke),
        boundary_levels=slice(1, ke1),
        turbulent_velocity_scale=velocity_scale,
        stability_length_for_momentum=stability_m,
        stability_length_for_scalars=stability_h,
        circulation_acceleration=circulation,
        supersaturation_standard_deviation=sdss,
        diffusion_coefficient_for_momentum=tkvm,
        diffusion_coefficient_for_scalars=tkvh,
    )


# ------------------------------------------------------------------- what the section writes --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_3_writes_only_the_five_expected_fields(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The section's true output set, measured rather than read off the source.

    'solve_turb_budgets' has six output blocks and this configuration reaches three of them. The
    three it does not reach are the eddy dissipation rate ('lpres_edr .OR. ltmpcor', both false,
    so 'td_edr' is untouched) and the two flux-conversion blocks ('lexpcor' and 'lcpfluc', both
    false, so 'zvari' components 1..5 are untouched). Neither is inferred here: what is asserted
    is which serialized names differ.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="3", date=date)

    assert utils.fields_that_changed(data_provider, before, after) == {
        "td_tke",
        "td_tkvm",
        "td_tkvh",
        "td_rcld",
        "td_zvari",
    }


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_turbulent_flux_conversion_does_not_run_in_this_capture(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Only component 0 of 'zvari' changes, so the effective gradients are the vertical ones.

    'solve_turb_budgets' would convert the fluxes of the quasi-conserved variables into fluxes
    of the usual ones (turb_utilities.f90:1750-1770, :1795-1817) and correct them for a
    fluctuating heat capacity (:1819-1832). All three blocks need 'lexpcor', 'laddcnv' or
    'lcpfluc', which this port freezes off. This states that as a property of the reference data,
    so that a re-capture with a different namelist fails here rather than in whichever section
    first reads a gradient that was silently converted.
    """
    before = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    columns = slice(before.ivstart(), before.ivend())

    for component in range(1, 6):
        np.testing.assert_array_equal(
            before.raw_zvari(component).asnumpy()[columns],
            after.raw_zvari(component).asnumpy()[columns],
            err_msg=f"'zvari' component {component} changed across section 3)",
        )


def test_the_k_loop_of_solve_turb_budgets_has_no_vertical_neighbour_access() -> None:
    """No subscript in 'solve_turb_budgets' indexes the vertical axis with anything but 'k'.

    This is the assumption the whole section rests on: the k-loop at turb_utilities.f90:1384 is
    declared 'LOOP SEQ' and is nevertheless fully parallel in k, so the port is field operators
    and not a 'scan_operator' (port spec section 3.2). Asserting it against the Fortran source
    rather than in a comment means that an ICON which grows a 'k-1' here fails loudly.

    The check is textual on purpose: it looks for any occurrence of 'k' followed by '+' or '-'
    and a digit anywhere in the subroutine, comments included, and for any argument of any
    subscript that mentions 'k' without being exactly 'k'. Both are over-broad, which is the
    right direction for a canary.

    Skipped when the ICON checkout is not next to the icon4py one, since the package must remain
    testable on its own.
    """
    source = _icon_source()
    if source is None:
        pytest.skip(f"no ICON checkout above this file holding '{_TURB_UTILITIES}'")

    lines = source.read_text(errors="replace").splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("SUBROUTINE solve_turb_budgets")
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("END SUBROUTINE solve_turb_budgets"))
    body = lines[start:end]

    offsets = [f"{start + i + 1}: {line.strip()}" for i, line in enumerate(body) if re.search(r"\bk\s*[-+]\s*\d", line)]
    assert not offsets, "'solve_turb_budgets' now has a vertical offset:\n" + "\n".join(offsets)

    subscript = re.compile(r"\b[A-Za-z_]\w*\s*\(([^()]*)\)")
    suspicious = []
    for i, line in enumerate(body):
        code = line.split("!")[0]
        for match in subscript.finditer(code):
            for argument in match.group(1).split(","):
                argument = argument.strip()
                if argument not in ("k", "k_tvs") and re.search(r"\bk\b", argument):
                    suspicious.append(f"{start + i + 1}: {line.strip()}")
    assert not suspicious, "a subscript now derives an index from 'k':\n" + "\n".join(suspicious)


def test_the_capture_runs_the_compiled_in_closure_constants() -> None:
    """The four length-scale factors are the ones 'TurbulenceConfig()' defaults to.

    Every constant of section 3) is derived from 'a_heat', 'a_mom', 'd_heat' and 'd_mom' plus
    the four security factors, and 'turbdiff_nml' of exp.mch_icon-ch2_small sets none of the
    eight -- so the compiled-in defaults of mo_turbdiff_config.f90 are what produced the
    reference. None of the eight is serialized, so this cannot be read back from the archive;
    what it can do is fail if somebody changes a default in 'TurbulenceConfig' without noticing
    that the reference data was produced with the old one.
    """
    config = turbulence.TurbulenceConfig()

    assert (config.a_heat, config.a_mom) == (0.74, 0.92)
    assert (config.d_heat, config.d_mom) == (10.1, 16.6)
    assert (config.tkesmot, config.frcsecu) == (0.15, 1.00)
    assert (config.tkesecu, config.stbsecu) == (1.00, 0.01)
    assert config.vel_min == 0.01


# ---------------------------------------------------------------------- agreement with ICON --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_turbulent_velocity_scale_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_3(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_turbulent_velocity_scale",
        "tke(:,:,ntur)",
        run.turbulent_velocity_scale,
        run.after.tke(),
        columns=run.columns,
        levels=run.model_levels,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_set_turbulent_velocity_scale_at_model_top_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The model-top row, which is a copy of the row below and has its own oracle."""
    run = _run_section_3(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "set_turbulent_velocity_scale_at_model_top",
        "tke(:,1,ntur)",
        run.turbulent_velocity_scale,
        run.after.tke(),
        columns=run.columns,
        levels=slice(0, 1),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_stability_lengths_agrees_with_icon_through_the_next_fortran_statement(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'lsm' and 'lsh' have no oracle of their own, so they are checked one statement later.

    Section 3) overwrites the stability lengths with the diffusion coefficients before the
    section savepoint is written (turb_diffusion.f90:1868-1875), so 'turbdiff-3-exit' holds
    'lsm*tke' and 'MAX(lsh*tke, con_h)' and not the lengths themselves. This test applies
    exactly those two statements -- in numpy, not through the stencil that also does it -- to
    the computed lengths and the computed velocity scale, and requires the result to be ICON's
    bit for bit.

    That is a real check and not a tautology: it is independent of
    'compute_diffusion_coefficients_from_stability_lengths', and a wrong 'lsm' would have to be
    wrong by less than half an ulp of the product to survive it.
    """
    run = _run_section_3(data_provider, date, backend)
    window = (run.columns, run.model_levels)
    velocity_scale = run.turbulent_velocity_scale.asnumpy()[window]

    utils.assert_agrees_with_icon(
        "compute_stability_lengths",
        "lsm, through 'tkvm = lsm*tke'",
        run.stability_length_for_momentum.asnumpy()[window] * velocity_scale,
        run.after.tkvm().asnumpy()[window],
    )
    utils.assert_agrees_with_icon(
        "compute_stability_lengths",
        "lsh, through 'tkvh = MAX(lsh*tke, con_h)'",
        np.maximum(
            run.stability_length_for_scalars.asnumpy()[window] * velocity_scale,
            MOLECULAR_DIFFUSIVITY_FOR_SCALARS,
        ),
        run.after.tkvh().asnumpy()[window],
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_circulation_acceleration_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_3(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_circulation_acceleration",
        "zvari(:,:,0) [CKE gradient]",
        run.circulation_acceleration,
        run.after.effective_gradient(0),
        columns=run.columns,
        levels=run.boundary_levels,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_supersaturation_standard_deviation_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_3(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_supersaturation_standard_deviation",
        "rcld [SDSS]",
        run.supersaturation_standard_deviation,
        run.after.sdss(),
        columns=run.columns,
        levels=run.model_levels,
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_diffusion_coefficients_agrees_with_icon_within_its_gate(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    run = _run_section_3(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_diffusion_coefficients_from_stability_lengths",
        "tkvm",
        run.diffusion_coefficient_for_momentum,
        run.after.tkvm(),
        columns=run.columns,
        levels=run.model_levels,
    )
    utils.assert_agrees_with_icon(
        "compute_diffusion_coefficients_from_stability_lengths",
        "tkvh",
        run.diffusion_coefficient_for_scalars,
        run.after.tkvh(),
        columns=run.columns,
        levels=run.model_levels,
    )


# ---------------------------------------------------------------------- the untouched levels --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_3_leaves_the_surface_half_level_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'tke', 'tkvm', 'tkvh' and 'rcld' at 'ke1' are 'turbtran's, and section 3) must not move them.

    The Fortran stops at 'k_en = kem = ke' and says why for the SDSS (turb_utilities.f90:
    1774-1780): with 'k_en < k_sf' the surface value has already been produced by the call of
    'solve_turb_budgets' from 'turbtran', possibly as a tile aggregation, and recomputing it
    from aggregated inputs would be wrong. The same reasoning covers the other three. This is
    the row a naive translation overruns, because the circulation acceleration next to it does
    run to 'ke1'.
    """
    run = _run_section_3(data_provider, date, backend)
    surface = slice(run.ke, run.ke1)
    computed = {
        "tke": (run.turbulent_velocity_scale, run.after.tke()),
        "tkvm": (run.diffusion_coefficient_for_momentum, run.after.tkvm()),
        "tkvh": (run.diffusion_coefficient_for_scalars, run.after.tkvh()),
        "rcld": (run.supersaturation_standard_deviation, run.after.sdss()),
    }
    for name, (got, want) in computed.items():
        np.testing.assert_array_equal(
            got.asnumpy()[run.columns, surface],
            want.asnumpy()[run.columns, surface],
            err_msg=f"section 3) wrote the surface half level of '{name}'",
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_3_leaves_the_model_top_of_everything_but_the_velocity_scale_alone(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Only 'tke' has an upper boundary condition; the other four keep their entry state at row 0.

    'tke(:,1,ntur) = tke(:,2,ntur)' is the one statement of section 3) that touches the model
    top. 'tkvm', 'tkvh', 'rcld' and the circulation acceleration all start at Fortran 'k = 2',
    so their first row must come out as the section found it.
    """
    run = _run_section_3(data_provider, date, backend)
    top = slice(0, 1)
    computed = {
        "tkvm": (run.diffusion_coefficient_for_momentum, run.after.tkvm()),
        "tkvh": (run.diffusion_coefficient_for_scalars, run.after.tkvh()),
        "rcld": (run.supersaturation_standard_deviation, run.after.sdss()),
        "zvari(:,:,0)": (run.circulation_acceleration, run.after.effective_gradient(0)),
    }
    for name, (got, want) in computed.items():
        np.testing.assert_array_equal(
            got.asnumpy()[run.columns, top],
            want.asnumpy()[run.columns, top],
            err_msg=f"section 3) wrote the model top of '{name}'",
        )


# ------------------------------------------------------------------------- branch coverage ----


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_standard_stability_solution_never_fails_where_it_is_attempted(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """A branch-coverage canary, not a property this port relies on.

    The modified stability solution is taken when the standard one is not, and the standard one
    is guarded by four conjuncts: 'fh2 >= 0' and the positivity of the determinant and the two
    numerators. Raschendorfer asserts in a comment that the last three always hold at
    'fh2 >= 0' (turb_utilities.f90:1607), and over this capture they do -- so the reference data
    can distinguish 'fh2 >= 0' from the full condition in exactly zero places, and a port that
    dropped the three positivity tests would pass every other test in this module.

    The condition is translated as written anyway; this measures the gap instead of hiding it.
    Port spec section 5.4 asks for such gaps to be named. If this ever fails, the gap has closed
    and the assertion should be deleted rather than widened.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="3", date=date)
    window = (slice(before.ivstart(), before.ivend()), slice(1, entry.ke()))
    params = turbulence.TurbulenceParams(turbulence.TurbulenceConfig())

    thermal = before.thermal_forcing().asnumpy()[window]
    mechanical = before.mech_forcing().asnumpy()[window]
    time_scale = before.mixing_length().asnumpy()[window] / after.tke().asnumpy()[window]
    tim2 = time_scale * time_scale
    gh, gm = thermal * tim2, mechanical * tim2
    a11 = params.d_1 + (params.d_5 - params.d_4) * gh
    a12 = params.d_4 * gm
    a21 = (params.d_6 - params.d_4) * gh
    a22 = params.d_2 + params.d_3 * gh + params.d_4 * gm

    attempted = thermal >= 0.0
    assert (a11[attempted] * a22[attempted] - a12[attempted] * a21[attempted] > 0.0).all()
    assert (params.b_h * a22[attempted] - params.b_m * a12[attempted] > 0.0).all()
    assert (params.b_m * a11[attempted] - params.b_h * a21[attempted] > 0.0).all()


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_coherence_scaling_is_never_handed_an_exact_zero(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """A branch-coverage canary for the one place where 'SIGN' and a clamp could disagree.

    'compute_circulation_acceleration' writes Fortran's 'SIGN(1,x)*MIN(|x|,1)' as the clamp of
    'x' to [-1, 1]. The two agree everywhere including at 'x = -0.0', which is why the clamp was
    chosen -- but the reference data cannot show that, because the dimensionless virtual
    potential temperature gradient is never exactly zero anywhere in this capture. Recording it
    keeps the untested case visible.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    before = data_provider.from_savepoint_turbdiff_section(section="2c", date=date)
    window = (slice(before.ivstart(), before.ivend()), slice(1, entry.ke1()))

    gradient = (
        before.thermal_forcing().asnumpy()[window]
        * before.raw_zvari(0).asnumpy()[window]
        / (before.rhon().asnumpy()[window] * constants.GRAV * constants.GRAV)
    )

    assert (gradient != 0.0).all()
