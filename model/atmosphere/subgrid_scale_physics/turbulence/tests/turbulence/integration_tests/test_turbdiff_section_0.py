"""Datatests for section 0) of 'turbdiff': conserved variables, cloud cover, length scales.

The oracle is serialized ICON, not a hand-written reference: each stencil runs on the fields of
'turbdiff-entry' and is compared against 'turbdiff-0-exit', for all four timesteps of
exp.mch_icon-ch2_small.

Section 0) is the first thing 'turbdiff' does and the largest of its fifteen sections. It turns
ICON's model variables into the set the turbulence closure differentiates, diagnoses the
sub-grid cloud cover that the buoyancy factors depend on, puts everything the closure reads at
half levels there, and builds the turbulent master length scale. Seven programs, in the order
the Fortran runs them:

    compute_conserved_variables_and_factors_at_main_levels   adjust_satur_equil, 1st call
    compute_conserved_variables_and_factors_at_the_surface   adjust_satur_equil, 2nd call
    compute_layer_depth                                      dicke
    compute_horizontal_wind_including_the_zero_level         zvari(:,:,u_m), zvari(:,:,v_m)
    compute_half_level_interpolation_weight                  bound_level_interp, 'auxil'
    interpolate_variables_onto_half_levels                   bound_level_interp, seven '%bl'
    compute_turbulent_length_scale                           len_scale

Why the two 'adjust_satur_equil' calls are two programs rather than one 'concat_where' is in
their module docstrings and follows the boundary-row rule of the package README; why the wind
and the length scale ARE merged is in theirs.

WHAT THIS SECTION WRITES, MEASURED
----------------------------------
Among the twelve fields serialized at BOTH boundaries, exactly two differ: 'td_zvari' and
'td_rcld'. 'test_section_0_changes_only_the_conserved_variables_and_the_cloud_cover' asserts
that, and it also asserts that the four tendency arrays the two savepoints spell differently
('td_tketens_in' against 'td_tketens', and the three '_tens' pairs) are untouched.

The measurement cannot reach further, and the reason is worth stating: 'dicke', 'hlp',
'len_scale', 'rhon' and 'zaux' are 'turbdiff' locals that are UNDEFINED when section 0) begins,
so 'serialize_turbdiff_entry' does not write them at all and there is nothing to diff them
against. They are pinned positively instead, by the stencil comparisons below, which cover every
row of every one of them. The remaining nine fields of 'turbdiff-0-exit' -- 'edr', 'frh', 'frm',
'ftm', 'hor_scale', 'layr', 'lays', 'shv' and 'tket_hshr' -- belong to later sections and are
undefined here too; the savepoint reader's own docstring records which section writes each.

BIT-EXACTNESS: WHERE IT STOPS, AND WHY
--------------------------------------
Four of the seven programs are bit-exact, all four dates: 'compute_layer_depth',
'compute_horizontal_wind_including_the_zero_level', 'compute_half_level_interpolation_weight'
and 'compute_turbulent_length_scale'. On 'gtfn_cpu' and 'dace_cpu' that holds for all four; two
of them -- the wind and the length scale -- select a boundary row with 'concat_where' and so
carry 'uses_concat_where', which xfails them on 'embedded' (gt4py 1.1.10 cannot execute
'concat_where' there), so their exactness rests on the two compiled backends only.

The other three are not, and section 0) is the first section of the port that cannot be. It is
the first to evaluate a TRANSCENDENTAL function: Magnus' formula for the saturation vapour
pressure ('zpsat_w') and the Exner factor ('zexner') are an 'EXP' and an 'EXP(LOG())', and
nvhpc's libm -- which produced the reference -- does not round them the way the libm behind
GT4Py does. That is not a tolerance granted for want of an explanation; it is measured:

  * All three backends produce the SAME numbers, bit for bit, so the disagreement is on ICON's
    side of the comparison and not in any one backend's code generation.
  * 'test_the_disagreement_is_two_ulp_of_the_saturation_vapour_pressure' reproduces ICON's
    values EXACTLY -- all 662080 of them, on every date -- by perturbing the saturation vapour
    pressure by at most two units in the last place and changing nothing else. Everything else
    in the translation is bit-identical to the Fortran.
  * 'test_the_conserved_variables_are_bit_exact_where_no_exponential_is_involved' and
    'test_the_interpolation_is_bit_exact_where_its_inputs_are' assert exactness for every
    quantity of the three inexact programs that does NOT pass through an exponential. Those are
    what would catch a real mistranslation hiding behind the tolerance.

The relative errors are one rounding of 'exp' amplified by the near-cancellation 'dq = qt - qs'
inside the cloud diagnosis, which is why the liquid water content is the worst of them at
1.1e-10 while its own inputs agree to 1e-16.
"""

from __future__ import annotations

from typing import NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_conserved_variables_and_factors_at_main_levels import (
    compute_conserved_variables_and_factors_at_main_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_conserved_variables_and_factors_at_the_surface import (
    compute_conserved_variables_and_factors_at_the_surface,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_half_level_interpolation_weight import (
    compute_half_level_interpolation_weight,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_horizontal_wind_including_the_zero_level import (
    compute_horizontal_wind_including_the_zero_level,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_layer_depth import (
    compute_layer_depth,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_turbulent_length_scale import (
    compute_turbulent_length_scale,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.interpolate_variables_onto_half_levels import (
    interpolate_variables_onto_half_levels,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.thermodynamic_functions import (
    ThermoConstants,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: The two storage slots section 0) changes among the fields serialized at both boundaries.
SECTION_0_CHANGED_SLOTS = frozenset({"td_rcld", "td_zvari"})

#: The four arrays 'turbdiff' receives as 'INTENT(INOUT)' and the two hooks therefore serialize
#: under different names. They are compared explicitly because the intersection of the two field
#: lists cannot see them.
RENAMED_ACROSS_THE_BOUNDARY = (
    ("td_tketens_in", "td_tketens"),
    ("td_u_tens_in", "td_u_tens"),
    ("td_v_tens_in", "td_v_tens"),
    ("td_t_tens_in", "td_t_tens"),
)

#: 'zvari' component indices (mo_turbdiff_config.f90:62-77), as the reader takes them.
PRESSURE, U_M, V_M, TET_L, H2O_G, LIQ = 0, 1, 2, 3, 4, 5


#: The configuration parameters section 0) reads. 'q_crit' is the only one that
#: exp.mch_icon-ch2_small sets in 'turbdiff_nml' ("q_crit = 2.0 ! critical value for normalized
#: supersaturation"); the other five are the compiled-in defaults of mo_turbdiff_config.f90 and
#: match 'TurbulenceConfig'. They are spelled out here rather than read from a config object so
#: that what the reference run used is visible next to the assertions it backs.
C_SCLD = 1.0  # cloud-cover shape factor of the moist correction        (:254)
Q_CRIT = 2.0  # critical normalized super-saturation           turbdiff_nml of the experiment
CLC_DIAG = 0.5  # cloud cover at saturation                              (:250)
EPSI = 1.0e-6  # relative limit of accuracy                              (:152)
AKT = 0.4  # von Karman constant                                     (:230)
LEN_MIN = 1.0e-6  # minimal turbulent length scale [m]                     (:222)


class Section0(NamedTuple):
    """One timestep of section 0): the two savepoints, the bounds, and every output."""

    entry: sb.IconTurbdiffEntrySavepoint
    after: sb.IconTurbdiffSectionSavepoint
    nlev: int
    nlevp1: int
    #: Half-open range of columns 'turbdiff' actually computed; every comparison is masked with it.
    columns: slice

    #: 'zvari(:,:,tet_l)', 'zvari(:,:,h2o_g)' and 'zvari(:,:,liq)' after both calls of
    #: 'adjust_satur_equil'. Main levels from the first call, the zero level from the second.
    liquid_water_potential_temperature: gtx.Field
    total_water: gtx.Field
    liquid_water: gtx.Field
    #: The five storages the interpolation then overwrites on rows 1..nlev-1, before it runs.
    cloud_cover_on_main_levels: gtx.Field
    dqsat_dt_on_main_levels: gtx.Field
    buoyancy_factor_tet_l_on_main_levels: gtx.Field
    buoyancy_factor_h2o_g_on_main_levels: gtx.Field
    specific_heat_ratio: gtx.Field
    #: ... and after it.
    cloud_cover: gtx.Field
    exner_factor: gtx.Field
    dqsat_dt: gtx.Field
    buoyancy_factor_tet_l: gtx.Field
    buoyancy_factor_h2o_g: gtx.Field
    half_level_pressure: gtx.Field
    air_density: gtx.Field
    #: The two geometric quantities.
    layer_depth: gtx.Field
    interpolation_weight: gtx.Field


def _run_the_thermodynamics(data_provider, date: str, backend) -> Section0:
    """Run the five programs of section 0) that contain no 'concat_where'.

    The wind and the length scale are left out on purpose: they select their boundary row with
    'concat_where', which the embedded backend cannot execute, and running them here would make
    every test in this module xfail there instead of only their own two.

    The two inputs that come from another stencil of this section -- the interpolation weight
    'hlp' and, for the surface call, the main-level values one row up -- are taken from the EXIT
    savepoint rather than from the programs that produce them, so that a defect there cannot
    travel into these comparisons. The four MAIN-LEVEL inputs of the interpolation that the exit
    savepoint no longer holds (it holds their interpolated values) have to come from the
    main-level program; that one coupling is unavoidable and is what
    'test_the_interpolation_is_bit_exact_where_its_inputs_are' exists to bound.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    nlev, nlevp1 = entry.ke(), entry.ke1()
    ivstart, ivend = entry.ivstart(), entry.ivend()

    # 'zvari(:,:,{tet_l,h2o_g,liq})' start as their own entry state so that the rows section 0)
    # does not write come out unchanged. 'zaux' and 'rhon' have no entry state -- they are
    # 'turbdiff' locals the entry hook does not serialize -- so they start as NaN.
    liquid_water_potential_temperature = utils.copy_of(entry.zvari(TET_L), backend)
    total_water = utils.copy_of(entry.zvari(H2O_G), backend)
    liquid_water = utils.copy_of(entry.zvari(LIQ), backend)
    cloud_cover = utils.copy_of(entry.rcld(), backend)
    specific_heat_ratio = utils.nan_like(entry.rcld(), backend)
    dqsat_dt = utils.nan_like(entry.rcld(), backend)
    buoyancy_factor_tet_l = utils.nan_like(entry.rcld(), backend)
    buoyancy_factor_h2o_g = utils.nan_like(entry.rcld(), backend)
    exner_factor = utils.nan_like(entry.rcld(), backend)
    air_density = utils.nan_like(entry.rcld(), backend)

    # Fortran 'k_st=1, k_en=ke' over one-based main levels.
    compute_conserved_variables_and_factors_at_main_levels.with_backend(backend)(
        temperature=entry.t(),
        specific_humidity=entry.qv(),
        cloud_water=entry.qc(),
        pressure=entry.prs(),
        exner_factor=entry.epr(),
        supersaturation_deviation=entry.rcld(),
        cloud_cover_shape_factor=C_SCLD,
        critical_normalized_supersaturation=Q_CRIT,
        cloud_cover_at_saturation=CLC_DIAG,
        relative_accuracy_limit=EPSI,
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        cloud_cover=cloud_cover,
        specific_heat_ratio=specific_heat_ratio,
        dqsat_dt=dqsat_dt,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev),
        offset_provider={},
    )
    # Fortran 'k_st=ke1, k_en=ke1': the single row of the zero level. The four surface values it
    # starts from are what 'turb_setup' put into the surface row of 'zvari' before this section
    # began -- the only rows of that array that are defined at 'turbdiff-entry'.
    compute_conserved_variables_and_factors_at_the_surface.with_backend(backend)(
        surface_pressure=utils.surface_row(entry.zvari(PRESSURE), nlev, backend),
        surface_temperature=utils.surface_row(entry.zvari(TET_L), nlev, backend),
        surface_specific_humidity=utils.surface_row(entry.zvari(H2O_G), nlev, backend),
        surface_liquid_water=utils.surface_row(entry.zvari(LIQ), nlev, backend),
        liquid_water_potential_temperature_above=utils.surface_row(
            after.conserved_variable(TET_L), nlev - 1, backend
        ),
        total_water_above=utils.surface_row(after.conserved_variable(H2O_G), nlev - 1, backend),
        laminar_reduction_factor_for_scalars=entry.tfh(),
        supersaturation_deviation=entry.rcld(),
        cloud_cover_shape_factor=C_SCLD,
        critical_normalized_supersaturation=Q_CRIT,
        cloud_cover_at_saturation=CLC_DIAG,
        relative_accuracy_limit=EPSI,
        exner_factor=exner_factor,
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        cloud_cover=cloud_cover,
        air_density=air_density,
        specific_heat_ratio=specific_heat_ratio,
        dqsat_dt=dqsat_dt,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(nlev),
        vertical_end=gtx.int32(nlevp1),
        offset_provider={},
    )

    # Fortran 'DO k=1,ke'.
    layer_depth = utils.nan_like(entry.rcld(), backend)
    compute_layer_depth.with_backend(backend)(
        half_level_height=entry.hhl(),
        layer_depth=layer_depth,
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev),
        offset_provider={dims.Koff.value: dims.KDim},
    )
    # Fortran 'bound_level_interp(..., k_st=2, k_en=ke, ...)'.
    interpolation_weight = utils.nan_like(entry.rcld(), backend)
    compute_half_level_interpolation_weight.with_backend(backend)(
        layer_pressure_thickness=entry.dp0(),
        interpolation_weight=interpolation_weight,
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    # The seven interpolated storages start as what the two 'adjust_satur_equil' calls left in
    # them, which is what makes rows 0 and nlev come out as those calls wrote them -- the
    # Fortran interpolates in place over rows 1..nlev-1 only. The half-level pressure has no
    # main-level predecessor in its own storage and starts from its entry state, whose surface
    # row 'turb_setup' filled with 'ps' and whose model top is untouched memory.
    cloud_cover_on_half_levels = utils.copy_of(cloud_cover, backend)
    exner_factor_on_half_levels = utils.copy_of(exner_factor, backend)
    dqsat_dt_on_half_levels = utils.copy_of(dqsat_dt, backend)
    buoyancy_factor_tet_l_on_half_levels = utils.copy_of(buoyancy_factor_tet_l, backend)
    buoyancy_factor_h2o_g_on_half_levels = utils.copy_of(buoyancy_factor_h2o_g, backend)
    half_level_pressure = utils.copy_of(entry.zvari(PRESSURE), backend)
    air_density_on_half_levels = utils.copy_of(air_density, backend)
    interpolate_variables_onto_half_levels.with_backend(backend)(
        cloud_cover=cloud_cover,
        exner_factor=entry.epr(),
        dqsat_dt=dqsat_dt,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g,
        pressure=entry.prs(),
        air_density=entry.rhoh(),
        interpolation_weight=after.hlp(),
        cloud_cover_on_half_levels=cloud_cover_on_half_levels,
        exner_factor_on_half_levels=exner_factor_on_half_levels,
        dqsat_dt_on_half_levels=dqsat_dt_on_half_levels,
        buoyancy_factor_tet_l_on_half_levels=buoyancy_factor_tet_l_on_half_levels,
        buoyancy_factor_h2o_g_on_half_levels=buoyancy_factor_h2o_g_on_half_levels,
        pressure_on_half_levels=half_level_pressure,
        air_density_on_half_levels=air_density_on_half_levels,
        horizontal_start=gtx.int32(ivstart),
        horizontal_end=gtx.int32(ivend),
        vertical_start=gtx.int32(1),
        vertical_end=gtx.int32(nlev),
        offset_provider={dims.Koff.value: dims.KDim},
    )
    return Section0(
        entry=entry,
        after=after,
        nlev=nlev,
        nlevp1=nlevp1,
        columns=slice(ivstart, ivend),
        liquid_water_potential_temperature=liquid_water_potential_temperature,
        total_water=total_water,
        liquid_water=liquid_water,
        cloud_cover_on_main_levels=cloud_cover,
        dqsat_dt_on_main_levels=dqsat_dt,
        buoyancy_factor_tet_l_on_main_levels=buoyancy_factor_tet_l,
        buoyancy_factor_h2o_g_on_main_levels=buoyancy_factor_h2o_g,
        specific_heat_ratio=specific_heat_ratio,
        cloud_cover=cloud_cover_on_half_levels,
        exner_factor=exner_factor_on_half_levels,
        dqsat_dt=dqsat_dt_on_half_levels,
        buoyancy_factor_tet_l=buoyancy_factor_tet_l_on_half_levels,
        buoyancy_factor_h2o_g=buoyancy_factor_h2o_g_on_half_levels,
        half_level_pressure=half_level_pressure,
        air_density=air_density_on_half_levels,
        layer_depth=layer_depth,
        interpolation_weight=interpolation_weight,
    )


# ------------------------------------------------------------------------- the output set ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_section_0_changes_only_the_conserved_variables_and_the_cloud_cover(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Of everything both savepoints hold, only 'zvari' and 'rcld' differ.

    'utils.fields_that_changed' cannot be used here: it reads every name of the FIRST savepoint
    at the second, and 'turbdiff-entry' and 'turbdiff-0-exit' are two different hooks with two
    different field lists -- the entry one carries the scheme's arguments, the section one its
    working arrays. Only twelve names are common to both. The comparison is done over that
    intersection, plus the four 'INTENT(INOUT)' arrays the two hooks spell differently.

    What this can and cannot see is discussed in the module docstring.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    window = slice(entry.ivstart(), entry.ivend())

    def differs(before_name: str, after_name: str) -> bool:
        before = np.asarray(data_provider.serializer.read(before_name, entry.savepoint))
        exit_ = np.asarray(data_provider.serializer.read(after_name, after.savepoint))
        masked = window if before.ndim >= 2 and before.shape[0] > entry.ivend() else slice(None)
        return not np.array_equal(before[masked], exit_[masked])

    common = set(data_provider.serializer.fields_at_savepoint(entry.savepoint)) & set(
        data_provider.serializer.fields_at_savepoint(after.savepoint)
    )
    assert {name for name in common if differs(name, name)} == SECTION_0_CHANGED_SLOTS
    assert not [
        before for before, after_ in RENAMED_ACROSS_THE_BOUNDARY if differs(before, after_)
    ], "section 0) does not touch the tendency arrays it is handed"


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_roughness_layer_loop_is_dead(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """'kcm = ke1', so the canopy correction of the length scale never runs.

    'compute_turbulent_length_scale' accumulates from 'ke' downward and omits the loop
    'DO k=ke,kcm,-1' entirely, which is only correct while 'kcm > ke'. Raschendorfer says as
    much (turb_diffusion.f90:1105-1107, "Up to now it is kcm = ke+1 and the next vertical loop
    will not be executed!!"), but the value is a property of the run, so it is asserted against
    the run. Note that it is 'ke1' and not, as the savepoint reader's docstring for 'kcm()'
    says, 'ke1 + 1'.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)

    assert entry.kcm() == entry.ke1() > entry.ke()


# ----------------------------------------------------------------- the bit-exact programs ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_layer_depth_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The main-level depths, and nothing at the surface row."""
    run = _run_the_thermodynamics(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_layer_depth",
        "dicke",
        run.layer_depth,
        run.after.layer_depth(),
        columns=run.columns,
        levels=slice(0, run.nlev),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_half_level_interpolation_weight_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'hlp' on rows 1..nlev-1; the model top and the surface are not written."""
    run = _run_the_thermodynamics(data_provider, date, backend)

    utils.assert_agrees_with_icon(
        "compute_half_level_interpolation_weight",
        "hlp",
        run.interpolation_weight,
        run.after.hlp(),
        columns=run.columns,
        levels=slice(1, run.nlev),
    )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_horizontal_wind_including_the_zero_level_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """One program reproduces both wind components on all 'nlev + 1' rows.

    The output is allocated as a COPY OF THE ENTRY STATE of 'zvari' and compared over the whole
    column, so the comparison distinguishes the two ways the row selection can be wrong: the
    zero level taking the un-reduced wind, and the lowest main level taking the reduced one.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    nlev = entry.ke()
    columns = slice(entry.ivstart(), entry.ivend())

    zonal = utils.copy_of(entry.zvari(U_M), backend)
    meridional = utils.copy_of(entry.zvari(V_M), backend)
    compute_horizontal_wind_including_the_zero_level.with_backend(backend)(
        zonal_wind=entry.u(),
        meridional_wind=entry.v(),
        laminar_reduction_factor_for_momentum=entry.tfm(),
        nlev=gtx.int32(nlev),
        zonal_wind_on_conserved_variable_levels=zonal,
        meridional_wind_on_conserved_variable_levels=meridional,
        horizontal_start=gtx.int32(entry.ivstart()),
        horizontal_end=gtx.int32(entry.ivend()),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(entry.ke1()),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    for quantity, computed, reference in (
        ("zvari(:,:,u_m)", zonal, after.conserved_variable(U_M)),
        ("zvari(:,:,v_m)", meridional, after.conserved_variable(V_M)),
    ):
        utils.assert_agrees_with_icon(
            "compute_horizontal_wind_including_the_zero_level",
            quantity,
            computed,
            reference,
            columns=columns,
            levels=slice(0, entry.ke1()),
        )


@pytest.mark.datatest
@pytest.mark.uses_concat_where
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_turbulent_length_scale_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The whole length-scale profile, roughness length included.

    The layer depths come from the EXIT savepoint rather than from 'compute_layer_depth', so a
    defect there cannot travel into this comparison.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    columns = slice(entry.ivstart(), entry.ivend())

    length_scale = utils.nan_like(entry.rcld(), backend)
    compute_turbulent_length_scale.with_backend(backend)(
        layer_depth=after.layer_depth(),
        roughness_length_times_gravity=entry.gz0(),
        horizontal_length_scale_limit=entry.l_scal(),
        nlev=gtx.int32(entry.ke()),
        von_karman_constant=AKT,
        minimal_length_scale=LEN_MIN,
        turbulent_length_scale=length_scale,
        horizontal_start=gtx.int32(entry.ivstart()),
        horizontal_end=gtx.int32(entry.ivend()),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(entry.ke1()),
        offset_provider={dims.Koff.value: dims.KDim},
    )

    utils.assert_agrees_with_icon(
        "compute_turbulent_length_scale",
        "len_scale",
        length_scale,
        after.mixing_length(),
        columns=columns,
        levels=slice(0, entry.ke1()),
    )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_length_scale_is_accumulated_and_not_telescoped(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The cumulative sum of the layer depths is not the height difference it sums to.

    'dicke(k) = hhl(k) - hhl(k+1)', so summing it from the surface upward gives 'hhl(k) -
    hhl(ke1)' exactly in real arithmetic, and 'compute_turbulent_length_scale' could then be a
    one-line field operator instead of a scan. In floating point the two differ, and the port
    reproduces the accumulation because the reference did it that way. This measures the
    difference so that the scan is justified by a number and not by caution.
    """
    entry = data_provider.from_savepoint_turbdiff_entry(date=date)
    after = data_provider.from_savepoint_turbdiff_section(section="0", date=date)
    columns = slice(entry.ivstart(), entry.ivend())
    nlev = entry.ke()

    depth = after.layer_depth().asnumpy()[columns, :nlev]
    roughness = entry.gz0().asnumpy()[columns] * float(ThermoConstants.EDGRAV)
    accumulated = np.cumsum(depth[:, ::-1], axis=1)[:, ::-1] + roughness[:, np.newaxis]
    telescoped = (
        entry.hhl().asnumpy()[columns, :nlev]
        - entry.hhl().asnumpy()[columns, nlev, np.newaxis]
        + roughness[:, np.newaxis]
    )

    assert not np.array_equal(accumulated, telescoped), (
        "the telescoped height happens to be bit-identical to the accumulation in this capture; "
        "the scan is then unnecessary here, though still the faithful translation."
    )


# ------------------------------------------------- the three programs that use 'exp'/'log' ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_conserved_variables_and_factors_at_main_levels_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The first 'adjust_satur_equil' call, on the rows whose values survive the section.

    'tet_l', 'h2o_g' and 'liq' are compared over all main levels: 'zvari' is not in the
    'bound_level_interp' list, so what this program writes there is what section 0) ends with.
    The cloud cover and the three 'zaux' factors are compared at ROW 0 ONLY, because the
    interpolation overwrites rows 1..nlev-1 of their storages in place and the exit savepoint
    holds the interpolated values there; the rest of their profile is checked by
    'test_interpolate_variables_onto_half_levels_agrees_with_icon'. 'r_cpd' is the constant 1 on
    every main level and is compared over all of them.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    name = "compute_conserved_variables_and_factors_at_main_levels"
    main_levels = slice(0, run.nlev)
    model_top = slice(0, 1)

    for quantity, computed, reference, levels in (
        ("zvari(:,:,tet_l)", run.liquid_water_potential_temperature, TET_L, main_levels),
        ("zvari(:,:,h2o_g)", run.total_water, H2O_G, main_levels),
        ("zvari(:,:,liq)", run.liquid_water, LIQ, main_levels),
    ):
        utils.assert_agrees_with_icon(
            name,
            quantity,
            computed,
            run.after.conserved_variable(reference),
            columns=run.columns,
            levels=levels,
        )
    for quantity, computed, reference, levels in (
        ("rcld(:,1)", run.cloud_cover_on_main_levels, run.after.cloud_cover(), model_top),
        ("zaux(:,:,2) [r_cpd]", run.specific_heat_ratio, run.after.r_cpd(), main_levels),
        ("zaux(:,1,3) [dQs/dT]", run.dqsat_dt_on_main_levels, run.after.dqsat_dt(), model_top),
        ("zaux(:,1,4) [g_tet]", run.buoyancy_factor_tet_l_on_main_levels, run.after.g_tet_l(), model_top),
        ("zaux(:,1,5) [g_h2o]", run.buoyancy_factor_h2o_g_on_main_levels, run.after.g_h2o(), model_top),
    ):
        utils.assert_agrees_with_icon(
            name, quantity, computed, reference, columns=run.columns, levels=levels
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_compute_conserved_variables_and_factors_at_the_surface_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The second 'adjust_satur_equil' call: ten quantities on the single row of the zero level.

    'bound_level_interp' does not touch that row, so every one of them can be compared directly
    against the exit savepoint.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    name = "compute_conserved_variables_and_factors_at_the_surface"
    surface = slice(run.nlev, run.nlevp1)

    for quantity, computed, reference in (
        ("zaux(:,ke1,1) [exner]", run.exner_factor, run.after.exner_factor()),
        (
            "zvari(:,ke1,tet_l)",
            run.liquid_water_potential_temperature,
            run.after.conserved_variable(TET_L),
        ),
        ("zvari(:,ke1,h2o_g)", run.total_water, run.after.conserved_variable(H2O_G)),
        ("zvari(:,ke1,liq)", run.liquid_water, run.after.conserved_variable(LIQ)),
        ("rcld(:,ke1)", run.cloud_cover, run.after.cloud_cover()),
        ("rhon(:,ke1)", run.air_density, run.after.rhon()),
        ("zaux(:,ke1,2) [r_cpd]", run.specific_heat_ratio, run.after.r_cpd()),
        ("zaux(:,ke1,3) [dQs/dT]", run.dqsat_dt, run.after.dqsat_dt()),
        ("zaux(:,ke1,4) [g_tet]", run.buoyancy_factor_tet_l, run.after.g_tet_l()),
        ("zaux(:,ke1,5) [g_h2o]", run.buoyancy_factor_h2o_g, run.after.g_h2o()),
    ):
        utils.assert_agrees_with_icon(
            name, quantity, computed, reference, columns=run.columns, levels=surface
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_interpolate_variables_onto_half_levels_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The seven interpolated profiles on rows 1..nlev-1.

    Four of them are compared over their WHOLE column instead, because their storage holds the
    main-level value at row 0 and the zero-level value at row nlev and both must survive the
    interpolation untouched -- which is exactly what the Fortran's 'DO k=ke,2,-1' guarantees and
    what a wrong vertical domain here would destroy. The half-level pressure is compared over
    the whole column for the same reason; its model top is untouched memory on both sides, and
    the output buffer starts as the entry state so that "untouched" is asserted rather than
    skipped. The Exner factor and the density have no defined row 0 at all, so they stop at 1.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    name = "interpolate_variables_onto_half_levels"
    whole = slice(0, run.nlevp1)
    below_the_top = slice(1, run.nlevp1)

    for quantity, computed, reference, levels in (
        ("rcld", run.cloud_cover, run.after.cloud_cover(), whole),
        ("zaux(:,:,3) [dQs/dT]", run.dqsat_dt, run.after.dqsat_dt(), whole),
        ("zaux(:,:,4) [g_tet]", run.buoyancy_factor_tet_l, run.after.g_tet_l(), whole),
        ("zaux(:,:,5) [g_h2o]", run.buoyancy_factor_h2o_g, run.after.g_h2o(), whole),
        ("zvari(:,:,0) [prss]", run.half_level_pressure, run.after.conserved_variable(PRESSURE), whole),
        ("zaux(:,:,1) [exner]", run.exner_factor, run.after.exner_factor(), below_the_top),
        ("rhon", run.air_density, run.after.rhon(), below_the_top),
    ):
        utils.assert_agrees_with_icon(
            name, quantity, computed, reference, columns=run.columns, levels=levels
        )


# ------------------------------------------------------- what the tolerance is NOT hiding ---


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_conserved_variables_are_bit_exact_where_no_exponential_is_involved(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """'tet_l', 'h2o_g' and 'r_cpd' on the main levels agree with ICON bit for bit.

    Asserted without the gate, deliberately. These three are the outputs of the first
    'adjust_satur_equil' call that never pass through 'exp': 'q_h2o = qv + qc',
    'tet_l = (t - lhocp*qc)/epr' and the constant 1. If the gate's tolerance were covering a
    mistranslation rather than a libm difference, it would be visible here first, because these
    share every input with the quantities that do disagree.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    main_levels = slice(0, run.nlev)

    for quantity, computed, reference in (
        ("zvari(:,:,tet_l)", run.liquid_water_potential_temperature, run.after.conserved_variable(TET_L)),
        ("zvari(:,:,h2o_g)", run.total_water, run.after.conserved_variable(H2O_G)),
        ("zaux(:,:,2) [r_cpd]", run.specific_heat_ratio, run.after.r_cpd()),
    ):
        np.testing.assert_array_equal(
            computed.asnumpy()[run.columns, main_levels],
            reference.asnumpy()[run.columns, main_levels],
            err_msg=f"'{quantity}' contains no transcendental function and must be bit-exact",
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_interpolation_is_bit_exact_where_its_inputs_are(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """The three interpolated profiles whose main-level input is an ICON field, not a stencil's.

    The Exner factor, the pressure and the density are interpolated from 'epr', 'prs' and
    'rhoh', which come straight from the entry savepoint, so nothing that reaches them has been
    through an exponential. They are bit-exact, which says that the interpolation itself --
    including the choice of 'bound_level_interp's precomputed-weight branch over 'zbnd_val' --
    is exactly the Fortran's, and that the tolerance the other four need is inherited from their
    inputs and not produced here.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    rows = slice(1, run.nlev)

    for quantity, computed, reference in (
        ("zaux(:,:,1) [exner]", run.exner_factor, run.after.exner_factor()),
        ("zvari(:,:,0) [prss]", run.half_level_pressure, run.after.conserved_variable(PRESSURE)),
        ("rhon", run.air_density, run.after.rhon()),
    ):
        np.testing.assert_array_equal(
            computed.asnumpy()[run.columns, rows],
            reference.asnumpy()[run.columns, rows],
            err_msg=f"'{quantity}' is interpolated from an ICON field and must be bit-exact",
        )


@pytest.mark.cpu_only
@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_disagreement_is_two_ulp_of_the_saturation_vapour_pressure(
    date: str, *, data_provider: sb.IconSerialDataProvider, backend
) -> None:
    """Every value ICON produced is reproduced exactly by perturbing one 'exp' by <= 2 ULP.

    This is what turns "the gate is a tolerance" into "the gate is a libm". The cloud diagnosis
    is re-evaluated in numpy from the same inputs the stencil got, with the saturation vapour
    pressure -- the single 'EXP' in 'zpsat_w' -- moved by -2, -1, 0, +1 or +2 units in the last
    place and NOTHING ELSE changed. Every one of the 662080 liquid-water values of the reference
    is then matched bit for bit by one of the five.

    So the translation contains no second source of disagreement: not a reassociation, not a
    reciprocal, not a wrong constant. What is left is that nvhpc's 'EXP' and the one behind
    GT4Py round differently, which no arrangement of this code can fix.

    A numpy re-implementation is normally the wrong oracle for this port -- it restates the
    translation and agrees with it for the same reason it agrees with itself -- and it is used
    here for the opposite purpose: not to check the value, which the gated test above does, but
    to identify the mechanism of a KNOWN disagreement. The same argument is made in section 1b)
    for FMA contraction.

    'cpu_only' because the argument is host-libm-bound, not because the port is. The test first
    asserts that the unperturbed numpy re-evaluation reproduces what the BACKEND produced, bit
    for bit -- and that holds only while numpy and the backend call the same 'exp'. On the GPU
    backends they do not: numpy uses glibc on the host, the stencil uses CUDA's libm on the
    device, and the two disagree on 2342 of 662080 values by up to 5.9e-12 (measured, job
    833847, both 'dace_gpu' and 'gtfn_gpu', all four dates). That is the same order as the
    ICON-vs-port difference this test exists to explain, so on the GPU the five perturbations
    would be chasing two roundings at once and the argument would prove nothing.

    Nothing is lost by skipping it there: the GATES for this section run on the GPU backends and
    pass. What does not run is the explanation, and the explanation is a statement about nvhpc's
    'exp' versus the host's, which the host backends establish. Section 1b's canary needs no
    such restriction because it contains no transcendental -- numpy and CUDA agree bit for bit
    on 'a*b + c*d'.
    """
    run = _run_the_thermodynamics(data_provider, date, backend)
    entry, columns, nlev = run.entry, run.columns, run.nlev
    main = slice(0, nlev)

    lhocp = float(ThermoConstants.LHOCP)
    total_water = entry.qv().asnumpy()[columns, main] + entry.qc().asnumpy()[columns, main]
    liquid_water_temperature = (
        entry.t().asnumpy()[columns, main] - lhocp * entry.qc().asnumpy()[columns, main]
    )
    dry_air_pressure = (
        (1.0 - total_water)
        / (1.0 + float(ThermoConstants.RVD_M_O) * total_water)
        * entry.prs().asnumpy()[columns, main]
    )
    deviation_in = entry.rcld().asnumpy()[columns, main]

    def liquid_water_from(vapour_pressure: np.ndarray) -> np.ndarray:
        """'turb_cloud' at 'icldtyp = 2', taking the saturation vapour pressure as given."""
        saturating_cover = min(CLC_DIAG, 1.0 - EPSI)
        q_max = Q_CRIT * (1.0 / saturating_cover - 1.0)
        q_inv = 1.0 / (q_max + Q_CRIT)
        vapour = float(ThermoConstants.RDV) * vapour_pressure
        saturation = vapour / (dry_air_pressure + vapour)
        denominator = liquid_water_temperature - float(ThermoConstants.C4LES)
        # 'gam = z1/(z1 + lhocp*zdqsdt(tl, qs))' (turb_utilities.f90:2155), with 'zdqsdt' the
        # statement function of :3434-3443. The two multiplications must not be flattened into
        # one product: evaluating the statement function first and multiplying by 'lhocp'
        # afterwards is one rounding different, and it is what '_dqsat_dt' does inside the
        # stencil. Written flat, this re-evaluation disagrees with the stencil in ~11500 of the
        # 662080 values by one ULP and the two-ULP argument below fails on its own oracle.
        dqsat_dt = (
            float(ThermoConstants.C5LES)
            * (1.0 - saturation)
            * saturation
            / (denominator * denominator)
        )
        slope = 1.0 / (1.0 + lhocp * dqsat_dt)
        supersaturation = total_water - saturation
        deviation = np.minimum(float(ThermoConstants.RSIG_MAX) * saturation, deviation_in)
        saturated_content = np.minimum(total_water, deviation * q_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(
                deviation <= 0.0,
                np.where(supersaturation <= 0.0, -Q_CRIT, q_max),
                supersaturation / deviation,
            )
        cover = np.minimum(1.0, np.maximum(0.0, (normalized + Q_CRIT) * q_inv))
        return slope * np.where(normalized >= q_max, supersaturation, saturated_content) * (
            cover * cover
        )

    magnus = float(ThermoConstants.C1ES) * np.exp(
        float(ThermoConstants.C3LES)
        * (liquid_water_temperature - float(ThermoConstants.B3))
        / (liquid_water_temperature - float(ThermoConstants.C4LES))
    )
    reference = run.after.conserved_variable(LIQ).asnumpy()[columns, main]

    # The re-evaluation is only evidence about the STENCIL if it is the same translation, so
    # that is asserted first: unperturbed, it must reproduce what the backend produced, bit for
    # bit. Everything after this line is therefore a statement about the stencil too.
    np.testing.assert_array_equal(
        liquid_water_from(magnus),
        run.liquid_water.asnumpy()[columns, main],
        err_msg="the numpy re-evaluation below is not the same translation as the stencil",
    )

    reproduced = np.zeros_like(reference, dtype=bool)
    for shift in (-2, -1, 0, 1, 2):
        perturbed = magnus.copy()
        for _ in range(abs(shift)):
            perturbed = np.nextafter(perturbed, -np.inf if shift < 0 else np.inf)
        reproduced |= liquid_water_from(perturbed) == reference

    assert reproduced.all(), (
        f"{np.count_nonzero(~reproduced)} of {reproduced.size} liquid-water values are not "
        "explained by a two-ULP move of the saturation vapour pressure, so section 0) has a "
        "second source of disagreement with ICON and the gate is hiding it."
    )
