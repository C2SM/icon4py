# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Datatests for 'vertdiff': the implicit vertical diffusion of the first-order variables.

The oracle is a real ICON run: 'vertdiff-entry' supplies the inputs, 'vertdiff-exit' the
expected outputs, for the four timesteps that 'exp.mch_icon-ch2_small' serializes.

'vertdiff' (turb_vertdiff.f90:116-937) is one stage with one savepoint pair -- there are no
intermediate section boundaries the way 'turbdiff' has them -- and it is a loop over five
variables in two variable types, each of which calls 'vert_grad_diff' (turb_utilities.f90:2223),
which calls 'prep_impl_vert_diff' (:2690) and 'calc_impl_vert_diff' (:2865). Eleven programs
since the stencil merge, grouped by what they depend on:

    ONCE                                                        Fortran
     1 prepare_the_vertical_diffusion_matrix, six statements:
         rhon(:,ke1), eprs(:,ke1)                               turb_vertdiff.f90:536-542
         disc_mom                                               vert_grad_diff:2438-2455
         diff_dep [interior]                                    vert_grad_diff:2449-2453
         zvari(:,ke1,tem), zvari(:,ke1,vap)                     turb_vertdiff.f90:614-634
    ONCE PER VARIABLE TYPE (momentum: u,v with tkvm/tvm; scalars: t,qv,qc with tkvh/tvh)
     2 prep_impl_vert_diff, six statements:
         expl_mom [interior]                                    vert_grad_diff:2461-2469
         expl_mom(:,k_sf), diff_dep(:,k_sf)                     vert_grad_diff:2471-2478
         impl_mom                                               prep_impl_vert_diff:2764-2776
         expl_mom -= impl_mom, in place                         prep_impl_vert_diff:2778-2786
         invs_mom, a scan            (REUSED operator)          prep_impl_vert_diff:2830-2849
     3 invert_diffusion_momentum_at_the_surface_flux_level      prep_impl_vert_diff:2850-2858
     4 compute_diffusion_inversion_factor          (REUSED)     prep_impl_vert_diff:2842,2854
    ONCE PER VARIABLE
     5 compute_current_profile / ...potential_temperature...    turb_vertdiff.f90:646-694
     6 compute_surface_profile_value_from_flux_gradient         vert_grad_diff:2493-2503
     7 compute_explicit_flux_density                            calc_impl_vert_diff:2951-2959
     8 add_implicit_surface_flux_to_the_explicit_flux_density   calc_impl_vert_diff:2961-2973
     9 compute_diffusion_right_hand_side                        calc_impl_vert_diff:2975-2991
    10 solve_vertical_diffusion_equation                        calc_impl_vert_diff:3024-3052
    11 compute_and_apply_[potential_temperature_]diffusion_tendency
                                                    vert_grad_diff:2661-2670 + :773-799

The two surface gradients used to be a program of their own, run in the middle of the variable
loop where the Fortran runs them; they are statements of program 1 now, because the two 'zvari'
components they write are distinct and nothing between that point and each variable's use of its
row touches either. The elimination scan inside program 2 and program 4 are section 9)'s, unchanged: 'prep_impl_vert_diff' is the
same subroutine for the TKE and for the model variables, and those two are the parts of it whose
names and arguments say nothing about which. Program 3 is the loop section 9)'s docstring says it
does not translate because 'm = 1' makes it empty -- 'vertdiff' is where 'm = 2' occurs.

WHAT THE CONFIGURATION SWITCHES OFF
-----------------------------------
'vertdiff' has far more code than the list above. What the ported call site
(mo_nwp_turbdiff_interface.f90, through SUB 'nwp_turbdiff') leaves unreachable, measured from
the entry savepoint by 'test_vertdiff_runs_in_the_configuration_this_port_assumes':

    itndcon = 0        no explicit-tendency handling at all: the three 'itndcon' blocks of
                       'vert_grad_diff' and the 'old_prof'/'rhs_prof' selection collapse
    ldogrdcor = F      'ldoexpcor' and 'ldocirflx' are both false, so 'ncorr = 6 > mcorr = 5'
                       and 'igrdcon = 0' for every variable: no gradient correction, and
                       'zvari' plays no part in the solve
    l3dflxout = F      'leff_flux' is false for EVERY variable, because 'IF (n.LE.nmvar)
                       leff_flux = l3dflxout' overrides everything before it and
                       'ndiff = nmvar = 5'. The effective-flux integration at
                       calc_impl_vert_diff:3072-3090 never runs -- one scan fewer.
    lsfluse = T and tdc%lsflcnd = T
                       '.NOT.(lsfluse .AND. lsflcnd)' is false, so the 'shfl_s'/'qvfl_s' update
                       at turb_vertdiff.f90:850-895 is dead; both come out byte-identical
    ndtr = 0           no passive tracers. The 'ptr(:)' fan-out is empty and NOT serializable
    kcm = ke1          canopy off: 'vert_grad_diff's roughness-layer volume correction is an
                       empty loop, and so is the 'leff_flux = .TRUE.' it would force
    kstart_cloud = 1   cloud water is diffused from the model top like every other scalar
    lprecnd = F        no preconditioning; 'ldynimp = F' selects the precalculated implicit
                       weights; 'lfreeslip = F'; 'ilow_def_cond = 2'

TWO ORACLES, AND WHY THE SECOND ONE IS TRUSTWORTHY
--------------------------------------------------
The exit savepoint holds the five tendencies, 'rhon', 'zvari' and the SCALAR type's solver
workspace -- 'vertdiff' reuses one set of arrays for all five variables, so what survives is
the last variable of the last type. It holds nothing at all for the momentum type, and nothing
for the four profiles that are not cloud water.

'_reference' below is a numpy transcription of the Fortran. It is validated against every
quantity the savepoint does hold, bit for bit, by
'test_the_reference_reproduces_every_serialized_quantity' -- twelve of them, on all four dates
-- and it is only because it passes that test that its unserialized intermediates are used as
expected values here. Its one un-reproducible input is the surface Exner factor, the single
'EXP(LOG())' of the stage, where numpy's libm differs from nvhpc's by one ulp on 51..67 of the
8276 computed columns; the validation supplies ICON's value and
'test_the_surface_exner_factor_is_the_only_transcendental' measures both the difference and how
far it travels.

WHAT IS RUN, AND WHY IT IS THE WHOLE CHAIN
------------------------------------------
The ported sections of 'turbdiff' hand each program ICON's own inputs so that a failure names
one translation. That is not available here for most of this stage -- the savepoint has no
inputs for anything past the scalar workspace -- so '_run_vertdiff' runs the whole chain from
the entry savepoint and every intermediate is compared. With thirty comparisons along one chain
the first one that fails still names the program that produced it.

Every workspace buffer is allocated with 'nan_like'. These are '!$ACC CREATE' locals of
'vertdiff' with no entry state to copy, so there is nothing to poison them with except NaN --
which is what the copy-of-entry convention is for elsewhere, and which here means a row the port
fails to write cannot pass by accident. The five tendency buffers get the same treatment even
though they DO have an entry state, because a variable whose diffusion increment is zero (a dry
column's 'qc', routinely) would otherwise pass a copy-based comparison unwritten. 'rhon' is the
one field allocated as a copy: rows above the surface must come out untouched, and its surface
row is measurably different at the two savepoints, so the copy is not blind there.

NO 'concat_where' IS USED. The one row this stage treats differently -- the top of the
right-hand side, where the Fortran omits the outgoing-flux term because there is no flux level
above it -- is handled by writing that flux level as an explicit zero in
'compute_explicit_flux_density' instead. 'x - 0.0' is 'x' for every double, so the interior
expression then covers the whole range bit-exactly, and no program of this stage loses the embedded
backend. That is a deliberate departure from the package README's boundary-row rule and the
reasoning is in the two stencils' docstrings.

SEVEN ROWS OF ICON'S WORKSPACE ARE NOT THIS PORT'S TO REPRODUCE
---------------------------------------------------------------
'test_the_unwritten_workspace_rows_are_leftovers_not_results' measures them. 'expl_mom(:,0)' is
the depth of the top layer, because the Fortran parks the layer depth in that array before
overwriting rows 1 onward; 'disc_mom(:,nlev)', 'diff_dep(:,0)', 'invs_mom(:,nlev)',
'invs_fac(:,0)', 'invs_fac(:,nlev)' and 'dif_tend(:,nlev)' are never written by 'vertdiff' at
all. Those rows are excluded from the comparisons, and the test above is what keeps the
exclusion honest.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.add_implicit_surface_flux_to_the_explicit_flux_density import (
    add_implicit_surface_flux_to_the_explicit_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_and_apply_diffusion_tendency import (
    compute_and_apply_diffusion_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_and_apply_potential_temperature_diffusion_tendency import (
    compute_and_apply_potential_temperature_diffusion_tendency,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_current_potential_temperature_profile import (
    compute_current_potential_temperature_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_current_profile import (
    compute_current_profile,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_inversion_factor import (
    compute_diffusion_inversion_factor,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_diffusion_right_hand_side import (
    compute_diffusion_right_hand_side,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_explicit_flux_density import (
    compute_explicit_flux_density,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.compute_surface_profile_value_from_flux_gradient import (
    compute_surface_profile_value_from_flux_gradient,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.invert_diffusion_momentum_at_the_surface_flux_level import (
    invert_diffusion_momentum_at_the_surface_flux_level,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prep_impl_vert_diff import (
    prep_impl_vert_diff,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.prepare_the_vertical_diffusion_matrix import (
    prepare_the_vertical_diffusion_matrix,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.stencils.solve_vertical_diffusion_equation import (
    solve_vertical_diffusion_equation,
)
from icon4py.model.atmosphere.subgrid_scale_physics.turbulence.turbulence import TurbulenceConfig
from icon4py.model.common import constants, dimension as dims
from icon4py.model.testing import serialbox as sb

from .. import utils
from ..fixtures import *  # noqa: F403


#: Every stencil here differences or shifts across neighbouring vertical levels.
_KOFF = {dims.Koff.value: dims.KDim}

#: The five first-order variables, in the order 'vertdiff' loops over them, with the Fortran
#: index of the 'zvari' component each one writes ('u_m', 'v_m', 'tem', 'vap', 'liq').
MOMENTUM_VARIABLES = (("u", 1), ("v", 2))
SCALAR_VARIABLES = (("t", 3), ("qv", 4), ("qc", 5))
VARIABLES = MOMENTUM_VARIABLES + SCALAR_VARIABLES


def _implicit_weight_profile(vct_a: np.ndarray, nlev: int) -> np.ndarray:
    """'tdc%impl_weight', the fixed implicit weight of each flux level.

    Not a field of the turbulence scheme and not serialized: ICON fills it once at model
    initialisation (mo_nwp_phy_init.f90:1541-1547) and never changes it, so the test has to
    reproduce that initialisation from the same reference vertical coordinate 'vct_a' ICON takes
    'k1500m' from (:781-795). 'impl_s' and 'impl_t' come from the granule's own config
    defaults, which are mo_turbdiff_config.f90's, so a change to either is a change here too.

    'Turbulence._build_the_implicit_weight' is the same code inside the granule and
    'test_turbdiff_section_9.py' has a third copy; all three must agree, which
    'test_the_implicit_weight_is_the_one_icon_used' checks against the reference data rather
    than against them.
    """
    config = TurbulenceConfig()
    ramp_level = 1
    for level in range(nlev, 0, -1):  # Fortran 'DO jk = nlev,1,-1', one-based
        if vct_a[level - 1] >= 1500.0 and vct_a[level] < 1500.0:
            ramp_level = level
    weight = np.full(nlev + 1, config.impl_t, dtype=float)
    for level in range(ramp_level + 1, nlev + 1):
        weight[level - 1] = config.impl_t + (config.impl_s - config.impl_t) * (
            level - ramp_level
        ) / float(nlev - ramp_level)
    weight[nlev] = config.impl_s
    return weight


def _reference(
    entry: Any,
    implicit_weight: np.ndarray,
    surface_exner_factor: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """A numpy transcription of 'vertdiff' in the configuration the entry savepoint reports.

    This exists to supply expected values for the intermediates ICON does not serialize -- the
    momentum type's whole workspace, four of the five variable profiles, every explicit flux,
    every right-hand side and every solved profile. It is entitled to that role only because
    'test_the_reference_reproduces_every_serialized_quantity' shows it is bit-exact against the
    twelve quantities ICON does serialize, on all four dates.

    It is a transcription and not a reimplementation: each statement below is one Fortran
    statement, in the Fortran's order and with the Fortran's association. Two of those groupings
    were measured to matter, and both are one ulp:

    - 'fr_var = 1/dt_var' is formed once and multiplied by. 'rho*dz/dt_var' puts 'disc_mom' one
      ulp off on 250707 of 662080 values and the error reaches every tendency.
    - 'invs_mom' adds 'disc_mom + impl_mom(k+1)' before the elimination term, left to right.

    'surface_exner_factor' is the one input that can be supplied rather than computed. It is the
    only transcendental in the whole scheme, 'EXP(rdocp*LOG(p_s/p0))', and numpy's libm differs
    from nvhpc's by one ulp on a few dozen of the 8276 computed columns. Left to itself the
    reference is compared against ICON and that difference is measured; handed the value the
    port computed, everything downstream of it becomes a test of the arithmetic alone instead of
    a test of two libms. Both uses appear below.

    Returns a flat dict keyed by 'quantity' for the type- and call-independent parts and by
    'quantity:type' or 'quantity:variable' for the rest, so a failing comparison names what it
    was comparing.
    """
    nlev = entry.ke()
    reciprocal_time_step = 1.0 / entry.dt_var()

    def array(accessor: str) -> np.ndarray:
        return getattr(entry, accessor)().asnumpy()

    half_level_height = array("hhl")
    # The savepoint reader truncates the 'nproma' slab to the grid's cells, so the width comes
    # from a field and not from 'nvec'.
    columns = half_level_height.shape[0]
    air_density_on_main_levels = array("rhoh")
    exner_factor = array("epr")
    air_density = array("rhon").copy()
    surface_pressure, surface_humidity = array("ps"), array("qv_s")
    surface_temperature = array("t_g")
    sensible_heat_flux, water_vapour_flux = array("shfl_s"), array("qvfl_s")
    out: dict[str, np.ndarray] = {}

    def blank() -> np.ndarray:
        return np.full((columns, nlev + 1), np.nan)

    # turb_vertdiff.f90:536-542 -- air density and Exner factor at the lower boundary.
    virtual_factor = 1.0 + constants.RV_O_RD_MINUS_1 * surface_humidity
    air_density[:, nlev] = surface_pressure / (constants.RD * virtual_factor * surface_temperature)
    if surface_exner_factor is None:
        surface_exner_factor = np.exp(
            constants.RD_O_CPD * np.log(surface_pressure / constants.P0REF)
        )
    out["rhon"] = air_density
    out["eprs"] = surface_exner_factor

    # vert_grad_diff:2438-2455 -- discretisation momentum and interior diffusion depth. The
    # layer depth lives in the 'expl_mom' storage in the Fortran; here it is a local.
    layer_depth = half_level_height[:, :nlev] - half_level_height[:, 1 : nlev + 1]
    discretisation_momentum = blank()
    discretisation_momentum[:, :nlev] = (
        air_density_on_main_levels * layer_depth * reciprocal_time_step
    )
    diffusion_depth = blank()
    diffusion_depth[:, 1:nlev] = 0.5 * (layer_depth[:, : nlev - 1] + layer_depth[:, 1:nlev])
    out["disc_mom"] = discretisation_momentum

    # The gradients implied by the prescribed surface flux densities (turb_vertdiff.f90:614-634)
    # need the SCALAR type's diffusion coefficient, which is 'tkvh' whichever type is running.
    scalar_diffusion_coefficient = array("tkvh")
    transfer_momentum = air_density[:, nlev] * scalar_diffusion_coefficient[:, nlev]
    surface_gradient = {
        "t": sensible_heat_flux / (transfer_momentum * constants.CPD * surface_exner_factor),
        "qv": water_vapour_flux / transfer_momentum,
    }
    for name, gradient in surface_gradient.items():
        out[f"surface_gradient:{name}"] = gradient

    implicit_momentum = blank()  # one storage for both types, as in ICON
    for kind, coefficient, velocity, variables, surface_flux_condition in (
        ("mom", array("tkvm"), array("tvm"), MOMENTUM_VARIABLES, False),
        ("sca", scalar_diffusion_coefficient, array("tvh"), SCALAR_VARIABLES, True),
    ):
        # vert_grad_diff:2461-2478 -- the diffusion momentum and the surface transfer depth.
        diffusion_momentum = blank()
        diffusion_momentum[:, 1:nlev] = (
            coefficient[:, 1:nlev] * air_density[:, 1:nlev] / diffusion_depth[:, 1:nlev]
        )
        diffusion_momentum[:, nlev] = air_density[:, nlev] * velocity
        diffusion_depth[:, nlev] = coefficient[:, nlev] / velocity

        # prep_impl_vert_diff:2764-2786 -- the implicit/explicit split. 'k_sf+1-m' is 'nlev+1'
        # for a surface-concentration condition and 'nlev' for a surface-flux condition.
        implicit_end = nlev if surface_flux_condition else nlev + 1
        implicit_momentum[:, 1:implicit_end] = (
            diffusion_momentum[:, 1:implicit_end] * implicit_weight[1:implicit_end]
        )
        out[f"full_expl_mom:{kind}"] = diffusion_momentum.copy()
        diffusion_momentum[:, 1:nlev] = diffusion_momentum[:, 1:nlev] - implicit_momentum[:, 1:nlev]

        # prep_impl_vert_diff:2830-2858 -- the LU factorisation, once per type. The main
        # elimination loop stops at Fortran 'k_sf-m', one row short of the implicit part's
        # range, and the third loop finishes the last row when a flux condition removes the
        # sub-diagonal below it.
        factorisation_end = implicit_end - 1
        inverted_momentum, inversion_factor = blank(), blank()
        inverted_momentum[:, 0] = 1.0 / (discretisation_momentum[:, 0] + implicit_momentum[:, 1])
        for level in range(1, factorisation_end):
            inversion_factor[:, level] = (
                inverted_momentum[:, level - 1] * implicit_momentum[:, level]
            )
            inverted_momentum[:, level] = 1.0 / (
                discretisation_momentum[:, level]
                + implicit_momentum[:, level + 1]
                + implicit_momentum[:, level] * (1.0 - inversion_factor[:, level])
            )
        for level in range(factorisation_end, nlev):  # the surface-flux row; empty at 'm = 1'
            inversion_factor[:, level] = (
                inverted_momentum[:, level - 1] * implicit_momentum[:, level]
            )
            inverted_momentum[:, level] = 1.0 / (
                discretisation_momentum[:, level]
                + implicit_momentum[:, level] * (1.0 - inversion_factor[:, level])
            )
        out[f"expl_mom:{kind}"] = diffusion_momentum.copy()
        out[f"impl_mom:{kind}"] = implicit_momentum.copy()
        out[f"invs_mom:{kind}"] = inverted_momentum
        out[f"invs_fac:{kind}"] = inversion_factor
        out[f"diff_dep:{kind}"] = diffusion_depth.copy()

        for name, component in variables:
            # turb_vertdiff.f90:646-694 and vert_grad_diff:2493-2503 -- the current profile.
            current_profile = blank()
            current_profile[:, :nlev] = array("t") / exner_factor if name == "t" else array(name)
            if name in surface_gradient:
                current_profile[:, nlev] = (
                    current_profile[:, nlev - 1] - diffusion_depth[:, nlev] * surface_gradient[name]
                )
            else:
                current_profile[:, nlev] = 0.0

            # calc_impl_vert_diff:2951-2973 -- the explicit flux density, positive upward.
            explicit_flux = blank()
            explicit_flux[:, 1:] = diffusion_momentum[:, 1:] * (
                current_profile[:, 1:] - current_profile[:, :nlev]
            )
            if not surface_flux_condition:
                explicit_flux[:, nlev] = (
                    explicit_flux[:, nlev]
                    + implicit_momentum[:, nlev] * current_profile[:, nlev - 1]
                )

            # calc_impl_vert_diff:2975-2991 -- the right-hand side. Every read is of the flux
            # as it was before the loop, so this is not a recurrence.
            right_hand_side = explicit_flux.copy()
            right_hand_side[:, 0] = (
                discretisation_momentum[:, 0] * current_profile[:, 0] + explicit_flux[:, 1]
            )
            right_hand_side[:, 1:nlev] = (
                discretisation_momentum[:, 1:nlev] * current_profile[:, 1:nlev]
                + explicit_flux[:, 2 : nlev + 1]
                - explicit_flux[:, 1:nlev]
            )

            # calc_impl_vert_diff:3024-3052 -- the two substitutions.
            updated_profile = blank()
            updated_profile[:, 0] = right_hand_side[:, 0] * inverted_momentum[:, 0]
            for level in range(1, nlev):
                updated_profile[:, level] = (
                    right_hand_side[:, level]
                    + implicit_momentum[:, level] * updated_profile[:, level - 1]
                ) * inverted_momentum[:, level]
            for level in range(nlev - 2, -1, -1):
                updated_profile[:, level] = (
                    updated_profile[:, level]
                    + inversion_factor[:, level + 1] * updated_profile[:, level + 1]
                )

            # vert_grad_diff:2661-2670 and turb_vertdiff.f90:773-799 -- the tendencies.
            diffusion_tendency = (
                updated_profile[:, :nlev] - current_profile[:, :nlev]
            ) * reciprocal_time_step
            increment = exner_factor * diffusion_tendency if name == "t" else diffusion_tendency
            out[f"cur_prof:{name}"] = current_profile
            out[f"expl_flux:{name}"] = explicit_flux
            out[f"rhs:{name}"] = right_hand_side
            out[f"upd_prof:{name}"] = updated_profile
            out[f"dif_tend:{name}"] = diffusion_tendency
            out[f"tendency:{name}"] = getattr(entry, f"{name}_tens")().asnumpy() + increment
            out[f"zvari:{component}"] = right_hand_side
    return out


def _solve_and_apply(
    backend,
    *,
    bounds: dict[str, gtx.int32],
    nlev: int,
    name: str,
    entry: Any,
    reciprocal_time_step: float,
    right_hand_side: gtx.Field,
    implicit_momentum: gtx.Field,
    inverted_momentum: gtx.Field,
    inversion_factor: gtx.Field,
    current_profile: gtx.Field,
    updated_profile: gtx.Field,
    diffusion_tendency: gtx.Field,
    variable_tendency: gtx.Field,
) -> None:
    """The three scan-dependent programs of one variable, split out so they can be skipped.

    'solve_vertical_diffusion_equation' is two 'scan_operator's and the two tendency programs
    consume its output, so on a backend that cannot afford a scan these three are what has to go.
    Nothing upstream of them depends on them.
    """
    solve_vertical_diffusion_equation.with_backend(backend)(
        right_hand_side=right_hand_side,
        implicit_diffusion_momentum=implicit_momentum,
        inverted_diffusion_momentum=inverted_momentum,
        inversion_factor=inversion_factor,
        updated_profile=updated_profile,
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev),
        offset_provider=_KOFF,
        **bounds,
    )
    tendency_before = getattr(entry, f"{name}_tens")()
    if name == "t":
        compute_and_apply_potential_temperature_diffusion_tendency.with_backend(backend)(
            updated_profile=updated_profile,
            current_profile=current_profile,
            exner_factor=entry.epr(),
            temperature_tendency_before=tendency_before,
            reciprocal_time_step=reciprocal_time_step,
            diffusion_tendency=diffusion_tendency,
            temperature_tendency=variable_tendency,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev),
            offset_provider={},
            **bounds,
        )
    else:
        compute_and_apply_diffusion_tendency.with_backend(backend)(
            updated_profile=updated_profile,
            current_profile=current_profile,
            variable_tendency_before=tendency_before,
            reciprocal_time_step=reciprocal_time_step,
            diffusion_tendency=diffusion_tendency,
            variable_tendency=variable_tendency,
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(nlev),
            offset_provider={},
            **bounds,
        )


class Vertdiff(NamedTuple):
    """One timestep of 'vertdiff': the savepoints, the bounds, the reference and every output."""

    entry: sb.IconVertdiffEntrySavepoint
    after: sb.IconVertdiffExitSavepoint
    #: 'ke'; as a zero-based row index this is the surface half level.
    nlev: int
    #: Half-open range of columns 'vertdiff' actually computed. Every comparison is masked with
    #: it, since the rest of the slab is untouched memory holding plausible values.
    columns: slice
    #: The numpy transcription's answers, keyed as '_reference' documents.
    reference: dict[str, np.ndarray]
    #: Everything the GT4Py chain produced, under the same keys as 'reference'.
    computed: dict[str, gtx.Field]


def _run_vertdiff(
    data_provider, grid_savepoint, date: str, backend, *, with_the_scans: bool = True
) -> Vertdiff:
    """Run every program of 'vertdiff', in the order 'vertdiff' runs them.

    The whole chain rather than program-by-program, because past the first three programs the
    savepoint has no inputs to hand any of them: 'vertdiff' keeps one workspace for five
    variables and two types, and only the last variable of the last type survives to the exit.
    What replaces per-program isolation is the number of comparisons -- every intermediate of
    every variable is checked, so the first failure still names one program.

    The buffers follow ICON's aliasing exactly where it is observable: one 'impl_mom' for both
    types, so that its surface row still holds the momentum type's value at the end, which is
    what the exit savepoint has. They do NOT follow it where it would be wrong: the right-hand
    side is a different field from the explicit flux, because the Fortran's in-place overwrite
    reads rows it has not written yet and GT4Py has no row order to rely on.

    'with_the_scans' exists for one backend. The embedded backend runs a 'scan_operator' as a
    Python loop over every horizontal position AND every level
    (gt4py/next/embedded/operators.py:67), which for this grid is 662080 calls per scan and
    twelve scans per chain; one test did not finish in 35 minutes. When it is false, this
    returns after 'prepare_the_vertical_diffusion_matrix' and nothing else runs.

    IT USED TO STOP MUCH LATER, and the stencil merge is why it cannot. The LU factorisation is
    now a statement of 'prep_impl_vert_diff', so the four scan-free statements of that program --
    the diffusion momentum, its surface row, the surface diffusion depth and the implicit split
    -- cannot be run without it, and everything downstream reads what it produces. Before the
    merge, ten of the fifteen programs ran on 'embedded'; now it is the setup program alone.
    That is the price of merging producers INTO a scan, which is the direction that genuinely
    fuses, and it is paid in 'embedded' coverage rather than in coverage: every one of these
    comparisons still runs on 'gtfn_cpu', 'gtfn_gpu', 'dace_cpu' and 'dace_gpu'.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)
    nlev = entry.ke()
    horizontal_start, horizontal_end = gtx.int32(entry.ivstart()), gtx.int32(entry.ivend())
    bounds = {"horizontal_start": horizontal_start, "horizontal_end": horizontal_end}
    reciprocal_time_step = 1.0 / entry.dt_var()
    implicit_weight = _implicit_weight_profile(grid_savepoint.vct_a().asnumpy(), nlev)

    half = entry.rhon()  # a half-level field, for the shape of every workspace buffer
    main = entry.u()  # a main-level field
    computed: dict[str, gtx.Field] = {}

    # 'rhon' is the only in-out field of the scheme: rows above the surface must come out
    # unchanged, so this one starts as a copy rather than as NaN.
    air_density = utils.copy_of(entry.rhon(), backend)
    surface_exner_factor = utils.nan_like(half, backend)
    discretisation_momentum = utils.nan_like(half, backend)
    diffusion_depth = utils.nan_like(half, backend)
    surface_gradient = {
        "t": utils.nan_like(half, backend),
        "qv": utils.nan_like(half, backend),
    }
    # ONE PROGRAM, SIX STATEMENTS. The bound pair is the discretisation momentum's main-level
    # range; the diffusion depth starts one row lower and the four surface-row statements sit on
    # 'vertical_end'. The two gradient statements read the 'rhon' and 'eprs' rows the first two
    # wrote, which is why the four programs this replaces had to run in this order.
    prepare_the_vertical_diffusion_matrix.with_backend(backend)(
        surface_pressure=entry.ps(),
        surface_specific_humidity=entry.qv_s(),
        surface_temperature=entry.t_g(),
        air_density_at_main_levels=entry.rhoh(),
        half_level_height=entry.hhl(),
        reciprocal_time_step=reciprocal_time_step,
        diffusion_coefficient=entry.tkvh(),
        sensible_heat_flux=entry.shfl_s(),
        water_vapour_flux=entry.qvfl_s(),
        air_density=air_density,
        surface_exner_factor=surface_exner_factor,
        discretisation_momentum=discretisation_momentum,
        diffusion_depth=diffusion_depth,
        surface_temperature_gradient=surface_gradient["t"],
        surface_vapour_gradient=surface_gradient["qv"],
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(nlev),
        offset_provider=_KOFF,
        **bounds,
    )
    computed["rhon"] = air_density
    computed["eprs"] = surface_exner_factor
    computed["disc_mom"] = discretisation_momentum
    for name, gradient in surface_gradient.items():
        computed[f"surface_gradient:{name}"] = gradient
    # A copy, because the per-type factorisation writes this field's surface row.
    computed["diff_dep"] = utils.copy_of(diffusion_depth, backend)

    if not with_the_scans:
        return Vertdiff(
            entry=entry,
            after=after,
            nlev=nlev,
            columns=slice(entry.ivstart(), entry.ivend()),
            reference=_reference(
                entry,
                implicit_weight,
                surface_exner_factor=surface_exner_factor.asnumpy()[:, nlev],
            ),
            computed=computed,
        )

    implicit_weight_field = gtx.as_field((dims.KDim,), implicit_weight, allocator=backend)
    # One storage for both types, as 'zaux(:,:,3)' is in the Fortran.
    implicit_momentum = utils.nan_like(half, backend)

    for kind, coefficient, velocity, variables, surface_flux_condition in (
        ("mom", entry.tkvm(), entry.tvm(), MOMENTUM_VARIABLES, False),
        ("sca", entry.tkvh(), entry.tvh(), SCALAR_VARIABLES, True),
    ):
        diffusion_momentum = utils.nan_like(half, backend)
        inverted_momentum = utils.nan_like(half, backend)
        inversion_factor = utils.nan_like(half, backend)
        # ONE PROGRAM, SIX STATEMENTS, and one binding for both types: every range that depends
        # on the lower boundary condition is expressed on 'elimination_end', the Fortran's
        # 'k_sf-m'. The implicit part runs one row further than it; the subtraction runs to
        # 'nlev', one row short of the split, so the surface flux level keeps the WHOLE
        # diffusion momentum.
        elimination_end = nlev - 1 if surface_flux_condition else nlev
        prep_impl_vert_diff.with_backend(backend)(
            diffusion_coefficient=coefficient,
            air_density=air_density,
            surface_transfer_velocity=velocity,
            implicit_weight=implicit_weight_field,
            discretisation_momentum=discretisation_momentum,
            elimination_end=gtx.int32(elimination_end),
            diffusion_momentum=diffusion_momentum,
            diffusion_depth=diffusion_depth,
            implicit_diffusion_momentum=implicit_momentum,
            inverted_diffusion_momentum=inverted_momentum,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev),
            offset_provider=_KOFF,
            **bounds,
        )
        if surface_flux_condition:
            invert_diffusion_momentum_at_the_surface_flux_level.with_backend(backend)(
                discretisation_momentum=discretisation_momentum,
                implicit_diffusion_momentum=implicit_momentum,
                inverted_diffusion_momentum_above=inverted_momentum,
                inverted_diffusion_momentum=inverted_momentum,
                vertical_start=gtx.int32(nlev - 1),
                vertical_end=gtx.int32(nlev),
                offset_provider=_KOFF,
                **bounds,
            )
        compute_diffusion_inversion_factor.with_backend(backend)(
            inverted_diffusion_momentum=inverted_momentum,
            implicit_diffusion_momentum=implicit_momentum,
            inversion_factor=inversion_factor,
            vertical_start=gtx.int32(1),
            vertical_end=gtx.int32(nlev),
            offset_provider=_KOFF,
            **bounds,
        )
        computed[f"expl_mom:{kind}"] = diffusion_momentum
        computed[f"impl_mom:{kind}"] = utils.copy_of(implicit_momentum, backend)
        computed[f"invs_mom:{kind}"] = inverted_momentum
        computed[f"invs_fac:{kind}"] = inversion_factor
        computed[f"diff_dep:{kind}"] = utils.copy_of(diffusion_depth, backend)

        for name, component in variables:
            current_profile = utils.nan_like(half, backend)
            if name == "t":
                compute_current_potential_temperature_profile.with_backend(backend)(
                    temperature=entry.t(),
                    exner_factor=entry.epr(),
                    current_profile=current_profile,
                    vertical_start=gtx.int32(0),
                    vertical_end=gtx.int32(nlev),
                    offset_provider={},
                    **bounds,
                )
            else:
                compute_current_profile.with_backend(backend)(
                    variable=getattr(entry, name)(),
                    current_profile=current_profile,
                    vertical_start=gtx.int32(0),
                    vertical_end=gtx.int32(nlev),
                    offset_provider={},
                    **bounds,
                )
            if name in surface_gradient:
                compute_surface_profile_value_from_flux_gradient.with_backend(backend)(
                    current_profile_above=current_profile,
                    diffusion_depth=diffusion_depth,
                    surface_gradient=surface_gradient[name],
                    current_profile=current_profile,
                    vertical_start=gtx.int32(nlev),
                    vertical_end=gtx.int32(nlev + 1),
                    offset_provider=_KOFF,
                    **bounds,
                )

            explicit_flux = utils.nan_like(half, backend)
            compute_explicit_flux_density.with_backend(backend)(
                explicit_diffusion_momentum=diffusion_momentum,
                current_profile=current_profile,
                model_top_level=gtx.int32(0),
                explicit_flux_density=explicit_flux,
                vertical_start=gtx.int32(1),
                vertical_end=gtx.int32(nlev + 1),
                offset_provider=_KOFF,
                **bounds,
            )
            if not surface_flux_condition:
                add_implicit_surface_flux_to_the_explicit_flux_density.with_backend(backend)(
                    explicit_flux_density_at_the_surface=explicit_flux,
                    implicit_diffusion_momentum=implicit_momentum,
                    current_profile=current_profile,
                    explicit_flux_density=explicit_flux,
                    vertical_start=gtx.int32(nlev),
                    vertical_end=gtx.int32(nlev + 1),
                    offset_provider=_KOFF,
                    **bounds,
                )

            # The surface row of 'zvari' is the explicit flux and no program writes it again,
            # which is why the right-hand side starts as a copy of the flux rather than as NaN.
            right_hand_side = utils.copy_of(explicit_flux, backend)
            compute_diffusion_right_hand_side.with_backend(backend)(
                discretisation_momentum=discretisation_momentum,
                current_profile=current_profile,
                explicit_flux_density=explicit_flux,
                right_hand_side=right_hand_side,
                vertical_start=gtx.int32(0),
                vertical_end=gtx.int32(nlev),
                offset_provider=_KOFF,
                **bounds,
            )

            updated_profile = utils.nan_like(half, backend)
            diffusion_tendency = utils.nan_like(main, backend)
            variable_tendency = utils.nan_like(main, backend)
            _solve_and_apply(
                backend,
                bounds=bounds,
                nlev=nlev,
                name=name,
                entry=entry,
                reciprocal_time_step=reciprocal_time_step,
                right_hand_side=right_hand_side,
                implicit_momentum=implicit_momentum,
                inverted_momentum=inverted_momentum,
                inversion_factor=inversion_factor,
                current_profile=current_profile,
                updated_profile=updated_profile,
                diffusion_tendency=diffusion_tendency,
                variable_tendency=variable_tendency,
            )

            computed[f"cur_prof:{name}"] = current_profile
            computed[f"expl_flux:{name}"] = explicit_flux
            computed[f"rhs:{name}"] = right_hand_side
            computed[f"upd_prof:{name}"] = updated_profile
            computed[f"dif_tend:{name}"] = diffusion_tendency
            computed[f"tendency:{name}"] = variable_tendency
            computed[f"zvari:{component}"] = right_hand_side

    return Vertdiff(
        entry=entry,
        after=after,
        nlev=nlev,
        columns=slice(entry.ivstart(), entry.ivend()),
        reference=_reference(
            entry, implicit_weight, surface_exner_factor=surface_exner_factor.asnumpy()[:, nlev]
        ),
        computed=computed,
    )


def _surface(field: Any, nlev: int) -> np.ndarray:
    """The surface row of a half-level field, as a plain array.

    Several quantities of this stage are one row deep -- the Exner factor is declared
    '(nvec, ke1:ke1)' in Fortran and the savepoint hands it back as a cell field -- while the
    port carries them in a full half-level field so that their consumers need no rank change.
    """
    return field.asnumpy()[:, nlev]


# ------------------------------------------------------- what the configuration switches off --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_vertdiff_runs_in_the_configuration_this_port_assumes(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """Every switch the module docstring claims is off, measured at the entry savepoint.

    A port is only as good as the branch it took, and 'vertdiff' has four control integers and
    eight logicals. These are asserted rather than assumed so that a capture from a different
    configuration fails here, naming the switch, instead of failing later as a numerical
    disagreement nobody can place.

    'l3dflxout' is the one to watch. It is the last word on 'leff_flux' -- the Fortran overrides
    whatever the two branches above it decided, for every variable with 'n <= nmvar', and with
    'ndtr = 0' that is every variable there is -- so it alone decides whether the effective-flux
    integration at calc_impl_vert_diff:3072-3090 runs. It is false here, which is why this port
    has no vertical integration of the diffusion tendencies and 'zvari' comes out holding the
    right-hand side.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)

    assert entry.itndcon() == 0
    assert entry.lentire() is True
    assert entry.lsfluse() is True
    assert entry.ldoexpcor() is False
    assert entry.ldocirflx() is False
    assert entry.ldogrdcor() is False
    assert entry.l3dflxout() is False
    assert entry.lrunscm() is False
    assert entry.ndtr() == 0
    assert entry.ndiff() == 5
    assert entry.kcm() == entry.ke1()
    assert entry.kstart_cloud() == 1
    assert after.igrdcon() == 0
    assert after.ncorr() > after.mcorr()

    # The four config defaults the port depends on and the savepoints cannot show, stated here
    # against the granule's own config so that changing either is a change to this test.
    config = TurbulenceConfig()
    assert config.lsflcnd is True  # surface-FLUX condition for the scalars, 'm = 2'
    assert config.ilow_def_cond == 2  # zero surface value where there is none
    assert config.impl_s == 1.20
    assert config.impl_t == 0.75


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_implicit_weight_is_the_one_icon_used(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint
) -> None:
    """Recover 'tdc%impl_weight' from the reference data and check the reconstruction against it.

    The implicit weight is a selector of the same awkward kind wave 2b kept finding: ICON builds
    it once at model initialisation from the reference vertical coordinate and never serializes
    it, so a port that reproduces the initialisation wrongly is off by a smooth profile and
    still looks plausible.

    It does not have to be taken on trust. 'prep_impl_vert_diff' leaves 'impl_mom = expl_mom*w'
    and 'expl_mom = expl_mom - impl_mom' in the savepoint, so 'w = impl_mom/(expl_mom +
    impl_mom)' recovers it column by column. The sum rounds, so this is a close comparison and not a
    bit-exact one -- 1e-12 is nine orders of magnitude tighter than the 0.45 that separates
    'impl_t' from 'impl_s'.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(entry.ivstart(), entry.ivend())

    explicit = after.expl_mom().asnumpy()[columns, 1:nlev]
    implicit = after.impl_mom().asnumpy()[columns, 1:nlev]
    recovered = implicit / (explicit + implicit)

    expected = _implicit_weight_profile(grid_savepoint.vct_a().asnumpy(), nlev)[1:nlev]
    np.testing.assert_allclose(recovered, np.broadcast_to(expected, recovered.shape), rtol=1e-12)
    # The profile really does vary, so the agreement above is not vacuous.
    assert expected.min() == pytest.approx(0.75)
    assert expected.max() > 1.0


# ------------------------------------------------------- the reference, and what makes it one --


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_reference_reproduces_every_serialized_quantity(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint
) -> None:
    """'_reference' is bit-exact against ICON on everything ICON serializes.

    This is the test that entitles '_reference' to be the expected value for the intermediates
    ICON does NOT serialize -- the momentum type's whole workspace, four of the five profiles,
    every explicit flux and every solved profile. Without it the port would be checked against
    a second implementation of the same misunderstanding.

    Twelve quantities, and they cover every kind of arithmetic in the stage: the grid setup, the
    matrix, the factorisation, the profile of the last variable, its tendency, the right-hand
    sides of all five variables and all five accumulated tendencies.

    The reference is handed ICON's own surface Exner factor. That is the one input it cannot
    reproduce bit for bit -- 'EXP(rdocp*LOG(...))' through a different libm -- and supplying it
    is what makes this a test of the transcription and not of numpy's exponential. The two
    quantities that difference reaches, and how far, are measured by the test below.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(entry.ivstart(), entry.ivend())
    reference = _reference(
        entry,
        _implicit_weight_profile(grid_savepoint.vct_a().asnumpy(), nlev),
        surface_exner_factor=after.eprs().asnumpy(),
    )

    def agrees(quantity: str, computed: np.ndarray, expected: np.ndarray) -> None:
        got, want = computed[columns], expected[columns]
        assert np.array_equal(got, want), (
            f"the numpy reference differs from ICON on '{quantity}': max abs "
            f"{np.nanmax(np.abs(got - want))} over {np.count_nonzero(got != want)} of "
            f"{got.size} values."
        )

    interior = slice(0, nlev)
    agrees("rhon", reference["rhon"], after.rhon().asnumpy())
    agrees("disc_mom", reference["disc_mom"][:, interior], after.disc_mom().asnumpy()[:, interior])
    agrees(
        "diff_dep",
        reference["diff_dep:sca"][:, 1 : nlev + 1],
        after.diff_dep().asnumpy()[:, 1 : nlev + 1],
    )
    agrees(
        "expl_mom",
        reference["expl_mom:sca"][:, 1 : nlev + 1],
        after.expl_mom().asnumpy()[:, 1 : nlev + 1],
    )
    agrees(
        "impl_mom",
        reference["impl_mom:sca"][:, 1 : nlev + 1],
        after.impl_mom().asnumpy()[:, 1 : nlev + 1],
    )
    agrees(
        "invs_mom", reference["invs_mom:sca"][:, interior], after.invs_mom().asnumpy()[:, interior]
    )
    agrees(
        "invs_fac",
        reference["invs_fac:sca"][:, 1:nlev],
        after.invs_fac().asnumpy()[:, 1:nlev],
    )
    agrees("hlp (cur_prof of qc)", reference["cur_prof:qc"], after.cur_prof().asnumpy())
    agrees(
        "dicke (dif_tend of qc)",
        reference["dif_tend:qc"],
        after.dif_tend().asnumpy()[:, interior],
    )
    for name, component in VARIABLES:
        agrees(
            f"zvari(:,:,{component}) [{name}]",
            reference[f"zvari:{component}"],
            after.zvari(component).asnumpy(),
        )
        agrees(
            f"{name}_tens",
            reference[f"tendency:{name}"],
            getattr(after, f"{name}_tens")().asnumpy(),
        )
    # 'zvari(:,:,0)' is the circulation component; 'vertdiff' never touches it.
    agrees("zvari(:,:,0)", entry.zvari(0).asnumpy(), after.zvari(0).asnumpy())


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_surface_exner_factor_is_the_only_transcendental(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint
) -> None:
    """Where the numpy reference stops being bit-exact, and how far it is from being so.

    'zexner(ps) = EXP(rdocp*LOG(ps/p0ref))' is the only 'EXP' or 'LOG' in the whole of
    'vertdiff', and numpy's implementations of both differ from nvhpc's by a rounding on a few
    dozen columns. Measured on this capture: 51 to 67 of the 8276 computed columns, always by
    exactly one ulp.

    It is recorded rather than hidden because it is the one place the reference is NOT an oracle,
    and because it explains why '_run_vertdiff' hands the reference the Exner factor the port
    computed instead of letting it compute its own. What the difference does NOT do is reach the
    tendencies: they are bit-exact on all four dates, and 'zvari(:,:,3)' picks it up on two of
    the four in a single value of 670356.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(entry.ivstart(), entry.ivend())
    reference = _reference(entry, _implicit_weight_profile(grid_savepoint.vct_a().asnumpy(), nlev))

    got, want = reference["eprs"][columns], after.eprs().asnumpy()[columns]
    differing = np.count_nonzero(got != want)
    assert differing < 100, f"{differing} of {got.size} columns differ, which is more than a ulp"
    np.testing.assert_allclose(got, want, rtol=4.0 * np.finfo(np.float64).eps, atol=0.0)

    # Where it goes. The Exner factor divides the prescribed sensible heat flux into a surface
    # temperature gradient, so it reaches the temperature profile's lower boundary value, its
    # explicit surface flux -- which is 'zvari(:,ke1,tem)' at the exit -- and from there the
    # right-hand side of the temperature solve. It gets no further: all five tendencies are
    # bit-exact on all four dates either way, and 'zvari(:,:,tem)' differs in at most a single
    # value of 670356 in the whole slab.
    for name, component in VARIABLES:
        icon = after.zvari(component).asnumpy()[columns]
        deviating = np.count_nonzero(reference[f"zvari:{component}"][columns] != icon)
        assert deviating <= (1 if name == "t" else 0), (
            f"the numpy Exner factor moves 'zvari(:,:,{component})' [{name}] in {deviating} "
            "values, which is more than the surface row it can reach."
        )
        np.testing.assert_allclose(
            reference[f"tendency:{name}"][columns],
            getattr(after, f"{name}_tens")().asnumpy()[columns],
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_unwritten_workspace_rows_are_leftovers_not_results(
    date: str, *, data_provider: sb.IconSerialDataProvider
) -> None:
    """The rows of ICON's workspace that this port deliberately does not produce.

    'vertdiff' declares its workspace '!$ACC CREATE' and each of its loops writes only the rows
    the quantity is defined on; six rows are therefore excluded from the comparisons below, and
    this is where the exclusion is argued rather than assumed.

    The interesting one is 'expl_mom(:,0)', and it is not undefined at all: it is the depth of
    the top model layer, because 'vert_grad_diff' parks the layer depth in the 'expl_mom'
    storage (:2442) and then overwrites rows 1 onward with the diffusion momentum. Reproducing
    it would mean writing a length into a field of mass fluxes. It is asserted exactly, so the
    claim is a measurement and not a reading of the source.

    For the other five there is nothing to compute a value from -- 'disc_mom(:,ke1)' would need
    an air density on a level that does not exist -- so what is checked is the one alternative a
    comparison could not see: that the Fortran writes the row by copying its neighbour. It does
    not. 'dif_tend(:,ke1)' gets the sharper version of the same question, since the Fortran DOES
    write it under 'itndcon >= 1' ('dif_tend(i,k_sf) = cur_prof(i,k_sf)', :2652) and 'itndcon'
    is 0 here.
    """
    entry = data_provider.from_savepoint_vertdiff_entry(date=date)
    after = data_provider.from_savepoint_vertdiff_exit(date=date)
    nlev = entry.ke()
    columns = slice(entry.ivstart(), entry.ivend())
    height = entry.hhl().asnumpy()

    top_layer_depth = height[columns, 0] - height[columns, 1]
    assert np.array_equal(after.expl_mom().asnumpy()[columns, 0], top_layer_depth)

    for quantity, row, neighbour in (
        ("disc_mom", nlev, nlev - 1),
        ("diff_dep", 0, 1),
        ("invs_mom", nlev, nlev - 1),
        ("invs_fac", 0, 1),
        ("invs_fac", nlev, nlev - 1),
        ("dif_tend", nlev, nlev - 1),
    ):
        slab = getattr(after, quantity)().asnumpy()[columns]
        assert not np.array_equal(slab[:, row], slab[:, neighbour]), (
            f"'{quantity}(:,{row})' is a copy of its neighbour, so the Fortran may write it "
            "after all and excluding it from the comparison would hide a missing write."
        )

    # 'dif_tend(:,ke1)' is 'cur_prof(:,ke1)' when explicit tendencies are considered.
    assert entry.itndcon() == 0
    assert not np.array_equal(
        after.dif_tend().asnumpy()[columns, nlev], after.cur_prof().asnumpy()[columns, nlev]
    )


# ------------------------------------------------- the port, on every backend (no scan needed) --
#
# ONLY THE FIRST TEST BELOW IS SCAN-FREE, and that is a change the stencil merge made. The LU
# factorisation is now a statement of 'prep_impl_vert_diff', so nothing past the setup program
# can be run without a scan and everything past it is marked 'embedded_too_slow'. What survives
# on 'embedded' is 'prepare_the_vertical_diffusion_matrix'; what is unaffected is every compiled
# backend. See '_run_vertdiff' for the measurement behind the marker.


@pytest.mark.datatest
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_setup_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The four comparisons that need nothing but 'prepare_the_vertical_diffusion_matrix'.

    THE ONLY TEST OF THIS FILE THAT RUNS ON 'embedded' since the stencil merge, which is why it
    was split off from the diffusion-momentum comparisons that used to sit with it.

    THE SETUP IS ONE PROGRAM AND ITS GATE IS 'Tol', so the three quantities that earned no
    tolerance are asserted BIT-EXACT here, directly, and not through 'assert_agrees_with_icon'.
    Only 'eprs' has an exponential in it; 'rhon', 'disc_mom' and 'diff_dep' have none, and
    letting the merged program's gate cover them would widen a check that three separate 'Exact'
    entries used to make. The pattern is section 0)'s, and 'rhon' already used it inside this
    very program before the merge.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend, with_the_scans=False)
    nlev, columns = run.nlev, run.columns
    after, computed = run.after, run.computed

    for quantity, got, want in (
        ("rhon", computed["rhon"].asnumpy()[columns], after.rhon().asnumpy()[columns]),
        (
            "disc_mom",
            computed["disc_mom"].asnumpy()[columns, 0:nlev],
            after.disc_mom().asnumpy()[columns, 0:nlev],
        ),
        (
            "diff_dep [interior]",
            computed["diff_dep"].asnumpy()[columns, 1:nlev],
            after.diff_dep().asnumpy()[columns, 1:nlev],
        ),
    ):
        assert np.array_equal(got, want), (
            f"'{quantity}' has no transcendental in it and must be bit-exact, but "
            f"{np.count_nonzero(got != want)} of {got.size} values differ by up to "
            f"{np.nanmax(np.abs(got - want))}"
        )
    utils.assert_agrees_with_icon(
        "prepare_the_vertical_diffusion_matrix",
        "eprs",
        _surface(computed["eprs"], nlev),
        after.eprs(),
        columns=columns,
    )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_diffusion_momentum_agrees_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The scalar type's factorisation workspace, against the exit savepoint.

    The savepoint holds the SCALAR type's workspace, because that is the last type 'vertdiff'
    runs; the momentum type's is compared against the reference in the next test. One row of it
    is the momentum type's all the same -- 'impl_mom(:,ke1)', which the scalar pass does not
    write because a surface-flux condition has no sub-diagonal to the surface -- and it is
    included here, since a port that wrote it would differ.

    Four ranges, and all four now come out of one program, so all four name it. What used to
    attribute them -- one stencil name per range -- is preserved in the quantity string instead;
    the Fortran statement each one belongs to is in 'prep_impl_vert_diff's docstring.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    after, computed = run.after, run.computed

    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "diff_dep(:,ke1)",
        computed["diff_dep:sca"],
        after.diff_dep(),
        columns=columns,
        levels=slice(nlev, nlev + 1),
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "expl_mom [interior]",
        computed["expl_mom:sca"],
        after.expl_mom(),
        columns=columns,
        levels=slice(1, nlev),
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "expl_mom(:,ke1)",
        computed["expl_mom:sca"],
        after.expl_mom(),
        columns=columns,
        levels=slice(nlev, nlev + 1),
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "impl_mom",
        computed["impl_mom:sca"],
        after.impl_mom(),
        columns=columns,
        levels=slice(1, nlev + 1),
    )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_momentum_type_diffusion_momentum_agrees_with_the_reference(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The half of the setup that ICON overwrites before the exit savepoint.

    'vertdiff' runs the momentum type first and the scalar type second into the same storage, so
    nothing of the momentum matrix survives except the one 'impl_mom' row the scalar pass leaves
    alone. Everything else here is compared against '_reference', which
    'test_the_reference_reproduces_every_serialized_quantity' has shown to be bit-exact against
    ICON wherever ICON can be seen.

    The momentum type is not a duplicate of the scalar one. It runs with 'lsflucond = .FALSE.',
    so its implicit momentum extends one row further than the scalar type's -- the difference
    that 'invert_diffusion_momentum_at_the_surface_flux_level' exists to carry, and the one that
    would make 'u_tens' and 'v_tens' wrong with nothing before them saying so.

    'full_expl_mom' -- the diffusion momentum before the implicit part is subtracted -- USED TO
    BE COMPARED HERE and no longer can be: 'prep_impl_vert_diff' subtracts in place, exactly as
    the Fortran does, so the unreduced value never leaves the program. WHAT COVERS IT INSTEAD is
    'impl_mom = full_expl_mom * impl_weight', which is compared below over the same rows: the
    weight is strictly positive on every one of them (asserted here, since the argument depends
    on it), so an error in the unreduced momentum scales 'impl_mom' by the same factor and
    cannot hide. That is a weaker attribution and the same coverage.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    reference, computed = run.reference, run.computed

    weight = _implicit_weight_profile(grid_savepoint.vct_a().asnumpy(), nlev)
    assert np.all(weight[1 : nlev + 1] > 0.0), (
        "'impl_mom' only carries an error in the unreduced diffusion momentum where the "
        "implicit weight is non-zero, and it is zero somewhere -- so the comparison below no "
        "longer covers 'compute_diffusion_momentum'."
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "diff_dep(:,ke1) [mom]",
        computed["diff_dep:mom"],
        reference["diff_dep:mom"],
        columns=columns,
        levels=slice(nlev, nlev + 1),
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "expl_mom [mom]",
        computed["expl_mom:mom"],
        reference["expl_mom:mom"],
        columns=columns,
        levels=slice(1, nlev + 1),
    )
    utils.assert_agrees_with_icon(
        "prep_impl_vert_diff",
        "impl_mom [mom]",
        computed["impl_mom:mom"],
        reference["impl_mom:mom"],
        columns=columns,
        levels=slice(1, nlev + 1),
    )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_profiles_and_the_explicit_fluxes_agree(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The five variable profiles and the five explicit flux densities.

    The profiles are where the five variables stop being five instances of one thing. Three of
    them get a zero lower boundary value and two get one reconstructed from a prescribed surface
    flux; one of them is diffused in units it does not arrive in. Cloud water's profile is the
    only one ICON serializes ('hlp'), so it is compared against the savepoint and the other four
    against the reference.

    The explicit fluxes carry the second per-type difference: the momentum type adds the
    implicit surface coupling to its surface row and the scalar type does not, which is the
    lower boundary condition of the two systems and the reason 'u' and 'qv' are not solved by
    the same code path.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    after, reference, computed = run.after, run.reference, run.computed

    # Bit-exact and ungated, for the reason 'test_the_setup_and_the_diffusion_momentum_agree_
    # with_icon' gives: these come out of the merged setup program, whose gate is 'Tol' for the
    # Exner factor alone. The temperature gradient does divide by 'eprs' -- but the reference is
    # handed the SAME 'eprs' the port computed, so this comparison is not the one that would
    # notice a difference in it.
    for name in ("t", "qv"):
        got = _surface(computed[f"surface_gradient:{name}"], nlev)[columns]
        want = np.asarray(reference[f"surface_gradient:{name}"])[columns]
        assert np.array_equal(got, want), (
            f"'zvari(:,ke1,{name})' before the solve overwrites it is not bit-exact: "
            f"{np.count_nonzero(got != want)} of {got.size} values differ by up to "
            f"{np.nanmax(np.abs(got - want))}"
        )

    utils.assert_agrees_with_icon(
        "compute_current_profile",
        "hlp (cur_prof of qc)",
        computed["cur_prof:qc"],
        after.cur_prof(),
        columns=columns,
    )
    for name, _ in VARIABLES:
        stencil = (
            "compute_current_potential_temperature_profile"
            if name == "t"
            else "compute_current_profile"
        )
        utils.assert_agrees_with_icon(
            stencil,
            f"cur_prof [{name}] on the main levels",
            computed[f"cur_prof:{name}"],
            reference[f"cur_prof:{name}"],
            columns=columns,
            levels=slice(0, nlev),
        )
        surface_stencil = (
            "compute_surface_profile_value_from_flux_gradient"
            if name in ("t", "qv")
            else "compute_current_profile"
        )
        utils.assert_agrees_with_icon(
            surface_stencil,
            f"cur_prof(:,ke1) [{name}]",
            computed[f"cur_prof:{name}"],
            reference[f"cur_prof:{name}"],
            columns=columns,
            levels=slice(nlev, nlev + 1),
        )

        utils.assert_agrees_with_icon(
            "compute_explicit_flux_density",
            f"explicit flux [{name}] on the interior flux levels",
            computed[f"expl_flux:{name}"],
            reference[f"expl_flux:{name}"],
            columns=columns,
            levels=slice(1, nlev),
        )
        surface_stencil = (
            "add_implicit_surface_flux_to_the_explicit_flux_density"
            if name in ("u", "v")
            else "compute_explicit_flux_density"
        )
        utils.assert_agrees_with_icon(
            surface_stencil,
            f"explicit flux(:,ke1) [{name}]",
            computed[f"expl_flux:{name}"],
            reference[f"expl_flux:{name}"],
            columns=columns,
            levels=slice(nlev, nlev + 1),
        )

    # The upper boundary condition this port states as a value rather than as a missing term.
    for name, _ in VARIABLES:
        assert np.array_equal(
            computed[f"expl_flux:{name}"].asnumpy()[columns, 0],
            np.zeros(columns.stop - columns.start),
        ), name


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_right_hand_sides_agree_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The five right-hand sides, which is the one thing 'zvari' still holds for every variable.

    'zvari' is not the effective flux density its name and its declaration suggest: 'leff_flux'
    is false for all five variables here, so the vertical integration that would turn the
    storage into a flux never runs and what survives is the right-hand side on the diffused rows
    and the explicit surface flux on the row below them. Both are compared, each against the
    program that wrote it.

    The right-hand side does NOT depend on the factorisation -- it is built from the explicit
    flux, the discretisation momentum and the profile -- which is what lets this test keep the
    embedded backend while the solve cannot.

    THE TEMPERATURE'S SURFACE ROW IS THE ONE EXCEPTION and it is compared against the reference
    rather than against ICON. It is 'expl_mom(ke1)*(cur_prof(ke1) - cur_prof(ke))' with a
    surface value reconstructed through the Exner factor, so it carries the port's 'EXP(LOG())'
    into a comparison with nvhpc's; measured, that moves one value of 8276 by 1.5e-17 on two of
    the four dates. 'test_the_surface_exner_factor_is_the_only_transcendental' is where that is
    stated against ICON with a tolerance; here the arithmetic is checked without it.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    after, reference, computed = run.after, run.reference, run.computed

    for name, component in VARIABLES:
        utils.assert_agrees_with_icon(
            "compute_diffusion_right_hand_side",
            f"zvari(:,:,{component}) [{name}], the right-hand side",
            computed[f"zvari:{component}"],
            after.zvari(component),
            columns=columns,
            levels=slice(0, nlev),
        )
        surface_stencil = (
            "add_implicit_surface_flux_to_the_explicit_flux_density"
            if name in ("u", "v")
            else "compute_explicit_flux_density"
        )
        utils.assert_agrees_with_icon(
            surface_stencil,
            f"zvari(:,ke1,{component}) [{name}], the explicit surface flux",
            computed[f"zvari:{component}"],
            reference[f"zvari:{component}"] if name == "t" else after.zvari(component),
            columns=columns,
            levels=slice(nlev, nlev + 1),
        )

    # 'vertdiff' does not touch the circulation component of 'zvari'.
    assert np.array_equal(run.entry.zvari(0).asnumpy()[columns], after.zvari(0).asnumpy()[columns])


# --------------------------------------------------- the port, where a 'scan_operator' is needed --
#
# 'embedded_too_slow' on all three: the embedded backend runs a scan as a Python loop over every
# horizontal position and every level, which for this grid is 662080 calls per scan and twelve
# scans per chain. One test, one date had not finished after 35 minutes. The measurement and the
# reference are in '_run_vertdiff'.


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_factorisation_agrees(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The LU factorisation of both variable types, in the three ranges the Fortran writes it in.

    'prep_impl_vert_diff' has three loops and they differ by one row each. The elimination
    proper stops at 'k_sf-m'; the third loop covers the row from there to 'k_sf-1', which is
    empty for the momentum type and one row for the scalar type, and drops the term that couples
    the lowest main level to the surface. The inversion factor is the same product on every row
    of both and is one program covering the union.

    Getting the three ranges wrong by a row is this section's quietest failure mode, so each is
    compared separately and against the savepoint where the savepoint has it.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    after, reference, computed = run.after, run.reference, run.computed

    utils.assert_agrees_with_icon(
        "compute_inverted_diffusion_momentum",
        "invs_mom [the elimination]",
        computed["invs_mom:sca"],
        after.invs_mom(),
        columns=columns,
        levels=slice(0, nlev - 1),
    )
    utils.assert_agrees_with_icon(
        "invert_diffusion_momentum_at_the_surface_flux_level",
        "invs_mom(:,ke) [the surface-flux row]",
        computed["invs_mom:sca"],
        after.invs_mom(),
        columns=columns,
        levels=slice(nlev - 1, nlev),
    )
    utils.assert_agrees_with_icon(
        "compute_diffusion_inversion_factor",
        "invs_fac",
        computed["invs_fac:sca"],
        after.invs_fac(),
        columns=columns,
        levels=slice(1, nlev),
    )
    utils.assert_agrees_with_icon(
        "compute_inverted_diffusion_momentum",
        "invs_mom [mom]",
        computed["invs_mom:mom"],
        reference["invs_mom:mom"],
        columns=columns,
        levels=slice(0, nlev),
    )
    utils.assert_agrees_with_icon(
        "compute_diffusion_inversion_factor",
        "invs_fac [mom]",
        computed["invs_fac:mom"],
        reference["invs_fac:mom"],
        columns=columns,
        levels=slice(1, nlev),
    )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_solved_profiles_agree_with_the_reference(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """The five solved profiles, which no savepoint holds: the Fortran turns them into tendencies
    in place.

    They are worth comparing separately from the tendencies because the tendency is a difference
    of two profiles that agree to several digits: an error in the solve that is a rounding here
    can be orders of magnitude larger in relative terms by the time it reaches 'dif_tend', and
    it is easier to read at this end.

    One matrix, five right-hand sides -- two for the momentum type and three for the scalars --
    which is the whole point of factorising once per variable type. If the port had rebuilt the
    matrix per variable this test would still pass, so the thing it does show is that reusing it
    is legitimate.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    reference, computed = run.reference, run.computed

    for name, _ in VARIABLES:
        utils.assert_agrees_with_icon(
            "solve_vertical_diffusion_equation",
            f"upd_prof [{name}]",
            computed[f"upd_prof:{name}"],
            reference[f"upd_prof:{name}"],
            columns=columns,
            levels=slice(0, nlev),
        )


@pytest.mark.datatest
@pytest.mark.embedded_too_slow
@utils.experiment_for_turbulence
@pytest.mark.parametrize("date", utils.TURBDIFF_DATES)
def test_the_tendencies_agree_with_icon(
    date: str, *, data_provider: sb.IconSerialDataProvider, grid_savepoint, backend
) -> None:
    """What 'vertdiff' is for: five accumulated tendencies, and the diffusion tendency itself.

    'dicke' is the diffusion tendency of the last variable, cloud water, and the only one of the
    five ICON serializes; the other four are compared against the reference. It is worth its own
    comparison because it is the quantity the accumulation is built from, so a failure here and
    a pass on 'qc_tens' would mean the accumulation cancelled an error.

    't_tens' is bit-exact against ICON despite being the one variable whose profile passes
    through the surface Exner factor: the rounding that reaches 'zvari(:,ke1,tem)' does not
    survive the right-hand side. That is measured rather than assumed -- it is asserted here at
    the same 'Exact()' gate as the other four.
    """
    run = _run_vertdiff(data_provider, grid_savepoint, date, backend)
    nlev, columns = run.nlev, run.columns
    after, reference, computed = run.after, run.reference, run.computed

    utils.assert_agrees_with_icon(
        "compute_and_apply_diffusion_tendency",
        "dicke (dif_tend of qc)",
        computed["dif_tend:qc"],
        after.dif_tend(),
        columns=columns,
        levels=slice(0, nlev),
    )
    for name, _ in VARIABLES:
        stencil = (
            "compute_and_apply_potential_temperature_diffusion_tendency"
            if name == "t"
            else "compute_and_apply_diffusion_tendency"
        )
        utils.assert_agrees_with_icon(
            stencil,
            f"dif_tend [{name}]",
            computed[f"dif_tend:{name}"],
            reference[f"dif_tend:{name}"],
            columns=columns,
        )
        utils.assert_agrees_with_icon(
            stencil,
            f"{name}_tens",
            computed[f"tendency:{name}"],
            getattr(after, f"{name}_tens")(),
            columns=columns,
        )
