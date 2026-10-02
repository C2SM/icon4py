# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The GABLS1 single column: a synthetic state, a surface layer, and nine hours of time steps.

WHY THIS EXISTS, AND WHY IT IS UNLIKE EVERYTHING ELSE IN THIS PACKAGE. Every other test of the
turbulence granule is either a savepoint comparison or a LOCAL algebraic invariant evaluated at
one instant. Neither can see a defect that is small in one step and secular over hours, and
neither can L3: 'ICON4PY_MODE_VERIFY' restarts from the Fortran state at every step precisely so
that errors cannot compound, which makes it structurally blind to this class. GABLS1 integrates
the granule forward for 3240 steps and asks whether the boundary layer it produces is the one
eleven large-eddy simulations produced. It is the only thing here that tests the scheme as a
trajectory.

STATUS (2026-10-02): THERE IS NO TEST, AND THE COLUMN DOES NOT REPRODUCE BEARE. This module is
the driver alone. The agent writing it was killed on 2026-09-03 before 'test_gabls1.py' existed,
so every assertion the comments below once attributed to that file was planned and never
written; each of them now says so. A 15-variant sweep of this driver (job 845490, 'gtfn_cpu')
gives 'h = 324 m' against Beare's 158-211 m, with 'u*' and 'H' outside the published spread as
well. The LLDC floors, not 'tur_len', set the depth, and the surface TKE is inert -- see THE ONE
FREE CHOICE below, and `Gabls1Column._zero_the_tendencies` for a suspected driver defect that
would explain the inertness. Measurements, the open attribution and the next steps are in
'docs/superpowers/notes/2026-10-01-gabls1-status.md' in the workspace. Do not turn this into a
passing test by loosening a tolerance: the failure is the result.

WHAT IT DOES NOT TEST, stated first so that a green run is not over-read:

* NOT 'turbtran', and not ICON's surface coupling. The surface layer below is Beare's own
  Monin-Obukhov specification, evaluated on the host. The granule CONSUMES thirteen surface
  quantities and COMPUTES none of them, so supplying them is filling an interface slot rather
  than working around a missing routine -- and ICON's own SCM does the same thing at the same
  point in the call sequence, 'set_scm_bnd' at 'mo_nwp_turbdiff_interface.f90:510', seventy
  lines above the 'turbdiff' call the granule replaces. But our surface layer is not ICON's:
  'tfh = 1' where ICON has 'tfh < 1' from 'rlam_heat = 10', 'tkvh(ke1) = kappa*u*z0' where
  'turbtran' has '1.26' times that, and the resistance integral is log-linear rather than
  ICON's interpolated profile function. So a disagreement here would not be attributable to
  the granule, and this test cannot be used to defer phase 3.
* NOT a localisation. A GABLS1 failure names no stencil. Diagnosis runs through the savepoint
  suite; this is a gate.
* NOT the unstable half-plane. A cooling surface under a warm layer never reaches 'fh2 < 0', so
  ICON's 'lcorr'/'gama' branch goes unexercised, exactly as it does in every other test here.
* NOT a production configuration, and NOT a performance measurement. 'tur_len = 100', 'tfh = 1',
  'l_pat = 0', every deformation input zero, a dry column, 'dt = 10 s'.

THE ONE FREE CHOICE IS THE SURFACE TKE. Ten of the thirteen surface quantities are fixed by
Beare's own specification and two more by an internal consistency identity of the scheme
('lays = tv/(tkv*tf)' is '1/dz_0a', 'turb_diffusion.f90:1157-1158', so 'tkvm(ke1) = tvm*dz_0a_m'
is forced to be 'kappa*u*z0'). The thirteenth, 'tke(ke1)', is not: 'turbtran' obtains it from a
local-equilibrium TKE budget rather than from a flux law. Its neutral limit is exact and is the
Mellor-Yamada surface value 'q = d_mom**(1/3) * u_star', and under stable stratification
'turbtran' returns LESS than that -- so this choice biases the boundary layer DEEPER, in the
same direction as the mixing-length excess of D-L1. `SURFACE_TKE_FACTOR` is that constant and
`Gabls1Config.surface_tke_factor` is the knob a sensitivity run turns. Turned, it does nothing:
at 1.5 and 2.0 the sweep reproduces the base run in every printed digit, and a 20-step probe on
'gtfn_cpu' moved the surface TKE row from 1.347 to 0.792 and left 'u', 'T' and every other TKE
row bit-identical (2026-09-03, recovered 2026-10-01 from the session transcripts). ICON's TKE
diffusion has a fixed-value lower boundary at the surface level ('lsflucond = .FALSE.',
'k_sf = ke1', 'turb_diffusion.f90:2419-2433'), so that row should reach the interior within a
step; why it does not here is the question `Gabls1Column._zero_the_tendencies` answers.

THE COLUMN STATE LIVES ON THE HOST. The surface layer has to be solved in host arithmetic
anyway -- it is a fixed-point iteration over eighteen columns of eighteen values -- and the
fields are 18 x 321, so a round trip per step costs nothing measurable. Keeping 'u', 'v', 't'
and 'tke' as numpy arrays and pushing them into the granule's fields with 'overwrite_with' is
what makes the time loop readable, and it is the only way the mutation of `TkeCarry` can be
expressed as what it is: a defect of the COMPOSITION, not of any stencil.

THREE TRAPS, ALL ALREADY DOCUMENTED ELSEWHERE IN THE PACKAGE, ALL HIT HERE.
'TurbulenceMetricState.dp0' is bound into two stencils BY IDENTITY when the granule is
constructed, so it must be written in place and never replaced -- here it is never written at
all, see `_hydrostatic_column`. The tendency containers are 'INTENT(INOUT)' and accumulate, so
all six are zeroed before every call. And 'input_state' is frozen, so every update goes through
'overwrite_with' rather than through 'dataclasses.replace' or a raw '__setitem__', which on a
GPU backend would raise.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np

from icon4py.model.atmosphere.subgrid_scale_physics.turbulence import (
    turbulence,
    turbulence_options as options,
    turbulence_states as states,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import simple, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc

from .. import utils as package_utils
from . import gabls1_reference as ref


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


__all__ = [
    "HORIZONTAL_LENGTH_SCALE",
    "SURFACE_TKE_FACTOR",
    "Gabls1Column",
    "Gabls1Config",
    "Gabls1History",
    "SurfaceLayer",
    "TkeCarry",
    "boundary_layer_depth",
    "normalised_flux_exponent",
    "solve_the_surface_layer",
]


#: 'q = d_mom**(1/3) * u_star', the neutral surface-layer turbulent velocity of Mellor-Yamada
#: (1982) Eq. (44a), which is what 'turbtran's local-equilibrium budget reduces to at 'fh2 = 0'
#: ('turb_utilities.f90:1541'). It is 'TurbulenceParams.c_tke', reproduced here as a literal
#: because the surface layer is the driver's and not the granule's, and because a sensitivity
#: run has to be able to move it without moving the granule's own constant.
SURFACE_TKE_FACTOR: float = 16.6 ** (1.0 / 3.0)

#: 'l_hori' [m]: the mean characteristic length of ICON's grid, which the SCM torus for this
#: case ('Torus_Triangles_4x4_2500m.nc') sets to 2500 m. It enters the scheme only through
#: 'l_scal = min(0.5*l_hori, tur_len)' and the shear floor
#: 'fc_min = (vel_min/max(l_hori, tur_len))**2 = 1.6e-11', so it is chosen but not influential.
#: Measured, not asserted -- 'test_gabls1.py', which was to assert it, was never written (STATUS
#: in the module docstring): 'l_hori = 10000' reproduces the base run in every printed digit
#: (sweep job 845490).
HORIZONTAL_LENGTH_SCALE: float = 2500.0

#: 'tkvm = tkvh' at 't = 0' [m2/s]. THE GRANULE HAS NO COLD START, so the caller has to
#: supply one. 'turbdiff' turns the coefficients back into stability lengths through
#: 'l = K/q' ('compute_stability_lengths_from_diffusion_coefficients'), which is '0/0' if both
#: are zero, so an all-zero initial state produces NaN on the first step and not a slow spin-up.
#: ICON does not meet this: 'mo_nwp_phy_init.f90:1726' calls 'turbdiff' with 'iini = 1', which
#: runs the section 1c) this granule does not carry ("NO 'lini' ANYWHERE", 'turbulence.py').
#: One metre squared per second is the order of magnitude of the LLDC floors themselves and is
#: therefore not a large perturbation; the boundary layer equilibrates "within only 0.5 x 10^4 s
#: (1.5 h)" (Beare p. 254), five and a half hours before the window Table IV reports. The
#: sensitivity was measured, not recorded in a test ('test_gabls1.py' was never written): 0.1
#: instead of 1.0 reproduces the base run in every printed digit (sweep job 845490).
INITIAL_DIFFUSION_COEFFICIENT: float = 1.0

#: How far a fixed-point iterate may move before the surface layer is called converged.
_SURFACE_LAYER_TOLERANCE: float = 1.0e-12

#: How many iterations it may take. The contraction factor of the iteration below is
#: 'C*(2*beta_m - beta_h)' with 'C' the bulk stability parameter, which is ~0.05 for this case,
#: so convergence is reached in single digits and a limit of 100 is generous rather than tight.
_SURFACE_LAYER_ITERATIONS: int = 100


class TkeCarry(enum.Enum):
    """Which rows of the turbulent velocity scale the driver carries from one step to the next.

    THE GRANULE HAS TWO TKE FIELDS WHERE THE FORTRAN HAS ONE. ICON runs the scheme with
    'ntim = 1', so 'tke(:,:,nvor)' and 'tke(:,:,ntur)' are the same storage and the Fortran
    updates it in place. ADR-0001 forbids an icon4py physics component writing into its input
    state, so the port splits them: 'input_state.tke' is read, 'diagnostic_state.updated_tke' is
    written, and THE CALLER is what closes the loop between one step and the next. That makes
    the closing a property of the composition, which is precisely the class of defect no
    savepoint test can see -- 'test_turbdiff_granule.py' and 'test_turbulence_granule.py'
    receive both fields from the reference and run one step, so they never exercise a handoff at
    all -- and precisely the class L3 is blind to, because 'ICON4PY_MODE_VERIFY' restarts from
    the Fortran state every step.
    """

    #: Correct: every row of 'updated_tke' becomes the next step's 'tke'.
    WHOLE_COLUMN = "whole_column"
    #: Rows '0..nlev-1' only. This is the mutation the design of 2026-08-31 proposed -- a
    #: translator seeing section 3) write '1..ke' and not noticing that 'ke1' has to be brought
    #: across from 'nvor' makes exactly this mistake. IT DOES NOT BITE IN THIS DRIVER, and that
    #: is a result rather than an oversight: the Monin-Obukhov surface layer prescribes
    #: 'tke[nlev]' afresh at every step, so the row the mutation drops is overwritten before it
    #: is read. Measured, not asserted: the sweep's 'tkeinterior' run reproduces the base run in
    #: every printed digit (job 845490). 'test_gabls1.py', which was to assert it, was never
    #: written, so this comment is where the negative result is recorded.
    INTERIOR_ONLY = "interior_only"
    #: Nothing is carried: every step starts from the initial TKE profile. The same
    #: transcription mistake one field wider, and the one that does bite -- THE WRONG WAY: it
    #: gives 'h = 503.9 m' against the base run's 324.0 (sweep job 845490), deeper, where the
    #: design of 2026-08-31 predicted a collapse below Beare's envelope.
    NONE = "none"


@dataclasses.dataclass(frozen=True)
class Gabls1Config:
    """The GABLS1 column, as a set of choices, with the published ones taken from the reference.

    Only what a test may legitimately vary appears here. Everything Beare fixes -- the initial
    profiles, the cooling rate, the roughness length, the MO constants -- comes from
    `gabls1_reference` and is not a parameter.
    """

    #: Number of main levels. 320 over a 2000 m top is a uniform 'dz = 6.25 m', which is the
    #: resolution Beare's Table IV reports for the full ensemble, so its 158-211 m band is the
    #: directly comparable one. It also puts ~29 layers inside the boundary layer, so the
    #: 5 %-stress crossing is quantised at ~3.5 % rather than the ~13 % of ICON's own 87-level
    #: SCM grid. A COARSE GRID MAKES THE DEPTH ASSERTION MEANINGLESS.
    num_levels: int = 320
    #: Model top [m]. NOT Beare's 400 m domain: the granule ramps the TKE-diffusion implicit
    #: weight from 'impl_t' down to 'impl_s' between 1500 m and the surface, and a 400 m column
    #: never crosses 1500 m, so 'ramp_level' stays 1 and the ramp spreads over the whole column
    #: -- a purely numerical difference introduced by the domain choice. 2000 m is also ICON's
    #: own GABLS1 SCM 'top_height'.
    model_top_height: float = 2000.0
    #: Time step [s], for the diffusion and for the TKE alike. ICON's 'run_SCM_GABLS1' uses 10.
    time_step: float = 10.0
    #: Length of the integration [s].
    run_length: float = ref.RUN_LENGTH
    #: 'tur_len' [m], the ASYMPTOTIC TURBULENT DISTANCE. The asymptotic MIXING length is
    #: 'akt*tur_len', so 100 m here is 40 m of mixing length, which is exactly the 'lambda_0' of
    #: Beare Eq. (5) -- the identical functional form, 'z+z0' included. ICON's default of 500 m
    #: is 200 m, five times Beare's. 'test_gabls1.py' was to run both and was never written; the
    #: sweep ran them, and 'h' is 324.0, 324.3 and 324.3 m at 100, 300 and 500 (job 845490). So
    #: 'tur_len' does NOT set the depth here, and the 100/500 pair discriminates nothing.
    tur_len: float = 100.0
    #: 'tkhmin', 'tkmmin' [m2/s], the lower limits of the diffusion coefficients. ICON's default
    #: 0.75 is a floor no LES has, in exactly the regime -- strongly stable, small K -- where
    #: the closure would otherwise decouple levels. This comment called it the SECOND
    #: over-mixing mechanism beside the mixing length; measured, it is the dominant one: 'h' is
    #: 324 m with it, 59-90 m without it (at each of 'tur_len' 100, 300, 500) and 133 m at 0.01
    #: (sweep job 845490), and no tested setting lands in Beare's 158-211 m.
    tkhmin: float = 0.75
    tkmmin: float = 0.75
    #: 'l_hori' [m]; see `HORIZONTAL_LENGTH_SCALE`.
    l_hori: float = HORIZONTAL_LENGTH_SCALE
    #: 'tke[nlev] = surface_tke_factor * u_star'; see the module docstring and
    #: `SURFACE_TKE_FACTOR`.
    surface_tke_factor: float = SURFACE_TKE_FACTOR
    #: 'tkvh[nlev] = scalar_diffusivity_ratio * kappa * u_star * z0'. One by the consistency
    #: identity, because Beare's MO layer has a neutral turbulent Prandtl number of one;
    #: 'turbtran' uses 'S_h/S_m = 1/Pr_n = 1.26' instead, and the sensitivity to that 26 % is
    #: measured rather than argued. Measured, it is nil: 1.26 reproduces the base run in every
    #: printed digit (sweep job 845490), and in the 20-step probe it moved the surface 'tkvh'
    #: from 0.02112 to 0.02661 and left 'tke' and 'T' bit-identical -- the same inertness as
    #: `surface_tke_factor` (module docstring).
    scalar_diffusivity_ratio: float = 1.0
    #: Which rows of the TKE the driver carries across a step; see `TkeCarry`.
    tke_carry: TkeCarry = TkeCarry.WHOLE_COLUMN
    #: How often the boundary-layer depth is diagnosed, in steps. Every step would be correct
    #: and pointless: 'h' is a smooth function of a 9-hour integration and the profile has to
    #: come back to the host to compute it.
    diagnostic_interval: int = 30

    @property
    def num_steps(self) -> int:
        """Number of time steps in the integration."""
        return round(self.run_length / self.time_step)

    @property
    def layer_thickness(self) -> float:
        """Uniform layer thickness [m]."""
        return self.model_top_height / self.num_levels


@dataclasses.dataclass(frozen=True)
class SurfaceLayer:
    """One evaluation of Beare's Monin-Obukhov surface layer, per column.

    Every member is a numpy array of one value per column. The names are the physical ones; the
    translation into the granule's thirteen surface slots happens in `Gabls1Column._apply`.
    """

    #: Friction velocity [m/s].
    friction_velocity: np.ndarray
    #: Temperature scale 'theta_star' [K]. Positive under stable stratification.
    temperature_scale: np.ndarray
    #: Obukhov length [m]. Positive under stable stratification, infinite at neutral.
    obukhov_length: np.ndarray
    #: 'z_a/L', the stability parameter at the lowest main level [1].
    stability_parameter: np.ndarray
    #: 'tvm', the turbulent transfer velocity for momentum [m/s].
    transfer_velocity_for_momentum: np.ndarray
    #: 'tvh', the turbulent transfer velocity for heat [m/s].
    #:
    #: WRITTEN AS 'kappa*u*/(A + beta_h*zeta)' AND NOT AS 'u_star*theta_star/(theta_a -
    #: theta_s)'. The two are algebraically the same and the second is '0/0' at 't = 0', where
    #: GABLS1 starts exactly neutral. A zero transfer velocity is not the neutral limit -- it is
    #: an INFINITE surface resistance, which 'vertdiff' turns into an infinite diffusion depth
    #: and then into a NaN temperature tendency on the very first step.
    transfer_velocity_for_scalars: np.ndarray
    #: Wind speed at the lowest main level, floored at 'vel_min' [m/s].
    wind_speed: np.ndarray
    #: 'theta_a - theta_s' [K].
    potential_temperature_difference: np.ndarray
    #: Number of fixed-point iterations the slowest column needed.
    iterations: int


def solve_the_surface_layer(
    *,
    wind_speed: np.ndarray,
    potential_temperature_difference: np.ndarray,
    reference_height: float,
    roughness_length: float = ref.ROUGHNESS_LENGTH,
    von_karman: float = ref.VON_KARMAN,
    beta_m: float = ref.BETA_M,
    beta_h: float = ref.BETA_H,
    reference_theta: float = ref.REFERENCE_THETA,
    gravity: float = constants.GRAV,
) -> SurfaceLayer:
    """Beare's lower boundary condition, iterated to a fixed point.

    The system is the log-linear Monin-Obukhov surface layer with the constants the
    intercomparison prescribes (Beare p. 250, 'beta_m = 4.8', 'beta_h = 7.8', 'z0 = 0.1 m' for
    momentum AND heat, 'kappa = 0.4'):

        zeta   = z_a/L
        u*     = kappa * U_a           / (ln((z_a + z0)/z0) + beta_m * zeta)
        theta* = kappa * (th_a - th_s) / (ln((z_a + z0)/z0) + beta_h * zeta)
        L      = u*^2 * theta_ref / (kappa * g * theta*)

    IT CONVERGES, AND THAT IS NOT OBVIOUS. Substituting gives
    'zeta = C * (A + beta_m*zeta)^2 / (A + beta_h*zeta)' with 'A = ln((z_a+z0)/z0)' and
    'C = z_a*g*(th_a - th_s)/(theta_ref*U_a^2)' a bulk stability parameter. The derivative of
    the right-hand side at 'zeta = 0' is 'C*(2*beta_m - beta_h) = 1.8*C', and 'C' stays below
    0.03 throughout this case, so the iteration is a contraction with factor ~0.05. Far from
    neutral it would not be -- the large-'zeta' slope is 'C*beta_m^2/beta_h = 2.95*C' -- which
    is why the iteration count is checked and a failure to converge RAISES rather than returning
    the last iterate, following '_iterate_to_the_fixed_point' of
    'test_neutral_surface_layer.py'.

    The same substitution makes the fixed point the root of a quadratic, which was to be the
    independent check -- two derivations of one number, neither reading the other's code -- in
    'test_gabls1.py::test_the_surface_layer_solves_its_own_defining_equation'. That test was
    never written; nothing checks this function today beyond the convergence guard below.

    Args:
        wind_speed: '|V|' at the lowest main level, already floored at 'vel_min' [m/s].
        potential_temperature_difference: 'theta_a - theta_s' [K].
        reference_height: 'z_a', the height of the lowest main level [m]. ICON's 'h_atm',
            'turb_transfer.f90:901-902' with 'xf = 1/2'.

    Raises:
        RuntimeError: If the iteration has not converged after `_SURFACE_LAYER_ITERATIONS`.
    """
    resistance = math.log((reference_height + roughness_length) / roughness_length)
    stability = np.zeros_like(wind_speed)
    movement = float("inf")
    iteration = 0
    converged = False
    while iteration < _SURFACE_LAYER_ITERATIONS:
        iteration += 1
        friction_velocity = von_karman * wind_speed / (resistance + beta_m * stability)
        temperature_scale = (
            von_karman * potential_temperature_difference / (resistance + beta_h * stability)
        )
        # 'L' is infinite at neutral and the stability parameter is then exactly zero, which is
        # what 'np.where' on the denominator produces without a division warning.
        buoyancy = von_karman * gravity * temperature_scale / reference_theta
        updated = np.where(
            buoyancy == 0.0,
            0.0,
            reference_height * buoyancy / (friction_velocity * friction_velocity),
        )
        movement = float(np.max(np.abs(updated - stability)))
        converged = movement < _SURFACE_LAYER_TOLERANCE
        stability = updated
        if converged:
            break
    if not converged:
        raise RuntimeError(
            f"The Monin-Obukhov surface layer did not converge in "
            f"{_SURFACE_LAYER_ITERATIONS} iterations; the last step moved the stability "
            f"parameter by {movement:.3e}. The bulk stability "
            f"parameter was "
            f"{np.max(reference_height * gravity * potential_temperature_difference / (reference_theta * wind_speed * wind_speed)):.4f}; "
            f"above 'beta_h/beta_m**2 = {beta_h / (beta_m * beta_m):.4f}' this system has no "
            f"positive root at all."
        )
    friction_velocity = von_karman * wind_speed / (resistance + beta_m * stability)
    temperature_scale = (
        von_karman * potential_temperature_difference / (resistance + beta_h * stability)
    )
    transfer_velocity_for_momentum = (
        von_karman * friction_velocity / (resistance + beta_m * stability)
    )
    transfer_velocity_for_scalars = (
        von_karman * friction_velocity / (resistance + beta_h * stability)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        obukhov_length = np.where(
            temperature_scale == 0.0,
            np.inf,
            friction_velocity
            * friction_velocity
            * reference_theta
            / (von_karman * gravity * temperature_scale),
        )
    return SurfaceLayer(
        friction_velocity=friction_velocity,
        temperature_scale=temperature_scale,
        obukhov_length=obukhov_length,
        stability_parameter=stability,
        transfer_velocity_for_momentum=transfer_velocity_for_momentum,
        transfer_velocity_for_scalars=transfer_velocity_for_scalars,
        wind_speed=wind_speed,
        potential_temperature_difference=potential_temperature_difference,
        iterations=iteration,
    )


@dataclasses.dataclass(frozen=True)
class Gabls1History:
    """What the integration recorded, in the units the paper quotes.

    Profiles are for column 0; `Gabls1Column.columns_agree` is what says the other seventeen
    are the same.
    """

    #: Time of every recorded sample [s].
    time: np.ndarray
    #: Friction velocity at every sample [m/s].
    friction_velocity: np.ndarray
    #: Surface sensible heat flux at every sample [W/m2], magnitude.
    heat_flux: np.ndarray
    #: Surface buoyancy flux at every sample [m2/s3], magnitude.
    buoyancy_flux: np.ndarray
    #: Boundary-layer depth at every sample [m], by Beare's 5 %-stress rule.
    depth: np.ndarray
    #: Half-level heights [m], top to surface, 'nlev + 1' entries.
    half_level_height: np.ndarray
    #: Main-level heights [m], top to surface, 'nlev' entries.
    main_level_height: np.ndarray
    #: Kinematic momentum flux magnitude on half levels, averaged over the final hour [m2/s2].
    mean_stress: np.ndarray
    #: Buoyancy flux on half levels, averaged over the final hour, magnitude [m2/s3].
    mean_buoyancy_flux: np.ndarray
    #: Momentum diffusion coefficient on half levels, averaged over the final hour [m2/s].
    mean_diffusion_momentum: np.ndarray
    #: Scalar diffusion coefficient on half levels, averaged over the final hour [m2/s].
    mean_diffusion_scalar: np.ndarray
    #: Wind speed on main levels, averaged over the final hour [m/s].
    mean_wind_speed: np.ndarray
    #: Wind speed on main levels, averaged over the penultimate hour [m/s].
    penultimate_mean_wind_speed: np.ndarray
    #: Potential temperature on main levels, averaged over the final hour [K].
    mean_potential_temperature: np.ndarray
    #: Friction velocity averaged over the final hour [m/s].
    final_hour_friction_velocity: float
    #: Surface heat flux averaged over the final hour [W/m2], magnitude.
    final_hour_heat_flux: float
    #: Surface buoyancy flux averaged over the final hour [m2/s3], magnitude.
    final_hour_buoyancy_flux: float
    #: Obukhov length averaged over the final hour [m].
    final_hour_obukhov_length: float
    #: Whether all eighteen columns held identical values at every recorded sample.
    columns_agree: bool

    @property
    def boundary_layer_depth(self) -> float:
        """The final-hour boundary-layer depth [m], from the final-hour mean stress profile.

        Beare p. 250 computes 'h' from the MEAN stress, and Table IV reports it for the last
        hour of the simulation, so this is the number Table IV is comparable with -- not the
        mean of the per-sample depths, which would average a quantised crossing.
        """
        return boundary_layer_depth(
            stress=self.mean_stress,
            surface_stress=self.final_hour_friction_velocity * self.final_hour_friction_velocity,
            height=self.half_level_height,
        )

    @property
    def wind_stationarity(self) -> float:
        """RMS difference of the wind speed between the 7-8 h and 8-9 h means [m/s].

        Beare p. 256's own quasi-steadiness criterion, which the paper reports as smaller than
        0.1 m/s for every LES.
        """
        difference = self.mean_wind_speed - self.penultimate_mean_wind_speed
        return float(np.sqrt(np.mean(difference * difference)))


def boundary_layer_depth(*, stress: np.ndarray, surface_stress: float, height: np.ndarray) -> float:
    """Beare's boundary-layer depth: where the mean stress falls to 5 %, extrapolated by /0.95.

    'The boundary-layer depth (h) calculation involved first determining the height where the
    mean stress fell to 5 % of its surface value (h_0.05) followed by linear extrapolation:
    h = h_0.05/0.95' -- Beare p. 250.

    The crossing is found from the surface upward and interpolated linearly between the two
    half levels that bracket it, which is what makes the answer finer than the grid. Rows are
    ordered top to surface, as every field in this package is.

    THE SEARCH SKIPS THE TWO END ROWS, and it has to. Row 0 is the model top, which no section
    of 'turbdiff' writes a diffusion coefficient into, and row 'nlev' is the surface flux level,
    whose flux is the surface layer's business and not the column's -- `Gabls1Column.
    stress_and_buoyancy_flux` leaves both at zero. Searching from row 'nlev' would find a stress
    of zero at the first row it looked at and report a boundary layer of no depth at all.

    Args:
        stress: Kinematic momentum flux magnitude on half levels [m2/s2], top to surface.
        surface_stress: 'tau_0 = u_star**2' [m2/s2].
        height: Half-level heights [m], top to surface.

    Returns:
        'h' [m], or 'nan' if the stress never falls below the threshold anywhere in the column.
    """
    threshold = ref.DEPTH_STRESS_FRACTION * surface_stress
    # Bottom-up over the interior: index 'nlev - 1' is the lowest computed flux level, index 1
    # the highest.
    from_the_surface = np.arange(len(stress) - 2, 0, -1)
    for position, level in enumerate(from_the_surface):
        if stress[level] < threshold:
            if position == 0:
                return 0.0
            below = from_the_surface[position - 1]
            span = stress[below] - stress[level]
            weight = 0.0 if span == 0.0 else (stress[below] - threshold) / span
            crossing = height[below] + weight * (height[level] - height[below])
            return ref.depth_extrapolated_from_the_stress_crossing(float(crossing))
    return float("nan")


def normalised_flux_exponent(
    *,
    flux: np.ndarray,
    height: np.ndarray,
    depth: float,
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    """The exponent 'm' of 'flux/flux_0 = (1 - z/h)**m', by least squares in log-log.

    Nieuwstadt (1985), reproduced at Beare p. 264 as Eqs. (3a) and (3b), with 'm_wh = 1' for the
    buoyancy flux and 'm_tau = 1.5' for the stress. The fit is restricted to
    '0.1 <= z/h <= 0.9': below that the surface layer is not self-similar and above it the
    logarithm of a vanishing flux is dominated by the discretisation.

    The surface value is taken from the fit range's own extrapolation rather than from the
    surface row, so that a mis-set surface flux cannot move the exponent -- the exponent is
    meant to be independent of both 'h' and the surface flux, which is exactly why Beare
    pp. 264-265 says the normalised profiles are where the LES agree.

    Returns:
        The exponent, or 'nan' if fewer than three levels fall inside the fit range.
    """
    normalised_height = height / depth
    inside = (normalised_height >= lower) & (normalised_height <= upper) & (flux > 0.0)
    minimum_points = 3
    if np.count_nonzero(inside) < minimum_points:
        return float("nan")
    abscissa = np.log(1.0 - normalised_height[inside])
    ordinate = np.log(flux[inside])
    slope, _ = np.polyfit(abscissa, ordinate, 1)
    return float(slope)


class Gabls1Column:
    """The GABLS1 column: build it, step it, and hand back what Beare's tables can judge.

    One instance owns one granule, one set of state containers and one host-side column state.
    `integrate` runs the whole nine hours; `step` is one time step and exists so that a test can
    watch the first few.
    """

    def __init__(self, config: Gabls1Config, backend: gtx_typing.Backend | None) -> None:
        self._config = config
        self._backend = backend
        self._nlev = config.num_levels
        self._grid = simple.simple_grid(allocator=backend, num_levels=config.num_levels)
        self._num_cells = self._grid.num_cells

        self._build_the_vertical_grid()
        self._build_the_initial_state()
        self._build_the_containers()
        self._build_the_granule()

    # ------------------------------------------------------------------ construction ---

    def _build_the_vertical_grid(self) -> None:
        """The uniform grid, derived analytically by icon4py's own equidistant branch.

        'get_vct_a_and_vct_b' with 'file_path = None' and 'lowest_layer_thickness <= 0.01'
        produces 'vct_a[k] = H*(N + 1 - k)/N' -- byte for byte ICON's 'mo_init_vgrid.f90:249-253'
        -- so no savepoint is needed for the vertical coordinate.
        """
        vertical_config = v_grid.VerticalGridConfig(
            num_levels=self._config.num_levels,
            model_top_height=self._config.model_top_height,
            lowest_layer_thickness=-10.0,
        )
        vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, self._backend)
        self._vertical_grid = v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b)
        self.half_level_height = data_alloc.as_numpy(vct_a).astype(float)
        self.main_level_height = 0.5 * (self.half_level_height[:-1] + self.half_level_height[1:])

    def _build_the_initial_state(self) -> None:
        """Beare's initial profiles, pp. 249-250, and the hydrostatic column that carries them."""
        height = self.main_level_height
        theta = np.where(
            height <= ref.MIXED_LAYER_DEPTH,
            ref.THETA_MIXED_LAYER,
            ref.THETA_MIXED_LAYER + ref.LAPSE_RATE * (height - ref.MIXED_LAYER_DEPTH),
        )
        exner, pressure, temperature, density, half_level_pressure = _hydrostatic_column(
            potential_temperature=theta,
            main_level_height=self.main_level_height,
            half_level_height=self.half_level_height,
            surface_pressure=ref.SURFACE_PRESSURE,
        )
        self._exner = exner
        self._pressure = pressure
        self._density = density
        self._layer_pressure_thickness = np.diff(half_level_pressure)
        self._surface_exner = float((ref.SURFACE_PRESSURE / constants.P0REF) ** constants.RD_O_CPD)
        self._surface_density = float(
            ref.SURFACE_PRESSURE / (constants.RD * ref.INITIAL_SURFACE_TEMPERATURE)
        )

        # 'q = sqrt(2*TKE)' from Beare's TKE profile, on HALF levels: the granule's field is the
        # turbulent velocity scale and not the energy ('turbulence_states.py', 'tke').
        tke_density = np.where(
            self.half_level_height < ref.INITIAL_TKE_DEPTH,
            ref.INITIAL_TKE_AMPLITUDE
            * (1.0 - self.half_level_height / ref.INITIAL_TKE_DEPTH) ** ref.INITIAL_TKE_EXPONENT,
            0.0,
        )
        # FLOORED AT 'vel_min', WHICH BEARE'S PROFILE IS NOT. Above 250 m his TKE is EXACTLY
        # zero, which an LES can carry and this scheme cannot: 'l = K/q' divides by it, and
        # ICON's own scheme keeps 'q' at or above 'tkesecu*vel_min = 0.01 m/s' for that reason.
        # The floor is 5e-5 m2/s2 of energy, four orders below the profile's amplitude.
        self.temperature = np.tile(temperature, (self._num_cells, 1))
        self.zonal_wind = np.full((self._num_cells, self._nlev), ref.GEOSTROPHIC_WIND, dtype=float)
        self.meridional_wind = np.zeros((self._num_cells, self._nlev), dtype=float)
        self.tke = np.tile(
            np.maximum(np.sqrt(2.0 * tke_density), turbulence.TurbulenceConfig().vel_min),
            (self._num_cells, 1),
        )
        self.time = 0.0

    def _build_the_containers(self) -> None:
        """The five state containers, every member allocated and every input given its value."""
        self._metric_state = states.TurbulenceMetricState(
            hhl=self._half(np.tile(self.half_level_height, (self._num_cells, 1))),
            dp0=self._main(np.tile(self._layer_pressure_thickness, (self._num_cells, 1))),
            l_hori=self._surface(np.full(self._num_cells, self._config.l_hori)),
            # 73 deg N: the tropical smoothing of the TKE forcing does not apply, and
            # 'frcsmot = 0' would switch it off in any case.
            trop_mask=self._surface(np.zeros(self._num_cells)),
            innertrop_mask=self._surface(np.zeros(self._num_cells)),
        )
        zero_main = np.zeros((self._num_cells, self._nlev))
        self._input_state = states.TurbulenceInputState(
            u=self._main(self.zonal_wind),
            v=self._main(self.meridional_wind),
            t=self._main(self.temperature),
            w=self._half(np.zeros((self._num_cells, self._nlev + 1))),
            qv=self._main(zero_main),
            qc=self._main(zero_main),
            prs=self._main(np.tile(self._pressure, (self._num_cells, 1))),
            rhoh=self._main(np.tile(self._density, (self._num_cells, 1))),
            epr=self._main(np.tile(self._exner, (self._num_cells, 1))),
            tke=self._half(self.tke),
            tracers=(),
            ut_sso=self._main(zero_main),
            vt_sso=self._main(zero_main),
            tket_conv=self._half(np.zeros((self._num_cells, self._nlev + 1))),
            hdef2=self._half(np.zeros((self._num_cells, self._nlev + 1))),
            hdiv=self._half(np.zeros((self._num_cells, self._nlev + 1))),
            dwdx=self._half(np.zeros((self._num_cells, self._nlev + 1))),
            dwdy=self._half(np.zeros((self._num_cells, self._nlev + 1))),
        )
        ones = np.ones(self._num_cells)
        zeros = np.zeros(self._num_cells)
        self._surface_state = states.TurbulenceSurfaceState(
            t_g=self._surface(np.full(self._num_cells, ref.INITIAL_SURFACE_TEMPERATURE)),
            qv_s=self._surface(zeros),
            ps=self._surface(np.full(self._num_cells, ref.SURFACE_PRESSURE)),
            fr_land=self._surface(ones),
            l_lake=data_alloc.zero_field(
                self._grid, dims.CellDim, dtype=bool, allocator=self._backend
            ),
            l_sice=data_alloc.zero_field(
                self._grid, dims.CellDim, dtype=bool, allocator=self._backend
            ),
            # Zero switches the circulation term's acceleration off exactly, and only because
            # the column is dry: 'compute_circulation_acceleration' forms
            # 'max(l_pat, sqrt(coherence*L*l_hori))' with 'coherence = 1 - 2*|clc - 0.5|', which
            # is zero at 'clc = 0'. 'pat_len' stays at its default so 'lcircterm' is still true
            # and both circulation programs still run -- with a known answer of zero.
            l_pat=self._surface(zeros),
            urb_isa=self._surface(zeros),
            rlamh_fac=self._surface(ones),
            z0_waves=self._surface(zeros),
        )
        half_zeros = np.zeros((self._num_cells, self._nlev + 1))
        self._diagnostic_state = states.TurbulenceDiagnosticState(
            gz0=self._surface(np.full(self._num_cells, constants.GRAV * ref.ROUGHNESS_LENGTH)),
            tvm=self._surface(zeros),
            tvh=self._surface(zeros),
            # 'tfm = tfh = 1'. 'tfm = dz_0a_m/dz_sa_m' and 'dz_s0_m = 0' whenever
            # 'rlam_mom <= 0', which is ICON's default, so 'tfm = 1' in a standard run too --
            # and it is what makes the surface wind 'u(ke1) = u(ke)*(1 - tfm)' vanish, which is
            # what Beare's MO lower boundary means. 'tfh = 1' is a DELIBERATE DIVERGENCE from
            # ICON's 'rlam_heat = 10': Beare has no excess resistance for heat, 'z0h = z0m'.
            tfm=self._surface(ones),
            tfh=self._surface(ones),
            tfv=self._surface(zeros),
            tkred_sfc=self._surface(ones),
            tkred_sfc_h=self._surface(ones),
            # See `INITIAL_DIFFUSION_COEFFICIENT`: zero here is not a small initial condition,
            # it is a division by zero on the first step.
            tkvm=self._half(np.full_like(half_zeros, INITIAL_DIFFUSION_COEFFICIENT)),
            tkvh=self._half(np.full_like(half_zeros, INITIAL_DIFFUSION_COEFFICIENT)),
            # 'rcld' is the supersaturation deviation; the column is dry, so its surface row is
            # zero and the granule writes the rest.
            rcld=self._half(half_zeros),
            # 'turbdiff' writes rows 1..nlev; row 0 is the model top and no section touches it,
            # so it is given the density of the layer below rather than a zero that a stray read
            # would turn into an infinity.
            rhon=self._half(
                np.tile(
                    np.concatenate(
                        (
                            self._density[:1],
                            0.5 * (self._density[:-1] + self._density[1:]),
                            self._density[-1:],
                        )
                    ),
                    (self._num_cells, 1),
                )
            ),
            updated_tke=self._half(half_zeros),
            shfl_s=self._surface(zeros),
            qvfl_s=self._surface(zeros),
            tcm=self._surface(zeros),
            tch=self._surface(zeros),
            umfl_s=self._surface(zeros),
            vmfl_s=self._surface(zeros),
            t_2m=self._surface(zeros),
            qv_2m=self._surface(zeros),
            td_2m=self._surface(zeros),
            rh_2m=self._surface(zeros),
            u_10m=self._surface(zeros),
            v_10m=self._surface(zeros),
            tprn=self._half(half_zeros),
            edr=self._half(half_zeros),
            tur_len_scale=self._half(half_zeros),
        )
        self._tendency_state = states.TurbulenceTendencyState(
            ddt_u=self._main(zero_main),
            ddt_v=self._main(zero_main),
            ddt_t=self._main(zero_main),
            ddt_qv=self._main(zero_main),
            ddt_qc=self._main(zero_main),
            ddt_tke=self._half(half_zeros),
            ddt_tracers=(),
            tket_hshr=self._half(half_zeros),
        )

    def _build_the_granule(self) -> None:
        """The configuration, and the three switches the granule forces.

        'itype_sher = 2', 'ltkeshs' and 'ltkesso' are refused at any other value, and ICON's own
        GABLS1 SCM namelist sets the opposite of all three. That is not a contradiction: every
        input those three terms read -- 'dwdx', 'dwdy', 'hdiv', 'hdef2', 'ut_sso', 'vt_sso' --
        is identically zero here, so the mandatory configuration and ICON's produce the same
        numbers. That is still an argument: 'test_gabls1.py', which was to assert it, was never
        written.
        """
        self.config_used = turbulence.TurbulenceConfig(
            itype_sher=options.ShearProductionType.VERTICAL_AND_VERTICAL_VELOCITY,
            tur_len=self._config.tur_len,
            tkhmin=self._config.tkhmin,
            tkmmin=self._config.tkmmin,
        )
        self.granule = turbulence.Turbulence(
            grid=self._grid,
            config=self.config_used,
            params=turbulence.TurbulenceParams(self.config_used),
            vertical_grid=self._vertical_grid,
            metric_state=self._metric_state,
            backend=self._backend,
        )

    # --------------------------------------------------------------------- allocation ---

    def _half(self, values: np.ndarray) -> gtx.Field:
        field = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=self._backend
        )
        package_utils.overwrite_with(field, values, self._backend)
        return field

    def _main(self, values: np.ndarray) -> gtx.Field:
        field = data_alloc.zero_field(self._grid, dims.CellDim, dims.KDim, allocator=self._backend)
        package_utils.overwrite_with(field, values, self._backend)
        return field

    def _surface(self, values: np.ndarray) -> gtx.Field:
        field = data_alloc.zero_field(self._grid, dims.CellDim, allocator=self._backend)
        package_utils.overwrite_with(field, values, self._backend)
        return field

    # ------------------------------------------------------------------- the time loop ---

    def surface_potential_temperature(self, time: float) -> float:
        """'theta_s(t) = 265 K - 0.25 K/h * t', Beare p. 249.

        Beare specifies a Boussinesq case, so his surface value is a POTENTIAL temperature. The
        granule's 't_g' is an absolute temperature, from which it forms
        'theta_s = t_g/Pi_s' with the Exner factor of the SURFACE PRESSURE
        ('compute_conserved_variables_and_factors_at_the_surface': the roughness layer is
        treated as massless, so one Exner factor serves both levels). So 't_g' is set to
        'theta_s * Pi_s' and the surface layer here reads it back the same way -- otherwise the
        column would start 1 K stably stratified instead of neutral, which is not the case.
        """
        return ref.INITIAL_SURFACE_TEMPERATURE - ref.SURFACE_COOLING_RATE * time

    def _apply(self, layer: SurfaceLayer) -> None:
        """Write the thirteen surface quantities the granule reads, in place."""
        nlev = self._nlev
        friction_velocity = layer.friction_velocity
        transfer_momentum = layer.transfer_velocity_for_momentum
        transfer_scalar = layer.transfer_velocity_for_scalars
        # The consistency identity of 'turb_diffusion.f90:1157-1158':
        # 'lays = tv/(tkv*tf) = 1/dz_0a', so 'tkv(ke1) = tv*dz_0a = kappa*u_star*z0' with
        # 'tf = 1' and the MO resistance integral. It is a determination, not a choice.
        surface_diffusivity = ref.VON_KARMAN * friction_velocity * ref.ROUGHNESS_LENGTH
        package_utils.overwrite_with(self._diagnostic_state.tvm, transfer_momentum, self._backend)
        package_utils.overwrite_with(self._diagnostic_state.tvh, transfer_scalar, self._backend)
        self._write_row(self._diagnostic_state.tkvm, nlev, surface_diffusivity)
        self._write_row(
            self._diagnostic_state.tkvh,
            nlev,
            self._config.scalar_diffusivity_ratio * surface_diffusivity,
        )
        self._write_row(self._diagnostic_state.rcld, nlev, np.zeros_like(friction_velocity))
        # The one free choice; see the module docstring.
        self.tke[:, nlev] = self._config.surface_tke_factor * friction_velocity
        # 'shfl_s = rho_s * c_pd * Pi_s * tvh * (th_a - th_s)', POSITIVE DOWNWARD. It is the
        # closed form 'turbtran' itself reduces to ('turb_transfer.f90:1682-1684'), and in a
        # stable layer 'th_a > th_s' so it comes out positive.
        heat_flux = (
            self._surface_density
            * constants.CPD
            * self._surface_exner
            * transfer_scalar
            * layer.potential_temperature_difference
        )
        package_utils.overwrite_with(self._diagnostic_state.shfl_s, heat_flux, self._backend)
        package_utils.overwrite_with(
            self._diagnostic_state.qvfl_s, np.zeros_like(heat_flux), self._backend
        )

    def _write_row(self, field: gtx.Field, level: int, values: np.ndarray) -> None:
        """Overwrite one K row of a (Cell, K) field, on the backend's own device."""
        array_ns = data_alloc.import_array_ns(self._backend)
        field.ndarray[: self._num_cells, level] = array_ns.asarray(values)

    def evaluate_the_surface_layer(self) -> SurfaceLayer:
        """Solve the surface layer for the current column state."""
        nlev = self._nlev
        speed = np.maximum(
            self.config_used.vel_min,
            np.hypot(self.zonal_wind[:, nlev - 1], self.meridional_wind[:, nlev - 1]),
        )
        potential_temperature = self.temperature[:, nlev - 1] / self._exner[nlev - 1]
        return solve_the_surface_layer(
            wind_speed=speed,
            potential_temperature_difference=(
                potential_temperature - self.surface_potential_temperature(self.time)
            ),
            reference_height=float(self.main_level_height[nlev - 1]),
        )

    def step(self) -> SurfaceLayer:
        """One time step: surface layer, granule, explicit update, TKE carry.

        The Coriolis and geostrophic terms are the DRIVER'S. The granule diffuses; it does not
        know about 'f'. 'f*dt = 1.39e-3' per step, so a forward-explicit rotation is accurate
        over 3240 steps: it amplifies a pure rotation by 'sqrt(1 + (f*dt)**2)' per step, about
        0.3 % over the run. That is argued, not verified: the control run that was to show it,
        conserving '|V - V_g|' with the granule's tendencies discarded, was planned for
        'test_gabls1.py', which was never written.
        """
        layer = self.evaluate_the_surface_layer()
        self._apply(layer)
        theta_s = self.surface_potential_temperature(self.time)
        package_utils.overwrite_with(
            self._surface_state.t_g,
            np.full(self._num_cells, theta_s * self._surface_exner),
            self._backend,
        )
        package_utils.overwrite_with(self._input_state.u, self.zonal_wind, self._backend)
        package_utils.overwrite_with(self._input_state.v, self.meridional_wind, self._backend)
        package_utils.overwrite_with(self._input_state.t, self.temperature, self._backend)
        package_utils.overwrite_with(self._input_state.tke, self.tke, self._backend)
        self._zero_the_tendencies()

        self.granule.run(
            input_state=self._input_state,
            surface_state=self._surface_state,
            diagnostic_state=self._diagnostic_state,
            tendency_state=self._tendency_state,
            dt_var=self._config.time_step,
            dt_tke=self._config.time_step,
        )

        dt = self._config.time_step
        coriolis = ref.CORIOLIS_PARAMETER
        ddt_u = data_alloc.as_numpy(self._tendency_state.ddt_u)[: self._num_cells]
        ddt_v = data_alloc.as_numpy(self._tendency_state.ddt_v)[: self._num_cells]
        ddt_t = data_alloc.as_numpy(self._tendency_state.ddt_t)[: self._num_cells]
        zonal = self.zonal_wind + dt * (ddt_u + coriolis * self.meridional_wind)
        meridional = self.meridional_wind + dt * (
            ddt_v - coriolis * (self.zonal_wind - ref.GEOSTROPHIC_WIND)
        )
        self.zonal_wind = zonal
        self.meridional_wind = meridional
        self.temperature = self.temperature + dt * ddt_t
        self._carry_the_tke()
        self.time += dt
        return layer

    def _zero_the_tendencies(self) -> None:
        """All six accumulate; 'ddt_tke' is read on entry as the transport tendency 'tvt'.

        SUSPECTED DRIVER DEFECT, found 2026-10-02 by reading the source; neither measured nor
        fixed. Zeroing 'ddt_tke' here throws away the TKE-diffusion tendency the previous call
        wrote in section 10). ICON never resets it between calls: with TKE advection on,
        'mo_nwp_turbdiff_interface.f90:288' adds the advection tendency ONTO it, and the next
        'turbdiff' reads it in section 3) as 'tvt', the "turbulent transport of turbulent
        velocity scale" ('turb_diffusion.f90:1829', 'turb_utilities.f90:1143', used at ':1499').
        So in ICON the TKE diffusion acts one step late, and here it never acts: section 9)'s
        result reaches nothing, and the surface TKE row, whose only way into the interior is
        that diffusion, cannot matter -- which is the inertness the sweep measured (module
        docstring). The likely fix, carrying 'ddt_tke' and zeroing only the other five, will
        move every number in that sweep, so it needs the sweep rerun; see
        'docs/superpowers/notes/2026-10-01-gabls1-status.md'.
        """
        zero_main = np.zeros((self._num_cells, self._nlev))
        zero_half = np.zeros((self._num_cells, self._nlev + 1))
        for field, zero in (
            (self._tendency_state.ddt_u, zero_main),
            (self._tendency_state.ddt_v, zero_main),
            (self._tendency_state.ddt_t, zero_main),
            (self._tendency_state.ddt_qv, zero_main),
            (self._tendency_state.ddt_qc, zero_main),
            (self._tendency_state.ddt_tke, zero_half),
        ):
            package_utils.overwrite_with(field, zero, self._backend)

    def _carry_the_tke(self) -> None:
        """Close the loop between 'updated_tke' and the next step's 'tke'; see `TkeCarry`."""
        if self._config.tke_carry is TkeCarry.NONE:
            return
        updated = data_alloc.as_numpy(self._diagnostic_state.updated_tke)[: self._num_cells]
        rows = self._nlev if self._config.tke_carry is TkeCarry.INTERIOR_ONLY else self._nlev + 1
        self.tke[:, :rows] = updated[:, :rows]

    # --------------------------------------------------------------------- diagnostics ---

    def stress_and_buoyancy_flux(self) -> tuple[np.ndarray, np.ndarray]:
        """The two fluxes Nieuwstadt's similarity profiles are written in, on half levels.

        Kinematic momentum flux 'tau = K_m*|dV/dz|' [m2/s2] and buoyancy flux magnitude
        '(g/theta_ref)*K_h*dtheta/dz' [m2/s3], both for column 0, both from the granule's own
        diffusion coefficients and the driver's own gradients. Row 0 and row 'nlev' are zero:
        the model top has no level above it and the surface flux level's gradient is the
        surface layer's business, not the column's.
        """
        nlev = self._nlev
        diffusion_momentum = data_alloc.as_numpy(self._diagnostic_state.tkvm)[0]
        diffusion_scalar = data_alloc.as_numpy(self._diagnostic_state.tkvh)[0]
        spacing = self.main_level_height[:-1] - self.main_level_height[1:]
        shear_u = (self.zonal_wind[0, :-1] - self.zonal_wind[0, 1:]) / spacing
        shear_v = (self.meridional_wind[0, :-1] - self.meridional_wind[0, 1:]) / spacing
        theta = self.temperature[0] / self._exner
        gradient = (theta[:-1] - theta[1:]) / spacing
        stress = np.zeros(nlev + 1)
        buoyancy = np.zeros(nlev + 1)
        stress[1:nlev] = diffusion_momentum[1:nlev] * np.hypot(shear_u, shear_v)
        buoyancy[1:nlev] = (
            constants.GRAV / ref.REFERENCE_THETA * diffusion_scalar[1:nlev] * np.abs(gradient)
        )
        return stress, buoyancy

    def integrate(self) -> Gabls1History:
        """Run the whole case and return what Beare's tables can judge."""
        config = self._config
        final_hour_start = config.run_length - 3600.0
        penultimate_hour_start = config.run_length - 7200.0
        samples: list[tuple[float, float, float, float, float]] = []
        columns_agree = True

        stress_sum = np.zeros(self._nlev + 1)
        buoyancy_sum = np.zeros(self._nlev + 1)
        momentum_sum = np.zeros(self._nlev + 1)
        scalar_sum = np.zeros(self._nlev + 1)
        wind_sum = np.zeros(self._nlev)
        theta_sum = np.zeros(self._nlev)
        penultimate_wind_sum = np.zeros(self._nlev)
        final_count = 0
        penultimate_count = 0
        friction_sum = 0.0
        heat_sum = 0.0
        buoyancy_flux_sum = 0.0
        obukhov_sum = 0.0

        for index in range(config.num_steps):
            layer = self.step()
            time = self.time
            heat_flux = float(
                self._surface_density
                * constants.CPD
                * layer.friction_velocity[0]
                * layer.temperature_scale[0]
            )
            buoyancy_flux = float(
                constants.GRAV
                / ref.REFERENCE_THETA
                * layer.friction_velocity[0]
                * layer.temperature_scale[0]
            )
            if index % config.diagnostic_interval == 0 or index == config.num_steps - 1:
                stress, _ = self.stress_and_buoyancy_flux()
                depth = boundary_layer_depth(
                    stress=stress,
                    surface_stress=float(layer.friction_velocity[0] ** 2),
                    height=self.half_level_height,
                )
                samples.append(
                    (time, float(layer.friction_velocity[0]), heat_flux, buoyancy_flux, depth)
                )
                columns_agree = columns_agree and self._columns_agree()
            if time > penultimate_hour_start and time <= final_hour_start:
                penultimate_wind_sum += np.hypot(self.zonal_wind[0], self.meridional_wind[0])
                penultimate_count += 1
            if time > final_hour_start:
                stress, buoyancy = self.stress_and_buoyancy_flux()
                stress_sum += stress
                buoyancy_sum += buoyancy
                momentum_sum += data_alloc.as_numpy(self._diagnostic_state.tkvm)[0]
                scalar_sum += data_alloc.as_numpy(self._diagnostic_state.tkvh)[0]
                wind_sum += np.hypot(self.zonal_wind[0], self.meridional_wind[0])
                theta_sum += self.temperature[0] / self._exner
                friction_sum += float(layer.friction_velocity[0])
                heat_sum += heat_flux
                buoyancy_flux_sum += buoyancy_flux
                obukhov_sum += float(layer.obukhov_length[0])
                final_count += 1

        recorded = np.array(samples, dtype=float)
        return Gabls1History(
            time=recorded[:, 0],
            friction_velocity=recorded[:, 1],
            heat_flux=recorded[:, 2],
            buoyancy_flux=recorded[:, 3],
            depth=recorded[:, 4],
            half_level_height=self.half_level_height,
            main_level_height=self.main_level_height,
            mean_stress=stress_sum / final_count,
            mean_buoyancy_flux=buoyancy_sum / final_count,
            mean_diffusion_momentum=momentum_sum / final_count,
            mean_diffusion_scalar=scalar_sum / final_count,
            mean_wind_speed=wind_sum / final_count,
            penultimate_mean_wind_speed=penultimate_wind_sum / penultimate_count,
            mean_potential_temperature=theta_sum / final_count,
            final_hour_friction_velocity=friction_sum / final_count,
            final_hour_heat_flux=heat_sum / final_count,
            final_hour_buoyancy_flux=buoyancy_flux_sum / final_count,
            final_hour_obukhov_length=obukhov_sum / final_count,
            columns_agree=columns_agree,
        )

    def _columns_agree(self) -> bool:
        """Whether all eighteen columns still hold identical values.

        Free evidence that the granule is column-independent: the eighteen columns of
        'simple_grid' are initialised identically and nothing in the driver distinguishes them,
        so any difference is the granule mixing horizontally -- which it must not, being a 1D
        scheme.
        """
        for values in (self.zonal_wind, self.meridional_wind, self.temperature, self.tke):
            if not np.array_equal(values, np.tile(values[0], (self._num_cells, 1))):
                return False
        return True


def _hydrostatic_column(
    *,
    potential_temperature: np.ndarray,
    main_level_height: np.ndarray,
    half_level_height: np.ndarray,
    surface_pressure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exner, pressure, temperature, density on main levels and pressure on half levels.

    Integrated upward from the surface in the Exner function, 'd(Pi)/dz = -g/(c_pd*theta)', with
    the layer's own potential temperature held constant across the layer. For the 320-level grid
    this puts the lowest main level at 'z = 3.125 m', 'T = 265.964 K', 'p = 101279 Pa' and
    'rho = 1.3266 kg/m3' -- 0.3 % from Beare's Boussinesq reference density of 1.3223, which is
    as close as a compressible column gets to an incompressible specification.

    THE PROFILE IS BUILT ONCE AND HELD FIXED FOR THE WHOLE INTEGRATION, which is a departure
    from the design note of 2026-08-31 and is deliberate. GABLS1 IS A BOUSSINESQ CASE: Beare
    p. 250 gives one reference density for the whole domain and the LES do not let it evolve. A
    column that re-integrated its own hydrostatic balance every step would introduce a
    density-temperature coupling the reference does not have, for a 0.9 % density change over
    nine hours. Holding it fixed also makes 'TurbulenceMetricState.dp0' genuinely static, which
    removes the identity-binding trap that container documents rather than merely respecting it.
    """
    exner_main = np.empty_like(potential_temperature)
    exner_half = np.empty(len(half_level_height))
    exner_half[-1] = (surface_pressure / constants.P0REF) ** constants.RD_O_CPD
    for level in range(len(potential_temperature) - 1, -1, -1):
        exner_main[level] = (
            exner_half[level + 1]
            - constants.GRAV_O_CPD
            * (main_level_height[level] - half_level_height[level + 1])
            / potential_temperature[level]
        )
        exner_half[level] = (
            exner_main[level]
            - constants.GRAV_O_CPD
            * (half_level_height[level] - main_level_height[level])
            / potential_temperature[level]
        )
    pressure = constants.P0REF * exner_main**constants.CPD_O_RD
    half_level_pressure = constants.P0REF * exner_half**constants.CPD_O_RD
    temperature = potential_temperature * exner_main
    density = pressure / (constants.RD * temperature)
    return exner_main, pressure, temperature, density, half_level_pressure
