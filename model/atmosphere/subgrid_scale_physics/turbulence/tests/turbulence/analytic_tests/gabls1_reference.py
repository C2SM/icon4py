# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The published GABLS1 numbers, transcribed as literals, with the citation beside each one.

THIS MODULE CONTAINS NO LOGIC AND IMPORTS NOTHING. It is pure transcription, and it is separate
from the test that consumes it for one reason: a compute node cannot fetch anything. The GABLS1
archive host 'www.gabls.org' returns 502 today and 'gabls.metoffice.gov.uk' presents a
certificate that does not match its hostname, so the reference values exist here or they do not
exist at all. Anything a test compares against must be a committed literal.

THE SOURCE. R. J. Beare, M. K. MacVean, A. A. M. Holtslag, J. Cuxart, I. Esau, J.-C. Golaz,
M. A. Jimenez, M. Khairoutdinov, B. Kosovic, D. Lewellen, T. S. Lund, J. K. Lundquist,
A. McCabe, A. F. Moene, Y. Noh, S. Raasch and P. Sullivan, "An Intercomparison of Large-Eddy
Simulations of the Stable Boundary Layer", Boundary-Layer Meteorology 118, 247-272 (2006),
doi:10.1007/s10546-004-2820-6. Page numbers below are journal pages. The full text was obtained
from the corresponding author's own copy and read directly; the transcription and the
verification against ICON's own setup script are recorded in
'docs/superpowers/notes/2026-08-31-turbulence-scheme-primary-sources.md' section 4.

WHAT IS BEARE'S AND WHAT IS ICON'S. 'icon/scripts/scm/init_create/get_GABLS1_data.py' builds
ICON's SCM initial file for this case, and every value it shares with the paper agrees. It also
adds two things the paper does not have, and those are marked ICON-SCM below rather than GABLS1:
a 20 km domain top with a 0.005 K/m free-atmosphere potential-temperature gradient above 1500 m,
and a 12-level vertical grid. Neither may be quoted as part of the case.

WHAT IS PUBLISHED AND WHAT IS OURS. Every number in the first four groups is a literal from the
paper. The last group is not: the paper publishes only an ORDERING for the Nieuwstadt exponents,
and any tolerance placed around them was chosen here. Each such value carries 'OURS' in its
comment, and a reader who distrusts the test should start with exactly those.

THE TOLERANCE IS NOT OURS TO CHOOSE EITHER. Table IV gives eleven LES models at five
resolutions; at 6.25 m they span 158-211 m about an ensemble mean of 188 m, which is +-14 %. A
single-column tolerance tighter than the reference ensemble's own spread would be asserting
precision the reference does not have.
"""

from __future__ import annotations


__all__ = [
    "BETA_H",
    "BETA_M",
    "BUOYANCY_FLUX_RANGE",
    "CORIOLIS_PARAMETER",
    "DEPTH_STRESS_FRACTION",
    "ENSEMBLE_MEAN_DEPTH",
    "FRICTION_VELOCITY_RANGE",
    "GEOSTROPHIC_WIND",
    "HEAT_FLUX_RANGE",
    "INITIAL_SURFACE_TEMPERATURE",
    "INITIAL_TKE_AMPLITUDE",
    "INITIAL_TKE_DEPTH",
    "INITIAL_TKE_EXPONENT",
    "LAPSE_RATE",
    "LATITUDE",
    "MIXED_LAYER_DEPTH",
    "MIXING_LENGTH_ASYMPTOTE",
    "MODEL_DEPTH_RANGE",
    "MOMENTUM_FLUX_RANGE",
    "NIEUWSTADT_BUOYANCY_EXPONENT",
    "NIEUWSTADT_EXPONENT_TOLERANCE",
    "NIEUWSTADT_STRESS_EXPONENT",
    "REFERENCE_DENSITY",
    "REFERENCE_GRAVITY",
    "REFERENCE_THETA",
    "RESOLUTION_OF_THE_REPORTED_SPREAD",
    "ROUGHNESS_LENGTH",
    "RUN_LENGTH",
    "SPINUP_DEPTH_RANGE",
    "SURFACE_COOLING_RATE",
    "SURFACE_PRESSURE",
    "THETA_MIXED_LAYER",
    "VON_KARMAN",
    "WIND_STATIONARITY_THRESHOLD",
    "Z_LESS_DIFFUSIVITY_RANGE",
    "depth_extrapolated_from_the_stress_crossing",
]


# --------------------------------------------------------------------------------------------
# 1. The case specification.  Beare et al. (2006), pp. 249-250.
# --------------------------------------------------------------------------------------------
#
#   "The initial potential temperature profile consisted of a mixed layer (with potential
#    temperature 265 K) up to 100 m with an overlying inversion of strength 0.01 K m^-1. A
#    prescribed surface cooling of 0.25 K h^-1 was applied for 9 h so that a quasi-equilibrium
#    state was approached."  -- p. 249

THETA_MIXED_LAYER = 265.0
"""K. Potential temperature of the initial mixed layer. Beare p. 249."""

MIXED_LAYER_DEPTH = 100.0
"""m. Depth of the initial mixed layer. Beare p. 249."""

LAPSE_RATE = 0.01
"""K/m. Potential-temperature gradient of the inversion above the mixed layer. Beare p. 249.

Carried to the model top. The 0.005 K/m kink above 1500 m in 'get_GABLS1_data.py' is an
ICON-SCM extension and is NOT part of GABLS1.
"""

SURFACE_COOLING_RATE = 0.25 / 3600.0
"""K/s, i.e. 0.25 K/h. Prescribed surface cooling. Beare p. 249.

Written as the published 0.25 K/h divided by 3600 rather than as a decimal, so that the
published number is visible in the source.
"""

INITIAL_SURFACE_TEMPERATURE = 265.0
"""K. Surface temperature at t = 0, i.e. continuous with the mixed layer. Beare p. 249.

'get_GABLS1_data.py:107-108' has 't_surfs = [265.0, 262.75]' over 'times = [0, 32400]', and
265.0 - 0.25/3600 * 32400 = 262.75 exactly, so the two specifications agree.
"""

RUN_LENGTH = 9 * 3600.0
"""s. Length of the simulation: 9 hours. Beare p. 249."""

#   "The geostrophic wind was set to 8 m s^-1 in the east-west direction, with a Coriolis
#    parameter of 1.39 x 10^-4 s^-1 (corresponding to latitude 73 deg N). The initial wind
#    profile was geostrophic except at the bottom grid point where it was zero. In order to
#    stimulate turbulence, a random potential temperature perturbation of amplitude 0.1 K and
#    zero mean was applied below height 50 m. For models with a turbulent kinetic energy (TKE)
#    subgrid closure, the TKE field was initialised as 0.4(1 - z/250)^3 m^2 s^-2, below a
#    height (z) of 250 m."  -- pp. 249-250

GEOSTROPHIC_WIND = 8.0
"""m/s. Zonal geostrophic wind; the meridional component is zero. Beare p. 249."""

CORIOLIS_PARAMETER = 1.39e-4
"""1/s. Beare p. 249, stated explicitly alongside the latitude."""

LATITUDE = 73.0
"""degrees north. Beare p. 249."""

INITIAL_TKE_AMPLITUDE = 0.4
"""m^2/s^2. Amplitude of the initial TKE profile 0.4*(1 - z/250)^3. Beare p. 250.

This is TURBULENT KINETIC ENERGY, not the turbulent velocity scale. A scheme whose prognostic
variable is q = sqrt(2*TKE) must convert.
"""

INITIAL_TKE_DEPTH = 250.0
"""m. Depth below which the initial TKE profile is non-zero. Beare p. 250."""

INITIAL_TKE_EXPONENT = 3
"""Exponent of the initial TKE profile 0.4*(1 - z/250)^3. Beare p. 250."""

#   "Monin-Obukhov similarity was applied at the bottom boundary (with recommended constants:
#    beta_m = 4.8 and beta_h = 7.8) using a surface roughness length of 0.1 m for momentum and
#    heat, and a von Karman constant (kappa) of 0.4. The reference surface potential temperature
#    was 263.5 K, density 1.3223 kg m^-3, gravity 9.81 m s^-2. The domain size was set to
#    400 m x 400 m x 400 m."  -- p. 250

ROUGHNESS_LENGTH = 0.1
"""m. Surface roughness length, for MOMENTUM AND HEAT ALIKE. Beare p. 250.

The paper is explicit that the same value applies to heat, which 'get_GABLS1_data.py:110's
single 'z0 = 0.1' implies but does not state. It means GABLS1 has NO excess resistance for
heat, unlike ICON's default 'rlam_heat = 10'.
"""

VON_KARMAN = 0.4
"""von Karman constant. Beare p. 250. Equal to ICON's default 'akt = 0.4'."""

BETA_M = 4.8
"""Stability coefficient of the Monin-Obukhov momentum profile function. Beare p. 250."""

BETA_H = 7.8
"""Stability coefficient of the Monin-Obukhov heat profile function. Beare p. 250."""

REFERENCE_THETA = 263.5
"""K. Reference surface potential temperature of the Boussinesq buoyancy. Beare p. 250."""

REFERENCE_DENSITY = 1.3223
"""kg/m^3. Beare's Boussinesq reference density, p. 250.

Recorded for comparison only. A compressible column integrated hydrostatically from
'SURFACE_PRESSURE' does not reproduce it exactly, and need not: the two differ by ~0.3 %.
"""

REFERENCE_GRAVITY = 9.81
"""m/s^2. Beare p. 250.

Recorded for comparison only. A test must use its own model's gravity -- icon4py's
'constants.GRAV = 9.80665' -- so that a roughness length recovered internally as 'gz0/g' comes
back as exactly 0.1 m.
"""

SURFACE_PRESSURE = 101320.0
"""Pa. 'get_GABLS1_data.py:105'.

NOT in the paper, which specifies a Boussinesq case with a reference density rather than a
surface pressure. It is ICON's choice, and it is recorded here because a compressible column
needs one.
"""


# --------------------------------------------------------------------------------------------
# 2. The boundary-layer depth: definition, and Table IV.  Beare et al. (2006), pp. 250, 253.
# --------------------------------------------------------------------------------------------
#
#   "The boundary-layer depth (h) calculation involved first determining the height where the
#    mean stress fell to 5 % of its surface value (h_0.05) followed by linear extrapolation:
#    h = h_0.05/0.95. This was the same method as used by Kosovic and Curry (2000)"  -- p. 250

DEPTH_STRESS_FRACTION = 0.05
"""The fraction of the surface stress whose crossing height defines h_0.05. Beare p. 250."""


def depth_extrapolated_from_the_stress_crossing(
    height_of_the_five_percent_crossing: float,
) -> float:
    """Beare's linear extrapolation 'h = h_0.05/0.95', p. 250.

    The one function in this module. It is here rather than in the test because it is part of
    the published definition of h -- omitting it understates the depth by 5 %, which is a third
    of the ensemble spread -- and because writing '/0.95' at the call site is exactly the kind
    of step that gets dropped.
    """
    return height_of_the_five_percent_crossing / (1.0 - DEPTH_STRESS_FRACTION)


#   Table IV, p. 253: "Boundary-layer heights (in metres) for last hour of simulation and each
#   of the models at different resolutions."
#
#     model     1 m    2 m   3.125 m  6.25 m  12.5 m
#     MO        164    162     171      204     263
#     CSU        -      -      197      211     237
#     IMUK      149    162     168      158      -
#     LLNL       -      -      169      194     257
#     NERSC      -      -      179      188     204
#     WVU        -      -       -       201     197
#     NCAR       -     197     204       -       -
#     UIB        -     163     173      174     191
#     CORA       -     187     195      211      -
#     WU         -      -       -       178     158
#     COAMPS     -      -       -       161      -
#     MEAN      157    174     182      188     215

ENSEMBLE_MEAN_DEPTH = {
    1.0: 157.0,
    2.0: 174.0,
    3.125: 182.0,
    6.25: 188.0,
    12.5: 215.0,
}
"""m, keyed by LES resolution in m. Ensemble-mean SBL depth, Beare Table IV, p. 253."""

MODEL_DEPTH_RANGE = {
    1.0: (149.0, 164.0),
    2.0: (162.0, 197.0),
    3.125: (168.0, 204.0),
    6.25: (158.0, 211.0),
    12.5: (158.0, 263.0),
}
"""m, keyed by LES resolution in m. Min and max over the models reporting at that resolution.

THIS IS THE TOLERANCE, and it is the paper's, not ours. At 6.25 m the eight reporting models
span 158-211 m about a mean of 188 m -- +-14 %. Asserting a single-column depth to better than
that would claim more precision than the reference ensemble has.
"""

RESOLUTION_OF_THE_REPORTED_SPREAD = 6.25
"""m. The resolution whose spread a single-column test should be judged against.

It is the coarsest at which the full ensemble reports, so its 158-211 m band rests on eight
models rather than two, and it is fine enough that an SBL of ~180 m is resolved by ~29 layers.
"""

#   "There is significant variability in the boundary-layer depth, fluctuating between about
#    150 and 200 m up until 9 h."  -- p. 254

SPINUP_DEPTH_RANGE = (150.0, 200.0)
"""m. The range the depth fluctuates through during the run, before the final hour. p. 254."""

#   "this was assessed by taking the root-mean-square difference in the wind speed averaged
#    over 7-8 h and 8-9 h. This difference was smaller than 0.1 m s^-1 for all simulations, so
#    a reasonable quasi-steady state was considered to have been achieved."  -- p. 256

WIND_STATIONARITY_THRESHOLD = 0.1
"""m/s. The paper's own stationarity criterion: RMS wind-speed difference between the 7-8 h and
8-9 h averages. Beare p. 256.

It also fixes what an SCM should average over -- p. 250: "Profiles averaged over the horizontal
domain and over the final and penultimate hours of the simulation were calculated; in general,
the mean profile will refer to averages over the final hour."
"""


# --------------------------------------------------------------------------------------------
# 3. Surface fluxes and friction velocity.  Beare et al. (2006), p. 253.
# --------------------------------------------------------------------------------------------
#
#   "At the surface, the mean buoyancy fluxes are 3.5 to 5.5 x 10^-4 m^2 s^-3, corresponding to
#    a heat flux of 12.5 to 19.6 W m^-2, and the magnitude of the mean momentum fluxes are
#    0.06-0.08 m^2 s^-2, corresponding to a friction velocity of 0.24-0.28 m s^-1."  -- p. 253

FRICTION_VELOCITY_RANGE = (0.24, 0.28)
"""m/s. Ensemble range of the surface friction velocity in the final hour. Beare p. 253."""

MOMENTUM_FLUX_RANGE = (0.06, 0.08)
"""m^2/s^2. Ensemble range of the surface momentum flux magnitude. Beare p. 253."""

BUOYANCY_FLUX_RANGE = (3.5e-4, 5.5e-4)
"""m^2/s^3. Ensemble range of the surface buoyancy flux magnitude. Beare p. 253."""

HEAT_FLUX_RANGE = (12.5, 19.6)
"""W/m^2. Ensemble range of the surface sensible heat flux magnitude. Beare p. 253.

Downward, in the sign convention where a stable boundary layer cools from below.
"""


# --------------------------------------------------------------------------------------------
# 4. The normalised flux profiles.  Beare et al. (2006), pp. 264-265, after Nieuwstadt (1985).
# --------------------------------------------------------------------------------------------
#
#   "The theoretical model of Nieuwstadt (1985) predicts the following similarity profiles for
#    mean buoyancy and momentum flux:
#        w'b'/w'b'_0 = (1 - z/h)^{m_wh}      (3a)
#        tau/tau_0   = (1 - z/h)^{m_tau}     (3b)
#    where the subscript 0 indicates the surface values. The analysis of Nieuwstadt (1985)
#    gives the following values for the exponents: m_wh = 1, m_tau = 1.5."  -- p. 264
#
#   "At both 6.25-m and 2-m resolution, the normalised profiles have a much smaller spread than
#    the standard deviation of the observations [...] The fact that the normalised fluxes have
#    much less spread compared with the non-normalised fluxes in Figures 4 and 5, indicates
#    that much of the spread is due to variations in the boundary-layer depth and surface
#    fluxes."  -- pp. 264-265

NIEUWSTADT_STRESS_EXPONENT = 1.5
"""m_tau in tau/tau_0 = (1 - z/h)^m_tau. Nieuwstadt (1985), quoted at Beare p. 264."""

NIEUWSTADT_BUOYANCY_EXPONENT = 1.0
"""m_wh in w'b'/w'b'_0 = (1 - z/h)^m_wh. Nieuwstadt (1985), quoted at Beare p. 264."""

NIEUWSTADT_EXPONENT_TOLERANCE = 0.35
"""OURS, NOT PUBLISHED. Absolute tolerance on a fitted exponent against the two above.

THE PAPER PUBLISHES NO UNCERTAINTY ON m_tau OR m_wh. It gives the two values and states that
the LES normalised profiles "lie close to" them, and its Figures 12 and 13 are the only
quantification -- vector figures that could not be digitised. So any +- bound is a choice made
here, and this is it.

What IS published is the ORDERING 'm_wh < m_tau': buoyancy flux falls off linearly, stress as
the 3/2 power, so stress vanishes higher. That ordering is the assertion with a source behind
it; this tolerance is the assertion without one, and a reviewer should weigh the two
differently. 0.35 is a little under half the gap between the two exponents, so a fit that
satisfies both this bound and the ordering cannot have swapped them.
"""

Z_LESS_DIFFUSIVITY_RANGE = (0.06, 0.08)
"""The z-less limit of phi_KM = K_m^eff/(Lambda_local^2 * S). Beare p. 263.

    "The results of IMUK and MO at 1 m (not shown) favoured the lower limiting values of
     phi_KM between 0.06 and 0.08."  -- p. 263

The neighbouring result, Beare p. 267, is Brost and Wyngaard's (1978) normalised diffusivity,
reproduced there as their Equation (9) -- normalised momentum diffusivity K_m/(u* h) and
normalised heat diffusivity K_h/(1.2 u* h) both equal to the product of kappa, z and the 3/2
power of (1 - z/h), divided by h and by (1 + 4.7 z/L). Of it the paper says: "Profiles for
normalised momentum and heat diffusion are close and cluster fairly evenly around the profile
given by (9)." It is written out in words rather than as a formula because a formula in a
comment is indistinguishable from commented-out code, which is a lint error here.
"""


# --------------------------------------------------------------------------------------------
# 5. One number from the intercomparison that is about the SCHEME rather than the case.
# --------------------------------------------------------------------------------------------
#
#   Beare et al. Eq. (5), p. 265, gives the Met Office mixing length in the same harmonic-blend
#   form ICON uses, with lambda_0 = 40 m; p. 267: "since the local Obukhov length in the
#   interior of the SBL is much smaller than 40 m, a smaller value of lambda_0 may be more
#   appropriate."

MIXING_LENGTH_ASYMPTOTE = 40.0
"""m. The asymptotic mixing length of the Met Office LES in this intercomparison. Beare p. 265.

It is the mixing length itself, so a scheme parametrised by an asymptotic turbulent DISTANCE
must divide by von Karman's constant to match it: 40 m of mixing length is 100 m of distance.
"""
