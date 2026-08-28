# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-stencil numerical-agreement gates for the turbulence port.

The turbulence granule is validated against serialized ICON reference data, and the default
expectation is bit-exact agreement with the Fortran. That expectation is not always met: GT4Py
re-associates expressions while inlining and fusing, substitutes reciprocals, and lowers scans
differently, none of which strict IEEE compilation constrains. Which stencils fall short is
discovered by measurement, not predicted (see the port spec, section 5.3).

So the exceptions are declared here, one entry per stencil, rather than inferred at test time.
A stencil with no entry is a test failure: 'gate_for' refuses to guess, because guessing is how a
tolerance appears in a diff without anyone deciding it should. Downgrading an entry from 'Exact'
to 'Tol' is a reviewed change with a measured error behind it -- the procedure is in
'model/atmosphere/subgrid_scale_physics/turbulence/docs/gates.md'.

A per-stencil gate is necessary but not sufficient: 'tke' is prognostic and temporally smoothed,
so a per-call tolerance compounds over a forecast. The trajectory drift check in
'integration_tests/test_trajectory_drift.py' covers what this registry structurally cannot.
"""

import dataclasses
import enum
from typing import TypeAlias


__all__ = [
    "DOWNGRADE_PROCEDURE_DOC",
    "GATES",
    "REGISTRY_FILE",
    "Exact",
    "Gate",
    "Reason",
    "Tol",
    "UnregisteredStencilError",
    "gate_for",
]


_PACKAGE = "model/atmosphere/subgrid_scale_physics/turbulence"

#: This file, so the error messages below can say where to add the missing entry.
REGISTRY_FILE = f"{_PACKAGE}/tests/turbulence/gate_registry.py"

#: Where the downgrade procedure is written down, quoted in the error messages below.
DOWNGRADE_PROCEDURE_DOC = f"{_PACKAGE}/docs/gates.md"


class Reason(str, enum.Enum):
    """The closed set of admissible reasons for not being bit-exact.

    Each member names a mechanism by which GT4Py legitimately produces a different rounding than
    the Fortran. The set is closed on purpose: an inexplicable disagreement is a bug to be
    investigated, not a tolerance to be granted. Extending the set is a spec-level decision.
    """

    #: 'exp', 'log', 'pow', 'tanh', '**' resolve to a different libm than the Fortran's, which is
    #: free to differ by a few ULP even under strict IEEE.
    TRANSCENDENTAL = "transcendental function via a different libm"

    #: Inlining and fusion regroup an expression tree, so additions and multiplications happen in
    #: a different order than the mirrored Fortran statement.
    REASSOCIATION = "expression re-association"

    #: A division is turned into a multiplication by a reciprocal, which rounds twice where the
    #: Fortran rounds once.
    RECIPROCAL_SUBSTITUTION = "reciprocal substitution"

    #: A 'scan_operator' carry is evaluated in a different order or accumulated differently than
    #: the sequential Fortran k-loop it was translated from.
    SCAN_LOWERING = "scan lowering"


@dataclasses.dataclass(frozen=True)
class Exact:
    """Bit-exact agreement with the Fortran reference is required. The default for every stencil."""


@dataclasses.dataclass(frozen=True)
class Tol:
    """Tolerant agreement, allowed only for a measured and reviewed reason.

    Attributes:
        rtol: The relative tolerance the stencil test asserts.
        reason: Why bit-exactness is unattainable here; must be a 'Reason' member. A plain string
            equal to a member's value is accepted and normalized; anything else is rejected.
        measured_max_rel_err: The largest relative error actually observed over the reference data.
            Recording it makes the headroom between observation and assertion visible, so a
            later widening of 'rtol' has to argue against a number instead of against nothing.
    """

    rtol: float
    reason: Reason
    measured_max_rel_err: float

    def __post_init__(self) -> None:
        try:
            reason = Reason(self.reason)
        except ValueError:
            allowed = ", ".join(f"'{member.value}'" for member in Reason)
            raise ValueError(
                f"Invalid argument 'reason': got '{self.reason}', expected one of: {allowed}."
            ) from None
        object.__setattr__(self, "reason", reason)

        if not self.rtol > 0.0:
            raise ValueError(
                f"Invalid argument 'rtol': should be a positive tolerance, got {self.rtol}."
            )
        if not self.measured_max_rel_err > 0.0:
            raise ValueError(
                "Invalid argument 'measured_max_rel_err': should be a positive measured error, "
                f"got {self.measured_max_rel_err}."
            )
        if self.measured_max_rel_err > self.rtol:
            raise ValueError(
                f"Invalid argument 'measured_max_rel_err': {self.measured_max_rel_err} exceeds "
                f"'rtol' {self.rtol}, so the gate would assert a tolerance tighter than the error "
                "it records."
            )


Gate: TypeAlias = Exact | Tol


class UnregisteredStencilError(LookupError):
    """Raised when a stencil is validated without a gate declared for it."""


#: The gate for every turbulence stencil, keyed by stencil name. Every stencil is added here as
#: 'Exact()' when it lands, and downgraded only per 'DOWNGRADE_PROCEDURE_DOC'. A diff of this dict
#: is the record of where the numerics moved.
#:
#: Measured on 'exp.mch_icon-ch2_small', all four serialized timesteps, on the 'embedded',
#: 'gtfn_cpu' and 'dace_cpu' backends. The GPU backends ('dace_gpu', 'gtfn_gpu') are validated
#: separately on a GPU node and may need their own entries: nvcc contracts multiply-add by
#: default ('--fmad=true'), which these CPU measurements cannot see.
#:
#: 'embedded' does not cover every entry. Stencils that select a boundary row with 'concat_where'
#: xfail there in gt4py 1.1.10 (see the package README, "Boundary rows"), so their gate rests on
#: the two compiled backends.
GATES: dict[str, Gate] = {
    # turbdiff section 1a) -- vertical gradients of the conserved variables. Section 1a contains
    # only '-', '*' and '/' with no multiply-add pattern, so bit-exactness here is not evidence
    # that FMA contraction is absent; the first fused expression is in section 1b.
    "compute_surface_transfer_ratios": Exact(),
    "compute_inverse_layer_depth_and_tke_discretisation_momentum": Exact(),
    # Covers the whole gradient profile including the surface row, which the Fortran writes in a
    # separate loop and this port selects with 'concat_where' -- see the boundary-row convention
    # in the package README.
    "compute_vertical_gradients_of_conserved_variables": Exact(),
    # turbdiff section 1b) -- the two TKE forcing terms. Both are 'a*b + c*d' expressions and
    # were the reason the reference was recaptured without FMA contraction (v02).
    "compute_thermal_forcing": Exact(),
    "compute_mechanical_forcing": Exact(),

    # turbdiff section 0) conserved variables, cloud cover and turbulent length scales.
    # Measured on 'exp.mch_icon-ch2_small' v02, all four dates, on 'embedded', 'gtfn_cpu' and
    # 'dace_cpu' -- which produce bit-identical results here, so the disagreement is on ICON's
    # side of the comparison and not in any backend's code generation. The three tolerant
    # entries are the programs that evaluate 'EXP' (Magnus' formula for the saturation vapour
    # pressure) or 'EXP(LOG())' (the Exner factor); every quantity of theirs that does NOT pass
    # through an exponential is bit-exact and is asserted so, ungated, by
    # 'test_the_conserved_variables_are_bit_exact_where_no_exponential_is_involved' and
    # 'test_the_interpolation_is_bit_exact_where_its_inputs_are'.
    #
    # 'rtol' is ~9.4x the measured maximum. The error is a single rounding of 'exp' amplified by
    # the near-cancellation 'dq = qt - qs' in the cloud diagnosis, so it is the amplification and
    # not the ULP that varies with the data: across the four dates the worst value spans a factor
    # of 18 (5.9e-12 to 1.07e-10), and a decade of headroom is that spread, not more.
    "compute_conserved_variables_and_factors_at_main_levels": Tol(
        rtol=1e-9,
        reason=Reason.TRANSCENDENTAL,
        measured_max_rel_err=1.0654e-10,  # 'zvari(:,:,liq)', 2020-12-10T06:01:20
    ),
    "compute_conserved_variables_and_factors_at_the_surface": Tol(
        rtol=5e-12,
        reason=Reason.TRANSCENDENTAL,
        measured_max_rel_err=5.3564e-13,  # 'zvari(:,ke1,liq)', 2020-12-10T06:02:00
    ),
    "interpolate_variables_onto_half_levels": Tol(
        rtol=5e-10,
        reason=Reason.TRANSCENDENTAL,
        measured_max_rel_err=5.3270e-11,  # 'rcld', 2020-12-10T06:01:20
    ),
    "compute_half_level_interpolation_weight": Exact(),
    "compute_horizontal_wind_including_the_zero_level": Exact(),
    "compute_layer_depth": Exact(),
    "compute_turbulent_length_scale": Exact(),

    # turbdiff section 2c) final preparations. The single live statement turns the diffusion
    # coefficients back into stability lengths, and it is the exact inverse of section 3's
    # 'compute_diffusion_coefficients_from_stability_lengths'. Bit-exactness here depends on
    # reproducing ICON's reciprocal: it forms '1/tke' once and multiplies, and writing 'x / q'
    # instead misses 156546 of 653804 values in 'ls_m' by up to 1 ulp. That all three backends
    # agree is also evidence GT4Py did not substitute the reciprocal away.
    "compute_stability_lengths_from_diffusion_coefficients": Exact(),

    # turbdiff section 3) turbulent budgets ('solve_turb_budgets').
    # PROVISIONAL: entered as the documented default so the section datatests can run at
    # all -- 'gate_for' refuses to default, so without an entry the tests error instead
    # of reporting. The agents that wrote these stencils were killed before validating
    # them, so no measurement stands behind these yet. A failure here is a real finding.
    "compute_circulation_acceleration": Exact(),
    "compute_diffusion_coefficients_from_stability_lengths": Exact(),
    "compute_stability_lengths": Exact(),
    "compute_supersaturation_standard_deviation": Exact(),
    "compute_turbulent_velocity_scale": Exact(),
    "set_turbulent_velocity_scale_at_model_top": Exact(),

    # turbdiff section 6) q-diffusion tendency.
    # PROVISIONAL: entered as the documented default so the section datatests can run at
    # all -- 'gate_for' refuses to default, so without an entry the tests error instead
    # of reporting. The agents that wrote these stencils were killed before validating
    # them, so no measurement stands behind these yet. A failure here is a real finding.
    "compute_cke_flux_at_main_levels": Exact(),
    "compute_cke_flux_density": Exact(),
    "compute_explicit_tke_diffusion_momentum": Exact(),
    "compute_saved_tke_profile": Exact(),

    # turbdiff section 9) TKE profile update through the diffusion tendency.
    # PROVISIONAL: entered as the documented default so the section datatests can run at
    # all -- 'gate_for' refuses to default, so without an entry the tests error instead
    # of reporting. The agents that wrote these stencils were killed before validating
    # them, so no measurement stands behind these yet. A failure here is a real finding.
    "add_virtual_diffusion_increment_to_tke_profile": Exact(),
    "compute_diffusion_inversion_factor": Exact(),
    "compute_explicit_tke_flux_density": Exact(),
    "compute_implicit_part_of_tke_diffusion_momentum": Exact(),
    "compute_inverted_diffusion_momentum": Exact(),
    "compute_tke_diffusion_right_hand_side": Exact(),
    "solve_tke_diffusion_equation": Exact(),
    "subtract_implicit_part_of_tke_diffusion_momentum": Exact(),

    # turbdiff section 10) the q tendency of the TKE diffusion. No transcendental and no
    # multiply-add exposure -- 'sqrt' is correctly rounded and '2*x' is exact -- so this was
    # established in plain numpy against the archive before any GT4Py was written, and the
    # backends only had to confirm it. Measured on 'gtfn_cpu' and 'dace_cpu', all four dates;
    # 'embedded' xfails the programs that use 'concat_where'.
    "compute_turbulent_velocity_scale_tendency": Exact(),

    # turbdiff section 11) interpolation of the SDSS back to main levels.
    # Measured bit-exact on embedded, gtfn_cpu and dace_cpu, all four dates.
    "interpolate_supersaturation_deviation_to_main_levels": Exact(),
}


def gate_for(stencil_name: str) -> Gate:
    """Return the declared gate for a stencil, refusing to default.

    Args:
        stencil_name: Name of the stencil under test, as keyed in 'GATES'.

    Returns:
        The declared 'Exact' or 'Tol' gate.

    Raises:
        UnregisteredStencilError: If the stencil has no entry, since a silent default is exactly
            the drift this registry exists to prevent.
    """
    try:
        return GATES[stencil_name]
    except KeyError:
        raise UnregisteredStencilError(
            f"Stencil '{stencil_name}' has no entry in 'GATES': add one to "
            f"'{REGISTRY_FILE}', 'Exact()' unless a measurement says otherwise, and see "
            f"'{DOWNGRADE_PROCEDURE_DOC}' for the downgrade procedure."
        ) from None
