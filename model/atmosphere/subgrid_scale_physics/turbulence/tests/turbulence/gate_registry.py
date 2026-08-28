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
#: Measured on 'exp.mch_icon-ch2_small', all four serialized timesteps, on the 'embedded' and
#: 'gtfn_cpu' backends. The GPU backends ('dace_gpu', 'dace_cpu', 'gtfn_gpu') are validated
#: separately on a GPU node and may need their own entries: nvcc contracts multiply-add by
#: default ('--fmad=true'), which these CPU measurements cannot see.
GATES: dict[str, Gate] = {
    # turbdiff section 1a) -- vertical gradients of the conserved variables. Section 1a contains
    # only '-', '*' and '/' with no multiply-add pattern, so bit-exactness here is not evidence
    # that FMA contraction is absent; the first fused expression is in section 1b.
    "compute_surface_transfer_ratios": Exact(),
    "compute_inverse_layer_depth_and_tke_discretisation_momentum": Exact(),
    "compute_vertical_gradients_of_conserved_variables": Exact(),
    "compute_surface_gradients_of_conserved_variables": Exact(),
    # turbdiff section 1b) -- the two TKE forcing terms. Both are 'a*b + c*d' expressions and
    # were the reason the reference was recaptured without FMA contraction (v02).
    "compute_thermal_forcing": Exact(),
    "compute_mechanical_forcing": Exact(),
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
