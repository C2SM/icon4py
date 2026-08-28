# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import pytest

from .. import gate_registry


def test_unknown_stencil_is_an_error_not_a_default() -> None:
    with pytest.raises(gate_registry.UnregisteredStencilError) as excinfo:
        gate_registry.gate_for("compute_turbulent_length_scale")

    message = str(excinfo.value)
    assert "compute_turbulent_length_scale" in message
    assert "GATES" in message
    assert "gates.md" in message


def test_registered_stencil_returns_its_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = gate_registry.Exact()
    monkeypatch.setitem(gate_registry.GATES, "compute_turbulent_length_scale", gate)

    assert gate_registry.gate_for("compute_turbulent_length_scale") is gate


def test_every_entry_declares_a_gate_of_the_expected_type() -> None:
    """Stencils have landed, so the registry is no longer empty and its contents are checkable.

    This assertion replaced 'test_registry_starts_empty', which is the review step the registry
    exists to force (port spec D11). It deliberately does not pin the set of keys: that would
    make every new stencil edit this file twice, and the thing worth protecting is that no entry
    is malformed, not that the count is a particular number.
    """
    assert gate_registry.GATES, "the registry is empty; stencils have landed, so entries are due"

    for name, gate in gate_registry.GATES.items():
        assert isinstance(gate, (gate_registry.Exact, gate_registry.Tol)), (
            f"entry '{name}' is a {type(gate).__name__}, not a declared gate type"
        )


def test_every_entry_is_reachable_through_gate_for() -> None:
    """A key nobody can look up is a gate nobody applies."""
    for name, gate in gate_registry.GATES.items():
        assert gate_registry.gate_for(name) is gate


def test_exact_is_frozen() -> None:
    gate = gate_registry.Exact()
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.rtol = 1.0e-12  # type: ignore[misc]  # assigning to a frozen instance on purpose


def test_exact_is_hashable_and_compares_by_value() -> None:
    assert gate_registry.Exact() == gate_registry.Exact()
    assert len({gate_registry.Exact(), gate_registry.Exact()}) == 1


def test_tol_is_frozen() -> None:
    gate = gate_registry.Tol(
        rtol=1.0e-13,
        reason=gate_registry.Reason.TRANSCENDENTAL,
        measured_max_rel_err=4.0e-14,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.rtol = 1.0e-3  # type: ignore[misc]  # assigning to a frozen instance on purpose


def test_tol_is_hashable_and_compares_by_value() -> None:
    def make() -> gate_registry.Tol:
        return gate_registry.Tol(
            rtol=1.0e-13,
            reason=gate_registry.Reason.SCAN_LOWERING,
            measured_max_rel_err=4.0e-14,
        )

    assert make() == make()
    assert len({make(), make()}) == 1


@pytest.mark.parametrize("reason", list(gate_registry.Reason))
def test_tol_accepts_every_listed_reason(reason: gate_registry.Reason) -> None:
    gate = gate_registry.Tol(rtol=1.0e-13, reason=reason, measured_max_rel_err=4.0e-14)

    assert gate.reason is reason


def test_tol_accepts_a_listed_reason_spelled_out() -> None:
    gate = gate_registry.Tol(
        rtol=1.0e-13,
        reason="reciprocal substitution",
        measured_max_rel_err=4.0e-14,
    )

    assert gate.reason is gate_registry.Reason.RECIPROCAL_SUBSTITUTION


def test_tol_rejects_an_unlisted_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        gate_registry.Tol(
            rtol=1.0e-13,
            reason="the test was flaky",
            measured_max_rel_err=4.0e-14,
        )


@pytest.mark.parametrize("measured_max_rel_err", [0.0, -4.0e-14, float("nan")])
def test_tol_rejects_a_non_positive_measured_error(measured_max_rel_err: float) -> None:
    with pytest.raises(ValueError, match="measured_max_rel_err"):
        gate_registry.Tol(
            rtol=1.0e-13,
            reason=gate_registry.Reason.REASSOCIATION,
            measured_max_rel_err=measured_max_rel_err,
        )


def test_tol_rejects_a_measured_error_exceeding_rtol() -> None:
    with pytest.raises(ValueError, match="measured_max_rel_err"):
        gate_registry.Tol(
            rtol=1.0e-13,
            reason=gate_registry.Reason.REASSOCIATION,
            measured_max_rel_err=2.0e-13,
        )


def test_tol_accepts_a_measured_error_equal_to_rtol() -> None:
    gate = gate_registry.Tol(
        rtol=1.0e-13,
        reason=gate_registry.Reason.REASSOCIATION,
        measured_max_rel_err=1.0e-13,
    )

    assert gate.measured_max_rel_err == gate.rtol


@pytest.mark.parametrize("rtol", [0.0, -1.0e-13])
def test_tol_rejects_a_non_positive_rtol(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        gate_registry.Tol(
            rtol=rtol,
            reason=gate_registry.Reason.REASSOCIATION,
            measured_max_rel_err=4.0e-14,
        )
