# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic tests of the turbulence stencils: no serialized ICON data anywhere.

WHY THIS DIRECTORY EXISTS. Every other test of this package compares against a savepoint. That
establishes "we reproduce the Fortran", which is the right first question and not the last one:
a savepoint test cannot see a faithful port of a wrong implementation, it says nothing about a
regime the capture never visited, and it pins the current stencil decomposition, so a merging
pass invalidates it. An algebraic invariant does none of those things.

WHAT COUNTS AS ANALYTIC HERE. The state is built from scratch by 'utils.py'; the expected value
is derived from the equations or from the closure constants, in the test, in closed form. A
numpy transcription of the Fortran is NOT analytic -- it can share a misreading with the port,
which is the weakness this directory exists to cover -- and the one such transcription in the
package ('_vert_smooth_as_the_fortran_writes_it', 'integration_tests/test_turbdiff_section_2c.py')
stays where it is.

EVERY INVARIANT SHIPS WITH A MUTATION THAT BREAKS IT. 'broken_stencils.py' holds deliberately
defective copies of the stencils tested here, and each test module asserts that its invariant
fails on the broken copy -- by calling the SAME assertion function the passing test calls,
inside 'pytest.raises', so that what the mutation breaks is literally what the port is held to.
A passing test says nothing until it has been seen to fail for the right reason; the pattern is
David Strassmann's 'BrokenPiecewiseParabolicMethod' in the advection convergence study.

AND EVERY MUTATION IS TESTED WHERE IT IS *NOT* SEEN, TOO. Several of the mutations here are
exactly the identity in some regime -- the complementary implicit weight at Crank-Nicolson, the
negated buoyancy cofactor at 'Ri = 0', the exchanged time-smoothing weights at any steady state
-- and each of those blind spots is asserted rather than written down, because a limitation
that is only in a comment gets rediscovered. Between them they say which test earns its place:
the amplification factor catches what conservation cannot, the second moment catches what
neither can, the stratified equilibrium catches what no neutral state can, and the decay
catches what no equilibrium can.

Not a 'datatest': nothing here reads 'ICON4PY_TEST_DATA_PATH', so all of it runs with no
serialized archive present.
"""
