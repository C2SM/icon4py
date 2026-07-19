# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for `codegen/collection_gen.py` (M2b Task 4): split-operator
codegen for the collection (coalescence) RATE-MATRIX kernel, the first
REAL (non-demonstration) consumer of `codegen/generate.py`/
`codegen/templates.py` (M1 Task 6).

Read `collection_gen.py`'s own module docstring first for the DSL/numpy
split decision this file's tests assume: the generated split operators
compute `KC_ij = E_c_ij*(vtm_i-vtm_j)*A_c_ij*con_j*dt` (col_level==1 only)
taking `E_c_ij` as a plain INPUT field (produced by the existing numpy
`collision_kernel.collision_efficiency`, unchanged); the LUT bilinear
gather itself and `core/coalescence.py::coalesce_rain`'s
categorization/fixed-point scatter both stay numpy, out of this file's
scope (both already have their own dedicated test files).

Per `generate.py`'s module docstring, tests never call `emit_module`
(which writes to disk) -- only the read-only `check_regenerated` and
`load_generated_module`.

Real `AmpsConfig.cloudlab().num_h_bins[0] == 40` liquid bins is used
throughout (the M0-gate size), chunked at the default `chunk_size=8` ->
`(40/8)**2 == 25` generated field_operators, matching the committed
`core/generated/collection_rate_matrix.py`.
"""

from __future__ import annotations

import os

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next.program_processors.runners.gtfn import run_gtfn_cached

from icon4py.model.atmosphere.subgrid_scale_physics.amps.codegen import (
    collection_gen,
    generate,
    templates,
)
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import collision_kernel
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.liquid_diag import LiquidDiag
from icon4py.model.atmosphere.subgrid_scale_physics.amps.core.lookup_tables import (
    AmpsLuts,
    load_luts,
)
from icon4py.model.common import dimension as dims


NBINS = 40  # AmpsConfig.cloudlab().num_h_bins[0] -- the M0-gate liquid bin count.
CHUNK_SIZE = 8
NCELLS, NLEV = 6, 4  # small but non-trivial -- keeps 25-chunk embedded runs fast.


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_luts() -> AmpsLuts:
    return load_luts()


def _realistic_diag(nbins: int, ncells: int, nlev: int) -> LiquidDiag:
    """A `(nbins, npoints)` `LiquidDiag` with a monotonically-growing,
    real-magnitude rain size distribution (diameters ~0.02-0.30cm,
    terminal velocities ~50-350cm/s -- the same order of magnitude as
    `test_collision_kernel.py`'s own real-magnitude fixtures) plus a
    per-point multiplicative wobble (deterministic, seeded) so every
    (bin, cell, level) combination gets a genuinely distinct value, not a
    column-constant broadcast. `nre` is kept within `AmpsLuts.drpdrp`'s
    own `log10(Nre)` domain (`adrpdrp.ys=-3.9379` to `~1.78`, see
    `core/lookup_tables.py`) so the bilinear gather driving `E_c` isn't
    degenerately edge-clamped everywhere."""
    npoints = ncells * nlev
    rng = np.random.default_rng(4021)
    wobble = rng.uniform(0.9, 1.1, size=npoints)

    base_length = np.linspace(0.02, 0.30, nbins)
    base_vtm = np.linspace(50.0, 350.0, nbins)
    base_nre = np.linspace(0.02, 0.45, nbins)

    length = base_length[:, None] * wobble[None, :]
    vtm = base_vtm[:, None] * wobble[None, :]
    nre = base_nre[:, None] * wobble[None, :]

    ones = np.ones((nbins, npoints))
    zeros = np.zeros((nbins, npoints))
    return LiquidDiag(
        mean_mass=zeros,
        length=length,
        a_len=zeros,
        c_len=zeros,
        density=ones,
        terminal_velocity=vtm,
        capacitance=zeros,
        ventilation_fv=ones,
        ventilation_fh=ones,
        ventilation_fkn=ones,
        vapdep_coef1=zeros,
        vapdep_coef2=zeros,
        nre=nre,
    )


def _realistic_con(nbins: int, ncells: int, nlev: int) -> np.ndarray:
    """`(nbins, npoints)` per-volume number density, ~50-900 cm^-3
    (matching `test_collision_kernel.py`'s own real-magnitude range),
    with the same per-point wobble idiom as `_realistic_diag`."""
    npoints = ncells * nlev
    rng = np.random.default_rng(918)
    wobble = rng.uniform(0.8, 1.2, size=npoints)
    base_con = np.linspace(50.0, 900.0, nbins)
    return base_con[:, None] * wobble[None, :]


def _field(arr_1d_over_points: np.ndarray, ncells: int, nlev: int) -> gtx.Field:
    reshaped: np.ndarray = arr_1d_over_points.reshape(ncells, nlev)
    return gtx.as_field((dims.CellDim, dims.KDim), reshaped)


def _chunk_kwargs(
    diag: LiquidDiag,
    con: np.ndarray,
    e_c_full: np.ndarray,
    *,
    ilo: int,
    ihi: int,
    jlo: int,
    jhi: int,
    ncells: int,
    nlev: int,
) -> dict[str, gtx.Field]:
    """Build the exact kwarg set `collection_gen`'s `(ilo,ihi,jlo,jhi)`
    chunk operator expects, sliced from the full `(nbins,npoints)`
    `diag`/`con` and `(nbins,nbins,npoints)` `e_c_full`."""
    kwargs: dict[str, gtx.Field] = {}
    for i in range(ilo, ihi):
        kwargs[f"len_i{i:02d}"] = _field(diag.length[i], ncells, nlev)
        kwargs[f"vtm_i{i:02d}"] = _field(diag.terminal_velocity[i], ncells, nlev)
    for j in range(jlo, jhi):
        kwargs[f"len_j{j:02d}"] = _field(diag.length[j], ncells, nlev)
        kwargs[f"vtm_j{j:02d}"] = _field(diag.terminal_velocity[j], ncells, nlev)
        kwargs[f"con_j{j:02d}"] = _field(con[j], ncells, nlev)
    for i in range(ilo, ihi):
        for j in range(jlo, jhi):
            kwargs[f"ec_i{i:02d}_j{j:02d}"] = _field(e_c_full[i, j], ncells, nlev)
    return kwargs


# ---------------------------------------------------------------------------
# Builder structural properties
# ---------------------------------------------------------------------------


class TestBuilderChunking:
    def test_operator_count_is_chunk_grid_squared(self):
        src = collection_gen.build_collision_rate_matrix(NBINS, chunk_size=CHUNK_SIZE)
        n_chunks = len(templates.chunk_bins(NBINS, CHUNK_SIZE))
        assert src.count("@gtx.field_operator") == n_chunks * n_chunks == 25

    def test_never_exceeds_chunk_size_squared_pairs_per_operator(self):
        for ilo, ihi in templates.chunk_bins(NBINS, CHUNK_SIZE):
            for jlo, jhi in templates.chunk_bins(NBINS, CHUNK_SIZE):
                assert (ihi - ilo) * (jhi - jlo) <= CHUNK_SIZE * CHUNK_SIZE

    def test_deterministic_chunk_names_present(self):
        module = generate.load_generated_module("collection_rate_matrix")
        for ilo, ihi in templates.chunk_bins(NBINS, CHUNK_SIZE):
            for jlo, jhi in templates.chunk_bins(NBINS, CHUNK_SIZE):
                name = collection_gen.kc_chunk_name(ilo, ihi, jlo, jhi)
                assert hasattr(module, name), f"missing generated operator {name}"

    def test_builder_is_deterministic(self):
        """No randomness -- required for `check_regenerated`'s byte-equality
        drift guard to mean anything."""
        first = collection_gen.build_collision_rate_matrix(NBINS, chunk_size=CHUNK_SIZE)
        second = collection_gen.build_collision_rate_matrix(NBINS, chunk_size=CHUNK_SIZE)
        assert first == second


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


class TestDriftGuard:
    def test_collection_rate_matrix_matches_committed_file(self):
        generate.check_regenerated(
            "collection_rate_matrix", collection_gen.build_collision_rate_matrix, nbins=NBINS
        )

    def test_mismatched_params_are_detected(self):
        with pytest.raises(AssertionError):
            generate.check_regenerated(
                "collection_rate_matrix",
                collection_gen.build_collision_rate_matrix,
                nbins=NBINS,
                chunk_size=4,
            )


# ---------------------------------------------------------------------------
# Embedded execution reproduces the numpy Task-2 reference (collision_kernel).
# ---------------------------------------------------------------------------


class TestEmbeddedExecutionMatchesNumpy:
    """Every one of the 25 generated `(i_chunk, j_chunk)` operators, run on
    the embedded (pure-Python/numpy) gt4py backend, must reproduce
    `core/collision_kernel.py::collision_kernel`'s own `KC_ij` (Task 2,
    col_level=1 -- cloudlab's own setting) to 1e-12, given the SAME
    (numpy-computed) `E_c` fed in as a plain input field. This isolates
    the DSL-codegen'd ASSEMBLY arithmetic from the (deliberately
    numpy-only, separately-tested) LUT gather -- see module/collection_gen
    docstrings for why."""

    def test_all_25_chunks(self, real_luts):
        module = generate.load_generated_module("collection_rate_matrix")
        diag = _realistic_diag(NBINS, NCELLS, NLEV)
        con = _realistic_con(NBINS, NCELLS, NLEV)
        dt = 2.0

        e_c_full = collision_kernel.collision_efficiency(diag, diag, real_luts)
        kc_full = collision_kernel.collision_kernel(diag, diag, con, dt, real_luts, col_level=1)
        assert kc_full.shape == (NBINS, NBINS, NCELLS * NLEV)

        chunks = templates.chunk_bins(NBINS, CHUNK_SIZE)
        n_pairs_checked = 0
        for ilo, ihi in chunks:
            for jlo, jhi in chunks:
                op = getattr(module, collection_gen.kc_chunk_name(ilo, ihi, jlo, jhi))
                kwargs = _chunk_kwargs(
                    diag,
                    con,
                    e_c_full,
                    ilo=ilo,
                    ihi=ihi,
                    jlo=jlo,
                    jhi=jhi,
                    ncells=NCELLS,
                    nlev=NLEV,
                )
                kwargs["dt"] = dt

                pairs = [(i, j) for i in range(ilo, ihi) for j in range(jlo, jhi)]
                outs = tuple(_field(np.zeros(NCELLS * NLEV), NCELLS, NLEV) for _ in pairs)
                op(**kwargs, out=outs, offset_provider={})

                for out_field, (i, j) in zip(outs, pairs, strict=True):
                    got = out_field.asnumpy().reshape(NCELLS * NLEV)
                    expected = kc_full[i, j]
                    np.testing.assert_allclose(
                        got, expected, rtol=1e-12, atol=1e-12, err_msg=f"KC mismatch at ({i},{j})"
                    )
                    n_pairs_checked += 1

        assert n_pairs_checked == NBINS * NBINS  # every one of the 1600 bin pairs, exactly once


# ---------------------------------------------------------------------------
# gtfn_cpu compile feasibility -- the M0-gate question for collection.
#
# Opt-in (AMPS_COLLECTION_CODEGEN_GTFN=1), not run by default -- same
# rationale as the repo's own `gtfn_too_slow` marker (a single compile
# already costs O(minutes), see below). Real numbers WERE measured in
# this environment (it has a working C++/ninja/cmake toolchain -- `cmake`,
# `g++`, and a `ninja` executable via the `ninja` pip package are all
# present in `.venv`): a single 8x8-bin (64-pair, 104-field-param) chunk
# cold-compiled + ran on gtfn_cpu in ~150.6s (steady-state call ~18ms
# after that). Extrapolated (not re-measured for all 25 chunks, to keep
# this task's own runtime bounded): ~25*150.6s ~= 63min aggregate if all
# 25 nbins=40 chunks were gtfn-compiled serially. See
# `.superpowers/sdd/m2b-task-4-report.md` for the full writeup, including
# why this is a ONE-TIME, cacheable (GT4PY persistent build cache) AOT
# cost rather than a per-run cost, and why per-operator compile time
# staying bounded (no RecursionError, no need to raise Python's recursion
# limit, unlike the monolithic 40-bin/1600-pair operator M0 measured at
# ~2579s) is itself the win SPLIT operators were mandated for, even though
# the AGGREGATE serial compile time across all 25 chunks is not obviously
# smaller than the monolithic number.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("AMPS_COLLECTION_CODEGEN_GTFN"),
    reason=(
        "gtfn_cpu compile timing is opt-in (set AMPS_COLLECTION_CODEGEN_GTFN=1) -- a single "
        "chunk's cold compile costs O(minutes); see this module's own comment block and "
        ".superpowers/sdd/m2b-task-4-report.md for numbers already measured in this "
        "environment."
    ),
)
class TestGtfnCompileFeasibility:
    def test_one_chunk_compiles_and_matches_numpy(self, real_luts):
        module = generate.load_generated_module("collection_rate_matrix")
        op = getattr(module, collection_gen.kc_chunk_name(0, 8, 0, 8)).with_backend(run_gtfn_cached)

        diag = _realistic_diag(NBINS, NCELLS, NLEV)
        con = _realistic_con(NBINS, NCELLS, NLEV)
        dt = 2.0
        e_c_full = collision_kernel.collision_efficiency(diag, diag, real_luts)
        kc_full = collision_kernel.collision_kernel(diag, diag, con, dt, real_luts, col_level=1)

        kwargs = _chunk_kwargs(
            diag, con, e_c_full, ilo=0, ihi=8, jlo=0, jhi=8, ncells=NCELLS, nlev=NLEV
        )
        kwargs["dt"] = dt
        pairs = [(i, j) for i in range(8) for j in range(8)]
        outs = tuple(_field(np.zeros(NCELLS * NLEV), NCELLS, NLEV) for _ in pairs)

        op(**kwargs, out=outs, offset_provider={})  # first call: pays the cold-compile cost

        for out_field, (i, j) in zip(outs, pairs, strict=True):
            got = out_field.asnumpy().reshape(NCELLS * NLEV)
            np.testing.assert_allclose(got, kc_full[i, j], rtol=1e-12, atol=1e-12)
