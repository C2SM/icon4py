# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The FFSL-WENO schemes against the Fortran capture of Jocksch's moving-cylinder test.

Reference: one ICON run per (ihadv_tracer, itype_hlimit) of A. Jocksch's live cylinder
block (icon-exclaim branch transport_ajocksch_capture, built with -Kieee -Mnofma
-gpu=nofma, nproma = 1, 10 identical levels) on the shared 20 x 22 torus with 5 km
edges: 100 calls of step_advection at one model date with a uniform 1 m/s wind, air mass 1
and dt = 999.99999995 s, tracers 1-4 advected with the case's scheme and tracer 5 kept as
the initial cylinder. Savepoints: 'lsq-coefficients' (init time) and 'advection-init' /
'advection-exit' with a 'step' key. Registered as
'test_defs.Experiments.jocksch_cylinder(ihadv_tracer, itype_hlimit)'; the data is not
downloadable, see the scope note for the test-data layout.

Three levels, each gated at twice the worst agreement measured on gtfn_cpu, dace_cpu and
gtfn_cpu with FMA contraction off (santis, GCC 14.3, numpy with OpenBLAS; the measured
values stand next to every gate), not looser: a factor two at round-off level is still
round-off, and the backends differ from each other in the last digit
(docs/running_the_jocksch_reference_tests.md has the tables per backend):

L1  every init-time coefficient 'weno_least_squares' produces (9-point stencil, moments,
    row weights, the full quadratic and linear pseudoinverses, the 27 + 3 candidate
    pseudoinverses, l_weights_s) against 'lsq-coefficients' of the ihadv103 run; the
    stencil is compared index by index because Jocksch's create_stencil_c9 order is what
    the port reproduces (it does, so no permutation is applied);
L2  one 'Advection.run' from 'advection-init' step n against 'advection-exit' step n, for
    steps 1, 2, 50, 100, on the granule the driver builds (grid, geometry, interpolation
    and reconstruction coefficients all from icon4py, the grid file being the one ICON
    read). Tracers 1-4 of the capture are identical by construction (the same initial
    cylinder advected with the same scheme, icon-ajocksch/CAPTURE_NOTES.md, 'tracers 1..5
    = 1 where ...'), so the step is run for tracer 0 only and the exit savepoint's tracers
    1-3 (tracer and flux) are asserted bit-equal to its tracer 0;
L3  the 100-step trajectory from step 1 with the new tracer fed back, against the exit
    savepoints along the way, to see how round-off compounds: the per-step difference
    grows over the first tens of steps and then saturates. Step 1 -> worst step
    (gtfn_cpu): (2,0) 8.9e-16 -> 8.7e-15 (96), (3,0) 2.1e-14 -> 5.8e-14 (98), (102,0)
    1.1e-16 -> 5.9e-15 (97), (103,0) 2.5e-9 -> 1.3e-8 (12), (3,3) 2.2e-16 -> 4.9e-14
    (97), (3,4) 2.1e-14 -> 5.9e-14 (97), (2,4) 8.9e-16 -> 1.4e-14 (61); so up to 220x
    the step-1 level where step 1 is at 1e-16, but the maximum over the 100 steps stays
    within 2x of the trajectory's own value at step 100 for every case (1.0x .. 1.4x;
    2.2x for 103, whose worst step is early) and within 2x of its step-50 value except
    for 103 (3.7x).

Measured (gtfn_cpu, dace_cpu): schemes 2, 102 and the limited runs of 3 agree to 1e-15 per
step, the unlimited quadratic scheme 3 to 2e-14 (the SVD round-off of its pseudoinverse,
7e-13, propagated), and the quadratic WENO scheme 103 only to 3.5e-9 in the tracer and
8e-9 in the flux, at a handful of edges on the cylinder boundary (the hybrid scheme 132,
which selects between the quadratic fit and the 103 blend per cell, to 2.4e-9 / 4.2e-9
for the same reason). That is not the port:
substituting the Fortran candidate pseudoinverses changes nothing, and perturbing the
tracer at 1e-14 moves the flux by 2e-14. The Fortran evaluates the smoothness indicator
of every candidate in single precision, by declaration: mo_advection_hflux.f90:2643
(upwind_hflux_miura3_weno) declares 'REAL(sp) :: zlc(6), z_lsq_smooth(6), area', and
:3007 assigns 'DOT_PRODUCT(z_lsq_smooth, real(z_quad_vector_sum))', a single-precision
dot product, to the working-precision smoothness; icon4py evaluates it in double. A
single-precision indicator of an O(1) tracer carries a relative round-off of 1e-7 into
the nonlinear weights, which the blend turns into the 1e-9 seen (the earlier port
stretch reported an emulation experiment of the same size; it is not in the tree and has
not been reproduced here, so only the source-level argument stands). The ihadv103 gate
is therefore the size of the Fortran's single-precision round-off, not of a
double-precision port. With FMA contraction off
(ICON4PY_FP_CONTRACT_OFF=1, the Fortran's -Mnofma) the numbers change in the last printed
digit only, except that the step-1 tracer of scheme 102 becomes bit-identical (one
subnormal residue, 7e-42): the 1e-16 seen with contraction is the port's own FMA.

The grid is built from the grid file (as the driver does), not from the 'icon-grid'
savepoint: with nproma = 1 ICON's neighbour_idx is identically 1 and the block index that
carries the cell number is not serialized, and the WENO states need C2E2C2E2C, which the
savepoint grid lacks.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Final

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.tracer_advection import tracer_advection, weno_least_squares as weno
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.config import config_io
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import geometry_attributes as geometry_attrs
from icon4py.model.common.interpolation import (
    interpolation_attributes,
    interpolation_factory,
    interpolation_fields,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver
from icon4py.model.testing import (
    config as test_config,
    definitions as test_defs,
    grid_utils as gridtest_utils,
    serialbox as sb,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    process_props,
)

from ..fixtures import advection_exit_savepoint, advection_init_savepoint
from ..utils import construct_diagnostic_init_state, construct_prep_adv


_HADV = tracer_advection.HorizontalAdvectionType
_HLIM = tracer_advection.HorizontalAdvectionLimiter

#: ihadv_tracer -> icon4py scheme (weno_idealized_scope.md, scheme numbering)
SCHEMES: Final[dict[int, _HADV]] = {
    2: _HADV.LINEAR_2ND_ORDER,
    3: _HADV.QUADRATIC_3RD_ORDER,
    102: _HADV.LINEAR_2ND_ORDER_WENO,
    103: _HADV.QUADRATIC_3RD_ORDER_WENO,
    132: _HADV.QUADRATIC_3RD_ORDER_WENO_HYBRID,
}
#: itype_hlimit -> icon4py limiter; schemes 2 and 3 use ICON's own limiters, so these
#: mean the same thing on both sides (the WENO routines would select Jocksch's cell-local
#: limiter for 4, which icon4py does not have; no such case is listed below)
LIMITERS: Final[dict[int, _HLIM]] = {
    0: _HLIM.NO_LIMITER,
    3: _HLIM.MONOTONIC,
    4: _HLIM.POSITIVE_DEFINITE,
}

#: the model date of every advection savepoint; the calls are told apart by 'step'
DATE: Final = "2001-01-01T00:16:40.000"
NUM_STEPS: Final = 100
#: the levels of the capture; the columns are identical
NUM_LEVELS: Final = 10
#: tracers 1-4 (0-based 0-3) are advected, tracer 5 is the initial cylinder
ADVECTED_TRACERS: Final = (0, 1, 2, 3)
EXPERIMENT_CONFIG: Final = test_config.EXPERIMENT_CONFIG_PATH / "jocksch_cylinder.yaml"

#: L1 gates, max |py - f90| / max |f90| per array (per candidate for the candidate sets);
#: measured: bit-identical for the stencil, moments, moments_hat, row weights and
#: l_weights_s; the SVD pseudoinverses differ by LAPACK-vs-ICON round-off: 7.2e-13 full
#: quadratic, 2.5e-12 worst of the 27 candidates, 3.5e-16 linear full, 2.7e-16 linear
#: candidates. The 'weno_least_squares' quantities are pure numpy and therefore the same
#: on every backend; the linear full pseudoinverse is the interpolation factory's and is
#: computed on the backend's array namespace (see its gate).
L1_TOLERANCE_QUADRATIC_PSEUDOINV: Final = 8e-13
L1_TOLERANCE_QUADRATIC_CANDIDATES: Final = 3e-12
#: 3.5e-16 on CPU; this SVD is `interpolation_fields.py` `array_ns.linalg.svd`, cupy on GPU
L1_TOLERANCE_LINEAR_PSEUDOINV: Final = 8e-16
L1_TOLERANCE_LINEAR_CANDIDATES: Final = 3e-16

#: L2 gates per case, (max |q_py - q_f90|, max |F_py - F_f90| / max |F_f90|) for tracer 0
#: over the steps 1, 2, 50, 100, and the trajectory gate on max |q_py - q_f90| over all
#: 100 steps: twice the worst value measured on gtfn_cpu, dace_cpu and gtfn_cpu with
#: -ffp-contract=off (in brackets, per case), see the module docstring for the ihadv103 case
L2_TOLERANCES: Final[dict[tuple[int, int], tuple[float, float]]] = {
    (2, 0): (2e-15, 3e-15),  # 8.9e-16, 1.3e-15
    (3, 0): (5e-14, 1e-13),  # 2.1e-14, 4.7e-14
    (102, 0): (
        2e-15,
        3e-15,
    ),  # 4.4e-16 (2 ulp of an O(1) tracer; 4 ulp leaves room for GPU reordering), 1.3e-15
    (103, 0): (7e-9, 2e-8),  # 3.5e-9, 7.8e-9
    (3, 3): (8e-15, 2e-14),  # 3.7e-15, 8.2e-15
    (3, 4): (5e-14, 1e-13),  # 2.1e-14, 4.7e-14
    (2, 4): (2e-15, 3e-15),  # 8.9e-16, 1.3e-15
    (132, 0): (5e-9, 9e-9),  # 2.4e-9, 4.2e-9 (the 103 candidates' REAL(sp) indicator again)
}
TRAJECTORY_TOLERANCES: Final[dict[tuple[int, int], float]] = {
    (2, 0): 2e-14,  # 8.8e-15 (step 96)
    (3, 0): 1.2e-13,  # 5.8e-14 (step 98)
    (102, 0): 1.3e-14,  # 6.2e-15 (step 97; 5.9e-15 contracted)
    (103, 0): 3e-8,  # 1.3e-8 (step 12)
    (3, 3): 1e-13,  # 4.9e-14 (step 97)
    (3, 4): 1.2e-13,  # 6.0e-14 (step 97)
    (2, 4): 3e-14,  # 1.4e-14 (step 61)
    (132, 0): 1.4e-8,  # 7.0e-9 (step 3)
}

CASES: Final = [
    pytest.param((2, 0), id="ihadv2_hlim0"),
    pytest.param((3, 0), id="ihadv3_hlim0"),
    pytest.param((102, 0), id="ihadv102_hlim0"),
    pytest.param((103, 0), id="ihadv103_hlim0"),
    pytest.param((3, 3), id="ihadv3_hlim3"),
    pytest.param((3, 4), id="ihadv3_hlim4"),
    pytest.param((2, 4), id="ihadv2_hlim4"),
    pytest.param((132, 0), id="ihadv132_hlim0"),
]
STEPS: Final = [1, 2, 50, 100]
TRAJECTORY_REPORT_STEPS: Final = (1, 10, 50, 100)


def scale_relative_difference(values: np.ndarray, reference: np.ndarray) -> float:
    """max |values - reference| / max |reference| (0 for an all-zero reference)."""
    scale = float(np.abs(reference).max())
    difference = float(np.abs(values - reference).max())
    return difference / scale if scale > 0.0 else difference


def jocksch_experiment_config(
    case: tuple[int, int], output_path: pathlib.Path
) -> driver_config.ExperimentConfig:
    """The driver configuration of the Python cylinder experiment, in the capture's setup."""
    ihadv_tracer, itype_hlimit = case
    config = config_io.read_yaml_str(EXPERIMENT_CONFIG.read_text(), driver_config.ExperimentConfig)
    return config.with_overrides(
        vertical_grid={"num_levels": NUM_LEVELS},
        driver={"output_path": output_path, "enable_output": False},
        tracer_advection={
            "horizontal_advection_type": SCHEMES[ihadv_tracer],
            "horizontal_advection_limiter": LIMITERS[itype_hlimit],
        },
    )


@pytest.fixture
def case(request: pytest.FixtureRequest) -> tuple[int, int]:
    """(ihadv_tracer, itype_hlimit); set by parametrization."""
    return request.param


@pytest.fixture
def experiment_description(case: tuple[int, int]) -> test_defs.ExperimentDescription:
    return test_defs.Experiments.jocksch_cylinder(*case)


@pytest.fixture
def experiment(
    experiment_description: test_defs.ExperimentDescription,
    case: tuple[int, int],
    download_ser_data: None,  # checks the data is in place as side-effect
    tmp_path: pathlib.Path,
) -> test_defs.Experiment:
    # the capture directory has no Fortran namelist dictionaries: the configuration is
    # the Python experiment's, so the granule under test is the one the driver runs
    return test_defs.Experiment(
        experiment_description=experiment_description,
        experiment_config=jocksch_experiment_config(case, tmp_path / "driver_output"),
    )


@pytest.fixture
def date() -> str:
    return DATE


@dataclasses.dataclass(frozen=True)
class _Granule:
    advection: tracer_advection.Advection
    grid: object


_granules: dict[tuple[str, str], _Granule] = {}


@pytest.fixture
def advection_granule(
    experiment: test_defs.Experiment,
    backend: gtx_typing.Backend | None,
    process_props: decomposition.ProcessProperties,
) -> _Granule:
    """The tracer advection granule the driver builds for the case, cached per backend."""
    key = (experiment.name, data_alloc.backend_name(backend))
    if key not in _granules:
        grid_manager = gridtest_utils.get_grid_manager_from_identifier(
            experiment.grid,
            num_levels=NUM_LEVELS,
            keep_skip_values=True,
            allocator=model_backends.get_allocator(backend),
        )
        icon4py_driver = driver.initialize_driver(
            config=experiment.config,
            grid_manager=grid_manager,
            process_props=process_props,
            backend=backend,
        )
        assert icon4py_driver.granules.tracer_advection is not None
        _granules[key] = _Granule(icon4py_driver.granules.tracer_advection, grid_manager.grid)
    return _granules[key]


# ---- L1 ----


@pytest.mark.datatest
@pytest.mark.parametrize("case", [pytest.param((103, 0), id="ihadv103_hlim0")], indirect=True)
def test_lsq_coefficients_match_reference(
    case: tuple[int, int],
    *,
    experiment: test_defs.Experiment,
    data_provider: sb.IconSerialDataProvider,
    backend: gtx_typing.Backend | None,
) -> None:
    reference = data_provider.from_lsq_coefficients_savepoint()
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = geometry.grid
    domain_length = grid.grid_params.domain_length
    domain_height = grid.grid_params.domain_height
    assert domain_length is not None and domain_height is not None
    c2e2c = grid.get_connectivity("C2E2C").asnumpy()
    c2v = grid.get_connectivity("C2V").asnumpy()
    cell_center_x = geometry.get(geometry_attrs.CELL_CENTER_X).asnumpy()
    cell_center_y = geometry.get(geometry_attrs.CELL_CENTER_Y).asnumpy()
    quadratic, linear = reference.QUADRATIC, reference.LINEAR

    # the 9-point stencil, index by index in Jocksch's position order
    stencil_c9 = weno.create_stencil_c9(c2e2c, c2v)
    np.testing.assert_array_equal(stencil_c9, reference.stencil(quadratic))
    np.testing.assert_array_equal(c2e2c, reference.stencil(linear))

    # moments: the torus polygon integrals, and the stencil cells' moments shifted to the
    # center cell frame (f90 2217-2241)
    lsq_moments = weno.compute_lsq_moments_torus(
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        vertex_x=geometry.get(geometry_attrs.VERTEX_X).asnumpy(),
        vertex_y=geometry.get(geometry_attrs.VERTEX_Y).asnumpy(),
        c2v=c2v,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    np.testing.assert_array_equal(lsq_moments, reference.moments(quadratic))
    # llsq_lin_consv = .false.: the linear set has no moments
    np.testing.assert_array_equal(reference.moments(linear), 0.0)
    z_dist = weno.compute_torus_distance_vectors(
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        neighbor_table=stencil_c9,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    moments_hat = weno.compute_lsq_moments_hat(
        lsq_moments=lsq_moments, stencil_c9=stencil_c9, z_dist=z_dist
    )
    np.testing.assert_array_equal(moments_hat, reference.moments_hat(quadratic))

    # row weights: the full distance weights and the 27 candidate sets
    np.testing.assert_array_equal(
        interpolation_fields.compute_lsq_weights_c(z_dist, weno.LSQ_WGT_EXP_QUADRATIC),
        reference.weights_c(quadratic),
    )
    np.testing.assert_array_equal(
        weno.compute_candidate_weights_quadratic(z_dist), reference.weights_c_3(quadratic)
    )
    np.testing.assert_array_equal(weno.L_WEIGHTS_S, reference.l_weights_s(quadratic))

    # pseudoinverses: SVD round-off only
    pseudoinv = weno.compute_lsq_pseudoinverse_quadratic(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    difference = scale_relative_difference(pseudoinv, reference.pseudoinv(quadratic))
    print(f"\nL1 quadratic pseudoinverse: {difference:.3e}")
    assert difference <= L1_TOLERANCE_QUADRATIC_PSEUDOINV

    candidates = weno.compute_weno_pseudoinverse_quadratic(
        stencil_c9=stencil_c9,
        lsq_moments=lsq_moments,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    candidates_ref = reference.pseudoinv_3(quadratic)
    per_candidate = [
        scale_relative_difference(candidates[:, cand], candidates_ref[:, cand])
        for cand in range(reference.NUM_QUADRATIC_CANDIDATES)
    ]
    print("L1 quadratic candidates:", " ".join(f"{d:.1e}" for d in per_candidate))
    assert max(per_candidate) <= L1_TOLERANCE_QUADRATIC_CANDIDATES

    linear_candidates = weno.compute_weno_pseudoinverse_linear(
        c2e2c=c2e2c,
        cell_center_x=cell_center_x,
        cell_center_y=cell_center_y,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    difference = scale_relative_difference(linear_candidates, reference.pseudoinv_3(linear))
    print(f"L1 linear candidates: {difference:.3e}")
    assert difference <= L1_TOLERANCE_LINEAR_CANDIDATES

    # the linear full pseudoinverse is the interpolation factory's (miura, ihadv_tracer=2)
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=NUM_LEVELS,
        keep_skip_values=True,
        allocator=model_backends.get_allocator(backend),
    )
    interpolation = interpolation_factory.InterpolationFieldsFactory(
        config=experiment.config.interpolation,
        grid=grid,
        decomposition_info=grid_manager.decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomposition.single_node_exchange,
    )
    difference = scale_relative_difference(
        interpolation.get(interpolation_attributes.LSQ_PSEUDOINV).asnumpy(),
        reference.pseudoinv(linear),
    )
    print(f"L1 linear pseudoinverse: {difference:.3e}")
    assert difference <= L1_TOLERANCE_LINEAR_PSEUDOINV


# ---- L2 ----


def _run_step(
    granule: _Granule,
    init_savepoint: sb.AdvectionInitSavepoint,
    tracer: int,
    backend: gtx_typing.Backend | None,
    p_tracer_now: fa.CellKField[ta.wpfloat] | None = None,
) -> tuple[fa.CellKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat]]:
    """One advection step from the init savepoint's air mass and mass fluxes.

    The tracer is the savepoint's unless 'p_tracer_now' is given (the trajectory).
    Returns (p_tracer_new, hfl_tracer).
    """
    diagnostic_state = construct_diagnostic_init_state(
        granule.grid, init_savepoint, tracer, backend=backend
    )
    prep_adv = construct_prep_adv(init_savepoint)
    if p_tracer_now is None:
        p_tracer_now = init_savepoint.tracer(tracer)
    p_tracer_new = data_alloc.zero_field(granule.grid, dims.CellDim, dims.KDim, allocator=backend)
    granule.advection.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=init_savepoint.get_metadata("dtime")["dtime"],
    )
    return p_tracer_new, diagnostic_state.hfl_tracer


@pytest.mark.datatest
@pytest.mark.parametrize("case", CASES, indirect=True)
@pytest.mark.parametrize("step", STEPS)
def test_advection_step_matches_reference(
    case: tuple[int, int],
    step: int,
    *,
    advection_granule: _Granule,
    advection_init_savepoint: sb.AdvectionInitSavepoint,
    advection_exit_savepoint: sb.AdvectionExitSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    tracer = ADVECTED_TRACERS[0]
    p_tracer_new_ref = advection_exit_savepoint.tracer(tracer).asnumpy()
    hfl_tracer_ref = advection_exit_savepoint.hfl_tracer(tracer).asnumpy()
    # the capture advects four copies of the same cylinder with the same scheme (module
    # docstring, L2): one run of the granule covers them, provided they are the same
    for other in ADVECTED_TRACERS[1:]:
        np.testing.assert_array_equal(
            advection_exit_savepoint.tracer(other).asnumpy(), p_tracer_new_ref
        )
        np.testing.assert_array_equal(
            advection_exit_savepoint.hfl_tracer(other).asnumpy(), hfl_tracer_ref
        )

    p_tracer_new, hfl_tracer = _run_step(
        advection_granule, advection_init_savepoint, tracer, backend
    )
    tracer_difference = float(np.abs(p_tracer_new.asnumpy() - p_tracer_new_ref).max())
    flux_difference = scale_relative_difference(hfl_tracer.asnumpy(), hfl_tracer_ref)
    print(
        f"\nL2 ihadv{case[0]}_hlim{case[1]} step {step:3d} tracer {tracer}: "
        f"max|dq| = {tracer_difference:.3e}  max|dF|/max|F| = {flux_difference:.3e}"
    )
    tracer_tolerance, flux_tolerance = L2_TOLERANCES[case]
    assert tracer_difference <= tracer_tolerance
    assert flux_difference <= flux_tolerance


# ---- trajectory ----


@pytest.mark.datatest
@pytest.mark.parametrize("case", CASES, indirect=True)
def test_advection_trajectory_matches_reference(
    case: tuple[int, int],
    *,
    advection_granule: _Granule,
    data_provider: sb.IconSerialDataProvider,
    backend: gtx_typing.Backend | None,
) -> None:
    tracer = 0
    size = data_provider.grid_size
    p_tracer_now = data_provider.from_advection_init_savepoint(size=size, date=DATE, step=1).tracer(
        tracer
    )
    drift: dict[int, float] = {}
    for step in range(1, NUM_STEPS + 1):
        # the init savepoint of the step carries the (constant) air mass and mass fluxes;
        # the tracer is the Python trajectory's own
        init_savepoint = data_provider.from_advection_init_savepoint(
            size=size, date=DATE, step=step
        )
        p_tracer_new, _ = _run_step(
            advection_granule, init_savepoint, tracer, backend, p_tracer_now=p_tracer_now
        )
        reference = data_provider.from_advection_exit_savepoint(
            size=size, date=DATE, step=step
        ).tracer(tracer)
        drift[step] = float(np.abs(p_tracer_new.asnumpy() - reference.asnumpy()).max())
        p_tracer_now = p_tracer_new
    worst_step = max(drift, key=drift.get)
    print(
        f"\ntrajectory ihadv{case[0]}_hlim{case[1]} max|dq| at steps "
        + ", ".join(f"{s}: {drift[s]:.3e}" for s in TRAJECTORY_REPORT_STEPS)
        + f"; worst step {worst_step}: {drift[worst_step]:.3e}"
    )
    assert drift[worst_step] <= TRAJECTORY_TOLERANCES[case]
