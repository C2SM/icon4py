# WENO idealized experiments — scope note

Milestone 1 of the tracer-advection port: reproduce Andreas Jocksch's idealized FFSL-WENO
experiments in icon4py, with the init-time coefficients computed in Python, and show the
port is correct against (a) his Fortran run and (b) his paper. No ICON granule, no Fortran
binding in this milestone.

Sources. Paper: Jocksch et al., *A flux-form semi-Lagrangian WENO scheme on triangular
meshes implemented on GPUs*, PPAM 2026 (`PPAM_2026_paper_20.pdf` at the project root).
Fortran: `icon-exclaim` branch `transport_ajocksch` @ `dacecf46aa` (2026-07-20), checked out
as the `icon-ajocksch/` worktree; all `f90:` line numbers below refer to that commit. His
work is `git diff exclaim/icon-dsl...dacecf46aa` — 15 files under `src/`, nothing under
`run/`, so the mesh and namelists he used are not in git.

## Scheme numbering (Fortran → icon4py)

`hor_upwind_flux`, `src/advection/mo_advection_hflux.f90:301-425`:

| `ihadv_tracer` | Fortran routine | paper name | icon4py (`HorizontalAdvectionType`) |
|---|---|---|---|
| 2 | `upwind_hflux_miura` (f90:1642) | linear lsq | `LINEAR_2ND_ORDER` |
| 3 | `upwind_hflux_miura3` (f90:4378) | quadratic lsq | `QUADRATIC_3RD_ORDER` |
| 102 | `upwind_hflux_miura_weno` (f90:1165) | linear WENO, 3 three-point sub-stencils | `LINEAR_2ND_ORDER_WENO` |
| 103 | `upwind_hflux_miura3_weno` (f90:2532) | quadratic WENO, 27 six-point sub-stencils | `QUADRATIC_3RD_ORDER_WENO` |
| 132 | `upwind_hflux_miura_weno_hyb` (f90:3136) | hybrid, `c_sel` = 5e-5 (paper eq. 6) | **missing** |
| 202 | `upwind_hflux_miura_cell` (f90:702) | linear lsq, cell-based kernel | not planned (schedule variant) |
| 203 | `upwind_hflux_miura3_cell` (f90:3815) | quadratic lsq, cell-based kernel | not planned (schedule variant) |

Table 2 of the paper also distinguishes "WENO d_j = 1" from "WENO opt". Both are `103`;
they differ only in the linear weights `l_weights_s` set at init
(`src/shr_horizontal/mo_intp_coeffs_lsq_bln.f90:2590-2600`: the live set is the optimised
one, `0, 0, 0, 0, 2.991549980478795` for stencil types I–IV / V; type VI is fixed to 1 at
f90:2647). icon4py has only the optimised set (`weno_least_squares.py`, `L_WEIGHTS_S`).
**Missing: `d_j = 1` as a configuration option.**

## Limiters

- In the WENO and cell-based routines, `itype_hlimit = 4` (`ifluxl_sm`) selects **Andreas'
  cell-local positive-definite limiter**, applied inside the reconstruction/flux kernel
  (`f90:1056-1073` for 202, `f90:1533-1550` for 102): a reconstruction clamp
  (`flux = max(flux, 0)`) followed by a per-cell outflow scaling `rfac = min(1, q·ρ/(Σ
  outflow + ε))` (paper Alg. 1). ICON's standard `hflx_limiter_pd` call is commented out
  there (`f90:1138`, `f90:1615`). So every "limiter" row of Table 2 uses his limiter, and
  `itype_hlimit = 4` does **not** mean the same thing in 102/103/132/202/203 as in 2/3.
- `itype_hlimit = 3` (`ifluxl_m`) still calls ICON's `hflx_limiter_mo` (`f90:1129`,
  `f90:1606`); the paper's 3.371 is that limiter on scheme 3.
- icon4py has ICON's PD (4) and monotonic (3) limiters. **Missing: his cell-local limiter**
  as a distinct limiter value (it changes results, not only the schedule — do not alias it
  to 4).

## Init-time coefficients

`lsq_compute_coeff_cell_torus`, `mo_intp_coeffs_lsq_bln.f90:1389+` (torus geometry only;
the sphere branch is `lsq_compute_coeff_cell`, f90:596 dispatch):

- 27 quadratic candidate pseudoinverses `lsq_pseudoinv_3(9, 5, 27, cell, blk)` built per
  candidate from zeroed distance weights `lsq_weights_c_3` (f90:2161-2295, SVD at
  f90:2558-2575); linear WENO uses 3 candidates from `lsq_lin` (3-point stencils, exact
  inverses).
- Candidates 1–3 (type VI) are **assembled, not fitted**: initialised to the full 9-point
  pseudoinverse (f90:2586) and then reduced by the weighted sum of one 120°-group each,
  `pseudoinv_3(:,:,k) -= Σ_{i in group k} pseudoinv_3(:,:,i) · l_weights_s(i)`
  (f90:2647-2660), with `l_weights_s(1:3) = 1`.
- `lsq_error_3` (f90:2546) stores `A·A⁺` per candidate, used by the hybrid's selection
  criterion.
- The `.false.` block in `mo_nh_stepping.f90` that re-does this assembly at run time with
  a gradient-descent loop on `l_weights_s` (the weight optimisation) is **dead in the live
  path**; the live cylinder block does not touch the coefficients.

icon4py: `weno_least_squares.py` ports the torus branch (moments, stencil, 27 + 3 candidate
pseudoinverses, opt weights). Evidence so far: property tests and a numpy replica — **no
comparison with Fortran `lsq_pseudoinv_3` / `l_weights_s` exists yet** (L1 in W5).

Open question for the order study (W5): with this assembly, uniform smoothness gives
`Σ_j d_j A⁺_j = 3·A⁺_full`, normalised by `Σ_j d_j`; whether that reduces to the plain
quadratic scheme (paper §2.3 says it does for `d_j = 1`) is exactly what the torus-patch
study must measure rather than assume.

## The cylinder experiment (live block, `mo_nh_stepping.f90`, hunk after the
`serialize_all("step_advection", .TRUE.)` line, `if (.true.)` block)

- Mesh: 880 cells / 1320 edges, edge length a = 5000 m (`nblks_e = 1320` with
  `nproma = 1`; the "middle" edge sits at y = 4330 = a·√3/2). Only `20 × 22` gives 880
  cells with a periodic return after 20 edge lengths → `weno_data/grids/torus_20x22_res5000m.nc`
  (icon4py `grid_generator`, rectangular periodic layout, 100 km × 95.26 km). The paper's
  "2 × 1.67, d = 0.833" does not reduce cleanly to this; his actual file is being requested.
- Wind: uniform, |v| = 1 m/s, θ = 0 (x direction); edge mass flux `(-y_e·u_x + x_e·u_y)/|e|`
  with periodic wrap fix-ups; `vn_traj = mass_flx_me`; vertical mass flux 0; `ddqz = 1`,
  `airmass_now = airmass_new = 1`.
- IC: tracers 1–5 = 1 where `sqrt(x² + y²) < 25000 m` in ICON's `cells%cartesian_center`
  coordinates — the cylinder is centred at ICON's coordinate **origin**. How this grid
  file's coordinates (x from 0) map into ICON's torus coordinates is being established by
  the Fortran run (W1); the Python experiment keeps the centre as a parameter.
- Time step `dt = (a/2)·1.9999999999·CFL`, CFL = 0.2 → 1000 s; **100** calls of
  `step_advection`; tracers 1–4 copied `new → now` each call; tracer 5 keeps the IC.
- Error (printed as `"#", error(1)`): sum over cells and their three neighbours with
  `(ii > i .or. jj > j)` and centre distance `< 5000 m` of `(q5_now − q1_new)²` for both
  cells of the pair. With `nproma = 1` this is a sum over unordered neighbour pairs, each
  once; the distance test is always true on this mesh (centre distance a/√3 = 2887 m); on
  a torus every cell is in exactly three pairs, so the measure equals **3·Σ_i e_i²**.
  Whether Table 2 prints this sum or its square root is not stated — W1/W2 settle it.
- Paper Table 2 (100 steps, CFL 0.2): no limiter 4.023 / 3.859 / 3.402 / 3.310 / 3.310 /
  3.058 for linear lsq / linear WENO / quadratic lsq / WENO d_j=1 / hybrid / WENO opt;
  with his limiter 3.778 / 3.842 / 3.361 / 3.309 / 3.308 / 3.003; monotonic limiter on
  quadratic lsq 3.371.

## Dispersion-relation experiment (paper §4, `.false.` blocks in the same hunk)

Complex tracer `q = exp(i α x)` on cell centres, one `step_advection`, `ω = i·ln(q₁/q₀)/dt`;
θ = 30° needs the chequerboard-converged solution first. Full (non-WENO) stencils only.
Stability limit CFL ≈ 0.42 (Fig. 7). Planned as W4; needs no Fortran reference.

## Gap list for milestone 1

1. Fortran reference run + savepoints (W1): `advection-init/exit` hooks exist on his branch
   (`src/advection/mo_advection_stepping.f90:208,639`); need a `step` metainfo and a
   coefficient savepoint.
2. Python cylinder experiment with his error measure (W2).
3. `d_j = 1` weights, hybrid 132, his cell-local limiter (W3).
4. L1/L2/field-level tests vs the capture on all backends, Table 2 gates, 103 order
   study (W5). 103 currently launches 2 stencils × 27 candidates per step; fine at 880
   cells, restructure later.
5. Dispersion relation (W4).
