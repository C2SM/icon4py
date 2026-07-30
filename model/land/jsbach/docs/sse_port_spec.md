# SSE (soil-snow energy) — port spec

First vertical slice of the JSBACH port. This document is the requirements
reference extracted (and adversarially verified) from the Fortran source in
`icon-nwp/externals/jsbach/src/soil_snow_energy/`. All `file:line` citations are
point-in-time — verify against current source before relying on them.

Overall plan: see the design doc in the icon4py-knowledge repo
(`personal/jcanton/jsbach-port`). Scope decisions there: in-ICON coupling
(product) + two-tier oracle (validation); SSE chosen first because it exercises
every hard framework primitive (3D field, tridiagonal vertical solve,
multi-timestep prognostic state, one weighted tile aggregation) with low
transcendental density → a bit-exact validation gate is achievable.

## Ubiquitous language (Fortran name → domain name)

The port uses intention-revealing domain names; this table is the anticorruption
mapping back to the Fortran oracle for validation. Fortran names are kept only in
this table and in oracle-comparison code, not in the domain layer.

| Fortran                                | domain concept                    | vgrid               | state                      |
| -------------------------------------- | --------------------------------- | ------------------- | -------------------------- |
| `t_soil_sl`                            | soil temperature                  | `soil_depth_energy` | prognostic                 |
| `t_snow`                               | snow temperature                  | `snow_depth_energy` | prognostic                 |
| `snow_depth_sl`                        | snow depth per layer              | `snow_depth_energy` | prognostic                 |
| `t_soil_acoef`, `t_soil_bcoef`         | soil R&M tridiagonal coefficients | `soil_depth_energy` | internal state (see note)  |
| `t_snow_acoef`, `t_snow_bcoef`         | snow R&M tridiagonal coefficients | `snow_depth_energy` | internal state             |
| `hcap_grnd`                            | ground heat capacity              | surface (2D)        | prognostic                 |
| `grnd_hflx`                            | ground heat flux                  | surface (2D)        | prognostic (output to SEB) |
| `thaw_depth_max`, `thaw_depth_max_ym1` | max thaw depth (this/prev year)   | surface (2D)        | prognostic                 |
| `vol_heat_cap[_sl/_snow]`              | volumetric heat capacity          | soil/snow/2D        | diagnostic                 |
| `heat_cond[_sl/_snow]`                 | heat conductivity                 | soil/snow/2D        | diagnostic                 |
| `thaw_depth`                           | thaw depth                        | surface (2D)        | diagnostic                 |
| `hcap_grnd_old`, `grnd_hflx_old`       | previous-step ground values       | surface (2D)        | diagnostic                 |
| `t_srf` (from SEB)                     | surface temperature (top BC)      | surface (2D)        | forcing (input)            |

## True state set

Source: `mo_sse_memory_class.f90`. Restart default is `.TRUE.` unless
`lrestart=.FALSE.` is passed (`mo_jsb_memory_class.f90:240-242,438-439,640-641`).
Verified independently (adversarial pass, high confidence).

**Prognostic (must persist across timesteps):**
`t_soil_sl` (L179, init 280 K), `snow_depth_sl` (L201), `t_snow` (L211, init
273.15 K), `hcap_grnd` (L233), `grnd_hflx` (L249), `thaw_depth_max` (L274),
`thaw_depth_max_ym1` (L284). For the `jsbach_lite`+tmx scope the QUINCY-only
`bulk_dens_sl/sand_sl/silt_sl/clay_sl` (L296-339) are out of scope.

**Diagnostic (recomputed each step):** `vol_heat_cap` (L120), `vol_heat_cap_sl`
(L128), `vol_heat_cap_snow` (L136), `heat_cond` (L144), `heat_cond_sl` (L152),
`heat_cond_snow` (L160), `thermal_diffusivity` (L170, conditional),
`hcap_grnd_old` (L241), `grnd_hflx_old` (L257), `thaw_depth` (L265).

**Note on R&M coefficients:** `t_soil_acoef/bcoef` and `t_snow_acoef/bcoef`
default to restart-`.TRUE.` but are recomputed every step (a known JSBACH
over-inclusion in the restart set). They are two-time-level *carry* state within
the scheme, not independent prognostics — see the tridiagonal note.

## Aggregation

One operator only: `weighted_by_fract` (area-weighted), for all 23 fields
aggregated in `mo_sse_interface.f90` (routines `aggregate_soil_and_snow_temperature`
:639-696 and `aggregate_soil_and_snow_properties` :1010-1044). Confirmed: no other
aggregator appears. For the single-leaf `jsbach_lite` tile tree this reduces to a
no-op / identity in the first slice, but the operator must exist for correctness
when tiles are added.

## Vertical grids

- `soil_depth_energy` — `nsoil` = length of `soillev` in `ic_land_soil.nc`
  (`mo_sse_config_class.f90:235-256`); **data-dependent**, canonical 5 layers
  `dz = (0.065, 0.254, 0.913, 2.902, 5.700)` m (`:293`); resolved at runtime as
  `soil_e%n_levels` (`mo_sse_interface.f90:220`).
- `snow_depth_energy` — `nsnow` = namelist `nsnow`, default 5, `dz_snow = 0.05` m
  (`mo_sse_config_class.f90:141-143`), constraint `3 ≤ nsnow ≤ 20`.

## Kernels → GT4Py `scan_operator`

### `calc_soil_temperature` (`mo_sse_process.f90:353-757`)

Self-contained Richtmyer-Morton two-time-level implicit scheme. 19 required + 10
optional args (freeze/melt + supercooling gated by `PRESENT(wtr)`, `l_freeze`,
`l_supercool`). Numerically referentially transparent — every value flows through
explicit array/scalar args. Classified `needs_context` **only** for the host-only
water-budget check calling `finish`/`message` (`:557-592`, `#ifndef _OPENACC`)
and the `acc_stream` OpenACC async handle (non-numeric). No module state is
written.

Tridiagonal structure (the `scan_operator` spec):

- constants: `zd1` (`:436-438`), `heat_cap` (`:441-447`);
- **back-substitution** (top→bottom, from *previous* step's coefficients):
  top layer `t_soil_sl(1)=t_soil_top` (`:487-491`), then
  `t_soil_sl(is+1)=acoef(is)+bcoef(is)*t_soil_sl(is)` (`:494-504`); runs only when
  `.NOT. lstart`;
- **forward elimination** (bottom→top, building *next* step's coefficients):
  prep `zdz2`/`zdz1` (`:704-719`), bottom init (`:722-728`), sequential loop
  `DO is=nsoil-1,2,-1` with Thomas pivot `z1` (`:731-743`);
- ground flux / heat capacity from the top coefficient (`:748-753`).

GT4Py model: two KDim scans — (1) top→bottom reconstructing `t_soil_sl` from
lagged `acoef/bcoef`, (2) bottom→top producing new `acoef/bcoef`. The freeze/melt
(`:507-624`) and thaw-depth (`:642-687`) blocks sit between and depend on the
optional water/ice fields — **out of scope for the first non-freezing slice**
(start with `l_freeze=.FALSE.`, `l_supercool=.FALSE.`).

### `calc_snow_temperature` (`mo_sse_process.f90:763-860`)

Pure (no side effects; only the non-numeric `acc_stream`). Holds **only** the snow
back-substitution: top layer keeps `t_srf` (`:796-802`), then
`DO is=2,nsnow: IF (is>itop) t_snow(is)=acoef(is-1)+bcoef(is-1)*t_snow(is-1)`
(`:826-838`), then new top-soil temperature (`:841-849`). The `itop(ic)` variable
top-layer index masks absent upper snow layers (pass-through). Runs only when
`.NOT. lstart`.

The snow **coefficient build + forward elimination** live in a separate routine,
`calc_snow_abcoeff` (`:866-1035`, sweep `:1003-1017`). GT4Py: a top→bottom KDim
scan over `is=1..nsnow` guarded by `is>itop`, carrying `t_snow(is-1)`.

Both kernels: loop parallelism is SEQ over the layer index, GANG/VECTOR over cells
— i.e. the layer index is the scan (KDim) axis, cells are the horizontal field
axis.

## Inputs and the two oracle tiers

Both tiers share the SSE init path `sse_read_init_vars` (`mo_sse_init.f90:133-315`)
— only the top-boundary driver differs.

- `ic_land_soil.nc` (both tiers) — `surf_temp` (12-month climatology) initializes
  `t_soil_sl` via `init_soil_temperature` (`mo_sse_init.f90:303-308,539-544`);
  `soillev` defines the `soil_depth_energy` vgrid.
- `bc_land_soil.nc` (both tiers) — with the tmx config (`l_soil_texture=F`,
  `l_heat_cap_map=F`, `l_heat_cond_map=F`) the active read is the `fao` index →
  `fao_vol_hcap`/`fao_thermal_diff` tables → `vol_heat_cap` & `heat_cond`
  (`mo_sse_init.f90:260-263`, `:437-451`).
- SSE **consumes** `t_srf` from SEB (top BC, `mo_sse_interface.f90:269`) and
  **produces** `grnd_hflx`, `hcap_grnd` back to SEB (`:246-249`,
  `mo_sse_process.f90:750-751`). `t_soil_acoef/bcoef` are SSE-internal, not
  external forcing.
- Tier-1 (offline standalone) atmospheric forcing (`air_temp`, `precip`,
  `shortwave`, `longwave`, `CO2`, `wind_speed`) is read by `mo_jsb4_forcing.f90`
  and drives SEB one level above SSE — not read by SSE directly.

The exact `nsoil` and the IC/bc file contents live on the serialization machine
(no local Fortran builds); tier-2 golden capture needs JSBACH savepoints added to
`exp.aes_bubble_land_tmx` (none exist yet).
