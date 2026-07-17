# AMPS lookup-table data

`amps_luts.npz` is a converted copy of the AMPS Fortran collision-efficiency,
habit-frequency, diagnostic-map, and standard-normal lookup tables that the
scheme reads from disk at startup (`RDCETB`/`RDSTTB`,
`mod_amps_utility.F90`).

## Provenance

- Source: `/Users/jcanton/projects/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/`
  (`collision_data/` and `statpack/` subdirectories; the real data files
  used by the cloudlab reference run).

- Converter: `../../../../../../../codegen/convert_luts.py` (i.e.
  `model/atmosphere/subgrid_scale_physics/amps/codegen/convert_luts.py`
  from the repo root), run once via:

  ```sh
  uv run --frozen python \
      model/atmosphere/subgrid_scale_physics/amps/codegen/convert_luts.py
  ```

- Parsing follows the reader-subroutine formats transcribed verbatim in
  `docs/superpowers/facts/m1/lut-files.md` (the scale_amps repo's "F3" fact
  file) SS2 (RDCETB/RDSTTB) and SS3 (representative file heads). `RDAPTB`
  (`ap_act.bin.dat`, 7.9 MB) is dead code (bare `return` before any I/O,
  F3 SS2.2) and is deliberately NOT converted.

- Regenerated: 2026-07-17.

## Contents (16 tables, 33 npz keys: `<name>` + `<name>_aux` except `znorm`)

| key            | shape       | dtype   | source file(s)                       | aux fields                                                          |
| -------------- | ----------- | ------- | ------------------------------------ | ------------------------------------------------------------------- |
| `drpdrp`       | (201, 201)  | float64 | `drop_drop_Rey4.dat`                 | xs,dx,ys,dy,nr,nc                                                   |
| `hexdrp`       | (64, 71)    | float64 | `hex_drop_Nre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `bbcdrp`       | (64, 71)    | float64 | `bbc_drop_Nre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `coldrp`       | (62, 71)    | float64 | `col_drop_Nre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `gp1drp`       | (37, 125)   | float64 | `grp01_ratNre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `gp4drp`       | (27, 125)   | float64 | `grp04_ratNre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `gp8drp`       | (21, 125)   | float64 | `grp08_ratNre_Ec.dat`                | xs,dx,ys,dy,nr,nc                                                   |
| `pol_frq`      | (51, 101)   | float64 | `pol_frq.dat`                        | nr,nc                                                               |
| `pla_frq`      | (51, 101)   | float64 | `pla_frq.dat`                        | nr,nc                                                               |
| `col_frq`      | (51, 101)   | float64 | `col_frq.dat`                        | nr,nc                                                               |
| `ros_frq`      | (51, 101)   | float64 | `ros_frq.dat`                        | nr,nc                                                               |
| `ppo_frq`      | (51, 101)   | float64 | `ppo_frq.dat`                        | nr,nc                                                               |
| `mtac_map_col` | (50, 91, 2) | float64 | `tmp_map_col.dat`, `tmd_map_col.dat` | nr,nc                                                               |
| `mtac_map_pla` | (50, 91, 2) | float64 | `tmp_map_pla.dat`, `tmd_map_pla.dat` | nr,nc                                                               |
| `lmt_mass_col` | (50,)       | float64 | `lmt_mass_col.dat` (col 2 of 4 kept) | nr,nc (of the *source* 4-column file)                               |
| `lmt_mass_pla` | (50,)       | float64 | `lmt_mass_pla.dat` (col 2 of 4 kept) | nr,nc (of the *source* 4-column file)                               |
| `znorm`        | (451, 4)    | float64 | `stdnorm.dat` (cols 3-6 of 6 kept)   | none (headerless file; implicit x-grid = 0.01\*(row-1), not stored) |

`<name>_aux` layout:

- Collision-efficiency LUTs (`drpdrp`, `hexdrp`, `bbcdrp`, `coldrp`,
  `gp1drp`, `gp4drp`, `gp8drp`): 6-element float64 array
  `[xs, dx, ys, dy, nr, nc]`, the `col_lut_aux` field order
  (`class_Group.F90:179-183`).
- Everything else with an aux key (`*_frq`, `mtac_map_*`, `lmt_mass_*`):
  2-element float64 array `[nr, nc]` (no `xs/dx/ys/dy` -- these files have
  no axis-start/step header; F3 SS3: "grid is implicit").

`pol_frq`/`pla_frq`/`col_frq` are stored ALREADY normalized (clipped to
`>= 0`, then divided by the per-cell `pol+pla+col` sum), matching the
in-place transform `RDCETB` applies right after reading
(`mod_amps_utility.F90` lines 198-215, F3 SS6). `ros_frq`/`ppo_frq` are
clipped to `>= 0` only (no renormalization) -- also per `RDCETB`.

`mtac_map_col`/`mtac_map_pla` stack the two source files as
`[..., 0] = tmp_map_*` (log10(a/c-axis ratio)) and `[..., 1] = tmd_map_*`
(density m/v_cs), mirroring the Fortran `mtac_map_col/pla(:,:,1:2)`
common-block target (F3 SS4). Their file header declares `ncol=91`, smaller
than the Fortran-declared array's `ncol=101` dimension bound
(`mtac_map_col/pla(50,101,2)`); per F3 SS6's parser note, the file's own
`nrow x ncol` is kept as-is here rather than padding out to the larger
Fortran array bound (whose tail stays uninitialized/zero-equivalent
in the original code and carries no information).

## How this file is used

`core/lookup_tables.py`'s `load_luts()` reads this archive via
`importlib.resources` (packaged-data pattern) and merges it with several
*computed* tables (osmotic-coefficient LUTs, normal/inverse-normal CDF
LUTs, IGP spline knots) that need no data file at all -- see that module's
docstring for the F3 SS5 citations.

## Regenerating

Re-run the converter (see "Provenance" above) against a fresh AMPS_DATA
checkout and commit the resulting `amps_luts.npz`. The converter
cross-checks every file's header dims against the F3-documented target
shape and raises `ValueError` on any mismatch, so a successful run is a
self-consistency guarantee.

## Known limitations (NEEDS_CONTEXT, see task report)

- `core/lookup_tables.py`'s `init_osmo_par()` osmotic-coefficient curve
  interpolation nodes are a documented INFERENCE, not a verbatim
  transcription (F3 SS5.1 paraphrases the x-grid inconsistently); the
  y-data itself is verbatim.
- `core/lookup_tables.py`'s Low-List breakup fragment tables
  (`make_breakup_fragment_tables`) provide only the correctly-SIZED,
  zero-filled allocation (`bu_fd`, `bu_tmass`); the actual fragment
  physics fill requires `cal_Coalescence_Efficiency` +
  `cal_breakup_dis_LL` (`mod_amps_core.F90`), not quoted in F3 and out of
  scope for this LUT-conversion task.
