# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""AMPS configuration: one frozen dataclass mirroring BOTH Fortran namelists
consumed by the scheme, transcribed from
docs/superpowers/facts/m1/bins-config-indexmaps.md ("F2" in comments below):

* `/AMPS_param/` (F2 §4: `read_AMPSTASK`, the full field list; F2 §5: the
  actual cloudlab `AMPSTASK.F` file, quoted FULL, providing the cloudlab
  value for every field except `debug` -- see the NEEDS_CONTEXT note on
  that field below). This namelist has NO Fortran compile-time defaults
  anywhere in F2 (`read_AMPSTASK` unconditionally `open`s and `read`s
  `AMPSTASK.F`; most of its variables are declared without a Fortran
  initializer, e.g. F2 §1's supplementary `com_amps.F90` block:
  `integer :: level_comp, debug_level, ...` -- no `=`). Consequently every
  `AmpsConfig` field sourced from this namelist defaults to its cloudlab
  F2 §5 value (the only concrete value F2 provides for it), NOT to a
  genuine Fortran default -- documented per-field below.
* `PARAM_ATMOS_PHY_MP_AMPS_bin` (F2 §6a: module-level Fortran defaults,
  `scale_atmos_phy_mp_amps.F90` lines 121-179, quoted with real
  initializers; F2 §6b: the namelist's full field list with descriptive
  comments). These fields DO have genuine Fortran defaults, used as the
  `AmpsConfig()` defaults per the task brief ("Defaults = the Fortran
  defaults from F2 §6"). `cloudlab()`/`cloudlab_seeding()` override them
  with the literal `run.conf`/`restart_run.conf` values (cited from
  `/Users/jcanton/projects/scale_amps/scale-rm/test/case/cloudlab/scripts/
  run.conf` and `restart_run.conf`, whose exact line numbers are quoted in
  the M1 plan's Task 4 brief and independently re-verified against the
  checked-out files below).

NEEDS_CONTEXT (flagged per the task's ground-truth rule, not guessed;
non-blocking -- both are low-stakes fields with a documented, conservative
default):

* `debug` (a namelist member per F2 §4's field list) has NO value anywhere
  in F2 §5's quoted cloudlab `AMPSTASK.F` (every other namelist field is
  set there; `debug` simply never appears) and no Fortran compile-time
  default is shown either (`com_amps.F90`'s declaration block in F2 §1
  does not even list `debug` among its `integer ::`/`real(PS) ::` lines --
  it must be declared elsewhere, not quoted in F2). Defaulted here to
  `False` (F2 §1's own `binmicrosetup_scale` only ever uses `debug` to
  gate extra log writes -- see e.g. F2 §1 lines 155, 188 -- so this default
  has no numerical/physical consequence either way).
* `nbin_h` (a `PARAM_ATMOS_PHY_MP_AMPS_bin` member) has NO Fortran
  compile-time default (F2 §1's own supplementary note: "`nbin_h` has no
  compile-time default -- it is set only via namelist
  `PARAM_ATMOS_PHY_MP_AMPS_bin`"; `bin_grid.py` makes the identical
  observation and, there, leaves `nbin_h` a required keyword argument with
  no default at all). Here, since `AmpsConfig` needs *some* out-of-the-box
  default to remain constructible as `AmpsConfig()`, and F2's ONLY
  concrete value for `nbin_h` anywhere is the cloudlab run value (`20`,
  identical in both `run.conf:146` and `restart_run.conf:147` -- not a
  compile-time Fortran default, but the sole value F2 evidences), that
  value is used as the dataclass default and documented as such.

Micexfg index-16/17 naming: RESOLVED against call-site ground truth (this
supersedes an earlier draft of this docstring that followed F2 §5's
comment text and got it backwards -- comments lose to code). F2 §5's own
primary quoted Fortran comment text reads:

    ! 16: ice nucleation; homogenious freezing
    ! 17: ice nucleation; immersion freezing (heterogenious freezing)

but this is a STALE Fortran comment, contradicted by the actual call
path:

* `class_Cloud_Micro.F90:1131-1135` passes `CM%micexfg(14), CM%micexfg(15),
  CM%micexfg(16), CM%micexfg(17)` POSITIONALLY into `Ice_Nucleation1`'s
  dummy arguments `iflg_cfz, iflg_spl, iflg_ifz, iflg_hfz`
  (`mod_amps_core.F90:3078-3081`) -- i.e. slot 16 binds to `iflg_ifz`
  ("ifz" = immersion freezing) and slot 17 binds to `iflg_hfz` ("hfz" =
  homogeneous freezing), by dummy-argument name alone.
* Inside `Ice_Nucleation1`, `iflg_ifz` drives
  `select case(iflg_ifz)`: `case(2)` calls `immersion_mode` (the
  "standard" scheme), `case(1)` calls `immersion_mode_KC04`
  (`mod_amps_core.F90:3161-3171`) -- confirming slot 16 is the 3-way
  immersion-freezing scheme selector (`0`=off, `1`=KC04, `2`=standard),
  exactly as this task's ground-truth process mapping (design spec) and
  `docs/superpowers/facts/verification/rngv_other-run-quirks.md` both
  independently asserted.
* `iflg_hfz` drives a plain `if(iflg_hfz/=0)` guard around
  `homfreez_mode` (`mod_amps_core.F90:3176-3181`) -- any nonzero value
  triggers it, consistent with slot 17 being a simple on/off homogeneous-
  freezing switch (not a 3-way selector like slot 16).

So: slot 16 = immersion freezing (`ice_nucleation_immersion`, `int`,
0=off/1=KC04/2=standard -- see `ImmersionFreezingMode` below), slot 17 =
homogeneous freezing (`ice_nucleation_homogeneous`, `int`, any-nonzero=on).
F2 §5's own trailing editorial "Note" (appended below its comment block)
and the commented-out alternate `micexfg` line with value `2` at slot 16
(both noted in an earlier draft of this docstring as corroborating, not
conclusive, evidence) are now corroborated by direct call-site proof
rather than pattern-matching. The cloudlab array sets BOTH slots to `0`
(off), so this resolution has ZERO effect on any numeric value already
produced by `cloudlab()`/`micexfg_array()` -- it only fixes which
dataclass field carries which name/semantics.

Micexfg slot 19 is NOT unused: this OVERRIDES the design spec's "index
19-20 unused/dead" claim, which is wrong for slot 19 (a spec correction is
tracked separately; this module follows the call-site evidence).
`class_Cloud_Micro.F90:1285` passes `CM%micexfg(19)` into
`cal_aptact_var8_kc04dep`'s `iflg_dhf` dummy argument
(`mod_amps_core.F90:5776-5781`, whose own comment lists "iflg: 10, 13, 19"
as the three flags that procedure consumes), which then gates branches at
`mod_amps_core.F90` lines 6335, 6707, 7863, and 8958 -- all reachable in
cloudlab's `act_type=1` path (`cal_aptact_var8_kc04dep`, not the `_vec`
sibling, is the KC04-deposition variant; `act_type=1` selects the
implicit-vapor-prediction branch that calls it). So slot 19 is modeled as
`ice_nucleation_dhf` (the "DHF" flag `iflg_dhf`'s own dummy-argument name
implies -- F2 does not further define the acronym), cloudlab value `0`.
Slot 20 IS genuinely unused (repo-grep-confirmed: no call site anywhere
binds `micexfg(20)` to any dummy argument) -- `unused_20` is unaffected by
this correction.
"""

from __future__ import annotations

import dataclasses
import enum

from icon4py.model.atmosphere.subgrid_scale_physics.amps.core import bin_grid


class ImmersionFreezingMode(enum.IntEnum):
    """Immersion-freezing scheme selector, `micexfg` slot 16
    (`AmpsConfig.ice_nucleation_immersion`). Confirmed by call-site ground
    truth: `class_Cloud_Micro.F90:1131-1135` passes `CM%micexfg(16)`
    positionally as `Ice_Nucleation1`'s `iflg_ifz` dummy argument
    (`mod_amps_core.F90:3078-3081`); `select case(iflg_ifz)` there
    dispatches `case(2)` to `immersion_mode` (standard) and `case(1)` to
    `immersion_mode_KC04` (`mod_amps_core.F90:3161-3171`). The dataclass
    field itself stays typed `int` (not this enum) so any legacy/unknown
    integer value round-trips through `micexfg_array()` unchanged; use
    `ImmersionFreezingMode(cfg.ice_nucleation_immersion)` to interpret it.
    """

    OFF = 0
    KC04 = 1
    STANDARD = 2


@dataclasses.dataclass(frozen=True)
class AmpsConfig:
    """AMPS microphysics configuration: mirrors `/AMPS_param/` (AMPSTASK.F,
    F2 §4/§5) and `PARAM_ATMOS_PHY_MP_AMPS_bin` (F2 §6) in full.

    See the module docstring for the defaults policy (cloudlab literal vs.
    genuine Fortran default) and the two NEEDS_CONTEXT fields (`debug`,
    `nbin_h`). Every other field's default is traceable to an F2 citation
    in its own `#:` comment below.

    Runtime-only derived parameters (`dt_cl = dt/n_step_cl`,
    `dt_vp = dt_cl/n_step_vp` -- both need a run-time `dt` this config does
    not carry) are deliberately NOT modeled here; see the M1 plan Task 4
    brief ("Derived params in __post_init__ (dt_cl/dt_vp given dt at run
    time stay OUT -- runtime, not config)").
    """

    # -- 1. complexity / debug / output settings (F2 §4/§5) -----------------
    #: level of complexity. F2 §5's own comment (lines 642-647) documents
    #: only 1-5 (1=base; 2=+a/c-axis+dendritic length predicted; 3=+rosette
    #: /irregular length; 4=+aerosol mass components to level 2; 5=+aerosol
    #: mass components to level 3); the active cloudlab value is 7, but F2
    #: gives NO comment defining 6 or 7 -- not cited/documented, do not
    #: infer a meaning for them. F2 §5 line 649 (the `level_comp = 7` literal).
    level_comp: int = 7
    #: master debug switch. NEEDS_CONTEXT -- see module docstring.
    debug: bool = False
    #: debugging verbosity level; F2 §5 line 651.
    debug_level: int = 1
    #: sensitivity-test collision level; F2 §5 line 860.
    coll_level: int = 1
    #: 1=tendency only, 2=tendency+variable updated; F2 §5 line 655.
    out_type: int = 2
    #: time period for printing (s); F2 §5 line 660.
    T_print_period: int = 3_600_000
    #: output file format ('binary' or text); F2 §5 line 664.
    output_format: str = "binary"

    # -- 2. hydrometeor group descriptors (F2 §5) ----------------------------
    #: cloud-drop group token; F2 §5 line 667.
    token_c: int = 11
    #: cloud-drop group distribution type; F2 §5 line 669.
    dtype_c: int = 0
    #: cloud-drop hydrodynamic-breakup method; F2 §5 line 671.
    hbreak_c: int = 0
    #: cloud-drop prediction flag (1=fixed conc., 2=predicted); F2 §5 line 675.
    flagp_c: int = 2
    #: rain group token; F2 §5 line 679.
    token_r: int = 1
    #: rain group distribution type; F2 §5 line 681.
    dtype_r: int = 1
    #: rain hydrodynamic-breakup type; F2 §5 line 683.
    hbreak_r: int = 2
    #: rain prediction flag; F2 §5 line 685.
    flagp_r: int = 2
    #: solid-hydrometeor group token; F2 §5 line 689.
    token_s: int = 2
    #: solid-hydrometeor distribution type; F2 §5 line 691.
    dtype_s: int = 1
    #: solid-hydrometeor hydrodynamic-breakup type; F2 §5 line 693.
    hbreak_s: int = 2
    #: solid-hydrometeor prediction flag; F2 §5 line 695.
    flagp_s: int = 2
    #: aerosol group token; F2 §5 line 699.
    token_a: int = 3
    #: aerosol distribution type per category (4 categories); F2 §5 line 701.
    dtype_a: tuple[int, int, int, int] = (3, 4, 3, 3)
    #: aerosol prediction flag; F2 §5 line 714 (active; -3/1 alternates commented out).
    flagp_a: int = 2

    # -- 3. activation / habit ------------------------------------------------
    #: CCN activation model (1=implicit vapor prediction, 2=sat. adjustment); F2 §5 line 856.
    act_type: int = 1
    #: habit growth determination (0=max frequency, 1=random number method); F2 §5 line 719.
    ihabit_gm_random: int = 1

    # -- 4. size-distribution parameters (F2 §5) ------------------------------
    #: cloud-drop size ratio; F2 §5 line 726.
    srat_c: float = 0.0
    #: rain size ratio (for n=20); F2 §5 line 737.
    srat_r: float = 2.540068909
    #: solid-hydrometeor size ratio (20-bin config); F2 §5 line 750.
    srat_s: float = 4.1581061
    #: aerosol size ratio; F2 §5 line 758.
    srat_a: float = 1.0e-6
    #: cloud-drop size addition; F2 §5 line 728.
    sadd_c: float = 0.0
    #: rain size addition; F2 §5 line 739.
    sadd_r: float = 0.0
    #: solid-hydrometeor size addition; F2 §5 line 752.
    sadd_s: float = 0.0
    #: aerosol size addition; F2 §5 line 760.
    sadd_a: float = 0.0
    #: cloud-drop minimum mass (g); F2 §5 line 730.
    minmass_c: float = 0.0
    #: rain minimum mass (g); F2 §5 line 741.
    minmass_r: float = 4.188790e-09
    #: solid-hydrometeor minimum mass (g); F2 §5 line 754 (== MG1 in bin_grid.py bit-for-bit).
    minmass_s: float = 4.18879020478639e-12
    #: aerosol minimum mass (g); F2 §5 line 762.
    minmass_a: float = 1.0e-21
    #: cloud-drop fixed concentration; F2 §5 line 732.
    fcon_c: float = 25.0
    #: rain size-ratio exponent parameter (for n=20); F2 §5 line 743.
    sth_r: float = 1.02

    # -- 5. aerosol chemistry params, one value per of 4 aerosol categories --
    #: molecular weight of soluble part (g/mol, NH4HSO4); F2 §5 line 772.
    M_aps: tuple[float, float, float, float] = (115.11, 115.11, 115.11, 115.11)
    #: molecular weight of insoluble part (g/mol, AgI); F2 §5 line 775.
    M_api: tuple[float, float, float, float] = (234.77, 234.77, 234.77, 234.77)
    #: bulk density of soluble material (g/cm^3, NH4HSO4); F2 §5 line 784.
    den_aps: tuple[float, float, float, float] = (1.79, 1.79, 1.79, 1.79)
    #: bulk density of insoluble material (g/cm^3, AgI); F2 §5 line 787.
    den_api: tuple[float, float, float, float] = (5.683, 5.683, 5.683, 5.683)
    #: number of ions in dissociated solutes (NH4HSO4); F2 §5 line 796.
    nu_aps: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0)
    #: molal coefficient (average); F2 §5 line 798.
    phi_aps: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 0.75)
    #: log of standard deviation (natural log of radius, microns); F2 §5 line 812.
    ap_lnsig: tuple[float, float, float, float] = (
        0.712949807856125,
        0.0,
        0.916290731874155,
        0.916290731874155,
    )
    #: geometrical mean radius (cm); F2 §5 line 819.
    ap_mean: tuple[float, float, float, float] = (0.052e-4, 1.0e-4, 1.3e-4, 1.3e-4)
    #: initial number concentration of aerosol particles; F2 §5 line 832.
    N_ap_ini: tuple[float, float, float, float] = (317.0, 0.0, 0.0, 0.0)
    #: mass fraction of soluble material to total mass; F2 §5 line 801.
    eps_ap: tuple[float, float, float, float] = (1.0, 0.0, 0.05, 1.0)
    #: geometrical mean contact parameter (degrees, normal dist.); F2 §5 line 824.
    ap_mean_cp: tuple[float, float, float, float] = (132.0, 15.5, 132.0, 132.0)
    #: standard deviation of contact parameter (degrees, normal dist.); F2 §5 line 828.
    ap_sig_cp: tuple[float, float, float, float] = (20.0, 1.4, 20.0, 20.0)
    #: chemical name of each aerosol category (NH4HSO4); F2 §5 line 851.
    APSNAME: tuple[str, str, str, str] = ("NH4HSO4", "NH4HSO4", "NH4HSO4", "NH4HSO4")

    # -- 6. lookup-table directories (F2 §5) ----------------------------------
    #: directory of collision-coefficient LUT files; F2 §5 line 837.
    DRCETB: str = (
        "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/collision_data"
    )
    #: directory of aerosol-activation LUT files; F2 §5 line 843.
    DRAPTB: str = "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/apact"
    #: directory of the statistics lookup table; F2 §5 line 846.
    DRSTTB: str = (
        "/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/AMPS_DATA/statpack"
    )

    # -- 7. ice-nucleation scalar parameters (F2 §5) --------------------------
    #: max CCN; F2 §5 line 864.
    CCNMAX: float = 400.0
    #: dust fraction; F2 §5 line 868.
    frac_dust: float = 0.01
    #: ice nucleation half-life parameter; F2 §5 line 872.
    nucleation_halflife: float = 0.0001
    #: minimum dust radius (cm); F2 §5 line 876.
    CRIC_RN_IMM: float = 0.25e-4

    # -- 8. substep counts (F2 §5) --------------------------------------------
    #: # of collision-process substeps per dynamic step; F2 §5 line 889.
    n_step_cl: int = 1
    #: # of vapor-deposition substeps per collision substep; F2 §5 line 892.
    n_step_vp: int = 10

    # -- 9. micexfg(1:20) -- named process on/off flags, F2 §5 lines 894-920 -
    # Comment-block index labels quoted VERBATIM from F2 §5 (lines 899-916)
    # EXCEPT slots 16/17, whose F2 header comment is stale -- see the
    # module docstring for the call-site ground truth that resolves them.
    #: index 1: printing; F2 §5 line 899.
    print_flag: bool = True
    #: index 2: liq-liq (rain-rain) collision-coalescence process; F2 §5 line 900.
    rain_rain_coalescence: bool = True
    #: index 3: ice-ice collision-coalescence (aggregation) process; F2 §5 line 901.
    ice_ice_aggregation: bool = True
    #: index 4: liq-ice collision-coalescence (riming) process; F2 §5 line 902.
    ice_rain_riming: bool = True
    #: index 5: update surface temperature; F2 §5 line 903.
    update_surface_temperature: bool = True
    #: index 6: vapor deposition on liquid; F2 §5 line 904.
    vapor_deposition_liquid: bool = True
    #: index 7: vapor deposition on ice; F2 §5 line 905.
    vapor_deposition_ice: bool = True
    #: index 8: melting shedding process; F2 §5 line 906.
    melting_shedding: bool = True
    #: index 9: hydrodynamic breakup of ice particles; F2 §5 line 907.
    hydrodynamic_breakup_ice: bool = False
    #: index 10: ice nucleation process (master switch); F2 §5 line 908.
    ice_nucleation_master: bool = True
    #: index 11: hydrodynamic breakup of liquid hydrometeors (rain); F2 §5 line 909.
    hydrodynamic_breakup_rain: bool = False
    #: index 12: autoconversion of cloud droplet bin; F2 §5 line 910.
    autoconversion_cloud_droplet: bool = False
    #: index 13: ice nucleation, depositional nucleation; F2 §5 line 911.
    ice_nucleation_deposition: bool = True
    #: index 14: ice nucleation, contact freezing; F2 §5 line 912.
    ice_nucleation_contact: bool = False
    #: index 15: ice nucleation, splinter nucleation (Hallett-Mossop); F2 §5 line 913.
    ice_nucleation_hallett_mossop: bool = False
    #: index 16: ice nucleation, IMMERSION freezing scheme selector.
    #: Resolved from call-site ground truth (F2's own header comment at
    #: this slot is stale -- see module docstring):
    #: `class_Cloud_Micro.F90:1131-1135` passes `CM%micexfg(16)`
    #: positionally as `Ice_Nucleation1`'s `iflg_ifz` dummy
    #: (`mod_amps_core.F90:3078-3081`), which `select case`s to
    #: `immersion_mode`/`immersion_mode_KC04`
    #: (`mod_amps_core.F90:3161-3171`). `int`, not `bool`: genuinely
    #: 3-valued (0=off, 1=KC04, 2=standard) -- see `ImmersionFreezingMode`.
    ice_nucleation_immersion: int = 0
    #: index 17: ice nucleation, HOMOGENEOUS freezing (any nonzero value
    #: triggers `homfreez_mode` via a plain `if(iflg_hfz/=0)` guard,
    #: `mod_amps_core.F90:3176-3181`; `iflg_hfz` is `CM%micexfg(17)`,
    #: `class_Cloud_Micro.F90:1131-1135` / `mod_amps_core.F90:3078-3081`).
    #: `int` for namelist-type fidelity, though only 0/nonzero is
    #: meaningful (unlike slot 16's genuine 3-way selector).
    ice_nucleation_homogeneous: int = 0
    #: index 18: rain collisional breakup (Low-List). No F2 §5 comment text
    #: exists for index 18 (the header list stops at 17); name taken from
    #: the task's ground-truth process mapping (design spec), uncontradicted
    #: by anything in F2. F2 §5 line 920 (active `micexfg` line) value only.
    rain_collisional_breakup: bool = True
    #: index 19: NOT dead, despite the design spec's "index 19-20 unused/
    #: dead" claim (that claim is WRONG for slot 19 -- see module
    #: docstring; a spec correction is tracked separately). Call-site
    #: proof: `class_Cloud_Micro.F90:1285` passes `CM%micexfg(19)` into
    #: `cal_aptact_var8_kc04dep`'s `iflg_dhf` dummy argument
    #: (`mod_amps_core.F90:5776-5781`, whose own comment lists
    #: "iflg: 10, 13, 19"), which gates branches at `mod_amps_core.F90`
    #: lines 6335, 6707, 7863, 8958 -- reachable in cloudlab's `act_type=1`
    #: path. F2 §5 line 920 (active `micexfg` line) cloudlab value: `0`.
    ice_nucleation_dhf: int = 0
    #: index 20: unused/dead -- repo-grep-confirmed (unlike slot 19 above,
    #: no call site binds `micexfg(20)` to anything). F2 §5 line 920
    #: literal value.
    unused_20: int = 1

    # -- 10. PARAM_ATMOS_PHY_MP_AMPS_bin -- genuine Fortran defaults from ----
    # F2 §6a (scale_atmos_phy_mp_amps.F90 module-level initializers).
    #: liquid, ice bin counts; F2 §6a lines 983-993 ("for coarse (default)").
    num_h_bins: tuple[int, int] = (40, 20)
    #: haze-split bin count. NEEDS_CONTEXT -- see module docstring; default
    #: is F2's only concrete value (cloudlab run.conf), not a Fortran default.
    nbin_h: int = 20
    #: sedimentation scheme (1=Euler, 2=PPM); F2 §6a line 959; F2 §6b comment.
    iadvv: int = 1
    #: initial aerosol profile (1=SHEBA, 2=MPACE, 3=general); F2 §6a line 955.
    ini_aerosol_prf: int = 3
    #: whether restart fills aerosols; F2 §6a line 948.
    l_restart: bool = False
    #: invariant aerosols throughout integration; F2 §6a line 949.
    l_fix_aerosols: bool = True
    #: sediment on or off; F2 §6a line 950.
    l_sediment: bool = True
    #: ignore latent heat release from ice processes; F2 §6a line 951.
    l_no_ice_heat: bool = False
    #: fill aerosols in cloud-free region; F2 §6a line 952.
    l_fill_aerosols: bool = False
    #: whether bin shift is performed after advection; F2 §6a line 953.
    l_bin_shift: bool = False
    #: whether center-of-gravity axis limit=1 is applied; F2 §6a line 954.
    l_axis_limit: bool = True
    #: axis definition version (1: a,c,d,ag,cg); F2 §6a line 956.
    l_gaxis_version: int = 1
    #: advection scheme version for non-mass PPVs; F2 §6a line 957.
    l_aadv_version: int = 2
    #: radiation effective-radius formulation version; F2 §6a line 958.
    l_reff_version: int = 2
    #: which of the 4 aerosol types are held fixed if l_fix_aerosols; F2 §6a line 960.
    fix_aerosol_type: tuple[bool, bool, bool, bool] = (True, True, True, True)
    #: debugging on or off (bin-namelist copy, distinct from `debug` above); F2 §6a line 946.
    amps_debug: bool = False
    #: ignore AMPS microphysics or not; F2 §6a line 947.
    amps_ignore: bool = False
    #: master switch for M0 binary reference-data dumps (Fortran-side
    #: instrumentation only; has no effect on the Python port itself);
    #: F2 §6a line 963.
    l_amps_dump: bool = False
    #: dump output directory; F2 §6a line 964.
    amps_dump_dir: str = "."
    #: dump every Nth MP step; F2 §6a line 965.
    amps_dump_step_stride: int = 300
    #: local i-range start of dumped columns; F2 §6a line 966.
    amps_dump_is: int = 0
    #: local i-range end (ie<is disables); F2 §6a line 967.
    amps_dump_ie: int = -1
    #: local j-range start; F2 §6a line 968.
    amps_dump_js: int = 0
    #: local j-range end; F2 §6a line 969.
    amps_dump_je: int = -1

    def __post_init__(self) -> None:
        """Validate cross-field constraints. Frozen dataclass: raises
        ValueError rather than coercing/mutating."""
        nbins_liq, nbins_ice = self.num_h_bins
        if nbins_liq not in bin_grid.LIQUID_NBINS:
            raise ValueError(
                f"num_h_bins[0] (liquid) must be one of {bin_grid.LIQUID_NBINS}; got {nbins_liq}"
            )
        if nbins_ice not in bin_grid.ICE_NBINS:
            raise ValueError(
                f"num_h_bins[1] (ice) must be one of {bin_grid.ICE_NBINS}; got {nbins_ice}"
            )
        if not (1 <= self.nbin_h < nbins_liq):
            raise ValueError(
                f"nbin_h must satisfy 1 <= nbin_h < num_h_bins[0]; got nbin_h={self.nbin_h}, "
                f"num_h_bins[0]={nbins_liq}"
            )
        quad_fields = (
            ("dtype_a", self.dtype_a),
            ("M_aps", self.M_aps),
            ("M_api", self.M_api),
            ("den_aps", self.den_aps),
            ("den_api", self.den_api),
            ("nu_aps", self.nu_aps),
            ("phi_aps", self.phi_aps),
            ("ap_lnsig", self.ap_lnsig),
            ("ap_mean", self.ap_mean),
            ("N_ap_ini", self.N_ap_ini),
            ("eps_ap", self.eps_ap),
            ("ap_mean_cp", self.ap_mean_cp),
            ("ap_sig_cp", self.ap_sig_cp),
            ("APSNAME", self.APSNAME),
            ("fix_aerosol_type", self.fix_aerosol_type),
        )
        for name, value in quad_fields:
            if len(value) != 4:
                raise ValueError(
                    f"{name} must have exactly 4 entries (4 aerosol categories); got {len(value)}"
                )

    def micexfg_array(self) -> tuple[int, ...]:
        """Reconstruct the Fortran 20-slot `micexfg` integer array (1-based
        Fortran index i -> tuple position i-1), F2 §5 lines 894-920."""
        return (
            int(self.print_flag),
            int(self.rain_rain_coalescence),
            int(self.ice_ice_aggregation),
            int(self.ice_rain_riming),
            int(self.update_surface_temperature),
            int(self.vapor_deposition_liquid),
            int(self.vapor_deposition_ice),
            int(self.melting_shedding),
            int(self.hydrodynamic_breakup_ice),
            int(self.ice_nucleation_master),
            int(self.hydrodynamic_breakup_rain),
            int(self.autoconversion_cloud_droplet),
            int(self.ice_nucleation_deposition),
            int(self.ice_nucleation_contact),
            int(self.ice_nucleation_hallett_mossop),
            int(self.ice_nucleation_immersion),
            int(self.ice_nucleation_homogeneous),
            int(self.rain_collisional_breakup),
            int(self.ice_nucleation_dhf),
            int(self.unused_20),
        )

    @classmethod
    def cloudlab(cls) -> AmpsConfig:
        """The exact cloudlab warm spin-up configuration: AMPSTASK.F (F2
        §5, quoted FULL) values for the `/AMPS_param/` fields, plus
        `run.conf`'s `&PARAM_ATMOS_PHY_MP_AMPS_bin` values (checked-out
        `/Users/jcanton/projects/scale_amps/scale-rm/test/case/cloudlab/
        scripts/run.conf` lines 144-159, cited in the M1 plan's Task 4
        brief). Every AMPSTASK.F field is written out explicitly below
        (even where it duplicates the dataclass default -- see module
        docstring) so this classmethod is independently auditable against
        F2 without relying on `AmpsConfig()`'s own defaults.
        """
        return cls(
            # /AMPS_param/, F2 §5 (all explicit; `debug` excepted, see
            # module docstring NEEDS_CONTEXT note -- left at its default).
            level_comp=7,
            debug_level=1,
            coll_level=1,
            out_type=2,
            T_print_period=3_600_000,
            output_format="binary",
            token_c=11,
            dtype_c=0,
            hbreak_c=0,
            flagp_c=2,
            token_r=1,
            dtype_r=1,
            hbreak_r=2,
            flagp_r=2,
            token_s=2,
            dtype_s=1,
            hbreak_s=2,
            flagp_s=2,
            token_a=3,
            dtype_a=(3, 4, 3, 3),
            flagp_a=2,
            ihabit_gm_random=1,
            srat_c=0.0,
            sadd_c=0.0,
            minmass_c=0.0,
            fcon_c=25.0,
            srat_r=2.540068909,
            sadd_r=0.0,
            minmass_r=4.188790e-09,
            sth_r=1.02,
            srat_s=4.1581061,
            sadd_s=0.0,
            minmass_s=4.18879020478639e-12,
            srat_a=1.0e-6,
            sadd_a=0.0,
            minmass_a=1.0e-21,
            M_aps=(115.11, 115.11, 115.11, 115.11),
            M_api=(234.77, 234.77, 234.77, 234.77),
            den_aps=(1.79, 1.79, 1.79, 1.79),
            den_api=(5.683, 5.683, 5.683, 5.683),
            nu_aps=(2.0, 2.0, 2.0, 2.0),
            phi_aps=(0.75, 0.75, 0.75, 0.75),
            eps_ap=(1.0, 0.0, 0.05, 1.0),
            ap_lnsig=(0.712949807856125, 0.0, 0.916290731874155, 0.916290731874155),
            ap_mean=(0.052e-4, 1.0e-4, 1.3e-4, 1.3e-4),
            ap_mean_cp=(132.0, 15.5, 132.0, 132.0),
            ap_sig_cp=(20.0, 1.4, 20.0, 20.0),
            N_ap_ini=(317.0, 0.0, 0.0, 0.0),
            APSNAME=("NH4HSO4", "NH4HSO4", "NH4HSO4", "NH4HSO4"),
            DRCETB="/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/"
            "AMPS_DATA/collision_data",
            DRAPTB="/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/"
            "AMPS_DATA/apact",
            DRSTTB="/cluster/scratch/congchia/scale_amps/scale-rm/test/case/cloudlab/"
            "AMPS_DATA/statpack",
            act_type=1,
            CCNMAX=400.0,
            frac_dust=0.01,
            nucleation_halflife=0.0001,
            CRIC_RN_IMM=0.25e-4,
            n_step_cl=1,
            n_step_vp=10,
            # micexfg = 1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1 (F2 §5 line 920)
            print_flag=True,
            rain_rain_coalescence=True,
            ice_ice_aggregation=True,
            ice_rain_riming=True,
            update_surface_temperature=True,
            vapor_deposition_liquid=True,
            vapor_deposition_ice=True,
            melting_shedding=True,
            hydrodynamic_breakup_ice=False,
            ice_nucleation_master=True,
            hydrodynamic_breakup_rain=False,
            autoconversion_cloud_droplet=False,
            ice_nucleation_deposition=True,
            ice_nucleation_contact=False,
            ice_nucleation_hallett_mossop=False,
            ice_nucleation_immersion=0,
            ice_nucleation_homogeneous=0,
            rain_collisional_breakup=True,
            ice_nucleation_dhf=0,
            unused_20=1,
            # PARAM_ATMOS_PHY_MP_AMPS_bin, run.conf lines 144-159.
            num_h_bins=(40, 20),
            nbin_h=20,
            iadvv=1,
            ini_aerosol_prf=3,
            l_restart=False,
            l_fix_aerosols=False,
            l_sediment=True,
            l_no_ice_heat=False,
            l_fill_aerosols=False,
            l_bin_shift=False,
            l_axis_limit=True,
            l_gaxis_version=1,
            l_aadv_version=2,
            l_reff_version=2,
            fix_aerosol_type=(False, False, False, False),
        )

    @classmethod
    def cloudlab_seeding(cls) -> AmpsConfig:
        """The cloudlab ICE-SEEDING restart configuration
        (`restart_run.conf` lines 145-160): identical to `cloudlab()`
        except `num_h_bins=(40, 40)` (40 ice bins instead of 20,
        `restart_run.conf:146`) and `l_restart=True`
        (`restart_run.conf:150`). All other `&PARAM_ATMOS_PHY_MP_AMPS_bin`
        fields and every `/AMPS_param/` (AMPSTASK.F) field are unchanged
        (`AMPSTASK.F` is read from the run CWD and shared by both runs)."""
        return dataclasses.replace(cls.cloudlab(), num_h_bins=(40, 40), l_restart=True)
