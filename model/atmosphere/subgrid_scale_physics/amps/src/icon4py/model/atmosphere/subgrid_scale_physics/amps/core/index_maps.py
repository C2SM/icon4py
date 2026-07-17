# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""PPV (per-particle-variable) index maps and true compile-time dimension
parameters, transcribed VERBATIM from `par_amps.F90` (F2 §2, quoted FULL,
lines 1-102) and `maxdims.F90` (F2 §3, quoted FULL, lines 1-89), per
docs/superpowers/facts/m1/bins-config-indexmaps.md ("F2" in comments below).

`par_amps.F90` declares NINE separate 1-based Fortran index spaces, grouped
by hydrometeor category and by which array they index:

* the three "qXpv" property-vector spaces (`IcePPV`, `LiquidPPV`,
  `AerosolPPV`) -- indices into `qipv`/`qrpv`/`qapv`, the property-vector
  bin arrays that dominate the port (F2 §2 comments "array specfication for
  qipv/qrpv/qapv");
* three "mass-tendency" spaces (`IceMassIndex`, `LiquidMassIndex`,
  `AerosolMassIndex`) -- ONE shared index space per category for
  `g%MS%mass`, `g%MS%dmassdt`, and `new_M` (F2 §2's own comment groups all
  three arrays under one set of constants, e.g. "array specification for
  g%MS%mass,g%MS%dmassdt,new_M");
* two "ratio_M" spaces (`IceMassRatioIndex`, `LiquidMassRatioIndex`) --
  aerosol has none (F2 §2 has no `ratio_M` block under "array specfication
  for qapv");
* one "new_Q" (non-mass/axis property) space for ice only (`IceAxisIndex`)
  -- liquid and aerosol have no such block in F2 §2.

Every enum member's `.value` is the exact 1-based Fortran index literal;
`.py_idx` (via the `FortranIndex` base) gives the corresponding 0-based
Python/numpy index.

Excluded from this module, per the task's ground-truth instruction ("runtime-
set dims are NOT constants -- they derive from config"):

* F2 §2 lines 383-411 (`nvar_mcp_ice, i1_mcp_ice, i2_mcp_ice`, ... and the
  liquid/aerosol equivalents) -- declared WITHOUT Fortran initializers
  (`integer :: nvar_mcp_ice, ...`, no `=`), i.e. genuinely runtime-computed
  from the active configuration, not `parameter` constants.
* F2 §2 lines 415-416 (`isplit_bin_liq=10`, `isplit_bin_ice=0`) -- these
  ARE given initializers in the Fortran declaration, but F2 §1
  (`binmicrosetup_scale`, lines 281-297) overwrites them at runtime via a
  search loop; the declared values are just initial/placeholder defaults,
  not the true bin-split index for any real configuration. `bin_grid.py`'s
  own module docstring documents this same exclusion for the identical
  reason ("isplit_bin_liq/isplit_bin_ice ... needs BOTH a liquid and an ice
  grid simultaneously ... left out of scope for this task").
* F2 §2 lines 421-424 (`bug_time`, `bug_layer`) -- declared with no
  initializer at all, pure runtime debug state.
"""

from __future__ import annotations

import enum


class FortranIndex(enum.IntEnum):
    """Base class for 1-based Fortran array-index enums.

    Every member's `.value` IS the 1-based Fortran index literal from F2
    §2 -- do not renumber. `.py_idx` gives the 0-based equivalent for
    indexing numpy arrays / gt4py fields.
    """

    @property
    def py_idx(self) -> int:
        """0-based index (`.value - 1`) for numpy/gt4py array indexing."""
        return self.value - 1


# ---------------------------------------------------------------------------
# Ice, F2 §2 lines 342-361.
# ---------------------------------------------------------------------------


class IcePPV(FortranIndex):
    """qipv property-vector indices, F2 §2 lines 342-352 ("array
    specfication for qipv"). 16 members == max_nmoments_ice (F2 §6a)."""

    imt_q = 1  # total mass of ice particle (line 342)
    icon_q = 2  # number concentration (line 348)
    ivcs_q = 3  # circumscribing volume (dry) (line 350)
    iacr_q = 4  # a-axis length (line 350)
    iccr_q = 5  # c-axis length (line 350)
    idcr_q = 6  # dendritic length (line 350)
    iag_q = 7  # center of gravity, a-axis (polycrystals) (line 352)
    icg_q = 8  # center of gravity, c-axis (polycrystals) (line 352)
    inex_q = 9  # extra crystalline structure (line 352)
    imr_q = 10  # rime mass (line 344)
    imc_q = 11  # crystal mass (line 344)
    imw_q = 12  # melt water mass (line 344)
    imat_q = 13  # total aerosol mass (line 346)
    imas_q = 14  # soluble aerosol mass (line 346)
    ima_q = 15  # aggregate mass (line 344)
    imf_q = 16  # frozen water mass (nucleation) (line 344)


class IceMassIndex(FortranIndex):
    """Index space shared by g%MS%mass, g%MS%dmassdt, and new_M (ice), F2
    §2 lines 356-357."""

    imt = 1
    imr = 2
    ima = 3
    imc = 4
    imat = 5
    imas = 6
    imai = 7
    imw = 8
    imf = 9


class IceMassRatioIndex(FortranIndex):
    """Index space for ratio_M (ice), F2 §2 line 359."""

    imr_m = 1
    ima_m = 2
    imc_m = 3
    imat_m = 4
    imas_m = 5
    imai_m = 6
    imw_m = 7
    imf_m = 8


class IceAxisIndex(FortranIndex):
    """Index space for new_Q (ice non-mass/axis properties), F2 §2 line
    361. 7 members == mxnnonmc (F2 §3)."""

    ivcs = 1
    iacr = 2
    iccr = 3
    idcr = 4
    iag = 5
    icg = 6
    inex = 7


# ---------------------------------------------------------------------------
# Liquid, F2 §2 lines 365-372.
# ---------------------------------------------------------------------------


class LiquidPPV(FortranIndex):
    """qrpv property-vector indices, F2 §2 lines 365-367 ("array
    specfication for qrpv"). 4 members == max_nmoments_liq (F2 §6a)."""

    rmt_q = 1  # total mass (line 365)
    rcon_q = 2  # number concentration (line 367)
    rmat_q = 3  # total aerosol mass (line 366)
    rmas_q = 4  # soluble aerosol mass (line 366)


class LiquidMassIndex(FortranIndex):
    """Index space shared by g%MS%mass, g%MS%dmassdt, and new_M (liquid),
    F2 §2 lines 369-370."""

    rmt = 1
    rmat = 2
    rmas = 3
    rmai = 4


class LiquidMassRatioIndex(FortranIndex):
    """Index space for ratio_M (liquid), F2 §2 line 372."""

    rmat_m = 1
    rmas_m = 2
    rmai_m = 3


# ---------------------------------------------------------------------------
# Aerosol, F2 §2 lines 376-381.
# ---------------------------------------------------------------------------


class AerosolPPV(FortranIndex):
    """qapv property-vector indices, F2 §2 lines 376-378 ("array
    specfication for qapv"). 3 members == max_nmoments_aero (F2 §6a)."""

    amt_q = 1  # total mass (line 376)
    acon_q = 2  # number concentration (line 377)
    ams_q = 3  # soluble aerosol mass (line 378)


class AerosolMassIndex(FortranIndex):
    """Index space shared by g%MS%mass, g%MS%dmassdt, and new_M (aerosol),
    F2 §2 lines 380-381. No ratio_M block exists for aerosol in F2 §2."""

    amt = 1
    ams = 2
    ami = 3


# ---------------------------------------------------------------------------
# maxdims.F90 TRUE compile-time `integer,parameter` constants only (F2 §3,
# quoted FULL). Every other name declared in maxdims.F90 (n1mx, n2mx, n3mx,
# n4mx, mxln, mxlh, mxlv, nxpmax, nypmax, nzpmax, nzgmax, nprmx, nbrmx,
# ncrmx, npimx, nbimx, ncimx, npamx, nbamx, ncamx, LMAX, mxnbinr, mxnbini,
# mxnbina, mxnbin, mxnbinb) is a plain runtime-set `integer ::` variable
# (no `parameter` attribute) that "is set in init_AMPS subroutine" (F2 §3's
# own module-docstring comment) from the active bin-grid configuration --
# NOT ported here. See `config.py` (AmpsConfig.num_h_bins/nbin_h) and
# `bin_grid.py` for the runtime-derived equivalents.
# ---------------------------------------------------------------------------

MXNMASSCOMP = 8  # F2 §3 line 511: max # of mass components (ice)
MXNVOL = 2  # F2 §3 line 511: max # of volume components (ice)
MXNTEND = 12  # F2 §3 line 515: max # of microphysical tendencies (ice)
MXNAXIS = 5  # F2 §3 line 515: max # of axis lengths (ice)
MXNNONMC = 7  # F2 §3 line 515: max # of non-mass-component variables (ice)
MXNMASSCOMP_R = 3  # F2 §3 line 518: max # of mass components (liquid)
