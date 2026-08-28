# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Selectors of the NWP 1D turbulence scheme that vary across operational configurations.

One enum per integer selector that the DWD and MeteoSwiss operational setups actually change.
Each carries the full range the Fortran accepts, not only the ported subset: the granule
interface is the contract and must be able to express any namelist ICON can be given, while
'TurbulenceConfig._validate' states which members are implemented (port spec D5/D6).

The declarations and their German-to-English documentation are taken from the selector block of
'mo_turbdiff_config.f90'; 'itype_sher' and 'imode_tkesso' additionally reflect how the values are
branched on in 'turb_diffusion.f90'.
"""

import enum


__all__ = [
    "CharnockParameterType",
    "CloudRepresentationType",
    "ShearProductionType",
    "SsoTkeProductionType",
]


class ShearProductionType(int, enum.Enum):
    """Type of mean shear production for TKE.

    Called 'itype_sher' in mo_turbdiff_nml.f90, declared at mo_turbdiff_config.f90:312.
    """

    #: Only the vertical shear of the horizontal wind.
    VERTICAL_ONLY = 0
    #: As `VERTICAL_ONLY`, plus the horizontal shear correction.
    VERTICAL_AND_HORIZONTAL = 1
    #: As `VERTICAL_AND_HORIZONTAL`, plus the shear from the vertical velocity.
    VERTICAL_AND_VERTICAL_VELOCITY = 2
    #: Legacy DWD setting. The dedicated case was removed from the scheme in 2014
    #: (turb_diffusion.f90:108) and no branch tests for it any more: every use site tests
    #: '>= 1' or '== 2' (turb_diffusion.f90:1318,1350; turb_transfer.f90:1431;
    #: mo_nh_diffusion.f90:805,1172), so this behaves exactly like `VERTICAL_AND_HORIZONTAL`.
    LEGACY_VERTICAL_AND_HORIZONTAL = 3


class CloudRepresentationType(int, enum.Enum):
    """Mode of cloud representation in the turbulence parameterization.

    Called 'icldm_turb' in mo_turbdiff_nml.f90, declared at mo_turbdiff_config.f90:304.
    """

    #: Cloud water ignored completely (pure dry scheme).
    DRY = -1
    #: No clouds considered; all cloud water is evaporated.
    EVAPORATED = 0
    #: Only grid-scale condensation.
    GRID_SCALE = 1
    #: Sub-grid (turbulent) condensation considered as well.
    SUBGRID_SCALE = 2


class SsoTkeProductionType(int, enum.Enum):
    """Mode of calculating the SSO source term for TKE production.

    Called 'imode_tkesso' in mo_turbdiff_nml.f90, declared at mo_turbdiff_config.f90:343 and
    related to 'ltkesso'.
    """

    #: SSO source term off. Not a namelist value a user writes: 'mo_turbdiff_nml.f90:158' forces
    #: it when 'ltkesso = .FALSE.'.
    OFF = 0
    #: Original implementation.
    ORIGINAL = 1
    #: With a Richardson-number dependent reduction factor for Ri > 1.
    RICHARDSON_REDUCED = 2
    #: As `RICHARDSON_REDUCED`, with an additional reduction for mesh sizes below 2 km.
    RICHARDSON_AND_MESH_REDUCED = 3


class CharnockParameterType(int, enum.Enum):
    """Mode of estimating the Charnock parameter.

    Called 'imode_charpar' in mo_turbdiff_nml.f90, declared at mo_turbdiff_config.f90:182.
    """

    #: A constant value.
    CONSTANT = 1
    #: A wind-dependent value with a constant upper bound.
    WIND_DEPENDENT = 2
    #: As `WIND_DEPENDENT`, but reduced above 25 m/s for more realistic tropical-cyclone winds.
    WIND_DEPENDENT_CYCLONE_REDUCED = 3
