# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum


class MuphysScheme(enum.Enum):
    """Selects between the two graupel microphysics formulations.

    KOKKOS_MUPHYS follows the muphys C++/Kokkos reference implementation (the
    original source of this port, validated by the netCDF-based muphys tests).
    AES_GRAUPEL follows the newer MPIM rain-microphysics revisions carried by
    icon-nwp (mo_aes_graupel.f90), validated against the aes-graupel serialbox
    savepoints.
    """

    KOKKOS_MUPHYS = "kokkos_muphys"
    AES_GRAUPEL = "aes_graupel"


@dataclasses.dataclass(frozen=True)
class MuphysConfig:
    """Configuration for the muphys microphysics component."""

    qnc: float = 50.0e6  # cloud droplet number concentration [m^-3]
    scheme: MuphysScheme = MuphysScheme.AES_GRAUPEL
