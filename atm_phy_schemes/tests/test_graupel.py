# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test graupel in standalone mode using data serialized from ICON.

TODO:
1. ddt_tend_qg not in ser data
2. Pack variables into lists
"""

import os
from sys import exit, stderr

import numpy as np


try:
    import serialbox as ser
except ImportError:
    os.system(
        "git clone --recursive https://github.com/GridTools/serialbox; CC=`which gcc` CXX=`which g++` pip install serialbox/src/serialbox-python"
    )
    import serialbox as ser

from functools import reduce
from operator import add

from icon4py.atm_phy_schemes.gscp_data import gscp_set_coefficients
from icon4py.atm_phy_schemes.gscp_graupel import graupel
from icon4py.common.dimension import CellDim, KDim
from icon4py.testutils.utils import to_icon4py_field, zero_field
from icon4py.testutils.utils_serialbox import bcolors, field_test


# Configuration of serialized data
SER_DATA = os.path.join(os.path.dirname(__file__), "ser_data")
NUM_MPI_RANKS = 1  # TODO: Set to 6


def test_graupel_serialized_data():
    """Test graupel() against refernce data serialized from a FORTRAN run."""
    if not os.path.exists(SER_DATA):
        os.system(
            "wget -r --no-parent -nH -nc --cut-dirs=3 -q ftp://iacftp.ethz.ch/pub_read/davidle/ser_data_icon_graupel/ser_data/"
        )

    for rank in range(NUM_MPI_RANKS):
        print("=======================")
        print(f"Runing rank {str(rank)}")

        # Open Files
        try:
            serializer = ser.Serializer(
                ser.OpenModeKind.Read, SER_DATA, f"ref_rank_{str(rank + 1)}"
            )
            savepoints = serializer.savepoint_list()
        except ser.SerialboxError as e:
            print(f"serializer: error: {e}", file=stderr)
            exit(1)

        # Read serialized data for init.
        # The data read is a bit cumbersome due to the  inconsistent way it was serialized from ICON

        # Tech, Config

        # These were saved as ndarrays and need to be unpacked
        paramNames = (
            "inwp_gscp",
            "jg",
            "nproma",
            "nlev",
            "nblks_c",
            "idbg",
            "ithermo_water",
            "l_cv",
            "ldiag_ttend",
            "ldiag_qtend",
        )

        (
            inwp_gscp,
            jg,
            nproma,
            nlev,
            nblks_c,
            idbg,
            ithermo_water,
            l_cv,
            ldiag_ttend,
            ldiag_qtend,
        ) = [
            serializer.read_async(name, savepoint=savepoints[0])[0]
            for name in paramNames
        ]

        # Subtract 1 from these parameters to account for FORTRAN indexing
        paramNames = ("i_startblk", "i_endblk", "ivstart", "ivend", "kstart_moist")
        i_startblk, i_endblk, ivstart, ivend, kstart_moist = [
            serializer.read_async(name, savepoint=savepoints[0])[0] - 1
            for name in paramNames
        ]

        # These were not saved as ndarrays. Just read them.
        paramNames = (
            "tcall_gscp_jg",
            "mu_rain",
            "rain_n0_factor",
            "tune_zceff_min",
            "tune_zceff_min",
            "tune_v0snow",
            "tune_zvz0i",
            "tune_icesedi_exp",
            "qi0",
            "qc0",
        )
        (
            tcall_gscp_jg,
            tune_mu_rain,
            tune_rain_n0_factor,
            tune_zceff_min,
            tune_v0snow,
            tune_zvz0i,
            tune_zvz0i,
            tune_icesedi_exp,
            qi0,
            qc0,
        ) = [
            serializer.read_async(name, savepoint=savepoints[0]) for name in paramNames
        ]

        gscp_coefficients = gscp_set_coefficients(
            inwp_gscp,
            zceff_min=tune_zceff_min,
            v0snow=tune_v0snow,
            zvz0i=tune_zvz0i,
            mu_rain=tune_mu_rain,
            rain_n0_factor=tune_rain_n0_factor,
            icesedi_exp=tune_icesedi_exp,
        )

        # In
        paramNames = ("layer thickness", "pres", "moist air density")
        ddqz_z_full, pres, rho = [
            serializer.read_async(name, savepoint=savepoints[1])
            .swapaxes(1, 2)
            .reshape(nproma * nblks_c, nlev)
            for name in paramNames
        ]

        # Inout

        # 3D Fields
        paramNames = (
            "specific water vapor content",
            "specific cloud ice content",
            "specific cloud water content",
            "specific rain content",
            "specific snow content",
            "specific graupel content",
            "temperature",
        )

        qv, qi, qc, qr, qs, qg, temp = [
            serializer.read_async(name, savepoint=savepoints[2])
            .swapaxes(1, 2)
            .reshape(nproma * nblks_c, nlev)
            for name in paramNames
        ]

        # 1D Fields

        paramNames = (
            "precipitation rate of rain",
            "precipitation rate of snow",
            "precipitation rate of ice",
            "precipitation rate of graupel",
            "cloud number concentration",
        )

        rain_gsp_rate, snow_gsp_rate, ice_gsp_rate, graupel_gsp_rate, qnc_s = [
            serializer.read_async(name, savepoint=savepoints[2]).reshape(
                nproma * nblks_c
            )
            for name in paramNames
        ]

        # DL: Workaround Convert to 2D
        rain_gsp_rate, snow_gsp_rate, ice_gsp_rate, graupel_gsp_rate, qnc_s = [
            np.broadcast_to(fld, (nlev, nproma * nblks_c)).transpose()
            for fld in (
                rain_gsp_rate,
                snow_gsp_rate,
                ice_gsp_rate,
                graupel_gsp_rate,
                qnc_s,
            )
        ]

        # Out
        if ldiag_ttend:
            ddt_tend_t = (
                serializer.read_async("tendency temperature", savepoint=savepoints[2])
                .swapaxes(1, 2)
                .reshape(nproma * nblks_c, nlev)
            )
        else:
            ddt_tend_t = zero_field((nproma * nblks_c, nlev), CellDim, KDim)

        if ldiag_qtend:
            paramNames = (
                "tendency specific water vapor content",
                "tendency specific cloud water content",
                "tendency specific ice content",
                "tendency specific rain content",
                "tendency specific snow content",
            )

            ddt_tend_qv, ddt_tend_qc, ddt_tend_qi, ddt_tend_qr, ddt_tend_qs = [
                serializer.read_async(name, savepoint=savepoints[2]).reshape(
                    nproma * nblks_c, nlev
                )
                for name in paramNames
            ]
            ddt_tend_qg = zero_field((nproma * nblks_c, nlev), CellDim, KDim)
        else:
            (
                ddt_tend_qv,
                ddt_tend_qc,
                ddt_tend_qi,
                ddt_tend_qr,
                ddt_tend_qs,
                ddt_tend_qg,
            ) = [zero_field((nproma * nblks_c, nlev), CellDim, KDim) for _ in range(6)]

        # In
        ddqz_z_full, pres, rho, qv, qc, qi, qr, qs, qg, temp = [
            to_icon4py_field(field, CellDim, KDim)
            for field in (ddqz_z_full, pres, rho, qv, qc, qi, qr, qs, qg, temp)
        ]

        rain_gsp_rate, snow_gsp_rate, graupel_gsp_rate, ice_gsp_rate, qnc_s = [
            to_icon4py_field(field, CellDim, KDim)
            for field in (
                rain_gsp_rate,
                snow_gsp_rate,
                graupel_gsp_rate,
                ice_gsp_rate,
                qnc_s,
            )
        ]

        # Inout
        if ldiag_qtend:
            ddt_tend_t = to_icon4py_field(ddt_tend_t, CellDim, KDim)

        if ldiag_qtend:
            (
                ddt_tend_qv,
                ddt_tend_qc,
                ddt_tend_qi,
                ddt_tend_qr,
                ddt_tend_qs,
                ddt_tend_qg,
            ) = [
                to_icon4py_field(field, CellDim, KDim)
                for field in (
                    ddt_tend_qv,
                    ddt_tend_qc,
                    ddt_tend_qi,
                    ddt_tend_qr,
                    ddt_tend_qs,
                    ddt_tend_qg,
                )
            ]

        # Local automatic arrays TODO:remove
        temporaries = [
            zero_field((nproma * nblks_c, nlev), CellDim, KDim) for _ in range(14)
        ]

        # Workaround for missing index Fields in GT4Py.
        is_surface = np.zeros((nproma * nblks_c, nlev), dtype=bool)
        is_surface[:, -1] = True
        is_surface = to_icon4py_field(is_surface, CellDim, KDim)

        graupel(
            float(tcall_gscp_jg),
            ddqz_z_full,
            temp,
            pres,
            rho,
            qv,
            qi,
            qc,
            qr,
            qs,
            qg,
            qnc_s,
            float(qi0),
            float(qc0),
            ice_gsp_rate,
            rain_gsp_rate,
            snow_gsp_rate,
            graupel_gsp_rate,
            *temporaries,
            *gscp_coefficients,
            ddt_tend_t,
            ddt_tend_qv,
            ddt_tend_qc,
            ddt_tend_qi,
            ddt_tend_qr,
            ddt_tend_qs,
            ddt_tend_qg,
            is_surface,
            ldiag_ttend,
            ldiag_qtend,
            offset_provider={},
        )

        # TESTING

        # Initialize numErrors
        try:
            numErrors
        except NameError:
            numErrors = 0

        numErrors = 0
        # In fields: Should not change
        testFields = (
            (ddqz_z_full, "layer thickness"),
            (rho, "moist air density"),
            (pres, "pres"),
        )
        numErrors = reduce(
            add,
            [
                field_test(*field, serializer, savepoints[-3], numErrors=numErrors)
                for field in testFields
            ],
        )

        # Inout

        # Prognostics: 3D
        testFields = (
            (temp, "temperature"),
            (qv, "specific water vapor content"),
            (qi, "specific cloud ice content"),
            (qc, "specific cloud water content"),
            (qr, "specific rain content"),
            (qs, "specific snow content"),
            (qg, "specific graupel content"),
        )
        numErrors = reduce(
            add,
            [
                field_test(*field, serializer, savepoints[-2], numErrors=numErrors)
                for field in testFields
            ],
        )

        testFields = (
            (rain_gsp_rate[:, -1], "precipitation rate of rain"),
            (snow_gsp_rate[:, -1], "precipitation rate of snow"),
            (ice_gsp_rate[:, -1], "precipitation rate of ice"),
            (graupel_gsp_rate[:, -1], "precipitation rate of graupel"),
        )
        numErrors = reduce(
            add,
            [
                field_test(*field, serializer, savepoints[-2], numErrors=numErrors)
                for field in testFields
            ],
        )

        # Output diagnostics

        if ldiag_ttend:
            numErrors = field_test(
                ddt_tend_t,
                "tendency temperature",
                serializer,
                savepoints[-1],
                numErrors=numErrors,
            )

        if ldiag_qtend:
            testFields = (
                (
                    ddt_tend_qv,
                    "tendency specific water vapor content",
                ),
                (
                    ddt_tend_qc,
                    "tendency specific cloud watercontent",
                ),
                (ddt_tend_qi, "tendency specific ice content"),
                (ddt_tend_qr, "tendency specific rain content"),
                (ddt_tend_qs, "tendency specific snow content"),
            )

            numErrors = reduce(
                add,
                [
                    field_test(*field, serializer, savepoints[-1], numErrors=numErrors)
                    for field in testFields
                ],
            )

    assert (
        numErrors == 0
    ), f"{bcolors.FAIL}{numErrors} tests failed validation{bcolors.ENDC}"
