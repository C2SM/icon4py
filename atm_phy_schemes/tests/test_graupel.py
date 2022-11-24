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
"""Test graupel in standalone mode using data serialized from ICON."""
""" GT4Py hotfix:
    
    In _external_src/gt4py-functional/src/functional/iterator/transforms/pass_manager.py
    1. Exchange L49 with:
     inlined = InlineLambdas.apply(inlined, opcount_preserving=True) 
    2. Add "return inlined" below
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
        # ------------------------------
        # The data input is a bit cumbersome due to the inconsistent way it was originally serialized from ICON.

        # Read serialized tech. config
        ser_config_parameters = {}
        for param in (
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
            "i_startblk",
            "i_endblk",
            "ivstart",
            "ivend",
            "kstart_moist",
        ):
            ser_config_parameters[param] = serializer.read_async(
                param, savepoint=savepoints[0]
            )

            # Some parameters were saved as ndarrays and need to be unpacked
            if isinstance(ser_config_parameters[param], np.ndarray):
                ser_config_parameters[param] = ser_config_parameters[param][0]

            # Need to subtract 1 from some parameters to account for FORTRAN indexing
            if param in ["i_startblk", "i_endblk", "ivstart", "ivend", "kstart_moist"]:
                ser_config_parameters[param] -= 1

        shape_1D = ser_config_parameters["nproma"] * ser_config_parameters["nblks_c"]
        shape_2D = (shape_1D, ser_config_parameters["nlev"])

        ser_fields = {
            field: (
                serializer.read_async(field, savepoint=savepoints[1])
                .swapaxes(1, 2)
                .reshape(shape_2D)
            )
            for field in ("layer thickness", "pres", "moist air density")
        }

        # 3D fields
        for field in (
            "specific water vapor content",
            "specific cloud ice content",
            "specific cloud water content",
            "specific rain content",
            "specific snow content",
            "specific graupel content",
            "temperature",
        ):
            ser_fields[field] = (
                serializer.read_async(field, savepoint=savepoints[2])
                .swapaxes(1, 2)
                .reshape(shape_2D)
            )

        # 1D Fields

        for field in (
            "precipitation rate of rain",
            "precipitation rate of snow",
            "precipitation rate of ice",
            "precipitation rate of graupel",
            "cloud number concentration",
        ):

            data = serializer.read_async(field, savepoint=savepoints[2])
            data = np.expand_dims(data.reshape(shape_1D), 1)
            ser_fields[field] = np.broadcast_to(data, shape_2D)

        # Out
        if ser_config_parameters["ldiag_ttend"]:
            ser_fields["tendency temperature"] = (
                serializer.read_async("tendency temperature", savepoint=savepoints[2])
                .swapaxes(1, 2)
                .reshape(shape_2D)
            )
        else:  # DL: TODO Remove Workaround. No optional fields in GT4Py
            ser_fields["tendency temperature"] = np.zeros(shape_2D)

        field_names = (
            "tendency specific water vapor content",
            "tendency specific cloud water content",
            "tendency specific ice content",
            "tendency specific rain content",
            "tendency specific snow content",
        )
        if ser_config_parameters["ldiag_qtend"]:

            for field in field_names:
                ser_fields[field] = serializer.read_async(
                    field, savepoint=savepoints[2]
                ).reshape(shape_2D)

        else:  # DL: TODO Remove Workaround. No optional fields in GT4Py
            for field in field_names:
                ser_fields[field] = np.zeros(shape_2D)

        ser_fields["tendency specific graupel content"] = np.zeros(
            shape_2D
        )  # DL: Not serialized

        # Convert Numpy Arrays to GT4Py storages
        for fieldname, field in ser_fields.items():
            # ser_fields[fieldname] = to_icon4py_field(field, CellDim, KDim)
            ser_fields[fieldname] = to_icon4py_field(
                field[15:16, :], CellDim, KDim
            )  # DL: Debug single column

        # Local automatic arrays TODO:remove after scan is wrapped in fieldview
        # temporaries = [zero_field((shape_2D), CellDim, KDim) for _ in range(14)]
        temporaries = [
            zero_field((1, 90), CellDim, KDim) for _ in range(14)
        ]  # DL: Debug single column

        # Create index field. TODO: Remove after index fields are avail in fieldview
        # is_surface = np.zeros((shape_2D), dtype=bool)
        is_surface = np.zeros((1, 90), dtype=bool)  # DL: Debug single column
        is_surface[:, -1] = True
        is_surface = to_icon4py_field(is_surface, CellDim, KDim)

        # Compute Coefficients
        gscp_coefficients = gscp_set_coefficients(
            ser_config_parameters["inwp_gscp"],
            zceff_min=ser_config_parameters["tune_zceff_min"],
            v0snow=ser_config_parameters["tune_v0snow"],
            zvz0i=ser_config_parameters["tune_zvz0i"],
            mu_rain=ser_config_parameters["mu_rain"],
            rain_n0_factor=ser_config_parameters["rain_n0_factor"],
            icesedi_exp=ser_config_parameters["tune_icesedi_exp"],
        )

        # Run scheme
        graupel(
            float(ser_config_parameters["tcall_gscp_jg"]),
            ser_fields["layer thickness"],
            ser_fields["temperature"],
            ser_fields["pres"],
            ser_fields["moist air density"],
            ser_fields["specific water vapor content"],
            ser_fields["specific cloud ice content"],
            ser_fields["specific cloud water content"],
            ser_fields["specific rain content"],
            ser_fields["specific snow content"],
            ser_fields["specific graupel content"],
            ser_fields["cloud number concentration"],
            float(ser_config_parameters["qi0"]),
            float(ser_config_parameters["qc0"]),
            ser_fields["precipitation rate of ice"],
            ser_fields["precipitation rate of rain"],
            ser_fields["precipitation rate of snow"],
            ser_fields["precipitation rate of graupel"],
            *temporaries,
            *gscp_coefficients,
            ser_fields["tendency temperature"],
            ser_fields["tendency specific water vapor content"],
            ser_fields["tendency specific cloud water content"],
            ser_fields["tendency specific ice content"],
            ser_fields["tendency specific rain content"],
            ser_fields["tendency specific snow content"],
            ser_fields["tendency specific graupel content"],
            is_surface,
            ser_config_parameters["ldiag_ttend"],
            ser_config_parameters["ldiag_ttend"],
            offset_provider={},
        )

        # Test Fields against refernce

        # Initialize numErrors
        try:
            numErrors
        except NameError:
            numErrors = 0

        for fieldname, field in ser_fields.items():
            if fieldname in ("layer thickness", "moist air density", "pres"):
                numErrors = field_test(
                    field,
                    fieldname,
                    serializer,
                    savepoints[-3],
                    numErrors=numErrors,
                    shape_2D=shape_2D,
                    shape_1D=shape_1D,
                )
            elif fieldname.startswith("tendency"):
                if fieldname == "tendency specific graupel content":
                    continue

                numErrors = field_test(
                    field,
                    fieldname,
                    serializer,
                    savepoints[-1],
                    numErrors=numErrors,
                    shape_2D=shape_2D,
                    shape_1D=shape_1D,
                )

            else:
                numErrors = field_test(
                    field,
                    fieldname,
                    serializer,
                    savepoints[-2],
                    numErrors=numErrors,
                    shape_2D=shape_2D,
                    shape_1D=shape_1D,
                )

    assert (
        numErrors == 0
    ), f"{bcolors.FAIL}{numErrors} tests failed validation{bcolors.ENDC}"
