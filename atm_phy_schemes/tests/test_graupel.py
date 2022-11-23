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

        # Read tech. config
        config_parameters = {}
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
            config_parameters[param] = serializer.read_async(
                param, savepoint=savepoints[0]
            )

            # Some parameters were saved as ndarrays and need to be unpacked
            if isinstance(config_parameters[param], np.ndarray):
                config_parameters[param] = config_parameters[param][0]

            # Need to subtract 1 from some parameters to account for FORTRAN indexing
            if param in ["i_startblk", "i_endblk", "ivstart", "ivend", "kstart_moist"]:
                config_parameters[param] -= 1

        # Read Fields
        fields = {}
        shape_1D = config_parameters["nproma"] * config_parameters["nblks_c"]
        shape_2D = (shape_1D, config_parameters["nlev"])

        # In
        for field in ("layer thickness", "pres", "moist air density"):
            fields[field] = (
                serializer.read_async(field, savepoint=savepoints[1])
                .swapaxes(1, 2)
                .reshape(shape_2D)
            )

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
            fields[field] = (
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
            fields[field] = np.broadcast_to(data, shape_2D)

        # Out
        if config_parameters["ldiag_ttend"]:
            fields["tendency temperature"] = (
                serializer.read_async("tendency temperature", savepoint=savepoints[2])
                .swapaxes(1, 2)
                .reshape(shape_2D)
            )
        else:  # DL: TODO Remove Workaround. No optional fields in GT4Py
            fields["tendency temperature"] = np.zeros(shape_2D)

        field_names = (
            "tendency specific water vapor content",
            "tendency specific cloud water content",
            "tendency specific ice content",
            "tendency specific rain content",
            "tendency specific snow content",
        )
        if config_parameters["ldiag_qtend"]:

            for field in field_names:
                fields[field] = serializer.read_async(
                    field, savepoint=savepoints[2]
                ).reshape(shape_2D)

        else:  # DL: TODO Remove Workaround. No optional fields in GT4Py
            for field in field_names:
                fields[field] = np.zeros(shape_2D)

        fields["tendency specific graupel content"] = np.zeros(
            shape_2D
        )  # DL: Not serialized

        # Convert Numpy Arrays to GT4Py storages
        for fieldname, field in fields.items():
            if field.ndim == 1:  # DL: TODO  Workaroudn above
                fields[fieldname] = to_icon4py_field(field, CellDim, KDim)
            elif field.ndim == 2:
                fields[fieldname] = to_icon4py_field(field, CellDim, KDim)
            else:
                assert "Field dimension not supported"

        # Local automatic arrays TODO:remove after scan is wrapped in fieldview
        temporaries = [zero_field((shape_2D), CellDim, KDim) for _ in range(14)]

        # Create index field. TODO: Remove after index fields are avail in fieldview
        is_surface = np.zeros((shape_2D), dtype=bool)
        is_surface[:, -1] = True
        is_surface = to_icon4py_field(is_surface, CellDim, KDim)

        # Compute Coefficients
        gscp_coefficients = gscp_set_coefficients(
            config_parameters["inwp_gscp"],
            zceff_min=config_parameters["tune_zceff_min"],
            v0snow=config_parameters["tune_v0snow"],
            zvz0i=config_parameters["tune_zvz0i"],
            mu_rain=config_parameters["mu_rain"],
            rain_n0_factor=config_parameters["rain_n0_factor"],
            icesedi_exp=config_parameters["tune_icesedi_exp"],
        )

        # Run scheme
        graupel(
            float(config_parameters["tcall_gscp_jg"]),
            fields["layer thickness"],
            fields["temperature"],
            fields["pres"],
            fields["moist air density"],
            fields["specific water vapor content"],
            fields["specific cloud ice content"],
            fields["specific cloud water content"],
            fields["specific rain content"],
            fields["specific snow content"],
            fields["specific graupel content"],
            fields["cloud number concentration"],
            float(config_parameters["qi0"]),
            float(config_parameters["qc0"]),
            fields["precipitation rate of ice"],
            fields["precipitation rate of rain"],
            fields["precipitation rate of snow"],
            fields["precipitation rate of graupel"],
            *temporaries,
            *gscp_coefficients,
            fields["tendency temperature"],
            fields["tendency specific water vapor content"],
            fields["tendency specific cloud water content"],
            fields["tendency specific ice content"],
            fields["tendency specific rain content"],
            fields["tendency specific snow content"],
            fields["tendency specific graupel content"],
            is_surface,
            config_parameters["ldiag_ttend"],
            config_parameters["ldiag_ttend"],
            offset_provider={},
        )

        # Test Fields against refernce

        # Initialize numErrors
        try:
            numErrors
        except NameError:
            numErrors = 0

        for fieldname, field in fields.items():
            if fieldname in ("layer thickness", "moist air density", "pres"):
                numErrors = field_test(
                    field,
                    fieldname,
                    serializer,
                    savepoints[-3],
                    numErrors=numErrors,
                )
            elif fieldname.startswith("tendency"):
                if fieldname == "tendency specific graupel content":
                    continue

                numErrors = field_test(
                    field, fieldname, serializer, savepoints[-1], numErrors=numErrors
                )

            else:
                numErrors = field_test(
                    field, fieldname, serializer, savepoints[-2], numErrors=numErrors
                )

    assert (
        numErrors == 0
    ), f"{bcolors.FAIL}{numErrors} tests failed validation{bcolors.ENDC}"
