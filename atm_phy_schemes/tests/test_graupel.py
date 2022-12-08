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

GT4Py hotfix:

In _external_src/gt4py-functional/src/functional/iterator/transforms/pass_manager.py
1. Exchange L49 with: inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
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
from icon4py.testutils.utils import convert_numpy_field_to_icon4py_field, zero_field
from icon4py.testutils.utils_serialbox import bcolors, field_test


# Configuration of serialized data
SER_DATA = os.path.join(os.path.dirname(__file__), "ser_data")
NUM_MPI_RANKS = 1


def read_serialized_fields(fields: dict, fieldname: str, serializer) -> dict:
    """Return dict of Gt4Py arrays containing the field with name fieldname.
    Only works for 1D and 2D Fields. Assumes shape is nproma, nlev, nblocks"""

    savepoints = serializer.savepoint_list()

    field = serializer.read_async(fieldname, savepoint=savepoints[1])
    nlev = serializer.read_async(
        "nlev", savepoint=savepoints[1]
    )  # Needed for broadcasting 1D -> 2D below

    # Drop Last column since they are all 0.0
    field = np.delete(field, -1, axis=-1)
    
    # Special case for qnc_s, since in ICON it is obtained in the block loop
    if fieldname == "qnc_s" and np.all(field == 0):
        field[...] = float(serializer.read_async("cloud_num", savepoint=savepoints[1]))

    shape = np.shape(field)  # Assume shape is nproma, nlev, nblocks
    nblocks = np.shape(field)[-1]

    nCells = shape[0] * shape[-1]  # shape is ncells
    shape_2D = (nCells, int(nlev))  # shape os ncells, nlev

    if field.ndim == 2:
        field = np.expand_dims(field.reshape(nCells), 1)
        # TODO: For now, scan_operator currently does not accept 1D fields -> boradcast here
        field = np.broadcast_to(field, shape_2D)
    else:

        # Data was written from CPU run that employed blocking.  Thus they need to be unblocked.
        field = field.swapaxes(1, 2).reshape(shape_2D)

    # Convert to GT4Py array and append to dict
    fields[fieldname] = convert_numpy_field_to_icon4py_field(field, CellDim, KDim)

    return fields


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
                ser.OpenModeKind.Read,
                SER_DATA,
                f"reference_graupel_call_rank{str(rank)}",
            )
            savepoints = serializer.savepoint_list()
        except ser.SerialboxError as e:
            print(f"serializer: error: {e}", file=stderr)
            exit(1)

        # Read serialized data
        # --------------------

        # Read serialized tech. config
        in_config_parameters = {}

        for param in (
            "mu_rain",
            "rain_n0_factor",
            "tune_zceff_min",
            "tune_zceff_min",
            "tune_v0snow",
            "tune_zvz0i",
            "tune_icesedi_exp",
            "inwp_gscp",
        ):
            in_config_parameters[param] = serializer.read_async(
                param, savepoint=savepoints[0]
            )

        for param in (
            "tcall_gscp_jg",
            "qi0",
            "qc0",
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
            "i_startidx",
            "i_endidx",
            "kstart_moist",
        ):

            in_config_parameters[param] = serializer.read_async(
                param, savepoint=savepoints[1]
            )

        # Apply some corrections to input data
        for param_name, param in in_config_parameters.items():

            # Some parameters were saved as ndarrays and need to be unpacked
            if isinstance(param, np.ndarray):
                in_config_parameters[param_name] = param[0]

            # Need to subtract 1 from some parameters to account for FORTRAN indexing
            if param_name in [
                "i_startblk",
                "i_endblk",
                "ivstart",
                "ivend",
                "kstart_moist",
            ]:
                in_config_parameters[param_name] -= 1

        # Obtain shapes of 1D and 2D fields as needed for GT4Py implementaion

        # Read fields that wont be changed in routine
        in_fields = {}
        for fieldname in ("ddqz_z_full", "pres", "rho", "qnc_s"):
            in_fields = read_serialized_fields(in_fields, fieldname, serializer)

        # Read fields that will be changed in routine
        inout_fields = {}
        for fieldname in (
            # 1D Fields
            "rain_gsp_rate",
            "snow_gsp_rate",
            "ice_gsp_rate",
            "graupel_gsp_rate",
            # 2D Fields
            "temp",
            "qv",
            "qc",
            "qi",
            "qr",
            "qs",
            "qg",
            "temp",
        ):
            inout_fields = read_serialized_fields(inout_fields, fieldname, serializer)

        # Init diagnostics to 0.0.
        #  Needed in routine (optional fields not yet supported in GT4Py), but output not tested.
        nCells = np.shape(inout_fields["temp"])[0]
        shape_2D = np.shape(inout_fields["temp"])

        tendency_fields = {}
        for fieldname in (
            "ddt_tend_t",
            "ddt_tend_qv",
            "ddt_tend_qc",
            "ddt_tend_qi",
            "ddt_tend_qr",
            "ddt_tend_qs",
            "ddt_tend_qg",
            "qrsflux",
        ):
            tendency_fields[fieldname] = zero_field(shape_2D, CellDim, KDim)

        # Local automatic arrays TODO:remove after scan is wrapped in fieldview
        temporaries = [zero_field(shape_2D, CellDim, KDim) for _ in range(14)]

        # Create index field. TODO: Remove after index fields are avail in fieldview
        is_surface = np.zeros(shape_2D, dtype=bool)
        is_surface[:, -1] = True
        is_surface = convert_numpy_field_to_icon4py_field(is_surface, CellDim, KDim)

        # # Initialize runtime-constant coefficients
        gscp_coefficients = gscp_set_coefficients(
            in_config_parameters["inwp_gscp"],
            zceff_min=in_config_parameters["tune_zceff_min"],
            v0snow=in_config_parameters["tune_v0snow"],
            zvz0i=in_config_parameters["tune_zvz0i"],
            mu_rain=in_config_parameters["mu_rain"],
            rain_n0_factor=in_config_parameters["rain_n0_factor"],
            icesedi_exp=in_config_parameters["tune_icesedi_exp"],
        )

        # Run scheme
        # ----------

        graupel(
            float(in_config_parameters["tcall_gscp_jg"]),
            in_fields["ddqz_z_full"],
            inout_fields["temp"],
            in_fields["pres"],
            in_fields["rho"],
            inout_fields["qv"],
            inout_fields["qi"],
            inout_fields["qv"],
            inout_fields["qr"],
            inout_fields["qs"],
            inout_fields["qg"],
            in_fields["qnc_s"],
            float(in_config_parameters["qi0"]),
            float(in_config_parameters["qc0"]),
            inout_fields["ice_gsp_rate"],
            inout_fields["rain_gsp_rate"],
            inout_fields["snow_gsp_rate"],
            inout_fields["graupel_gsp_rate"],
            tendency_fields["qrsflux"],
            *temporaries,
            *gscp_coefficients,
            tendency_fields["ddt_tend_t"],
            tendency_fields["ddt_tend_qv"],
            tendency_fields["ddt_tend_qc"],
            tendency_fields["ddt_tend_qi"],
            tendency_fields["ddt_tend_qr"],
            tendency_fields["ddt_tend_qs"],
            tendency_fields["ddt_tend_qg"],
            is_surface,
            in_config_parameters["ldiag_ttend"],
            in_config_parameters["ldiag_ttend"],
            np.int32(nCells),
            np.int32(in_config_parameters["nlev"]),
            np.int32(in_config_parameters["kstart_moist"]),
            offset_provider={},
        )

        # Test fields against reference
        # -----------------------------

        # Initialize numErrors
        try:
            numErrors
        except NameError:
            numErrors = 0

        for fieldname, field in inout_fields.items():
            numErrors = field_test(
                field,
                fieldname,
                serializer,
                savepoints[-1],
                numErrors=numErrors,
                shape_2D=shape_2D,
                shape_1D=nCells,
            )

    assert (
        numErrors == 0
    ), f"{bcolors.FAIL}{numErrors} tests failed validation{bcolors.ENDC}"
