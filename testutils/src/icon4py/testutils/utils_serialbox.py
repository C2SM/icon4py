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

import numpy as np


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_comparison(fld, ref, idx, n):

    i = idx[n, 0]
    k = idx[n, 1]

    absDiff = fld[i, k] - ref[i, k]
    relDiff = np.max(np.abs(absDiff) / np.abs(ref[i, k]))
    print(
        f"[{i}, {k}]",
        f"Test = {fld[i, k]:.17e} |",
        f"Ref = {ref[i, k]:.17e} |",
        f"Abs. Diff = {absDiff:.17e} |",
        f"Rel. Diff = {relDiff:.17e}",
    )


def field_test(
    field,
    fieldname,
    serializer,
    savepoint,
    numErrors,
    atol=0.0,
    rtol=1e-15,
    shape_1D=None,
    shape_2D=None,
):
    """Test fields against serialized field."""
    fld = np.asarray(field)  # Convert Gt4Py array to numpy
    ref = serializer.read(fieldname, savepoint)

    if ref.ndim == 3:
        ref = ref.swapaxes(1, 2).reshape(fld.shape)
        # ref = ref.swapaxes(1, 2).reshape(shape_2D)[3231:3232, 29:] # DL: Debug single column
    elif ref.ndim == 2:  # DL: Debug single column
        CellDimSize = fld.shape[0] 
        # CellDimSize = shape_1D # DL: Debug single column

        ref = ref.reshape(CellDimSize)
        ref = np.expand_dims(ref, 1)

        ref = np.broadcast_to(ref, fld.shape) 
        # ref = np.broadcast_to(ref, shape_2D)[3231:3232, 29:] # DL: Debug single column

    try:
        np.testing.assert_allclose(fld, ref, atol=atol, rtol=rtol, verbose=False)
    except AssertionError as msg:
        print(f"[ {bcolors.FAIL}FAILED{bcolors.ENDC} ] {fieldname}")
        print(msg)

        isclose = np.isclose(fld, ref, atol=atol, rtol=rtol)
        idx = np.argwhere(~isclose)

        nSamples = 3
        nSamples = min(nSamples, len(idx))

        for n in range(nSamples):
            print_comparison(fld, ref, idx, n)

        print("  ...\n")

        for n in range(-nSamples, 0):
            print_comparison(fld, ref, idx, n)

        return numErrors + 1

    print(f"[   {bcolors.OKGREEN}OK{bcolors.ENDC}   ] {fieldname}")
    return numErrors
