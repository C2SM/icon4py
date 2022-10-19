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


def field_test(fld, fld_string, serializer, savepoint, numErrors, atol=0.0, rtol=1e-15):
    """Test fields against serialized field."""
    fld = np.asarray(fld)
    ref = serializer.read(fld_string, savepoint).reshape(fld.shape)

    try:
        np.testing.assert_allclose(fld, ref, atol=atol, rtol=rtol, verbose=False)
    except AssertionError as msg:
        print(f"[ {bcolors.FAIL}FAILED{bcolors.ENDC} ] {fld_string}")
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

    else:
        print(f"[   {bcolors.OKGREEN}OK{bcolors.ENDC}   ] {fld_string}")
        return numErrors
