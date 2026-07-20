# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import csv
import json
import sys
from pathlib import Path

import numpy


if len(sys.argv) < 2:
    print("Usage: python print_gt4py_timers.py <input_file> [--csv]")
    sys.exit(1)

input_file = Path(sys.argv[1])
with input_file.open() as f:
    data = json.load(f)

if len(sys.argv) > 2 and sys.argv[2] == "--csv":
    with Path("output.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "Median", "Std"])
        for k, v in data.items():
            if v.get("metrics").get("compute"):
                arr = numpy.array(v.get("metrics").get("compute")[1:])
                if len(arr) > 0:
                    median = numpy.median(arr)
                    if not numpy.isnan(median):
                        writer.writerow([k.split("<")[0], median, arr.std()])
else:
    for k, v in data.items():
        if v.get("metrics").get("compute"):
            arr = numpy.array(v.get("metrics").get("compute")[1:])
            if len(arr) > 0:
                median = numpy.median(arr)
                if not numpy.isnan(median):
                    print(f"{k.split('<')[0]}: Median = {median}, Std = {arr.std()}")
