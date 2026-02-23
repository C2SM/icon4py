# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import ast
import pathlib
import re


def read_namelist(path: pathlib.Path) -> dict:
    """ICON NAMELIST_ICON_output_atm reader.
    Returns a dictionary of dictionaries, where the keys are the namelist names
    and the values are dictionaries of key-value pairs from the namelist.

    Use as:
    namelists = read_namelist("/path/to/NAMELIST_ICON_output_atm")
    print(namelists["NWP_TUNING_NML"]["TUNE_ZCEFF_MIN"])
    """
    with path.open() as f:
        txt = f.read()
    blocks = re.findall(r"&(\w+)(.*?)\/", txt, re.S)
    out = {}
    for name, body in blocks:
        d = {}
        body_content = re.sub(r"!.*", "", body)  # remove comments
        for line in body_content.split(","):
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.replace("T", "True").replace("F", "False")
                try:
                    v = ast.literal_eval(v.strip())
                except Exception:
                    v = v.strip().strip("'")
                d[k.strip()] = v
        out[name] = d
    return out
