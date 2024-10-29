# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

def get_latex_macros(filename):
    latex_macros = {}
    with open(filename, 'r') as f:
        for line in f:
            macros = re.findall(r'\\(DeclareRobustCommand|newcommand|renewcommand){\\(.*?)}(\[(\d)\])?{(.+)}', line)
            for macro in macros:
                if len(macro[2]) == 0:
                    latex_macros[macro[1]] = "{"+macro[4]+"}"
                else:
                    latex_macros[macro[1]] = ["{"+macro[4]+"}", int(macro[3])]
    return latex_macros