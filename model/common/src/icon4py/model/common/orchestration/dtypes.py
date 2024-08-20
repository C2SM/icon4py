# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

try:
    import dace
except ImportError:
    from types import ModuleType
    from typing import Optional

    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


if dace:
    # Define DaCe Symbols
    CellDim_sym = dace.symbol("CellDim_sym")
    EdgeDim_sym = dace.symbol("EdgeDim_sym")
    VertexDim_sym = dace.symbol("VertexDim_sym")
    KDim_sym = dace.symbol("KDim_sym")

    DiffusionDiagnosticState_symbols = {
        f"DiffusionDiagnosticState_{member}_s{stride}_sym": dace.symbol(
            f"DiffusionDiagnosticState_{member}_s{stride}_sym"
        )
        for member in ["hdef_ic", "div_ic", "dwdx", "dwdy"]
        for stride in [0, 1]
    }
    PrognosticState_symbols = {
        f"PrognosticState_{member}_s{stride}_sym": dace.symbol(
            f"PrognosticState_{member}_s{stride}_sym"
        )
        for member in ["rho", "w", "vn", "exner", "theta_v"]
        for stride in [0, 1]
    }

    # Define DaCe Data Types
    DiffusionDiagnosticState_t = dace.data.Structure(
        dict(
            hdef_ic=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    DiffusionDiagnosticState_symbols[
                        f"DiffusionDiagnosticState_{'hdef_ic'}_s{0}_sym"
                    ],
                    DiffusionDiagnosticState_symbols[
                        f"DiffusionDiagnosticState_{'hdef_ic'}_s{1}_sym"
                    ],
                ],
            ),
            div_ic=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    DiffusionDiagnosticState_symbols[
                        f"DiffusionDiagnosticState_{'div_ic'}_s{0}_sym"
                    ],
                    DiffusionDiagnosticState_symbols[
                        f"DiffusionDiagnosticState_{'div_ic'}_s{1}_sym"
                    ],
                ],
            ),
            dwdx=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdx'}_s{0}_sym"],
                    DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdx'}_s{1}_sym"],
                ],
            ),
            dwdy=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdy'}_s{0}_sym"],
                    DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdy'}_s{1}_sym"],
                ],
            ),
        ),
        name="DiffusionDiagnosticState_t",
    )

    PrognosticState_t = dace.data.Structure(
        dict(
            rho=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    PrognosticState_symbols[f"PrognosticState_{'rho'}_s{0}_sym"],
                    PrognosticState_symbols[f"PrognosticState_{'rho'}_s{1}_sym"],
                ],
            ),
            w=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    PrognosticState_symbols[f"PrognosticState_{'w'}_s{0}_sym"],
                    PrognosticState_symbols[f"PrognosticState_{'w'}_s{1}_sym"],
                ],
            ),
            vn=dace.data.Array(
                dtype=dace.float64,
                shape=[EdgeDim_sym, KDim_sym],
                strides=[
                    PrognosticState_symbols[f"PrognosticState_{'vn'}_s{0}_sym"],
                    PrognosticState_symbols[f"PrognosticState_{'vn'}_s{1}_sym"],
                ],
            ),
            exner=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    PrognosticState_symbols[f"PrognosticState_{'exner'}_s{0}_sym"],
                    PrognosticState_symbols[f"PrognosticState_{'exner'}_s{1}_sym"],
                ],
            ),
            theta_v=dace.data.Array(
                dtype=dace.float64,
                shape=[CellDim_sym, KDim_sym],
                strides=[
                    PrognosticState_symbols[f"PrognosticState_{'theta_v'}_s{0}_sym"],
                    PrognosticState_symbols[f"PrognosticState_{'theta_v'}_s{1}_sym"],
                ],
            ),
        ),
        name="PrognosticState_t",
    )
