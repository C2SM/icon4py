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
from icon4py.model.common.settings import backend
from icon4py.model.common.orchestration.decorator import dace_orchestration

if dace_orchestration():
     import dace

     # Define DaCe Symbols
     CellDim_sym = dace.symbol('CellDim_sym')
     EdgeDim_sym = dace.symbol('EdgeDim_sym')
     VertexDim_sym = dace.symbol('VertexDim_sym')
     KDim_sym = dace.symbol('KDim_sym')

     DiffusionDiagnosticState_symbols = {f"DiffusionDiagnosticState_{member}_s{stride}_sym": dace.symbol(f"DiffusionDiagnosticState_{member}_s{stride}_sym") for member in ["hdef_ic", "div_ic", "dwdx", "dwdy"] for stride in [0, 1]}
     PrognosticState_symbols = {f"PrognosticState_{member}_s{stride}_sym": dace.symbol(f"PrognosticState_{member}_s{stride}_sym") for member in ["rho", "w", "vn", "exner", "theta_v"] for stride in [0, 1]}

     import inspect
     from icon4py.model.common import dimension
     from gt4py.next.ffront.fbuiltins import FieldOffset
     from gt4py.next.common import DimensionKind
     from gt4py.next.program_processors.runners.dace_iterator.utility import connectivity_identifier, field_size_symbol_name, field_stride_symbol_name
     OffsetProviders_t_members = [name for name, value in inspect.getmembers(dimension) if isinstance(value, FieldOffset) and value.source.kind == DimensionKind.HORIZONTAL]
     connectivity_table_size_symbols = {
          field_size_symbol_name(connectivity_identifier(k), axis): dace.symbol(
               field_size_symbol_name(connectivity_identifier(k), axis)
          )
          for k in OffsetProviders_t_members
          for axis in [0, 1]
     }
     connectivity_table_stride_symbols = {
          field_stride_symbol_name(connectivity_identifier(k), axis): dace.symbol(
               field_stride_symbol_name(connectivity_identifier(k), axis)
          )
          for k in OffsetProviders_t_members
          for axis in [0, 1]
     }
     OffsetProviders_symbols = {**connectivity_table_size_symbols, **connectivity_table_stride_symbols}

     # Define DaCe Data Types
     DiffusionDiagnosticState_t = dace.data.Structure(dict(hdef_ic=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'hdef_ic'}_s{0}_sym"], DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'hdef_ic'}_s{1}_sym"]]),
                                                           div_ic=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'div_ic'}_s{0}_sym"], DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'div_ic'}_s{1}_sym"]]),
                                                           dwdx=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdx'}_s{0}_sym"], DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdx'}_s{1}_sym"]]),
                                                           dwdy=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdy'}_s{0}_sym"], DiffusionDiagnosticState_symbols[f"DiffusionDiagnosticState_{'dwdy'}_s{1}_sym"]])),
                                                      name = 'DiffusionDiagnosticState_t')

     PrognosticState_t = dace.data.Structure(dict(rho=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_symbols[f"PrognosticState_{'rho'}_s{0}_sym"], PrognosticState_symbols[f"PrognosticState_{'rho'}_s{1}_sym"]]),
                                                      w=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_symbols[f"PrognosticState_{'w'}_s{0}_sym"], PrognosticState_symbols[f"PrognosticState_{'w'}_s{1}_sym"]]),
                                                      vn=dace.data.Array(dtype=dace.float64, shape=[EdgeDim_sym, KDim_sym], strides=[PrognosticState_symbols[f"PrognosticState_{'vn'}_s{0}_sym"], PrognosticState_symbols[f"PrognosticState_{'vn'}_s{1}_sym"]]),
                                                      exner=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_symbols[f"PrognosticState_{'exner'}_s{0}_sym"], PrognosticState_symbols[f"PrognosticState_{'exner'}_s{1}_sym"]]),
                                                      theta_v=dace.data.Array(dtype=dace.float64, shape=[CellDim_sym, KDim_sym], strides=[PrognosticState_symbols[f"PrognosticState_{'theta_v'}_s{0}_sym"], PrognosticState_symbols[f"PrognosticState_{'theta_v'}_s{1}_sym"]])),
                                                  name = 'PrognosticState_t')

     OffsetProviders_int64_t_dict = {member: dace.data.Array(dtype=dace.int64,
                                                             shape=[OffsetProviders_symbols[field_size_symbol_name(connectivity_identifier(member), 0)], OffsetProviders_symbols[field_size_symbol_name(connectivity_identifier(member), 1)]],
                                                             strides=[OffsetProviders_symbols[field_stride_symbol_name(connectivity_identifier(member), 0)], OffsetProviders_symbols[field_stride_symbol_name(connectivity_identifier(member), 1)]])
                                    for member in OffsetProviders_t_members}
     OffsetProviders_int32_t_dict = {member: dace.data.Array(dtype=dace.int32,
                                                             shape=[OffsetProviders_symbols[field_size_symbol_name(connectivity_identifier(member), 0)], OffsetProviders_symbols[field_size_symbol_name(connectivity_identifier(member), 1)]],
                                                             strides=[OffsetProviders_symbols[field_stride_symbol_name(connectivity_identifier(member), 0)], OffsetProviders_symbols[field_stride_symbol_name(connectivity_identifier(member), 1)]])
                                     for member in OffsetProviders_t_members}
     OffsetProviders_int64_t = dace.data.Structure(OffsetProviders_int64_t_dict, name='OffsetProviders_int64_t')
     OffsetProviders_int32_t = dace.data.Structure(OffsetProviders_int32_t_dict, name='OffsetProviders_int32_t')

     float64_t = dace.float64
     Field_f64_KDim_t = dace.data.Array(dtype=dace.float64, shape=[KDim_sym])
     self_t = dace.compiletime
     Connectivities_t = dace.compiletime
else:
     from icon4py.model.atmosphere.diffusion.diffusion_states import DiffusionDiagnosticState
     from icon4py.model.common.states.prognostic_state import PrognosticState
     from gt4py.next.ffront.fbuiltins import Field
     from gt4py.next import NeighborTableOffsetProvider
     from gt4py.next.common import Connectivity
     from icon4py.model.common.dimension import KDim
     
     DiffusionDiagnosticState_t = DiffusionDiagnosticState
     PrognosticState_t = PrognosticState
     OffsetProviders_int64_t = dict[str, NeighborTableOffsetProvider]
     OffsetProviders_int32_t = dict[str, NeighborTableOffsetProvider]
     float64_t = float
     Field_f64_KDim_t = Field[[KDim], float]
     self_t = None
     Connectivities_t = dict[str, Connectivity]
