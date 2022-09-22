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

from dataclasses import dataclass
from pathlib import Path

from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import Node, TemplatedGenerator, format_source

from icon4py.bindings.types import Field
from icon4py.bindings.utils import write_string


@dataclass
class CppDef:
    stencil_name: str
    fields: list[Field]
    levels_per_thread: int
    block_size: int

    def write(self, outpath: Path):
        definition = self._generate_definition()
        source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.cpp")

    def _generate_definition(self):
        definition = CppDefTemplate(
            includes=IncludeStatements(
                funcname=self.stencil_name,
                levels_per_thread=self.levels_per_thread,
                block_size=self.block_size,
            ),
            utility_functions=UtilityFunctions(),
            stencil_class=StencilClass(),  # todo
            run_func=RunFunc(),  # todo
            verify_func=VerifyFunc(),  # todo
            run_verify_func=RunAndVerifyFunc(),  # todo
            setup_func=SetupFunc(),  # todo
            free_func=FreeFunc(),  # todo
        )
        return definition


class IncludeStatements(Node):
    funcname: str
    levels_per_thread: int
    block_size: int


class UtilityFunctions(Node):
    ...


class StencilClass(Node):
    pass


class RunFunc(Node):
    pass


class VerifyFunc(Node):
    pass


class RunAndVerifyFunc(Node):
    pass


class SetupFunc(Node):
    pass


class FreeFunc(Node):
    pass


class CppDefTemplate(Node):
    includes: IncludeStatements
    utility_functions: UtilityFunctions
    stencil_class: StencilClass
    run_func: RunFunc
    verify_func: VerifyFunc
    run_verify_func: RunAndVerifyFunc
    setup_func: SetupFunc
    free_func: FreeFunc


class CppDefGenerator(TemplatedGenerator):
    CppDefTemplate = as_jinja(
        """\
        {{ includes }}
        {{ utility_functions }}
        {{ stencil_class }}
        extern "C" {
        {{ run_func }}
        {{ verify_func }}
        {{ run_verify_func }}
        {{ setup_func }}
        {{ free_func }}
        }
        """
    )

    IncludeStatements = as_jinja(
        """\
        #include "driver-includes/cuda_utils.hpp"
        #include "driver-includes/cuda_verify.hpp"
        #include "driver-includes/defs.hpp"
        #include "driver-includes/to_json.hpp"
        #include "driver-includes/to_vtk.h"
        #include "driver-includes/unstructured_domain.hpp"
        #include "driver-includes/unstructured_interface.hpp"
        #include "driver-includes/verification_metrics.hpp"
        #include \"{{ funcname }}.hpp\"
        #include <gridtools/common/array.hpp>
        #include <gridtools/fn/backend/gpu.hpp>
        #include <gridtools/fn/cartesian.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #define GRIDTOOLS_DAWN_NO_INCLUDE
        #include "driver-includes/math.hpp"
        #include <chrono>
        #define BLOCK_SIZE {{ block_size }}
        #define LEVELS_PER_THREAD {{ levels_per_thread }}
        namespace {
        template <int... sizes>
        using block_sizes_t = gridtools::meta::zip<
            gridtools::meta::iseq_to_list<
                std::make_integer_sequence<int, sizeof...(sizes)>,
                gridtools::meta::list, gridtools::integral_constant>,
            gridtools::meta::list<gridtools::integral_constant<int, sizes>...>>;

        using fn_backend_t =
            gridtools::fn::backend::gpu<block_sizes_t<BLOCK_SIZE, LEVELS_PER_THREAD>>;
        } // namespace
        using namespace gridtools::dawn;
        #define nproma 50000
        """
    )

    UtilityFunctions = as_jinja(
        """\
        template <int N> struct neighbor_table_fortran {
          const int *raw_ptr_fortran;
          __device__ friend inline constexpr gridtools::array<int, N>
          neighbor_table_neighbors(neighbor_table_fortran const &table, int index) {
            gridtools::array<int, N> ret{};
            for (int i = 0; i < N; i++) {
              ret[i] = table.raw_ptr_fortran[index + nproma * i];
            }
            return ret;
          }
        };

        template <int N> struct neighbor_table_4new_sparse {
          __device__ friend inline constexpr gridtools::array<int, N>
          neighbor_table_neighbors(neighbor_table_4new_sparse const &, int index) {
            gridtools::array<int, N> ret{};
            for (int i = 0; i < N; i++) {
              ret[i] = index + nproma * i;
            }
            return ret;
          }
        };

        template <class Ptr, class StrideMap>
        auto get_sid(Ptr ptr, StrideMap const &strideMap) {
          using namespace gridtools;
          using namespace fn;
          return sid::synthetic()
              .set<sid::property::origin>(sid::host_device::simple_ptr_holder<Ptr>(ptr))
              .template set<sid::property::strides>(strideMap)
              .template set<sid::property::strides_kind, sid::unknown_kind>();
        }
        """
    )
