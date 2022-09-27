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

from pathlib import Path
from typing import Sequence

from eve.codegen import JinjaTemplate as as_jinja
from eve.codegen import Node, TemplatedGenerator, format_source

from icon4py.bindings.codegen.header import (
    CppFreeFunc,
    CppRunAndVerifyFuncDeclaration,
    CppRunFuncDeclaration,
    CppSetupFuncDeclaration,
    CppVerifyFuncDeclaration,
    run_func_declaration,
    run_verify_func_declaration,
)
from icon4py.bindings.types import Field, Offset
from icon4py.bindings.utils import write_string


class CppDef:
    def __init__(
        self,
        stencil_name: str,
        fields: list[Field],
        offsets: list[Offset],
        levels_per_thread: int,
        block_size: int,
    ):
        self.stencil_name = stencil_name
        self.fields = fields
        self.levels_per_thread = levels_per_thread
        self.block_size = block_size
        self.offset_handler = GpuTriMeshOffsetHandler(offsets)

    def write(self, outpath: Path):
        definition = self._generate_definition()
        source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.cpp")

    def _generate_definition(self):
        output_fields = [field for field in self.fields if field.intent.out]

        definition = CppDefTemplate(
            includes=IncludeStatements(
                funcname=self.stencil_name,
                levels_per_thread=self.levels_per_thread,
                block_size=self.block_size,
            ),
            utility_functions=UtilityFunctions(),
            stencil_class=StencilClass(
                funcname=self.stencil_name,
                gpu_tri_mesh=GpuTriMesh(
                    table_vars=self.offset_handler.make_table_vars(),
                    neighbor_tables=self.offset_handler.make_neighbor_tables(),
                ),
            ),
            run_func=RunFunc(
                funcname=self.stencil_name,
                params=Params(fields=self.fields),
                run_func_declaration=CppRunFuncDeclaration(
                    funcname=self.stencil_name, fields=self.fields
                ),
            ),
            verify_func=VerifyFunc(
                funcname=self.stencil_name,
                verify_func_declaration=CppVerifyFuncDeclaration(
                    funcname=self.stencil_name, out_fields=output_fields
                ),
                metrics_serialisation=MetricsSerialisation(
                    funcname=self.stencil_name, out_fields=output_fields
                ),
            ),
            run_verify_func=RunAndVerifyFunc(
                funcname=self.stencil_name,
                run_verify_func_declaration=CppRunAndVerifyFuncDeclaration(
                    funcname=self.stencil_name,
                    fields=self.fields,
                    out_fields=output_fields,
                ),
                run_func_call=RunFuncCall(
                    funcname=self.stencil_name, fields=self.fields
                ),
                verify_func_call=VerifyFuncCall(
                    funcname=self.stencil_name, out_fields=output_fields
                ),
            ),
            setup_func=SetupFunc(
                funcname=self.stencil_name,
                out_fields=output_fields,
                func_declaration=CppSetupFuncDeclaration(
                    funcname=self.stencil_name, out_fields=output_fields
                ),
            ),
            free_func=FreeFunc(funcname=self.stencil_name),
        )
        return definition


class GpuTriMeshOffsetHandler:
    def __init__(self, offsets: list[Offset]):
        self.offsets = offsets

    def make_table_vars(self):
        if len(self.offsets) == 0:
            return ""
        return [self._make_table_var(offset) for offset in self.offsets]

    def make_neighbor_tables(self):
        if len(self.offsets) == 0:
            return ""

        return [
            (
                f"{self._make_table_var(offset)}Table = mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, bool>{{"
                f"{{{', '.join(self._make_location_type(offset))}}}, {1 if offset.includes_center else 0}}});"
            )
            for offset in self.offsets
        ]

    @staticmethod
    def _make_table_var(offset: Offset) -> str:
        return f"{offset.target[1].__str__().replace('2', '').lower()}{'o' if offset.includes_center else ''}"

    @staticmethod
    def _make_location_type(offset: Offset) -> list[str]:
        return [
            f"dawn::LocationType::{loc.location_type()}"
            for loc in offset.target[1].chain
        ]


class IncludeStatements(Node):
    funcname: str
    levels_per_thread: int
    block_size: int


class UtilityFunctions(Node):
    ...


class GpuTriMesh(Node):
    table_vars: list[str]
    neighbor_tables: list[str]


class StencilClass(Node):
    funcname: str
    gpu_tri_mesh: GpuTriMesh


class Params(Node):
    fields: Sequence[Field]


class RunFunc(Node):
    funcname: str
    params: Params
    run_func_declaration: CppRunFuncDeclaration


class MetricsSerialisation(CppVerifyFuncDeclaration):
    ...


class VerifyFunc(Node):
    funcname: str
    verify_func_declaration: CppVerifyFuncDeclaration
    metrics_serialisation: MetricsSerialisation


class RunFuncCall(CppRunFuncDeclaration):
    ...


class VerifyFuncCall(CppVerifyFuncDeclaration):
    ...


class SetupFunc(CppVerifyFuncDeclaration):
    func_declaration: CppSetupFuncDeclaration


class RunAndVerifyFunc(Node):
    funcname: str
    run_verify_func_declaration: CppRunAndVerifyFuncDeclaration
    run_func_call: RunFuncCall
    verify_func_call: VerifyFuncCall


class FreeFunc(CppFreeFunc):
    ...


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

    StencilClass = as_jinja(
        """\
        namespace dawn_generated {
        namespace cuda_ico {

        class {{ funcname }} {
        {{ gpu_tri_mesh }}

        }


        } // namespace cuda_ico
        } // namespace dawn_generated
        """
    )

    GpuTriMesh = as_jinja(
        """\
        public:
          struct GpuTriMesh {
            int NumVertices;
            int NumEdges;
            int NumCells;
            int VertexStride;
            int EdgeStride;
            int CellStride;
            {%- for var in _this_node.table_vars -%}
            int * {{ var }}Table;
            {%- endfor %}

            GpuTriMesh() {}

            GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
              NumVertices = mesh->NumVertices;
              NumCells = mesh->NumCells;
              NumEdges = mesh->NumEdges;
              VertexStride = mesh->VertexStride;
              CellStride = mesh->CellStride;
              EdgeStride = mesh->EdgeStride;
              {%- for table in _this_node.neighbor_tables -%}
              {{ table }}
              {%- endfor -%}
            }
          };
        """
    )

    CppRunFuncDeclaration = run_func_declaration

    RunFunc = as_jinja(
        """\
        {{run_func_declaration }} {
        dawn_generated::cuda_ico::{{ funcname }} s;
        s.copy_pointers({{ params }});
        s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
        return;
        }
        """
    )

    Params = as_jinja(
        """\
        {%- for field in _this_node.fields -%}
        {{ field.name }}
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%}
        """
    )

    CppVerifyFuncDeclaration = as_jinja(
        """\
        void verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }}_dsl,
        const {{ field.ctype('c++') }} {{ field.render_pointer() }} {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        const {{ field.ctype('c++')}} {{ field.name }}_rel_tol,
        const {{ field.ctype('c++')}} {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration) ;
        """
    )

    VerifyFunc = as_jinja(
        """\
        {{ verify_func_declaration }} {
        using namespace std::chrono;
        const auto &mesh = dawn_generated::cuda_ico::{{ funcname }}::getMesh();
        cudaStream_t stream = dawn_generated::cuda_ico::{{ funcname }}::getStream();
        int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_06::getKSize();
        high_resolution_clock::time_point t_start = high_resolution_clock::now();
        struct VerificationMetrics stencilMetrics;
        {{ metrics_serialisation }}
        }
        """
    )

    MetricsSerialisation = as_jinja(
        """\
        {%- for field in _this_node.out_fields %}
        int {{ field.name }}_kSize = dawn_generated::cuda_ico::{{ funcname }}::
        get_{{ field.name }}_KSize();
        stencilMetrics = ::dawn::verify_field(
            stream, (mesh.{{ field.stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", {{ field.name }}_rel_tol, {{ field.name }}_abs_tol, iteration);
        #ifdef __SERIALIZE_METRICS
        MetricsSerialiser serialiser_{{ field.name }}(
            stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
            \"{{ funcname }}\", \"{{ field.name }}\");
        serialiser_{{ field.name }}.writeJson(iteration);
        #endif
        if (!stencilMetrics.isValid) {
        #ifdef __SERIALIZE_ON_ERROR
        {{ field.serialise_func() }}(0, (mesh.{{ field.mesh_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.stride_type() }}), {{ field.name }},
                              \"{{ funcname }}\", \"{{ field.name }}\", iteration);
        {{ field.serialise_func() }}(0, (mesh.{{ field.mesh_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.stride_type() }}), {{ field.name }}_dsl,
                              \"{{ funcname }}\", \"{{ field.name }}_dsl\",
                              iteration);
        std::cout << "[DSL] serializing {{ field.name }} as error is high.\n" << std::flush;
        #endif
        }
        {%- endfor %}
        #ifdef __SERIALIZE_ON_ERROR
        serialize_flush_iter(\"{{ funcname }}\", iteration);
        #endif
        high_resolution_clock::time_point t_end = high_resolution_clock::now();
        duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
        std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n" << std::flush;
        return stencilMetrics.isValid;
        };
        """
    )

    CppRunAndVerifyFuncDeclaration = run_verify_func_declaration

    RunFuncCall = as_jinja(
        """\
        run_{{funcname}}(
        {%- for field in _this_node.fields -%}
        {{ field.name }},
        {%- endfor -%}
        const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) ;
        """
    )

    VerifyFuncCall = as_jinja(
        """\
        verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_before,
        {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_rel_tol,
        {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration) ;
        """
    )

    RunAndVerifyFunc = as_jinja(
        """\
        {{ run_verify_func_declaration }}
        static int iteration = 0;
        std::cout << \"[DSL] Running stencil {{ funcname }} (\"
                  << iteration << \") ...\\n\"
                  << std::flush;
        {{ run_func_call }}
        std::cout << \"[DSL] {{ funcname }} run time: \" << time
            << \"s\\n\"
            << std::flush;
        std::cout << \"[DSL] Verifying stencil {{ funcname }}...\\n\"
                  << std::flush;
        {{ verify_func_call }}
        iteration++;
        }
        """
    )

    CppSetupFuncDeclaration = as_jinja(
        """\
        void setup_{{funcname}}(
        dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_k_size
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%})
        """
    )

    SetupFunc = as_jinja(
        """\
        {{ func_declaration }} {
        dawn_generated::cuda_ico::{{ funcname }}::setup(mesh, k_size, stream,
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_size
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%}
        );
        }
        """
    )

    FreeFunc = as_jinja(
        """\
        void free_{{funcname}}() {
            dawn_generated::cuda_ico::{{ funcname }}::free();
        }
        """
    )
