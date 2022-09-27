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


@dataclass
class CppDef:
    stencil_name: str
    fields: list[Field]
    offsets: list[Offset]
    levels_per_thread: int
    block_size: int

    def write(self, outpath: Path):
        definition = self._generate_definition()
        source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
        write_string(source, outpath, f"{self.stencil_name}.cpp")

    def _generate_definition(self):
        output_fields = [field for field in self.fields if field.intent.out]
        # since we can vertical fields as dense fields for the purpose of this function, lets include them here
        dense_fields = [
            field
            for field in self.fields
            if field.is_dense() or (field.has_vertical_dimension and field.rank() == 1)
        ]
        sparse_fields = [field for field in self.fields if field.is_sparse()]
        compound_fields = [field for field in self.fields if field.is_compound()]
        sparse_offsets = [
            offset for offset in self.offsets if not offset.emit_strided_connectivity()
        ]
        strided_offsets = [
            offset for offset in self.offsets if offset.emit_strided_connectivity()
        ]
        parameters = [field for field in self.fields if field.rank() == 0]

        definition = CppDefTemplate(
            includes=IncludeStatements(
                funcname=self.stencil_name,
                levels_per_thread=self.levels_per_thread,
                block_size=self.block_size,
            ),
            utility_functions=UtilityFunctions(),
            stencil_class=StencilClass(
                funcname=self.stencil_name,
                gpu_tri_mesh=GpuTriMesh(fields=self.fields),
                run_fun=StenClassRunFun(
                    stencil_name=self.stencil_name,
                    all_fields=self.fields,
                    dense_fields=dense_fields,
                    sparse_fields=sparse_fields,
                    compound_fields=compound_fields,
                    sparse_connections=sparse_offsets,
                    strided_connections=strided_offsets,
                    all_connections=self.offsets,
                    parameters=parameters,
                ),
            ),  # todo
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


class IncludeStatements(Node):
    funcname: str
    levels_per_thread: int
    block_size: int


class UtilityFunctions(Node):
    ...


class GpuTriMesh(Node):
    fields: Sequence[Field]


class StenClassRunFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    dense_fields: Sequence[Field]
    sparse_fields: Sequence[Field]
    compound_fields: Sequence[Field]
    parameters: Sequence[Field]
    sparse_connections: Sequence[Offset]
    strided_connections: Sequence[Offset]
    all_connections: Sequence[Offset]


class StencilClass(Node):
    funcname: str
    gpu_tri_mesh: GpuTriMesh
    run_fun: StenClassRunFun


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
        {{ run_fun }}
        }


        } // namespace cuda_ico
        } // namespace dawn_generated
        """
    )

    GpuTriMesh = as_jinja(
        """/
        public:
          struct GpuTriMesh {
            int NumVertices;
            int NumEdges;
            int NumCells;
            int VertexStride;
            int EdgeStride;
            int CellStride;
            int *veTable;   # todo

            GpuTriMesh() {}

            GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
              NumVertices = mesh->NumVertices;
              NumCells = mesh->NumCells;
              NumEdges = mesh->NumEdges;
              VertexStride = mesh->VertexStride;
              CellStride = mesh->CellStride;
              EdgeStride = mesh->EdgeStride;
              veTable = mesh->NeighborTables.at(    # todo
                  std::tuple<std::vector<dawn::LocationType>, bool>{
                      {dawn::LocationType::Vertices, dawn::LocationType::Edges}, 0});
            }
          };
        """
    )

    StenClassRunFun = as_jinja(
        """
      void run(const int verticalStart, const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
      if (!is_setup_) {
          printf("{{stencil_name}} has not been set up! make sure setup() is called before run!\\n");
          return;
      }
      using namespace gridtools;
      using namespace fn;
      {% for field in _this_node.dense_fields -%}
        {% if field.is_sparse() == False %}
          auto {{field.name}}_sid = {{ field.render_sid() }};
        {% endif %}
      {% endfor -%}
      {% for parameter in _this_node.parameters -%}
        gridtools::stencil::global_parameter {{parameter.name}} { {{parameter.name}} };
      {% endfor -%}
      fn_backend_t cuda_backend{};
      cuda_backend.stream = stream_;
      {% for connection in _this_node.sparse_connections -%}
        neighbor_table_fortran<{{connection.num_nbh()}}> {{connection.render_lc_shorthand()}}_ptr{.raw_ptr_fortran = mesh_.{{connection.render_lc_shorthand()}}Table};
      {% endfor -%}
      {%- for connection in _this_node.strided_connections -%}
        neighbor_table_4new_sparse<{{connection.num_nbh()}}> {{connection.render_lc_shorthand()}}_ptr{};
      {% endfor -%}
      auto connectivities = gridtools::hymap::keys<
      {%- for connection in _this_node.all_connections -%}
        generated::{{connection.render_uc_shorthand()}}_t{%- if not loop.last -%}, {%- endif -%}
      {%- endfor -%}>::make_values(
      {%- for connection in _this_node.all_connections -%}
        {{connection.render_lc_shorthand()}}_ptr{%- if not loop.last -%}, {%- endif -%}
      {% endfor -%});
      {%- for field in _this_node.sparse_fields -%}
        {%- for i in range(0, field.num_nbh()) -%}
            double *{{field.name}}_{{i}} = &{{field.name}}[{{i}}*mesh_.{{field.stride_type()}}];
        {% endfor -%}
        {%- for i in range(0, field.num_nbh()) -%}
            auto {{field.name}}_sid_{{i}} = get_sid({{field.name}}_{{i}}, gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
        {% endfor -%}
        auto {{field.name}}_sid_comp = sid::composite::keys<
        {%- for i in range(0, field.num_nbh()) -%}
            integral_constant<int,{{i}}>{%- if not loop.last -%}, {%- endif -%}
        {%- endfor -%}>::make_values(
          {%- for i in range(0, field.num_nbh()) -%}
            {{field.name}}_sid_{{i}}{%- if not loop.last -%}, {%- endif -%}
        {%- endfor -%}
        );
      {%- endfor %}
      generated::{{stencil_name}}(connectivities)(cuda_backend,
      {%- for field in _this_node.all_fields -%}
        {{field.name}}_sid,
      {%- endfor -%}
            {%- for field in _this_node.parameters -%}
        {{field.name}}_gp,
      {%- endfor -%}
      horizontalStart, horizontalEnd, verticalStart, verticalEnd);
      #ifndef NDEBUG
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      #endif
      }

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
