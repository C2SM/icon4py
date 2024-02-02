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
from typing import Any, Sequence

from gt4py import eve
from gt4py.eve.codegen import JinjaTemplate as as_jinja
from gt4py.eve.codegen import Node, TemplatedGenerator, format_source

from icon4pytools.icon4pygen.bindings.codegen.header import (
    CppFreeFunc,
    CppRunAndVerifyFuncDeclaration,
    CppRunFuncDeclaration,
    CppSetupFuncDeclaration,
    CppVerifyFuncDeclaration,
    run_func_declaration,
    run_verify_func_declaration,
)
from icon4pytools.icon4pygen.bindings.codegen.render.offset import GpuTriMeshOffsetRenderer
from icon4pytools.icon4pygen.bindings.entities import Field, Offset
from icon4pytools.icon4pygen.bindings.utils import write_string


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
        #include <gridtools/fn/backend/gpu.hpp>

        #include "cuda_utils.hpp"
        #include "cuda_verify.hpp"
        #include "to_json.hpp"
        #include "to_vtk.h"
        #include "unstructured_interface.hpp"
        #include "verification_metrics.hpp"
        #include \"{{ funcname }}.hpp\"
        #include <gridtools/common/array.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #define GRIDTOOLS_DAWN_NO_INCLUDE
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
        """
    )

    UtilityFunctions = as_jinja(
        """\
        template <int N> struct neighbor_table_fortran {
          const int *raw_ptr_fortran;
          const int nproma;
          __device__ friend inline constexpr gridtools::array<int, N>
          neighbor_table_neighbors(neighbor_table_fortran const &table, int index) {
            gridtools::array<int, N> ret{};
            for (int i = 0; i < N; i++) {
              ret[i] = table.raw_ptr_fortran[index + table.nproma * i];
            }
            return ret;
          }
        };

        template <int N> struct neighbor_table_strided {
          const int nproma;
          __device__ friend inline constexpr gridtools::array<int, N>
          neighbor_table_neighbors(neighbor_table_strided const &table, int index) {
            gridtools::array<int, N> ret{};
            for (int i = 0; i < N; i++) {
              ret[i] = index + table.nproma * i;
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

    PublicUtilities = as_jinja(
        """
      static const GpuTriMesh & getMesh() {
        return mesh_;
      }

      static cudaStream_t getStream() {
        return stream_;
      }

      static int getKSize() {
        return kSize_;
      }

      static json *getJsonRecord() {
        return jsonRecord_;
      }

      static MeshInfoVtk *getMeshInfoVtk() {
        return mesh_info_vtk_;
      }

      static verify *getVerify() {
        return verify_;
      }

      {% for field in _this_node.fields %}
      static int get_{{field.name}}_KSize() {
      return {{field.name}}_kSize_;
      }
      {% endfor %}

      static void free() {
      }
      """
    )

    CopyPointers = as_jinja(
        """
      void copy_pointers(
        {%- for field in _this_node.fields %}
          {{field.renderer.render_ctype('c++')}}{{field.renderer.render_pointer()}} {{field.name}} {%- if not loop.last -%}, {%- endif -%}
        {% endfor %}
      ) {
        {% for field in _this_node.fields -%}
          {{field.name}}_ = {{field.name}};
        {% endfor %}
      }
      """
    )

    StencilClass = as_jinja(
        """
        namespace cuda_ico {

        class {{ funcname }} {
        {{ gpu_tri_mesh }}
        {{ private_members }}
        public:
        {{ public_utilities }}
        {{ setup_func }}
        {{ funcname }}() {}
        {{ run_fun }}
        {{ copy_pointers }}
        };
        } // namespace cuda_ico
        """
    )

    GpuTriMesh = as_jinja(
        """\
        public:
          struct GpuTriMesh {
            int Nproma;
            int NumVertices;
            int NumEdges;
            int NumCells;
            int VertexStride;
            int EdgeStride;
            int CellStride;
            {%- if has_offsets -%}
            {%- for var in _this_node.table_vars -%}
            int * {{ var }}Table;
            {%- endfor -%}
            {%- endif %}

            GpuTriMesh() {}

            GpuTriMesh(const GlobalGpuTriMesh *mesh) {
              Nproma = mesh->Nproma;
              NumVertices = mesh->NumVertices;
              NumCells = mesh->NumCells;
              NumEdges = mesh->NumEdges;
              VertexStride = mesh->VertexStride;
              CellStride = mesh->CellStride;
              EdgeStride = mesh->EdgeStride;
              {%- if has_offsets -%}
              {%- for table in _this_node.neighbor_tables -%}
              {{ table }}
              {%- endfor -%}
              {%- endif %}
            }
          };
        """
    )

    StencilClassSetupFunc = as_jinja(
        """\
        static void setup(
        const GlobalGpuTriMesh *mesh, int kSize, cudaStream_t stream, json *jsonRecord, MeshInfoVtk *mesh_info_vtk, verify *verify,
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_{{ suffix }}
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor %}) {
        mesh_ = GpuTriMesh(mesh);
        {{ suffix }}_ = {{ suffix }};
        is_setup_ = true;
        stream_ = stream;
        jsonRecord_ = jsonRecord;
        mesh_info_vtk_ = mesh_info_vtk;
        verify_ = verify;

        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ suffix }}_ = {{ field.name }}_{{ suffix }};
        {%- endfor -%}
        }
        """
    )

    PrivateMembers = as_jinja(
        """\
        private:
        {%- for field in _this_node.fields -%}
        {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_;
        {%- endfor -%}
        inline static int kSize_;
        inline static GpuTriMesh mesh_;
        inline static bool is_setup_;
        inline static cudaStream_t stream_;
        inline static json* jsonRecord_;
        inline static MeshInfoVtk* mesh_info_vtk_;
        inline static verify* verify_;
        {%- for field in _this_node.out_fields -%}
        inline static int {{ field.name }}_kSize_;
        {%- endfor %}

        dim3 grid(int kSize, int elSize, bool kparallel) {
            if (kparallel) {
              int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
              return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, dK, 1);
            } else {
              return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
            }
          }
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
      {% for field in _this_node.all_fields -%}
        {% if field.is_sparse() == False and field.rank() > 0 %}
          auto {{field.name}}_sid = get_sid({{field.name}}_, {{ field.renderer.render_sid() }});
        {% endif %}
      {% endfor -%}
      {%- for field in _this_node.sparse_fields -%}
        {%- for i in range(0, field.get_num_neighbors()) -%}
            {{field.renderer.render_ctype('c++')}} *{{field.name}}_{{i}} = &{{field.name}}_[{{i}}*mesh_.{{field.renderer.render_stride_type()}}{%- if field.has_vertical_dimension -%}*kSize_{%- endif -%}];
        {% endfor -%}
        {%- for i in range(0, field.get_num_neighbors()) -%}
            auto {{field.name}}_sid_{{i}} = get_sid({{field.name}}_{{i}}, {{ field.renderer.render_sid() }});
        {% endfor -%}
        auto {{field.name}}_sid_comp = sid::composite::keys<
        {%- for i in range(0, field.get_num_neighbors()) -%}
            integral_constant<int,{{i}}>{%- if not loop.last -%}, {%- endif -%}
        {%- endfor -%}>::make_values(
          {%- for i in range(0, field.get_num_neighbors()) -%}
            {{field.name}}_sid_{{i}}{%- if not loop.last -%}, {%- endif -%}
        {%- endfor -%}
        );
      {%- endfor %}
      {% for field in _this_node.all_fields -%}
        {%- if field.rank() == 0 -%}
          gridtools::stencil::global_parameter {{field.name}}_gp { {{field.name}}_ };
        {%- endif -%}
      {% endfor -%}
      fn_backend_t cuda_backend{};
      cuda_backend.stream = stream_;
      {% for connection in _this_node.sparse_connections -%}
        neighbor_table_fortran<{{connection.get_num_neighbors()}}> {{connection.renderer.render_lowercase_shorthand()}}_ptr{.raw_ptr_fortran = mesh_.{{connection.renderer.render_lowercase_shorthand()}}Table, .nproma = mesh_.Nproma};
      {% endfor -%}
      {%- for connection in _this_node.strided_connections -%}
        neighbor_table_strided<{{connection.get_num_neighbors()}}> {{connection.renderer.render_lowercase_shorthand()}}_ptr{.nproma = mesh_.Nproma};
      {% endfor -%}
      auto connectivities = gridtools::hymap::keys<
      {%- for connection in _this_node.all_connections -%}
        generated::{{connection.renderer.render_uppercase_shorthand()}}_t{%- if not loop.last -%}, {%- endif -%}
      {%- endfor -%}>::make_values(
      {%- for connection in _this_node.all_connections -%}
        {{connection.renderer.render_lowercase_shorthand()}}_ptr{%- if not loop.last -%}, {%- endif -%}
      {% endfor -%});

      int numCells = mesh_.NumCells;
      int numEdges = mesh_.NumEdges;
      int numVertices = mesh_.NumVertices;

      generated::{{stencil_name}}(connectivities)(cuda_backend,
      {%- for field in _this_node.all_fields -%}
        {%- if field.is_sparse() -%}
          {{field.name}}_sid_comp,
        {%- elif field.rank() == 0 -%}
          {{field.name}}_gp,
        {%- else -%}
          {{field.name}}_sid,
        {%- endif -%}
      {%- endfor -%}
      horizontalStart, horizontalEnd, verticalStart, verticalEnd, numCells, numEdges, numVertices);
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
        cuda_ico::{{ funcname }} s;
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
        bool verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }}_{{ suffix }},
        const {{ field.renderer.render_ctype('c++') }} {{ field.renderer.render_pointer() }} {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.tol_fields -%}
        const double {{ field.name }}_rel_tol,
        const double {{ field.name }}_abs_tol,
        {%- endfor -%}
        const int iteration)
        """
    )

    VerifyFunc = as_jinja(
        """\
        {{ verify_func_declaration }} {
        using namespace std::chrono;
        const auto &mesh = cuda_ico::{{ funcname }}::getMesh();
        cudaStream_t stream = cuda_ico::{{ funcname }}::getStream();
        int kSize = cuda_ico::{{ funcname }}::getKSize();
        MeshInfoVtk* mesh_info_vtk = cuda_ico::{{ funcname }}::getMeshInfoVtk();
        verify* verify = cuda_ico::{{ funcname }}::getVerify();
        high_resolution_clock::time_point t_start = high_resolution_clock::now();
        struct VerificationMetrics stencilMetrics;
        {{ metrics_serialisation }}
        }
        """
    )

    MetricsSerialisation = as_jinja(
        """\
        {%- for field in _this_node.out_fields %}
        int {{ field.name }}_kSize = cuda_ico::{{ funcname }}::
        get_{{ field.name }}_KSize();
        {% if field.is_integral() %}
        stencilMetrics = ::verify_field(
            stream, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", iteration);
        {% elif field.is_bool() %}
        stencilMetrics = ::verify_bool_field(
            stream, verify, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", iteration);
        {% else %}
        stencilMetrics = ::verify_field(
            stream, (mesh.{{ field.renderer.render_stride_type() }}) * {{ field.name }}_kSize, {{ field.name }}_dsl, {{ field.name }},
            \"{{ field.name }}\", {{ field.name }}_rel_tol, {{ field.name }}_abs_tol, iteration);
        {% endif %}
        #ifdef __SERIALIZE_METRICS
        MetricsSerialiser serialiser_{{ field.name }}(
            cuda_ico::{{ funcname }}::getJsonRecord(), stencilMetrics,
            \"{{ funcname }}\", \"{{ field.name }}\");
        serialiser_{{ field.name }}.writeJson(iteration);
        #endif
        if (!stencilMetrics.isValid) {
        #ifdef __SERIALIZE_ON_ERROR
        {{ field.renderer.render_serialise_func() }}(mesh_info_vtk, 0, (mesh.Num{{ field.location.render_location_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.renderer.render_stride_type() }}), {{ field.name }},
                              \"{{ funcname }}\", \"{{ field.name }}\", iteration);
        {{ field.renderer.render_serialise_func() }}(mesh_info_vtk, 0, (mesh.Num{{ field.location.render_location_type() }} - 1), {{ field.name }}_kSize,
                              (mesh.{{ field.renderer.render_stride_type() }}), {{ field.name }}_dsl,
                              \"{{ funcname }}\", \"{{ field.name }}_dsl\",
                              iteration);
        std::cout << "[DSL] serializing {{ field.name }} as error is high.\\n" << std::flush;
        #endif
        }
        {%- if loop.last -%}
        ;
        {%- endif -%}
        {%- endfor %}
        #ifdef __SERIALIZE_ON_ERROR
        serialize_flush_iter(\"{{ funcname }}\", iteration);
        #endif
        high_resolution_clock::time_point t_end = high_resolution_clock::now();
        duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
        std::cout << "[DSL] Verification took " << timing.count() << " seconds.\\n" << std::flush;
        return stencilMetrics.isValid;
        """
    )

    CppRunAndVerifyFuncDeclaration = run_verify_func_declaration

    RunFuncCall = as_jinja(
        """\
        run_{{funcname}}(
        {%- for field in _this_node.fields -%}
        {%- if field.intent.out -%}
        {{ field.name }}_before,
        {%- else -%}
        {{ field.name }},
        {%- endif -%}
        {%- endfor -%}
        verticalStart, verticalEnd, horizontalStart, horizontalEnd) ;
        """
    )

    VerifyFuncCall = as_jinja(
        """\
        verify_{{funcname}}(
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ suffix }},
        {{ field.name }},
        {%- endfor -%}
        {%- for field in _this_node.tol_fields -%}
        {{ field.name }}_rel_tol,
        {{ field.name }}_abs_tol,
        {%- endfor -%}
        iteration) ;
        """
    )

    RunAndVerifyFunc = as_jinja(
        """\
        {{ run_verify_func_declaration }} {
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
        GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream, json *json_record, MeshInfoVtk *mesh_info_vtk, verify *verify,
        {%- for field in _this_node.out_fields -%}
        const int {{ field.name }}_{{ suffix }}
        {%- if not loop.last -%}
        ,
        {%- endif -%}
        {%- endfor -%})
        """
    )

    SetupFunc = as_jinja(
        """\
        {{ func_declaration }} {
        cuda_ico::{{ funcname }}::setup(mesh, k_size, stream, json_record, mesh_info_vtk, verify,
        {%- for field in _this_node.out_fields -%}
        {{ field.name }}_{{ suffix }}
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
            cuda_ico::{{ funcname }}::free();
        }
        """
    )


class IncludeStatements(Node):
    funcname: str
    levels_per_thread: int
    block_size: int


class UtilityFunctions(Node):
    ...


class GpuTriMesh(Node):
    table_vars: list[str]
    neighbor_tables: list[str]
    has_offsets: bool


class StenClassRunFun(Node):
    stencil_name: str
    all_fields: Sequence[Field]
    dense_fields: Sequence[Field]
    sparse_fields: Sequence[Field]
    compound_fields: Sequence[Field]
    sparse_connections: Sequence[Offset]
    strided_connections: Sequence[Offset]
    all_connections: Sequence[Offset]


class PublicUtilities(Node):
    fields: Sequence[Field]


class CopyPointers(Node):
    fields: Sequence[Field]


class PrivateMembers(Node):
    fields: Sequence[Field]
    out_fields: Sequence[Field]


class StencilClassSetupFunc(CppSetupFuncDeclaration):
    ...


class StencilClass(Node):
    funcname: str
    gpu_tri_mesh: GpuTriMesh
    run_fun: StenClassRunFun
    public_utilities: PublicUtilities
    copy_pointers: CopyPointers
    private_members: PrivateMembers
    setup_func: StencilClassSetupFunc


class Params(Node):
    fields: Sequence[Field]


class RunFunc(Node):
    funcname: str
    params: Params
    run_func_declaration: CppRunFuncDeclaration


class MetricsSerialisation(Node):
    funcname: str
    out_fields: Sequence[Field]


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
    stencil_name: str
    fields: Sequence[Field]
    offsets: Sequence[Offset]
    levels_per_thread: int
    block_size: int

    includes: IncludeStatements = eve.datamodels.field(init=False)
    utility_functions: UtilityFunctions = eve.datamodels.field(init=False)
    stencil_class: StencilClass = eve.datamodels.field(init=False)
    run_func: RunFunc = eve.datamodels.field(init=False)
    verify_func: VerifyFunc = eve.datamodels.field(init=False)
    run_verify_func: RunAndVerifyFunc = eve.datamodels.field(init=False)
    setup_func: SetupFunc = eve.datamodels.field(init=False)
    free_func: FreeFunc = eve.datamodels.field(init=False)

    def _get_field_data(self) -> tuple:
        output_fields = [field for field in self.fields if field.intent.out]
        tolerance_fields = [field for field in output_fields if not field.is_integral()]
        # since we can vertical fields as dense fields for the purpose of this function, lets include them here
        dense_fields = [
            field
            for field in self.fields
            if field.is_dense() or (field.has_vertical_dimension and field.rank() == 1)
        ]
        sparse_fields = [field for field in self.fields if field.is_sparse()]
        compound_fields = [field for field in self.fields if field.is_compound()]
        sparse_offsets = [offset for offset in self.offsets if not offset.is_compound_location()]
        strided_offsets = [offset for offset in self.offsets if offset.is_compound_location()]
        all_fields = self.fields

        offsets = dict(sparse=sparse_offsets, strided=strided_offsets)
        fields = dict(
            output=output_fields,
            dense=dense_fields,
            sparse=sparse_fields,
            compound=compound_fields,
            all_fields=all_fields,
            tolerance=tolerance_fields,
        )
        return fields, offsets

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        fields, offsets = self._get_field_data()
        offset_renderer = GpuTriMeshOffsetRenderer(self.offsets)

        self.includes = IncludeStatements(
            funcname=self.stencil_name,
            levels_per_thread=self.levels_per_thread,
            block_size=self.block_size,
        )

        self.utility_functions = UtilityFunctions()

        self.stencil_class = StencilClass(
            funcname=self.stencil_name,
            gpu_tri_mesh=GpuTriMesh(
                table_vars=offset_renderer.make_table_vars(),
                neighbor_tables=offset_renderer.make_neighbor_tables(),
                has_offsets=offset_renderer.has_offsets,
            ),
            run_fun=StenClassRunFun(
                stencil_name=self.stencil_name,
                all_fields=fields["all_fields"],
                dense_fields=fields["dense"],
                sparse_fields=fields["sparse"],
                compound_fields=fields["compound"],
                sparse_connections=offsets["sparse"],
                strided_connections=offsets["strided"],
                all_connections=self.offsets,
            ),
            public_utilities=PublicUtilities(fields=fields["output"]),
            copy_pointers=CopyPointers(fields=self.fields),
            private_members=PrivateMembers(fields=self.fields, out_fields=fields["output"]),
            setup_func=StencilClassSetupFunc(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                suffix="kSize",
            ),
        )

        self.run_func = RunFunc(
            funcname=self.stencil_name,
            params=Params(fields=self.fields),
            run_func_declaration=CppRunFuncDeclaration(
                funcname=self.stencil_name, fields=self.fields
            ),
        )

        self.verify_func = VerifyFunc(
            funcname=self.stencil_name,
            verify_func_declaration=CppVerifyFuncDeclaration(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                suffix="dsl",
            ),
            metrics_serialisation=MetricsSerialisation(
                funcname=self.stencil_name, out_fields=fields["output"]
            ),
        )

        self.run_verify_func = RunAndVerifyFunc(
            funcname=self.stencil_name,
            run_verify_func_declaration=CppRunAndVerifyFuncDeclaration(
                funcname=self.stencil_name,
                fields=self.fields,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                suffix="before",
            ),
            run_func_call=RunFuncCall(funcname=self.stencil_name, fields=self.fields),
            verify_func_call=VerifyFuncCall(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                suffix="before",
            ),
        )

        self.setup_func = SetupFunc(
            funcname=self.stencil_name,
            out_fields=fields["output"],
            tol_fields=fields["tolerance"],
            func_declaration=CppSetupFuncDeclaration(
                funcname=self.stencil_name,
                out_fields=fields["output"],
                tol_fields=fields["tolerance"],
                suffix="k_size",
            ),
            suffix="k_size",
        )

        self.free_func = FreeFunc(funcname=self.stencil_name)


def generate_cpp_definition(
    stencil_name: str,
    fields: Sequence[Field],
    offsets: Sequence[Offset],
    levels_per_thread: int,
    block_size: int,
    outpath: Path,
) -> None:
    definition = CppDefTemplate(
        stencil_name=stencil_name,
        fields=fields,
        offsets=offsets,
        levels_per_thread=levels_per_thread,
        block_size=block_size,
    )
    source = format_source("cpp", CppDefGenerator.apply(definition), style="LLVM")
    write_string(source, outpath, f"{stencil_name}.cpp")
