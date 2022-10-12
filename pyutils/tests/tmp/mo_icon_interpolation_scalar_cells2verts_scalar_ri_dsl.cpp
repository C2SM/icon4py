#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.hpp"
#include <gridtools/common/array.hpp>
#include <gridtools/fn/backend/gpu.hpp>
#include <gridtools/fn/cartesian.hpp>
#include <gridtools/stencil/global_parameter.hpp>
#define GRIDTOOLS_DAWN_NO_INCLUDE
#include "driver-includes/math.hpp"
#include <chrono>
#define BLOCK_SIZE 128
#define LEVELS_PER_THREAD 1
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

namespace dawn_generated {
namespace cuda_ico {

class mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *vcTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      vcTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Vertices, dawn::LocationType::Cells}, 0});
    }
  };

private:
  double *p_cell_in_;
  double *c_intp_;
  double *p_vert_out_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int p_vert_out_kSize_;

  dim3 grid(int kSize, int elSize, bool kparallel) {
    if (kparallel) {
      int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD;
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, dK, 1);
    } else {
      return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    }
  }

public:
  static const GpuTriMesh &getMesh() { return mesh_; }

  static cudaStream_t getStream() { return stream_; }

  static int getKSize() { return kSize_; }

  static int get_p_vert_out_KSize() { return p_vert_out_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int p_vert_out_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    p_vert_out_kSize_ = p_vert_out_kSize;
  }

  mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl has not "
             "been set up! make sure setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto p_cell_in_sid = get_sid(
        p_cell_in_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_vert_out_sid = get_sid(
        p_vert_out_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<6> vc_ptr{.raw_ptr_fortran = mesh_.vcTable};
    auto connectivities =
        gridtools::hymap::keys<generated::V2C_t>::make_values(vc_ptr);
    double *c_intp_0 = &c_intp_[0 * mesh_.VertexStride];
    double *c_intp_1 = &c_intp_[1 * mesh_.VertexStride];
    double *c_intp_2 = &c_intp_[2 * mesh_.VertexStride];
    double *c_intp_3 = &c_intp_[3 * mesh_.VertexStride];
    double *c_intp_4 = &c_intp_[4 * mesh_.VertexStride];
    double *c_intp_5 = &c_intp_[5 * mesh_.VertexStride];
    auto c_intp_sid_0 = get_sid(
        c_intp_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_1 = get_sid(
        c_intp_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_2 = get_sid(
        c_intp_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_3 = get_sid(
        c_intp_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_4 = get_sid(
        c_intp_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_5 = get_sid(
        c_intp_5,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_intp_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>,
        integral_constant<int, 5>>::make_values(c_intp_sid_0, c_intp_sid_1,
                                                c_intp_sid_2, c_intp_sid_3,
                                                c_intp_sid_4, c_intp_sid_5);
    generated::mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        connectivities)(cuda_backend, p_cell_in_sid, c_intp_sid_comp,
                        p_vert_out_sid, horizontalStart, horizontalEnd,
                        verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *p_cell_in, double *c_intp, double *p_vert_out) {
    p_cell_in_ = p_cell_in;
    c_intp_ = c_intp;
    p_vert_out_ = p_vert_out;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    double *p_cell_in, double *c_intp, double *p_vert_out,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl s;
  s.copy_pointers(p_cell_in, c_intp, p_vert_out);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    const double *p_vert_out_dsl, const double *p_vert_out,
    const double p_vert_out_rel_tol, const double p_vert_out_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh = dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::getMesh();
  cudaStream_t stream = dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::getStream();
  int kSize = dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int p_vert_out_kSize = dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::
          get_p_vert_out_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.VertexStride) * p_vert_out_kSize,
                           p_vert_out_dsl, p_vert_out, "p_vert_out",
                           p_vert_out_rel_tol, p_vert_out_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_vert_out(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl", "p_vert_out");
  serialiser_p_vert_out.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(
        0, (mesh.NumVertices - 1), p_vert_out_kSize, (mesh.VertexStride),
        p_vert_out, "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl",
        "p_vert_out", iteration);
    serialize_dense_verts(
        0, (mesh.NumVertices - 1), p_vert_out_kSize, (mesh.VertexStride),
        p_vert_out_dsl,
        "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl",
        "p_vert_out_dsl", iteration);
    std::cout << "[DSL] serializing p_vert_out as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl",
                       iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    double *p_cell_in, double *c_intp, double *p_vert_out,
    double *p_vert_out_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double p_vert_out_rel_tol, const double p_vert_out_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil "
               "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
      p_cell_in, c_intp, p_vert_out_before, verticalStart, verticalEnd,
      horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl "
               "run time: "
            << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil "
               "mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl...\n"
            << std::flush;
  verify_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
      p_vert_out_before, p_vert_out, p_vert_out_rel_tol, p_vert_out_abs_tol,
      iteration);

  iteration++;
}

void setup_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_vert_out_k_size) {
  dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::setup(
          mesh, k_size, stream, p_vert_out_k_size);
}

void free_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl() {
  dawn_generated::cuda_ico::
      mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl::free();
}
}
