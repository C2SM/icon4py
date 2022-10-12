#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_intp_rbf_rbf_vec_interpol_vertex.hpp"
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

class mo_intp_rbf_rbf_vec_interpol_vertex {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *veTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      veTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Vertices, dawn::LocationType::Edges}, 0});
    }
  };

private:
  double *p_e_in_;
  double *ptr_coeff_1_;
  double *ptr_coeff_2_;
  double *p_u_out_;
  double *p_v_out_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int p_u_out_kSize_;
  inline static int p_v_out_kSize_;

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

  static int get_p_u_out_KSize() { return p_u_out_kSize_; }

  static int get_p_v_out_KSize() { return p_v_out_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int p_u_out_kSize,
                    const int p_v_out_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    p_u_out_kSize_ = p_u_out_kSize;
    p_v_out_kSize_ = p_v_out_kSize;
  }

  mo_intp_rbf_rbf_vec_interpol_vertex() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_intp_rbf_rbf_vec_interpol_vertex has not been set up! make "
             "sure setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto p_e_in_sid = get_sid(
        p_e_in_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto p_u_out_sid = get_sid(
        p_u_out_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    auto p_v_out_sid = get_sid(
        p_v_out_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<6> ve_ptr{.raw_ptr_fortran = mesh_.veTable};
    auto connectivities =
        gridtools::hymap::keys<generated::V2E_t>::make_values(ve_ptr);
    double *ptr_coeff_1_0 = &ptr_coeff_1_[0 * mesh_.VertexStride];
    double *ptr_coeff_1_1 = &ptr_coeff_1_[1 * mesh_.VertexStride];
    double *ptr_coeff_1_2 = &ptr_coeff_1_[2 * mesh_.VertexStride];
    double *ptr_coeff_1_3 = &ptr_coeff_1_[3 * mesh_.VertexStride];
    double *ptr_coeff_1_4 = &ptr_coeff_1_[4 * mesh_.VertexStride];
    double *ptr_coeff_1_5 = &ptr_coeff_1_[5 * mesh_.VertexStride];
    auto ptr_coeff_1_sid_0 = get_sid(
        ptr_coeff_1_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_1 = get_sid(
        ptr_coeff_1_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_2 = get_sid(
        ptr_coeff_1_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_3 = get_sid(
        ptr_coeff_1_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_4 = get_sid(
        ptr_coeff_1_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_5 = get_sid(
        ptr_coeff_1_5,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_1_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>,
        integral_constant<int, 5>>::make_values(ptr_coeff_1_sid_0,
                                                ptr_coeff_1_sid_1,
                                                ptr_coeff_1_sid_2,
                                                ptr_coeff_1_sid_3,
                                                ptr_coeff_1_sid_4,
                                                ptr_coeff_1_sid_5);
    double *ptr_coeff_2_0 = &ptr_coeff_2_[0 * mesh_.VertexStride];
    double *ptr_coeff_2_1 = &ptr_coeff_2_[1 * mesh_.VertexStride];
    double *ptr_coeff_2_2 = &ptr_coeff_2_[2 * mesh_.VertexStride];
    double *ptr_coeff_2_3 = &ptr_coeff_2_[3 * mesh_.VertexStride];
    double *ptr_coeff_2_4 = &ptr_coeff_2_[4 * mesh_.VertexStride];
    double *ptr_coeff_2_5 = &ptr_coeff_2_[5 * mesh_.VertexStride];
    auto ptr_coeff_2_sid_0 = get_sid(
        ptr_coeff_2_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_1 = get_sid(
        ptr_coeff_2_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_2 = get_sid(
        ptr_coeff_2_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_3 = get_sid(
        ptr_coeff_2_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_4 = get_sid(
        ptr_coeff_2_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_5 = get_sid(
        ptr_coeff_2_5,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto ptr_coeff_2_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>,
        integral_constant<int, 5>>::make_values(ptr_coeff_2_sid_0,
                                                ptr_coeff_2_sid_1,
                                                ptr_coeff_2_sid_2,
                                                ptr_coeff_2_sid_3,
                                                ptr_coeff_2_sid_4,
                                                ptr_coeff_2_sid_5);
    generated::mo_intp_rbf_rbf_vec_interpol_vertex(connectivities)(
        cuda_backend, p_e_in_sid, ptr_coeff_1_sid_comp, ptr_coeff_2_sid_comp,
        p_u_out_sid, p_v_out_sid, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *p_e_in, double *ptr_coeff_1, double *ptr_coeff_2,
                     double *p_u_out, double *p_v_out) {
    p_e_in_ = p_e_in;
    ptr_coeff_1_ = ptr_coeff_1;
    ptr_coeff_2_ = ptr_coeff_2;
    p_u_out_ = p_u_out;
    p_v_out_ = p_v_out;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_intp_rbf_rbf_vec_interpol_vertex(
    double *p_e_in, double *ptr_coeff_1, double *ptr_coeff_2, double *p_u_out,
    double *p_v_out, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_intp_rbf_rbf_vec_interpol_vertex s;
  s.copy_pointers(p_e_in, ptr_coeff_1, ptr_coeff_2, p_u_out, p_v_out);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_intp_rbf_rbf_vec_interpol_vertex(
    const double *p_u_out_dsl, const double *p_u_out, const double *p_v_out_dsl,
    const double *p_v_out, const double p_u_out_rel_tol,
    const double p_u_out_abs_tol, const double p_v_out_rel_tol,
    const double p_v_out_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_intp_rbf_rbf_vec_interpol_vertex::getMesh();
  cudaStream_t stream = dawn_generated::cuda_ico::
      mo_intp_rbf_rbf_vec_interpol_vertex::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_intp_rbf_rbf_vec_interpol_vertex::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int p_u_out_kSize = dawn_generated::cuda_ico::
      mo_intp_rbf_rbf_vec_interpol_vertex::get_p_u_out_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.VertexStride) * p_u_out_kSize, p_u_out_dsl, p_u_out,
      "p_u_out", p_u_out_rel_tol, p_u_out_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_u_out(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_intp_rbf_rbf_vec_interpol_vertex", "p_u_out");
  serialiser_p_u_out.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(
        0, (mesh.NumVertices - 1), p_u_out_kSize, (mesh.VertexStride), p_u_out,
        "mo_intp_rbf_rbf_vec_interpol_vertex", "p_u_out", iteration);
    serialize_dense_verts(0, (mesh.NumVertices - 1), p_u_out_kSize,
                          (mesh.VertexStride), p_u_out_dsl,
                          "mo_intp_rbf_rbf_vec_interpol_vertex", "p_u_out_dsl",
                          iteration);
    std::cout << "[DSL] serializing p_u_out as error is high.\n" << std::flush;
#endif
  }
  int p_v_out_kSize = dawn_generated::cuda_ico::
      mo_intp_rbf_rbf_vec_interpol_vertex::get_p_v_out_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.VertexStride) * p_v_out_kSize, p_v_out_dsl, p_v_out,
      "p_v_out", p_v_out_rel_tol, p_v_out_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_v_out(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_intp_rbf_rbf_vec_interpol_vertex", "p_v_out");
  serialiser_p_v_out.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(
        0, (mesh.NumVertices - 1), p_v_out_kSize, (mesh.VertexStride), p_v_out,
        "mo_intp_rbf_rbf_vec_interpol_vertex", "p_v_out", iteration);
    serialize_dense_verts(0, (mesh.NumVertices - 1), p_v_out_kSize,
                          (mesh.VertexStride), p_v_out_dsl,
                          "mo_intp_rbf_rbf_vec_interpol_vertex", "p_v_out_dsl",
                          iteration);
    std::cout << "[DSL] serializing p_v_out as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_intp_rbf_rbf_vec_interpol_vertex", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_intp_rbf_rbf_vec_interpol_vertex(
    double *p_e_in, double *ptr_coeff_1, double *ptr_coeff_2, double *p_u_out,
    double *p_v_out, double *p_u_out_before, double *p_v_out_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_u_out_rel_tol,
    const double p_u_out_abs_tol, const double p_v_out_rel_tol,
    const double p_v_out_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_intp_rbf_rbf_vec_interpol_vertex ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_intp_rbf_rbf_vec_interpol_vertex(
      p_e_in, ptr_coeff_1, ptr_coeff_2, p_u_out_before, p_v_out_before,
      verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_intp_rbf_rbf_vec_interpol_vertex run time: " << time
            << "s\n"
            << std::flush;
  std::cout
      << "[DSL] Verifying stencil mo_intp_rbf_rbf_vec_interpol_vertex...\n"
      << std::flush;
  verify_mo_intp_rbf_rbf_vec_interpol_vertex(
      p_u_out_before, p_u_out, p_v_out_before, p_v_out, p_u_out_rel_tol,
      p_u_out_abs_tol, p_v_out_rel_tol, p_v_out_abs_tol, iteration);

  iteration++;
}

void setup_mo_intp_rbf_rbf_vec_interpol_vertex(dawn::GlobalGpuTriMesh *mesh,
                                               int k_size, cudaStream_t stream,
                                               const int p_u_out_k_size,
                                               const int p_v_out_k_size) {
  dawn_generated::cuda_ico::mo_intp_rbf_rbf_vec_interpol_vertex::setup(
      mesh, k_size, stream, p_u_out_k_size, p_v_out_k_size);
}

void free_mo_intp_rbf_rbf_vec_interpol_vertex() {
  dawn_generated::cuda_ico::mo_intp_rbf_rbf_vec_interpol_vertex::free();
}
}
