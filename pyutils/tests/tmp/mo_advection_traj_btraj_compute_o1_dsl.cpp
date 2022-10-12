#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_advection_traj_btraj_compute_o1_dsl.hpp"
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

class mo_advection_traj_btraj_compute_o1_dsl {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      ecTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells}, 0});
    }
  };

private:
  double *p_vn_;
  double *p_vt_;
  int *cell_idx_;
  int *cell_blk_;
  double *pos_on_tplane_e_1_;
  double *pos_on_tplane_e_2_;
  double *primal_normal_cell_1_;
  double *dual_normal_cell_1_;
  double *primal_normal_cell_2_;
  double *dual_normal_cell_2_;
  int *p_cell_idx_;
  int *p_cell_blk_;
  double *p_distv_bary_1_;
  double *p_distv_bary_2_;
  double p_dthalf_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int p_cell_idx_kSize_;
  inline static int p_cell_blk_kSize_;
  inline static int p_distv_bary_1_kSize_;
  inline static int p_distv_bary_2_kSize_;

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

  static int get_p_cell_idx_KSize() { return p_cell_idx_kSize_; }

  static int get_p_cell_blk_KSize() { return p_cell_blk_kSize_; }

  static int get_p_distv_bary_1_KSize() { return p_distv_bary_1_kSize_; }

  static int get_p_distv_bary_2_KSize() { return p_distv_bary_2_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int p_cell_idx_kSize,
                    const int p_cell_blk_kSize, const int p_distv_bary_1_kSize,
                    const int p_distv_bary_2_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    p_cell_idx_kSize_ = p_cell_idx_kSize;
    p_cell_blk_kSize_ = p_cell_blk_kSize;
    p_distv_bary_1_kSize_ = p_distv_bary_1_kSize;
    p_distv_bary_2_kSize_ = p_distv_bary_2_kSize;
  }

  mo_advection_traj_btraj_compute_o1_dsl() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_advection_traj_btraj_compute_o1_dsl has not been set up! make "
             "sure setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto p_vn_sid = get_sid(
        p_vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto p_vt_sid = get_sid(
        p_vt_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto cell_idx_sid = get_sid(
        cell_idx_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto cell_blk_sid = get_sid(
        cell_blk_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto pos_on_tplane_e_1_sid = get_sid(
        pos_on_tplane_e_1_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto pos_on_tplane_e_2_sid = get_sid(
        pos_on_tplane_e_2_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto primal_normal_cell_1_sid = get_sid(
        primal_normal_cell_1_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto dual_normal_cell_1_sid = get_sid(
        dual_normal_cell_1_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto primal_normal_cell_2_sid = get_sid(
        primal_normal_cell_2_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto dual_normal_cell_2_sid = get_sid(
        dual_normal_cell_2_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto p_cell_idx_sid = get_sid(
        p_cell_idx_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto p_cell_blk_sid = get_sid(
        p_cell_blk_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto p_distv_bary_1_sid = get_sid(
        p_distv_bary_1_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto p_distv_bary_2_sid = get_sid(
        p_distv_bary_2_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    gridtools::stencil::global_parameter p_dthalf_gp{p_dthalf_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_4new_sparse<2> e2ec_ptr{};
    auto connectivities =
        gridtools::hymap::keys<, generated::E2EC_t>::make_values(, e2ec_ptr);
    generated::mo_advection_traj_btraj_compute_o1_dsl(connectivities)(
        cuda_backend, p_vn_sid, p_vt_sid, cell_idx_sid, cell_blk_sid,
        pos_on_tplane_e_1_sid, pos_on_tplane_e_2_sid, primal_normal_cell_1_sid,
        dual_normal_cell_1_sid, primal_normal_cell_2_sid,
        dual_normal_cell_2_sid, p_cell_idx_sid, p_cell_blk_sid,
        p_distv_bary_1_sid, p_distv_bary_2_sid, p_dthalf_gp, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *p_vn, double *p_vt, int *cell_idx, int *cell_blk,
                     double *pos_on_tplane_e_1, double *pos_on_tplane_e_2,
                     double *primal_normal_cell_1, double *dual_normal_cell_1,
                     double *primal_normal_cell_2, double *dual_normal_cell_2,
                     int *p_cell_idx, int *p_cell_blk, double *p_distv_bary_1,
                     double *p_distv_bary_2, double p_dthalf) {
    p_vn_ = p_vn;
    p_vt_ = p_vt;
    cell_idx_ = cell_idx;
    cell_blk_ = cell_blk;
    pos_on_tplane_e_1_ = pos_on_tplane_e_1;
    pos_on_tplane_e_2_ = pos_on_tplane_e_2;
    primal_normal_cell_1_ = primal_normal_cell_1;
    dual_normal_cell_1_ = dual_normal_cell_1;
    primal_normal_cell_2_ = primal_normal_cell_2;
    dual_normal_cell_2_ = dual_normal_cell_2;
    p_cell_idx_ = p_cell_idx;
    p_cell_blk_ = p_cell_blk;
    p_distv_bary_1_ = p_distv_bary_1;
    p_distv_bary_2_ = p_distv_bary_2;
    p_dthalf_ = p_dthalf;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_advection_traj_btraj_compute_o1_dsl(
    double *p_vn, double *p_vt, int *cell_idx, int *cell_blk,
    double *pos_on_tplane_e_1, double *pos_on_tplane_e_2,
    double *primal_normal_cell_1, double *dual_normal_cell_1,
    double *primal_normal_cell_2, double *dual_normal_cell_2, int *p_cell_idx,
    int *p_cell_blk, double *p_distv_bary_1, double *p_distv_bary_2,
    double p_dthalf, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_advection_traj_btraj_compute_o1_dsl s;
  s.copy_pointers(p_vn, p_vt, cell_idx, cell_blk, pos_on_tplane_e_1,
                  pos_on_tplane_e_2, primal_normal_cell_1, dual_normal_cell_1,
                  primal_normal_cell_2, dual_normal_cell_2, p_cell_idx,
                  p_cell_blk, p_distv_bary_1, p_distv_bary_2, p_dthalf);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_advection_traj_btraj_compute_o1_dsl(
    const int *p_cell_idx_dsl, const int *p_cell_idx, const int *p_cell_blk_dsl,
    const int *p_cell_blk, const double *p_distv_bary_1_dsl,
    const double *p_distv_bary_1, const double *p_distv_bary_2_dsl,
    const double *p_distv_bary_2, const double p_cell_idx_rel_tol,
    const double p_cell_idx_abs_tol, const double p_cell_blk_rel_tol,
    const double p_cell_blk_abs_tol, const double p_distv_bary_1_rel_tol,
    const double p_distv_bary_1_abs_tol, const double p_distv_bary_2_rel_tol,
    const double p_distv_bary_2_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::getMesh();
  cudaStream_t stream = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::getStream();
  int kSize = dawn_generated::cuda_ico::mo_advection_traj_btraj_compute_o1_dsl::
      getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int p_cell_idx_kSize = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::get_p_cell_idx_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * p_cell_idx_kSize, p_cell_idx_dsl, p_cell_idx,
      "p_cell_idx", p_cell_idx_rel_tol, p_cell_idx_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_cell_idx(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_advection_traj_btraj_compute_o1_dsl", "p_cell_idx");
  serialiser_p_cell_idx.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), p_cell_idx_kSize, (mesh.EdgeStride), p_cell_idx,
        "mo_advection_traj_btraj_compute_o1_dsl", "p_cell_idx", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_cell_idx_kSize,
                          (mesh.EdgeStride), p_cell_idx_dsl,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_cell_idx_dsl", iteration);
    std::cout << "[DSL] serializing p_cell_idx as error is high.\n"
              << std::flush;
#endif
  }
  int p_cell_blk_kSize = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::get_p_cell_blk_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * p_cell_blk_kSize, p_cell_blk_dsl, p_cell_blk,
      "p_cell_blk", p_cell_blk_rel_tol, p_cell_blk_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_cell_blk(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_advection_traj_btraj_compute_o1_dsl", "p_cell_blk");
  serialiser_p_cell_blk.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), p_cell_blk_kSize, (mesh.EdgeStride), p_cell_blk,
        "mo_advection_traj_btraj_compute_o1_dsl", "p_cell_blk", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_cell_blk_kSize,
                          (mesh.EdgeStride), p_cell_blk_dsl,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_cell_blk_dsl", iteration);
    std::cout << "[DSL] serializing p_cell_blk as error is high.\n"
              << std::flush;
#endif
  }
  int p_distv_bary_1_kSize = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::get_p_distv_bary_1_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * p_distv_bary_1_kSize, p_distv_bary_1_dsl,
      p_distv_bary_1, "p_distv_bary_1", p_distv_bary_1_rel_tol,
      p_distv_bary_1_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_distv_bary_1(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_advection_traj_btraj_compute_o1_dsl", "p_distv_bary_1");
  serialiser_p_distv_bary_1.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_distv_bary_1_kSize,
                          (mesh.EdgeStride), p_distv_bary_1,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_distv_bary_1", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_distv_bary_1_kSize,
                          (mesh.EdgeStride), p_distv_bary_1_dsl,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_distv_bary_1_dsl", iteration);
    std::cout << "[DSL] serializing p_distv_bary_1 as error is high.\n"
              << std::flush;
#endif
  }
  int p_distv_bary_2_kSize = dawn_generated::cuda_ico::
      mo_advection_traj_btraj_compute_o1_dsl::get_p_distv_bary_2_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * p_distv_bary_2_kSize, p_distv_bary_2_dsl,
      p_distv_bary_2, "p_distv_bary_2", p_distv_bary_2_rel_tol,
      p_distv_bary_2_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_distv_bary_2(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_advection_traj_btraj_compute_o1_dsl", "p_distv_bary_2");
  serialiser_p_distv_bary_2.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_distv_bary_2_kSize,
                          (mesh.EdgeStride), p_distv_bary_2,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_distv_bary_2", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), p_distv_bary_2_kSize,
                          (mesh.EdgeStride), p_distv_bary_2_dsl,
                          "mo_advection_traj_btraj_compute_o1_dsl",
                          "p_distv_bary_2_dsl", iteration);
    std::cout << "[DSL] serializing p_distv_bary_2 as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_advection_traj_btraj_compute_o1_dsl", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_advection_traj_btraj_compute_o1_dsl(
    double *p_vn, double *p_vt, int *cell_idx, int *cell_blk,
    double *pos_on_tplane_e_1, double *pos_on_tplane_e_2,
    double *primal_normal_cell_1, double *dual_normal_cell_1,
    double *primal_normal_cell_2, double *dual_normal_cell_2, int *p_cell_idx,
    int *p_cell_blk, double *p_distv_bary_1, double *p_distv_bary_2,
    double p_dthalf, int *p_cell_idx_before, int *p_cell_blk_before,
    double *p_distv_bary_1_before, double *p_distv_bary_2_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_cell_idx_rel_tol,
    const double p_cell_idx_abs_tol, const double p_cell_blk_rel_tol,
    const double p_cell_blk_abs_tol, const double p_distv_bary_1_rel_tol,
    const double p_distv_bary_1_abs_tol, const double p_distv_bary_2_rel_tol,
    const double p_distv_bary_2_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_advection_traj_btraj_compute_o1_dsl ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_advection_traj_btraj_compute_o1_dsl(
      p_vn, p_vt, cell_idx, cell_blk, pos_on_tplane_e_1, pos_on_tplane_e_2,
      primal_normal_cell_1, dual_normal_cell_1, primal_normal_cell_2,
      dual_normal_cell_2, p_cell_idx_before, p_cell_blk_before,
      p_distv_bary_1_before, p_distv_bary_2_before, p_dthalf, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_advection_traj_btraj_compute_o1_dsl run time: " << time
            << "s\n"
            << std::flush;
  std::cout
      << "[DSL] Verifying stencil mo_advection_traj_btraj_compute_o1_dsl...\n"
      << std::flush;
  verify_mo_advection_traj_btraj_compute_o1_dsl(
      p_cell_idx_before, p_cell_idx, p_cell_blk_before, p_cell_blk,
      p_distv_bary_1_before, p_distv_bary_1, p_distv_bary_2_before,
      p_distv_bary_2, p_cell_idx_rel_tol, p_cell_idx_abs_tol,
      p_cell_blk_rel_tol, p_cell_blk_abs_tol, p_distv_bary_1_rel_tol,
      p_distv_bary_1_abs_tol, p_distv_bary_2_rel_tol, p_distv_bary_2_abs_tol,
      iteration);

  iteration++;
}

void setup_mo_advection_traj_btraj_compute_o1_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_cell_idx_k_size, const int p_cell_blk_k_size,
    const int p_distv_bary_1_k_size, const int p_distv_bary_2_k_size) {
  dawn_generated::cuda_ico::mo_advection_traj_btraj_compute_o1_dsl::setup(
      mesh, k_size, stream, p_cell_idx_k_size, p_cell_blk_k_size,
      p_distv_bary_1_k_size, p_distv_bary_2_k_size);
}

void free_mo_advection_traj_btraj_compute_o1_dsl() {
  dawn_generated::cuda_ico::mo_advection_traj_btraj_compute_o1_dsl::free();
}
}
