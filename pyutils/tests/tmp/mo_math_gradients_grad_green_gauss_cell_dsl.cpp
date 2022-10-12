#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_math_gradients_grad_green_gauss_cell_dsl.hpp"
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

class mo_math_gradients_grad_green_gauss_cell_dsl {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *cecoTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      cecoTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Cells, dawn::LocationType::Edges,
               dawn::LocationType::Cells},
              1});
    }
  };

private:
  double *p_grad_1_u_;
  double *p_grad_1_v_;
  double *p_grad_2_u_;
  double *p_grad_2_v_;
  double *p_ccpr1_;
  double *p_ccpr2_;
  double *geofac_grg_x_;
  double *geofac_grg_y_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int p_grad_1_u_kSize_;
  inline static int p_grad_1_v_kSize_;
  inline static int p_grad_2_u_kSize_;
  inline static int p_grad_2_v_kSize_;

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

  static int get_p_grad_1_u_KSize() { return p_grad_1_u_kSize_; }

  static int get_p_grad_1_v_KSize() { return p_grad_1_v_kSize_; }

  static int get_p_grad_2_u_KSize() { return p_grad_2_u_kSize_; }

  static int get_p_grad_2_v_KSize() { return p_grad_2_v_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int p_grad_1_u_kSize,
                    const int p_grad_1_v_kSize, const int p_grad_2_u_kSize,
                    const int p_grad_2_v_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    p_grad_1_u_kSize_ = p_grad_1_u_kSize;
    p_grad_1_v_kSize_ = p_grad_1_v_kSize;
    p_grad_2_u_kSize_ = p_grad_2_u_kSize;
    p_grad_2_v_kSize_ = p_grad_2_v_kSize;
  }

  mo_math_gradients_grad_green_gauss_cell_dsl() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_math_gradients_grad_green_gauss_cell_dsl has not been set up! "
             "make sure setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto p_grad_1_u_sid = get_sid(
        p_grad_1_u_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_grad_1_v_sid = get_sid(
        p_grad_1_v_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_grad_2_u_sid = get_sid(
        p_grad_2_u_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_grad_2_v_sid = get_sid(
        p_grad_2_v_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_ccpr1_sid = get_sid(
        p_ccpr1_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto p_ccpr2_sid = get_sid(
        p_ccpr2_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<4> ceco_ptr{.raw_ptr_fortran = mesh_.cecoTable};
    auto connectivities =
        gridtools::hymap::keys<generated::C2E2CO_t>::make_values(ceco_ptr);
    double *geofac_grg_x_0 = &geofac_grg_x_[0 * mesh_.CellStride];
    double *geofac_grg_x_1 = &geofac_grg_x_[1 * mesh_.CellStride];
    double *geofac_grg_x_2 = &geofac_grg_x_[2 * mesh_.CellStride];
    double *geofac_grg_x_3 = &geofac_grg_x_[3 * mesh_.CellStride];
    auto geofac_grg_x_sid_0 = get_sid(
        geofac_grg_x_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_1 = get_sid(
        geofac_grg_x_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_2 = get_sid(
        geofac_grg_x_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_3 = get_sid(
        geofac_grg_x_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_x_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>,
        integral_constant<int, 3>>::make_values(geofac_grg_x_sid_0,
                                                geofac_grg_x_sid_1,
                                                geofac_grg_x_sid_2,
                                                geofac_grg_x_sid_3);
    double *geofac_grg_y_0 = &geofac_grg_y_[0 * mesh_.CellStride];
    double *geofac_grg_y_1 = &geofac_grg_y_[1 * mesh_.CellStride];
    double *geofac_grg_y_2 = &geofac_grg_y_[2 * mesh_.CellStride];
    double *geofac_grg_y_3 = &geofac_grg_y_[3 * mesh_.CellStride];
    auto geofac_grg_y_sid_0 = get_sid(
        geofac_grg_y_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_1 = get_sid(
        geofac_grg_y_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_2 = get_sid(
        geofac_grg_y_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_3 = get_sid(
        geofac_grg_y_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_grg_y_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>,
        integral_constant<int, 3>>::make_values(geofac_grg_y_sid_0,
                                                geofac_grg_y_sid_1,
                                                geofac_grg_y_sid_2,
                                                geofac_grg_y_sid_3);
    generated::mo_math_gradients_grad_green_gauss_cell_dsl(connectivities)(
        cuda_backend, p_grad_1_u_sid, p_grad_1_v_sid, p_grad_2_u_sid,
        p_grad_2_v_sid, p_ccpr1_sid, p_ccpr2_sid, geofac_grg_x_sid_comp,
        geofac_grg_y_sid_comp, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *p_grad_1_u, double *p_grad_1_v, double *p_grad_2_u,
                     double *p_grad_2_v, double *p_ccpr1, double *p_ccpr2,
                     double *geofac_grg_x, double *geofac_grg_y) {
    p_grad_1_u_ = p_grad_1_u;
    p_grad_1_v_ = p_grad_1_v;
    p_grad_2_u_ = p_grad_2_u;
    p_grad_2_v_ = p_grad_2_v;
    p_ccpr1_ = p_ccpr1;
    p_ccpr2_ = p_ccpr2;
    geofac_grg_x_ = geofac_grg_x;
    geofac_grg_y_ = geofac_grg_y;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_math_gradients_grad_green_gauss_cell_dsl(
    double *p_grad_1_u, double *p_grad_1_v, double *p_grad_2_u,
    double *p_grad_2_v, double *p_ccpr1, double *p_ccpr2, double *geofac_grg_x,
    double *geofac_grg_y, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_math_gradients_grad_green_gauss_cell_dsl s;
  s.copy_pointers(p_grad_1_u, p_grad_1_v, p_grad_2_u, p_grad_2_v, p_ccpr1,
                  p_ccpr2, geofac_grg_x, geofac_grg_y);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_math_gradients_grad_green_gauss_cell_dsl(
    const double *p_grad_1_u_dsl, const double *p_grad_1_u,
    const double *p_grad_1_v_dsl, const double *p_grad_1_v,
    const double *p_grad_2_u_dsl, const double *p_grad_2_u,
    const double *p_grad_2_v_dsl, const double *p_grad_2_v,
    const double p_grad_1_u_rel_tol, const double p_grad_1_u_abs_tol,
    const double p_grad_1_v_rel_tol, const double p_grad_1_v_abs_tol,
    const double p_grad_2_u_rel_tol, const double p_grad_2_u_abs_tol,
    const double p_grad_2_v_rel_tol, const double p_grad_2_v_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::getMesh();
  cudaStream_t stream = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::getStream();
  int kSize = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int p_grad_1_u_kSize = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::get_p_grad_1_u_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * p_grad_1_u_kSize, p_grad_1_u_dsl, p_grad_1_u,
      "p_grad_1_u", p_grad_1_u_rel_tol, p_grad_1_u_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_grad_1_u(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_1_u");
  serialiser_p_grad_1_u.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), p_grad_1_u_kSize, (mesh.CellStride), p_grad_1_u,
        "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_1_u", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), p_grad_1_u_kSize,
                          (mesh.CellStride), p_grad_1_u_dsl,
                          "mo_math_gradients_grad_green_gauss_cell_dsl",
                          "p_grad_1_u_dsl", iteration);
    std::cout << "[DSL] serializing p_grad_1_u as error is high.\n"
              << std::flush;
#endif
  }
  int p_grad_1_v_kSize = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::get_p_grad_1_v_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * p_grad_1_v_kSize, p_grad_1_v_dsl, p_grad_1_v,
      "p_grad_1_v", p_grad_1_v_rel_tol, p_grad_1_v_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_grad_1_v(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_1_v");
  serialiser_p_grad_1_v.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), p_grad_1_v_kSize, (mesh.CellStride), p_grad_1_v,
        "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_1_v", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), p_grad_1_v_kSize,
                          (mesh.CellStride), p_grad_1_v_dsl,
                          "mo_math_gradients_grad_green_gauss_cell_dsl",
                          "p_grad_1_v_dsl", iteration);
    std::cout << "[DSL] serializing p_grad_1_v as error is high.\n"
              << std::flush;
#endif
  }
  int p_grad_2_u_kSize = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::get_p_grad_2_u_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * p_grad_2_u_kSize, p_grad_2_u_dsl, p_grad_2_u,
      "p_grad_2_u", p_grad_2_u_rel_tol, p_grad_2_u_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_grad_2_u(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_2_u");
  serialiser_p_grad_2_u.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), p_grad_2_u_kSize, (mesh.CellStride), p_grad_2_u,
        "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_2_u", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), p_grad_2_u_kSize,
                          (mesh.CellStride), p_grad_2_u_dsl,
                          "mo_math_gradients_grad_green_gauss_cell_dsl",
                          "p_grad_2_u_dsl", iteration);
    std::cout << "[DSL] serializing p_grad_2_u as error is high.\n"
              << std::flush;
#endif
  }
  int p_grad_2_v_kSize = dawn_generated::cuda_ico::
      mo_math_gradients_grad_green_gauss_cell_dsl::get_p_grad_2_v_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * p_grad_2_v_kSize, p_grad_2_v_dsl, p_grad_2_v,
      "p_grad_2_v", p_grad_2_v_rel_tol, p_grad_2_v_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_p_grad_2_v(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_2_v");
  serialiser_p_grad_2_v.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), p_grad_2_v_kSize, (mesh.CellStride), p_grad_2_v,
        "mo_math_gradients_grad_green_gauss_cell_dsl", "p_grad_2_v", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), p_grad_2_v_kSize,
                          (mesh.CellStride), p_grad_2_v_dsl,
                          "mo_math_gradients_grad_green_gauss_cell_dsl",
                          "p_grad_2_v_dsl", iteration);
    std::cout << "[DSL] serializing p_grad_2_v as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_math_gradients_grad_green_gauss_cell_dsl",
                       iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_math_gradients_grad_green_gauss_cell_dsl(
    double *p_grad_1_u, double *p_grad_1_v, double *p_grad_2_u,
    double *p_grad_2_v, double *p_ccpr1, double *p_ccpr2, double *geofac_grg_x,
    double *geofac_grg_y, double *p_grad_1_u_before, double *p_grad_1_v_before,
    double *p_grad_2_u_before, double *p_grad_2_v_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double p_grad_1_u_rel_tol,
    const double p_grad_1_u_abs_tol, const double p_grad_1_v_rel_tol,
    const double p_grad_1_v_abs_tol, const double p_grad_2_u_rel_tol,
    const double p_grad_2_u_abs_tol, const double p_grad_2_v_rel_tol,
    const double p_grad_2_v_abs_tol) {
  static int iteration = 0;
  std::cout
      << "[DSL] Running stencil mo_math_gradients_grad_green_gauss_cell_dsl ("
      << iteration << ") ...\n"
      << std::flush;
  run_mo_math_gradients_grad_green_gauss_cell_dsl(
      p_grad_1_u_before, p_grad_1_v_before, p_grad_2_u_before,
      p_grad_2_v_before, p_ccpr1, p_ccpr2, geofac_grg_x, geofac_grg_y,
      verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_math_gradients_grad_green_gauss_cell_dsl run time: "
            << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil "
               "mo_math_gradients_grad_green_gauss_cell_dsl...\n"
            << std::flush;
  verify_mo_math_gradients_grad_green_gauss_cell_dsl(
      p_grad_1_u_before, p_grad_1_u, p_grad_1_v_before, p_grad_1_v,
      p_grad_2_u_before, p_grad_2_u, p_grad_2_v_before, p_grad_2_v,
      p_grad_1_u_rel_tol, p_grad_1_u_abs_tol, p_grad_1_v_rel_tol,
      p_grad_1_v_abs_tol, p_grad_2_u_rel_tol, p_grad_2_u_abs_tol,
      p_grad_2_v_rel_tol, p_grad_2_v_abs_tol, iteration);

  iteration++;
}

void setup_mo_math_gradients_grad_green_gauss_cell_dsl(
    dawn::GlobalGpuTriMesh *mesh, int k_size, cudaStream_t stream,
    const int p_grad_1_u_k_size, const int p_grad_1_v_k_size,
    const int p_grad_2_u_k_size, const int p_grad_2_v_k_size) {
  dawn_generated::cuda_ico::mo_math_gradients_grad_green_gauss_cell_dsl::setup(
      mesh, k_size, stream, p_grad_1_u_k_size, p_grad_1_v_k_size,
      p_grad_2_u_k_size, p_grad_2_v_k_size);
}

void free_mo_math_gradients_grad_green_gauss_cell_dsl() {
  dawn_generated::cuda_ico::mo_math_gradients_grad_green_gauss_cell_dsl::free();
}
}
