#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_61.hpp"
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

class mo_solve_nonhydro_stencil_61 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
    }
  };

private:
  double *rho_now_;
  double *grf_tend_rho_;
  double *theta_v_now_;
  double *grf_tend_thv_;
  double *w_now_;
  double *grf_tend_w_;
  double *rho_new_;
  double *exner_new_;
  double *w_new_;
  double dtime_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int rho_new_kSize_;
  inline static int exner_new_kSize_;
  inline static int w_new_kSize_;

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

  static int get_rho_new_KSize() { return rho_new_kSize_; }

  static int get_exner_new_KSize() { return exner_new_kSize_; }

  static int get_w_new_KSize() { return w_new_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int rho_new_kSize,
                    const int exner_new_kSize, const int w_new_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    rho_new_kSize_ = rho_new_kSize;
    exner_new_kSize_ = exner_new_kSize;
    w_new_kSize_ = w_new_kSize;
  }

  mo_solve_nonhydro_stencil_61() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_61 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto rho_now_sid = get_sid(
        rho_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto grf_tend_rho_sid = get_sid(
        grf_tend_rho_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_now_sid = get_sid(
        theta_v_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto grf_tend_thv_sid = get_sid(
        grf_tend_thv_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_now_sid = get_sid(
        w_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto grf_tend_w_sid = get_sid(
        grf_tend_w_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_new_sid = get_sid(
        rho_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_new_sid = get_sid(
        exner_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_new_sid = get_sid(
        w_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_61(connectivities)(
        cuda_backend, rho_now_sid, grf_tend_rho_sid, theta_v_now_sid,
        grf_tend_thv_sid, w_now_sid, grf_tend_w_sid, rho_new_sid, exner_new_sid,
        w_new_sid, dtime_gp, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *rho_now, double *grf_tend_rho, double *theta_v_now,
                     double *grf_tend_thv, double *w_now, double *grf_tend_w,
                     double *rho_new, double *exner_new, double *w_new,
                     double dtime) {
    rho_now_ = rho_now;
    grf_tend_rho_ = grf_tend_rho;
    theta_v_now_ = theta_v_now;
    grf_tend_thv_ = grf_tend_thv;
    w_now_ = w_now;
    grf_tend_w_ = grf_tend_w;
    rho_new_ = rho_new;
    exner_new_ = exner_new;
    w_new_ = w_new;
    dtime_ = dtime;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_61(
    double *rho_now, double *grf_tend_rho, double *theta_v_now,
    double *grf_tend_thv, double *w_now, double *grf_tend_w, double *rho_new,
    double *exner_new, double *w_new, double dtime, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61 s;
  s.copy_pointers(rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now,
                  grf_tend_w, rho_new, exner_new, w_new, dtime);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_61(
    const double *rho_new_dsl, const double *rho_new,
    const double *exner_new_dsl, const double *exner_new,
    const double *w_new_dsl, const double *w_new, const double rho_new_rel_tol,
    const double rho_new_abs_tol, const double exner_new_rel_tol,
    const double exner_new_abs_tol, const double w_new_rel_tol,
    const double w_new_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int rho_new_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::
      get_rho_new_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * rho_new_kSize, rho_new_dsl, rho_new,
      "rho_new", rho_new_rel_tol, rho_new_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_rho_new(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_61", "rho_new");
  serialiser_rho_new.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), rho_new_kSize,
                          (mesh.CellStride), rho_new,
                          "mo_solve_nonhydro_stencil_61", "rho_new", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), rho_new_kSize, (mesh.CellStride), rho_new_dsl,
        "mo_solve_nonhydro_stencil_61", "rho_new_dsl", iteration);
    std::cout << "[DSL] serializing rho_new as error is high.\n" << std::flush;
#endif
  }
  int exner_new_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::
      get_exner_new_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * exner_new_kSize, exner_new_dsl, exner_new,
      "exner_new", exner_new_rel_tol, exner_new_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_exner_new(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_61", "exner_new");
  serialiser_exner_new.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), exner_new_kSize, (mesh.CellStride), exner_new,
        "mo_solve_nonhydro_stencil_61", "exner_new", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), exner_new_kSize,
                          (mesh.CellStride), exner_new_dsl,
                          "mo_solve_nonhydro_stencil_61", "exner_new_dsl",
                          iteration);
    std::cout << "[DSL] serializing exner_new as error is high.\n"
              << std::flush;
#endif
  }
  int w_new_kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::get_w_new_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * w_new_kSize, w_new_dsl, w_new, "w_new",
      w_new_rel_tol, w_new_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_w_new(stencilMetrics,
                                     metricsNameFromEnvVar("SLURM_JOB_ID"),
                                     "mo_solve_nonhydro_stencil_61", "w_new");
  serialiser_w_new.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), w_new_kSize,
                          (mesh.CellStride), w_new,
                          "mo_solve_nonhydro_stencil_61", "w_new", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), w_new_kSize, (mesh.CellStride), w_new_dsl,
        "mo_solve_nonhydro_stencil_61", "w_new_dsl", iteration);
    std::cout << "[DSL] serializing w_new as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_61", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_61(
    double *rho_now, double *grf_tend_rho, double *theta_v_now,
    double *grf_tend_thv, double *w_now, double *grf_tend_w, double *rho_new,
    double *exner_new, double *w_new, double dtime, double *rho_new_before,
    double *exner_new_before, double *w_new_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double rho_new_rel_tol, const double rho_new_abs_tol,
    const double exner_new_rel_tol, const double exner_new_abs_tol,
    const double w_new_rel_tol, const double w_new_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_61 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_61(
      rho_now, grf_tend_rho, theta_v_now, grf_tend_thv, w_now, grf_tend_w,
      rho_new_before, exner_new_before, w_new_before, dtime, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_61 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_61...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_61(
      rho_new_before, rho_new, exner_new_before, exner_new, w_new_before, w_new,
      rho_new_rel_tol, rho_new_abs_tol, exner_new_rel_tol, exner_new_abs_tol,
      w_new_rel_tol, w_new_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_61(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int rho_new_k_size,
                                        const int exner_new_k_size,
                                        const int w_new_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::setup(
      mesh, k_size, stream, rho_new_k_size, exner_new_k_size, w_new_k_size);
}

void free_mo_solve_nonhydro_stencil_61() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_61::free();
}
}
