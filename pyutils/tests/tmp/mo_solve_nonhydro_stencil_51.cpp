#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_51.hpp"
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

class mo_solve_nonhydro_stencil_51 {
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
  double *z_q_;
  double *w_nnew_;
  double *vwind_impl_wgt_;
  double *theta_v_ic_;
  double *ddqz_z_half_;
  double *z_beta_;
  double *z_alpha_;
  double *z_w_expl_;
  double *z_exner_expl_;
  double dtime_;
  double cpd_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_q_kSize_;
  inline static int w_nnew_kSize_;

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

  static int get_z_q_KSize() { return z_q_kSize_; }

  static int get_w_nnew_KSize() { return w_nnew_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_q_kSize,
                    const int w_nnew_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_q_kSize_ = z_q_kSize;
    w_nnew_kSize_ = w_nnew_kSize;
  }

  mo_solve_nonhydro_stencil_51() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_51 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_q_sid = get_sid(
        z_q_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_nnew_sid = get_sid(
        w_nnew_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto vwind_impl_wgt_sid = get_sid(
        vwind_impl_wgt_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto theta_v_ic_sid = get_sid(
        theta_v_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddqz_z_half_sid = get_sid(
        ddqz_z_half_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_beta_sid = get_sid(
        z_beta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_alpha_sid = get_sid(
        z_alpha_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_w_expl_sid = get_sid(
        z_w_expl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_exner_expl_sid = get_sid(
        z_exner_expl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    gridtools::stencil::global_parameter cpd_gp{cpd_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_51(connectivities)(
        cuda_backend, z_q_sid, w_nnew_sid, vwind_impl_wgt_sid, theta_v_ic_sid,
        ddqz_z_half_sid, z_beta_sid, z_alpha_sid, z_w_expl_sid,
        z_exner_expl_sid, dtime_gp, cpd_gp, horizontalStart, horizontalEnd,
        verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_q, double *w_nnew, double *vwind_impl_wgt,
                     double *theta_v_ic, double *ddqz_z_half, double *z_beta,
                     double *z_alpha, double *z_w_expl, double *z_exner_expl,
                     double dtime, double cpd) {
    z_q_ = z_q;
    w_nnew_ = w_nnew;
    vwind_impl_wgt_ = vwind_impl_wgt;
    theta_v_ic_ = theta_v_ic;
    ddqz_z_half_ = ddqz_z_half;
    z_beta_ = z_beta;
    z_alpha_ = z_alpha;
    z_w_expl_ = z_w_expl;
    z_exner_expl_ = z_exner_expl;
    dtime_ = dtime;
    cpd_ = cpd;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_51(
    double *z_q, double *w_nnew, double *vwind_impl_wgt, double *theta_v_ic,
    double *ddqz_z_half, double *z_beta, double *z_alpha, double *z_w_expl,
    double *z_exner_expl, double dtime, double cpd, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51 s;
  s.copy_pointers(z_q, w_nnew, vwind_impl_wgt, theta_v_ic, ddqz_z_half, z_beta,
                  z_alpha, z_w_expl, z_exner_expl, dtime, cpd);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_51(
    const double *z_q_dsl, const double *z_q, const double *w_nnew_dsl,
    const double *w_nnew, const double z_q_rel_tol, const double z_q_abs_tol,
    const double w_nnew_rel_tol, const double w_nnew_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_q_kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::get_z_q_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * z_q_kSize, z_q_dsl, z_q,
                           "z_q", z_q_rel_tol, z_q_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_q(stencilMetrics,
                                   metricsNameFromEnvVar("SLURM_JOB_ID"),
                                   "mo_solve_nonhydro_stencil_51", "z_q");
  serialiser_z_q.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_q_kSize, (mesh.CellStride),
                          z_q, "mo_solve_nonhydro_stencil_51", "z_q",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_q_kSize, (mesh.CellStride),
                          z_q_dsl, "mo_solve_nonhydro_stencil_51", "z_q_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_q as error is high.\n" << std::flush;
#endif
  }
  int w_nnew_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::
      get_w_nnew_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * w_nnew_kSize, w_nnew_dsl, w_nnew, "w_nnew",
      w_nnew_rel_tol, w_nnew_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_w_nnew(stencilMetrics,
                                      metricsNameFromEnvVar("SLURM_JOB_ID"),
                                      "mo_solve_nonhydro_stencil_51", "w_nnew");
  serialiser_w_nnew.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), w_nnew_kSize,
                          (mesh.CellStride), w_nnew,
                          "mo_solve_nonhydro_stencil_51", "w_nnew", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), w_nnew_kSize, (mesh.CellStride), w_nnew_dsl,
        "mo_solve_nonhydro_stencil_51", "w_nnew_dsl", iteration);
    std::cout << "[DSL] serializing w_nnew as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_51", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_51(
    double *z_q, double *w_nnew, double *vwind_impl_wgt, double *theta_v_ic,
    double *ddqz_z_half, double *z_beta, double *z_alpha, double *z_w_expl,
    double *z_exner_expl, double dtime, double cpd, double *z_q_before,
    double *w_nnew_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double z_q_rel_tol, const double z_q_abs_tol,
    const double w_nnew_rel_tol, const double w_nnew_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_51 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_51(
      z_q_before, w_nnew_before, vwind_impl_wgt, theta_v_ic, ddqz_z_half,
      z_beta, z_alpha, z_w_expl, z_exner_expl, dtime, cpd, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_51 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_51...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_51(z_q_before, z_q, w_nnew_before, w_nnew,
                                      z_q_rel_tol, z_q_abs_tol, w_nnew_rel_tol,
                                      w_nnew_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_51(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_q_k_size,
                                        const int w_nnew_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::setup(
      mesh, k_size, stream, z_q_k_size, w_nnew_k_size);
}

void free_mo_solve_nonhydro_stencil_51() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_51::free();
}
}
