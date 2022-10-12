#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_44.hpp"
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

class mo_solve_nonhydro_stencil_44 {
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
  double *z_beta_;
  double *exner_nnow_;
  double *rho_nnow_;
  double *theta_v_nnow_;
  double *inv_ddqz_z_full_;
  double *z_alpha_;
  double *vwind_impl_wgt_;
  double *theta_v_ic_;
  double *rho_ic_;
  double dtime_;
  double rd_;
  double cvd_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_beta_kSize_;
  inline static int z_alpha_kSize_;

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

  static int get_z_beta_KSize() { return z_beta_kSize_; }

  static int get_z_alpha_KSize() { return z_alpha_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_beta_kSize,
                    const int z_alpha_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_beta_kSize_ = z_beta_kSize;
    z_alpha_kSize_ = z_alpha_kSize;
  }

  mo_solve_nonhydro_stencil_44() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_44 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_beta_sid = get_sid(
        z_beta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_nnow_sid = get_sid(
        exner_nnow_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_nnow_sid = get_sid(
        rho_nnow_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_nnow_sid = get_sid(
        theta_v_nnow_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto inv_ddqz_z_full_sid = get_sid(
        inv_ddqz_z_full_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_alpha_sid = get_sid(
        z_alpha_,
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

    auto rho_ic_sid = get_sid(
        rho_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    gridtools::stencil::global_parameter rd_gp{rd_};
    gridtools::stencil::global_parameter cvd_gp{cvd_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_44(connectivities)(
        cuda_backend, z_beta_sid, exner_nnow_sid, rho_nnow_sid,
        theta_v_nnow_sid, inv_ddqz_z_full_sid, z_alpha_sid, vwind_impl_wgt_sid,
        theta_v_ic_sid, rho_ic_sid, dtime_gp, rd_gp, cvd_gp, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_beta, double *exner_nnow, double *rho_nnow,
                     double *theta_v_nnow, double *inv_ddqz_z_full,
                     double *z_alpha, double *vwind_impl_wgt,
                     double *theta_v_ic, double *rho_ic, double dtime,
                     double rd, double cvd) {
    z_beta_ = z_beta;
    exner_nnow_ = exner_nnow;
    rho_nnow_ = rho_nnow;
    theta_v_nnow_ = theta_v_nnow;
    inv_ddqz_z_full_ = inv_ddqz_z_full;
    z_alpha_ = z_alpha;
    vwind_impl_wgt_ = vwind_impl_wgt;
    theta_v_ic_ = theta_v_ic;
    rho_ic_ = rho_ic;
    dtime_ = dtime;
    rd_ = rd;
    cvd_ = cvd;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_44(
    double *z_beta, double *exner_nnow, double *rho_nnow, double *theta_v_nnow,
    double *inv_ddqz_z_full, double *z_alpha, double *vwind_impl_wgt,
    double *theta_v_ic, double *rho_ic, double dtime, double rd, double cvd,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44 s;
  s.copy_pointers(z_beta, exner_nnow, rho_nnow, theta_v_nnow, inv_ddqz_z_full,
                  z_alpha, vwind_impl_wgt, theta_v_ic, rho_ic, dtime, rd, cvd);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_44(
    const double *z_beta_dsl, const double *z_beta, const double *z_alpha_dsl,
    const double *z_alpha, const double z_beta_rel_tol,
    const double z_beta_abs_tol, const double z_alpha_rel_tol,
    const double z_alpha_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_beta_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::
      get_z_beta_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_beta_kSize, z_beta_dsl, z_beta, "z_beta",
      z_beta_rel_tol, z_beta_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_beta(stencilMetrics,
                                      metricsNameFromEnvVar("SLURM_JOB_ID"),
                                      "mo_solve_nonhydro_stencil_44", "z_beta");
  serialiser_z_beta.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_beta_kSize,
                          (mesh.CellStride), z_beta,
                          "mo_solve_nonhydro_stencil_44", "z_beta", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_beta_kSize, (mesh.CellStride), z_beta_dsl,
        "mo_solve_nonhydro_stencil_44", "z_beta_dsl", iteration);
    std::cout << "[DSL] serializing z_beta as error is high.\n" << std::flush;
#endif
  }
  int z_alpha_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::
      get_z_alpha_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_alpha_kSize, z_alpha_dsl, z_alpha,
      "z_alpha", z_alpha_rel_tol, z_alpha_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_alpha(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_44", "z_alpha");
  serialiser_z_alpha.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_alpha_kSize,
                          (mesh.CellStride), z_alpha,
                          "mo_solve_nonhydro_stencil_44", "z_alpha", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_alpha_kSize, (mesh.CellStride), z_alpha_dsl,
        "mo_solve_nonhydro_stencil_44", "z_alpha_dsl", iteration);
    std::cout << "[DSL] serializing z_alpha as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_44", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_44(
    double *z_beta, double *exner_nnow, double *rho_nnow, double *theta_v_nnow,
    double *inv_ddqz_z_full, double *z_alpha, double *vwind_impl_wgt,
    double *theta_v_ic, double *rho_ic, double dtime, double rd, double cvd,
    double *z_beta_before, double *z_alpha_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_beta_rel_tol, const double z_beta_abs_tol,
    const double z_alpha_rel_tol, const double z_alpha_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_44 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_44(
      z_beta_before, exner_nnow, rho_nnow, theta_v_nnow, inv_ddqz_z_full,
      z_alpha_before, vwind_impl_wgt, theta_v_ic, rho_ic, dtime, rd, cvd,
      verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_44 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_44...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_44(
      z_beta_before, z_beta, z_alpha_before, z_alpha, z_beta_rel_tol,
      z_beta_abs_tol, z_alpha_rel_tol, z_alpha_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_44(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_beta_k_size,
                                        const int z_alpha_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::setup(
      mesh, k_size, stream, z_beta_k_size, z_alpha_k_size);
}

void free_mo_solve_nonhydro_stencil_44() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_44::free();
}
}
