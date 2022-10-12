#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_49.hpp"
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

class mo_solve_nonhydro_stencil_49 {
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
  double *z_rho_expl_;
  double *z_exner_expl_;
  double *rho_nnow_;
  double *inv_ddqz_z_full_;
  double *z_flxdiv_mass_;
  double *z_contr_w_fl_l_;
  double *exner_pr_;
  double *z_beta_;
  double *z_flxdiv_theta_;
  double *theta_v_ic_;
  double *ddt_exner_phy_;
  double dtime_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_rho_expl_kSize_;
  inline static int z_exner_expl_kSize_;

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

  static int get_z_rho_expl_KSize() { return z_rho_expl_kSize_; }

  static int get_z_exner_expl_KSize() { return z_exner_expl_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_rho_expl_kSize,
                    const int z_exner_expl_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_rho_expl_kSize_ = z_rho_expl_kSize;
    z_exner_expl_kSize_ = z_exner_expl_kSize;
  }

  mo_solve_nonhydro_stencil_49() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_49 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_rho_expl_sid = get_sid(
        z_rho_expl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_exner_expl_sid = get_sid(
        z_exner_expl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_nnow_sid = get_sid(
        rho_nnow_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto inv_ddqz_z_full_sid = get_sid(
        inv_ddqz_z_full_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_flxdiv_mass_sid = get_sid(
        z_flxdiv_mass_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_contr_w_fl_l_sid = get_sid(
        z_contr_w_fl_l_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_pr_sid = get_sid(
        exner_pr_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_beta_sid = get_sid(
        z_beta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_flxdiv_theta_sid = get_sid(
        z_flxdiv_theta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_ic_sid = get_sid(
        theta_v_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddt_exner_phy_sid = get_sid(
        ddt_exner_phy_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_49(connectivities)(
        cuda_backend, z_rho_expl_sid, z_exner_expl_sid, rho_nnow_sid,
        inv_ddqz_z_full_sid, z_flxdiv_mass_sid, z_contr_w_fl_l_sid,
        exner_pr_sid, z_beta_sid, z_flxdiv_theta_sid, theta_v_ic_sid,
        ddt_exner_phy_sid, dtime_gp, horizontalStart, horizontalEnd,
        verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_rho_expl, double *z_exner_expl, double *rho_nnow,
                     double *inv_ddqz_z_full, double *z_flxdiv_mass,
                     double *z_contr_w_fl_l, double *exner_pr, double *z_beta,
                     double *z_flxdiv_theta, double *theta_v_ic,
                     double *ddt_exner_phy, double dtime) {
    z_rho_expl_ = z_rho_expl;
    z_exner_expl_ = z_exner_expl;
    rho_nnow_ = rho_nnow;
    inv_ddqz_z_full_ = inv_ddqz_z_full;
    z_flxdiv_mass_ = z_flxdiv_mass;
    z_contr_w_fl_l_ = z_contr_w_fl_l;
    exner_pr_ = exner_pr;
    z_beta_ = z_beta;
    z_flxdiv_theta_ = z_flxdiv_theta;
    theta_v_ic_ = theta_v_ic;
    ddt_exner_phy_ = ddt_exner_phy;
    dtime_ = dtime;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_49(
    double *z_rho_expl, double *z_exner_expl, double *rho_nnow,
    double *inv_ddqz_z_full, double *z_flxdiv_mass, double *z_contr_w_fl_l,
    double *exner_pr, double *z_beta, double *z_flxdiv_theta,
    double *theta_v_ic, double *ddt_exner_phy, double dtime,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49 s;
  s.copy_pointers(z_rho_expl, z_exner_expl, rho_nnow, inv_ddqz_z_full,
                  z_flxdiv_mass, z_contr_w_fl_l, exner_pr, z_beta,
                  z_flxdiv_theta, theta_v_ic, ddt_exner_phy, dtime);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_49(
    const double *z_rho_expl_dsl, const double *z_rho_expl,
    const double *z_exner_expl_dsl, const double *z_exner_expl,
    const double z_rho_expl_rel_tol, const double z_rho_expl_abs_tol,
    const double z_exner_expl_rel_tol, const double z_exner_expl_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_rho_expl_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_49::get_z_rho_expl_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_rho_expl_kSize, z_rho_expl_dsl, z_rho_expl,
      "z_rho_expl", z_rho_expl_rel_tol, z_rho_expl_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_rho_expl(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_49", "z_rho_expl");
  serialiser_z_rho_expl.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_rho_expl_kSize, (mesh.CellStride), z_rho_expl,
        "mo_solve_nonhydro_stencil_49", "z_rho_expl", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_rho_expl_kSize,
                          (mesh.CellStride), z_rho_expl_dsl,
                          "mo_solve_nonhydro_stencil_49", "z_rho_expl_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_rho_expl as error is high.\n"
              << std::flush;
#endif
  }
  int z_exner_expl_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_49::get_z_exner_expl_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_exner_expl_kSize, z_exner_expl_dsl,
      z_exner_expl, "z_exner_expl", z_exner_expl_rel_tol, z_exner_expl_abs_tol,
      iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_exner_expl(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_49", "z_exner_expl");
  serialiser_z_exner_expl.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_exner_expl_kSize,
                          (mesh.CellStride), z_exner_expl,
                          "mo_solve_nonhydro_stencil_49", "z_exner_expl",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_exner_expl_kSize,
                          (mesh.CellStride), z_exner_expl_dsl,
                          "mo_solve_nonhydro_stencil_49", "z_exner_expl_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_exner_expl as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_49", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_49(
    double *z_rho_expl, double *z_exner_expl, double *rho_nnow,
    double *inv_ddqz_z_full, double *z_flxdiv_mass, double *z_contr_w_fl_l,
    double *exner_pr, double *z_beta, double *z_flxdiv_theta,
    double *theta_v_ic, double *ddt_exner_phy, double dtime,
    double *z_rho_expl_before, double *z_exner_expl_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_rho_expl_rel_tol,
    const double z_rho_expl_abs_tol, const double z_exner_expl_rel_tol,
    const double z_exner_expl_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_49 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_49(
      z_rho_expl_before, z_exner_expl_before, rho_nnow, inv_ddqz_z_full,
      z_flxdiv_mass, z_contr_w_fl_l, exner_pr, z_beta, z_flxdiv_theta,
      theta_v_ic, ddt_exner_phy, dtime, verticalStart, verticalEnd,
      horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_49 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_49...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_49(
      z_rho_expl_before, z_rho_expl, z_exner_expl_before, z_exner_expl,
      z_rho_expl_rel_tol, z_rho_expl_abs_tol, z_exner_expl_rel_tol,
      z_exner_expl_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_49(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_rho_expl_k_size,
                                        const int z_exner_expl_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49::setup(
      mesh, k_size, stream, z_rho_expl_k_size, z_exner_expl_k_size);
}

void free_mo_solve_nonhydro_stencil_49() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_49::free();
}
}
