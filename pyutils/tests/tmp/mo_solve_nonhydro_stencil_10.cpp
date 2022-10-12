#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_10.hpp"
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

class mo_solve_nonhydro_stencil_10 {
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
  double *w_;
  double *w_concorr_c_;
  double *ddqz_z_half_;
  double *rho_now_;
  double *rho_var_;
  double *theta_now_;
  double *theta_var_;
  double *wgtfac_c_;
  double *theta_ref_mc_;
  double *vwind_expl_wgt_;
  double *exner_pr_;
  double *d_exner_dz_ref_ic_;
  double *rho_ic_;
  double *z_theta_v_pr_ic_;
  double *theta_v_ic_;
  double *z_th_ddz_exner_c_;
  double dtime_;
  double wgt_nnow_rth_;
  double wgt_nnew_rth_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int rho_ic_kSize_;
  inline static int z_theta_v_pr_ic_kSize_;
  inline static int theta_v_ic_kSize_;
  inline static int z_th_ddz_exner_c_kSize_;

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

  static int get_rho_ic_KSize() { return rho_ic_kSize_; }

  static int get_z_theta_v_pr_ic_KSize() { return z_theta_v_pr_ic_kSize_; }

  static int get_theta_v_ic_KSize() { return theta_v_ic_kSize_; }

  static int get_z_th_ddz_exner_c_KSize() { return z_th_ddz_exner_c_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int rho_ic_kSize,
                    const int z_theta_v_pr_ic_kSize, const int theta_v_ic_kSize,
                    const int z_th_ddz_exner_c_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    rho_ic_kSize_ = rho_ic_kSize;
    z_theta_v_pr_ic_kSize_ = z_theta_v_pr_ic_kSize;
    theta_v_ic_kSize_ = theta_v_ic_kSize;
    z_th_ddz_exner_c_kSize_ = z_th_ddz_exner_c_kSize;
  }

  mo_solve_nonhydro_stencil_10() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_10 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto w_sid = get_sid(
        w_, gridtools::hymap::keys<
                unstructured::dim::horizontal,
                unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_concorr_c_sid = get_sid(
        w_concorr_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddqz_z_half_sid = get_sid(
        ddqz_z_half_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_now_sid = get_sid(
        rho_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_var_sid = get_sid(
        rho_var_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_now_sid = get_sid(
        theta_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_var_sid = get_sid(
        theta_var_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto wgtfac_c_sid = get_sid(
        wgtfac_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_ref_mc_sid = get_sid(
        theta_ref_mc_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto vwind_expl_wgt_sid = get_sid(
        vwind_expl_wgt_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto exner_pr_sid = get_sid(
        exner_pr_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto d_exner_dz_ref_ic_sid = get_sid(
        d_exner_dz_ref_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_ic_sid = get_sid(
        rho_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_theta_v_pr_ic_sid = get_sid(
        z_theta_v_pr_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_ic_sid = get_sid(
        theta_v_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_th_ddz_exner_c_sid = get_sid(
        z_th_ddz_exner_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    gridtools::stencil::global_parameter wgt_nnow_rth_gp{wgt_nnow_rth_};
    gridtools::stencil::global_parameter wgt_nnew_rth_gp{wgt_nnew_rth_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_10(connectivities)(
        cuda_backend, w_sid, w_concorr_c_sid, ddqz_z_half_sid, rho_now_sid,
        rho_var_sid, theta_now_sid, theta_var_sid, wgtfac_c_sid,
        theta_ref_mc_sid, vwind_expl_wgt_sid, exner_pr_sid,
        d_exner_dz_ref_ic_sid, rho_ic_sid, z_theta_v_pr_ic_sid, theta_v_ic_sid,
        z_th_ddz_exner_c_sid, dtime_gp, wgt_nnow_rth_gp, wgt_nnew_rth_gp,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *w, double *w_concorr_c, double *ddqz_z_half,
                     double *rho_now, double *rho_var, double *theta_now,
                     double *theta_var, double *wgtfac_c, double *theta_ref_mc,
                     double *vwind_expl_wgt, double *exner_pr,
                     double *d_exner_dz_ref_ic, double *rho_ic,
                     double *z_theta_v_pr_ic, double *theta_v_ic,
                     double *z_th_ddz_exner_c, double dtime,
                     double wgt_nnow_rth, double wgt_nnew_rth) {
    w_ = w;
    w_concorr_c_ = w_concorr_c;
    ddqz_z_half_ = ddqz_z_half;
    rho_now_ = rho_now;
    rho_var_ = rho_var;
    theta_now_ = theta_now;
    theta_var_ = theta_var;
    wgtfac_c_ = wgtfac_c;
    theta_ref_mc_ = theta_ref_mc;
    vwind_expl_wgt_ = vwind_expl_wgt;
    exner_pr_ = exner_pr;
    d_exner_dz_ref_ic_ = d_exner_dz_ref_ic;
    rho_ic_ = rho_ic;
    z_theta_v_pr_ic_ = z_theta_v_pr_ic;
    theta_v_ic_ = theta_v_ic;
    z_th_ddz_exner_c_ = z_th_ddz_exner_c;
    dtime_ = dtime;
    wgt_nnow_rth_ = wgt_nnow_rth;
    wgt_nnew_rth_ = wgt_nnew_rth;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_10(
    double *w, double *w_concorr_c, double *ddqz_z_half, double *rho_now,
    double *rho_var, double *theta_now, double *theta_var, double *wgtfac_c,
    double *theta_ref_mc, double *vwind_expl_wgt, double *exner_pr,
    double *d_exner_dz_ref_ic, double *rho_ic, double *z_theta_v_pr_ic,
    double *theta_v_ic, double *z_th_ddz_exner_c, double dtime,
    double wgt_nnow_rth, double wgt_nnew_rth, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10 s;
  s.copy_pointers(w, w_concorr_c, ddqz_z_half, rho_now, rho_var, theta_now,
                  theta_var, wgtfac_c, theta_ref_mc, vwind_expl_wgt, exner_pr,
                  d_exner_dz_ref_ic, rho_ic, z_theta_v_pr_ic, theta_v_ic,
                  z_th_ddz_exner_c, dtime, wgt_nnow_rth, wgt_nnew_rth);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_10(
    const double *rho_ic_dsl, const double *rho_ic,
    const double *z_theta_v_pr_ic_dsl, const double *z_theta_v_pr_ic,
    const double *theta_v_ic_dsl, const double *theta_v_ic,
    const double *z_th_ddz_exner_c_dsl, const double *z_th_ddz_exner_c,
    const double rho_ic_rel_tol, const double rho_ic_abs_tol,
    const double z_theta_v_pr_ic_rel_tol, const double z_theta_v_pr_ic_abs_tol,
    const double theta_v_ic_rel_tol, const double theta_v_ic_abs_tol,
    const double z_th_ddz_exner_c_rel_tol,
    const double z_th_ddz_exner_c_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int rho_ic_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::
      get_rho_ic_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * rho_ic_kSize, rho_ic_dsl, rho_ic, "rho_ic",
      rho_ic_rel_tol, rho_ic_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_rho_ic(stencilMetrics,
                                      metricsNameFromEnvVar("SLURM_JOB_ID"),
                                      "mo_solve_nonhydro_stencil_10", "rho_ic");
  serialiser_rho_ic.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), rho_ic_kSize,
                          (mesh.CellStride), rho_ic,
                          "mo_solve_nonhydro_stencil_10", "rho_ic", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), rho_ic_kSize, (mesh.CellStride), rho_ic_dsl,
        "mo_solve_nonhydro_stencil_10", "rho_ic_dsl", iteration);
    std::cout << "[DSL] serializing rho_ic as error is high.\n" << std::flush;
#endif
  }
  int z_theta_v_pr_ic_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_10::get_z_theta_v_pr_ic_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_theta_v_pr_ic_kSize, z_theta_v_pr_ic_dsl,
      z_theta_v_pr_ic, "z_theta_v_pr_ic", z_theta_v_pr_ic_rel_tol,
      z_theta_v_pr_ic_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_theta_v_pr_ic(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_10", "z_theta_v_pr_ic");
  serialiser_z_theta_v_pr_ic.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_theta_v_pr_ic_kSize,
                          (mesh.CellStride), z_theta_v_pr_ic,
                          "mo_solve_nonhydro_stencil_10", "z_theta_v_pr_ic",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_theta_v_pr_ic_kSize,
                          (mesh.CellStride), z_theta_v_pr_ic_dsl,
                          "mo_solve_nonhydro_stencil_10", "z_theta_v_pr_ic_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_theta_v_pr_ic as error is high.\n"
              << std::flush;
#endif
  }
  int theta_v_ic_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_10::get_theta_v_ic_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * theta_v_ic_kSize, theta_v_ic_dsl, theta_v_ic,
      "theta_v_ic", theta_v_ic_rel_tol, theta_v_ic_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_theta_v_ic(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_10", "theta_v_ic");
  serialiser_theta_v_ic.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), theta_v_ic_kSize, (mesh.CellStride), theta_v_ic,
        "mo_solve_nonhydro_stencil_10", "theta_v_ic", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), theta_v_ic_kSize,
                          (mesh.CellStride), theta_v_ic_dsl,
                          "mo_solve_nonhydro_stencil_10", "theta_v_ic_dsl",
                          iteration);
    std::cout << "[DSL] serializing theta_v_ic as error is high.\n"
              << std::flush;
#endif
  }
  int z_th_ddz_exner_c_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_10::get_z_th_ddz_exner_c_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_th_ddz_exner_c_kSize, z_th_ddz_exner_c_dsl,
      z_th_ddz_exner_c, "z_th_ddz_exner_c", z_th_ddz_exner_c_rel_tol,
      z_th_ddz_exner_c_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_th_ddz_exner_c(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_10", "z_th_ddz_exner_c");
  serialiser_z_th_ddz_exner_c.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_th_ddz_exner_c_kSize,
                          (mesh.CellStride), z_th_ddz_exner_c,
                          "mo_solve_nonhydro_stencil_10", "z_th_ddz_exner_c",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_th_ddz_exner_c_kSize,
                          (mesh.CellStride), z_th_ddz_exner_c_dsl,
                          "mo_solve_nonhydro_stencil_10",
                          "z_th_ddz_exner_c_dsl", iteration);
    std::cout << "[DSL] serializing z_th_ddz_exner_c as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_10", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_10(
    double *w, double *w_concorr_c, double *ddqz_z_half, double *rho_now,
    double *rho_var, double *theta_now, double *theta_var, double *wgtfac_c,
    double *theta_ref_mc, double *vwind_expl_wgt, double *exner_pr,
    double *d_exner_dz_ref_ic, double *rho_ic, double *z_theta_v_pr_ic,
    double *theta_v_ic, double *z_th_ddz_exner_c, double dtime,
    double wgt_nnow_rth, double wgt_nnew_rth, double *rho_ic_before,
    double *z_theta_v_pr_ic_before, double *theta_v_ic_before,
    double *z_th_ddz_exner_c_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double rho_ic_rel_tol, const double rho_ic_abs_tol,
    const double z_theta_v_pr_ic_rel_tol, const double z_theta_v_pr_ic_abs_tol,
    const double theta_v_ic_rel_tol, const double theta_v_ic_abs_tol,
    const double z_th_ddz_exner_c_rel_tol,
    const double z_th_ddz_exner_c_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_10 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_10(
      w, w_concorr_c, ddqz_z_half, rho_now, rho_var, theta_now, theta_var,
      wgtfac_c, theta_ref_mc, vwind_expl_wgt, exner_pr, d_exner_dz_ref_ic,
      rho_ic_before, z_theta_v_pr_ic_before, theta_v_ic_before,
      z_th_ddz_exner_c_before, dtime, wgt_nnow_rth, wgt_nnew_rth, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_10 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_10...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_10(
      rho_ic_before, rho_ic, z_theta_v_pr_ic_before, z_theta_v_pr_ic,
      theta_v_ic_before, theta_v_ic, z_th_ddz_exner_c_before, z_th_ddz_exner_c,
      rho_ic_rel_tol, rho_ic_abs_tol, z_theta_v_pr_ic_rel_tol,
      z_theta_v_pr_ic_abs_tol, theta_v_ic_rel_tol, theta_v_ic_abs_tol,
      z_th_ddz_exner_c_rel_tol, z_th_ddz_exner_c_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_10(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int rho_ic_k_size,
                                        const int z_theta_v_pr_ic_k_size,
                                        const int theta_v_ic_k_size,
                                        const int z_th_ddz_exner_c_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::setup(
      mesh, k_size, stream, rho_ic_k_size, z_theta_v_pr_ic_k_size,
      theta_v_ic_k_size, z_th_ddz_exner_c_k_size);
}

void free_mo_solve_nonhydro_stencil_10() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_10::free();
}
}
