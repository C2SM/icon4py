#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_42.hpp"
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

class mo_solve_nonhydro_stencil_42 {
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
  double *z_w_expl_;
  double *w_nnow_;
  double *ddt_w_adv_ntl1_;
  double *ddt_w_adv_ntl2_;
  double *z_th_ddz_exner_c_;
  double *z_contr_w_fl_l_;
  double *rho_ic_;
  double *w_concorr_c_;
  double *vwind_expl_wgt_;
  double dtime_;
  double wgt_nnow_vel_;
  double wgt_nnew_vel_;
  double cpd_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_w_expl_kSize_;
  inline static int z_contr_w_fl_l_kSize_;

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

  static int get_z_w_expl_KSize() { return z_w_expl_kSize_; }

  static int get_z_contr_w_fl_l_KSize() { return z_contr_w_fl_l_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_w_expl_kSize,
                    const int z_contr_w_fl_l_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_w_expl_kSize_ = z_w_expl_kSize;
    z_contr_w_fl_l_kSize_ = z_contr_w_fl_l_kSize;
  }

  mo_solve_nonhydro_stencil_42() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_42 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_w_expl_sid = get_sid(
        z_w_expl_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_nnow_sid = get_sid(
        w_nnow_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddt_w_adv_ntl1_sid = get_sid(
        ddt_w_adv_ntl1_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddt_w_adv_ntl2_sid = get_sid(
        ddt_w_adv_ntl2_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_th_ddz_exner_c_sid = get_sid(
        z_th_ddz_exner_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_contr_w_fl_l_sid = get_sid(
        z_contr_w_fl_l_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_ic_sid = get_sid(
        rho_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_concorr_c_sid = get_sid(
        w_concorr_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto vwind_expl_wgt_sid = get_sid(
        vwind_expl_wgt_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    gridtools::stencil::global_parameter dtime_gp{dtime_};
    gridtools::stencil::global_parameter wgt_nnow_vel_gp{wgt_nnow_vel_};
    gridtools::stencil::global_parameter wgt_nnew_vel_gp{wgt_nnew_vel_};
    gridtools::stencil::global_parameter cpd_gp{cpd_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_42(connectivities)(
        cuda_backend, z_w_expl_sid, w_nnow_sid, ddt_w_adv_ntl1_sid,
        ddt_w_adv_ntl2_sid, z_th_ddz_exner_c_sid, z_contr_w_fl_l_sid,
        rho_ic_sid, w_concorr_c_sid, vwind_expl_wgt_sid, dtime_gp,
        wgt_nnow_vel_gp, wgt_nnew_vel_gp, cpd_gp, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_w_expl, double *w_nnow, double *ddt_w_adv_ntl1,
                     double *ddt_w_adv_ntl2, double *z_th_ddz_exner_c,
                     double *z_contr_w_fl_l, double *rho_ic,
                     double *w_concorr_c, double *vwind_expl_wgt, double dtime,
                     double wgt_nnow_vel, double wgt_nnew_vel, double cpd) {
    z_w_expl_ = z_w_expl;
    w_nnow_ = w_nnow;
    ddt_w_adv_ntl1_ = ddt_w_adv_ntl1;
    ddt_w_adv_ntl2_ = ddt_w_adv_ntl2;
    z_th_ddz_exner_c_ = z_th_ddz_exner_c;
    z_contr_w_fl_l_ = z_contr_w_fl_l;
    rho_ic_ = rho_ic;
    w_concorr_c_ = w_concorr_c;
    vwind_expl_wgt_ = vwind_expl_wgt;
    dtime_ = dtime;
    wgt_nnow_vel_ = wgt_nnow_vel;
    wgt_nnew_vel_ = wgt_nnew_vel;
    cpd_ = cpd;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_42(
    double *z_w_expl, double *w_nnow, double *ddt_w_adv_ntl1,
    double *ddt_w_adv_ntl2, double *z_th_ddz_exner_c, double *z_contr_w_fl_l,
    double *rho_ic, double *w_concorr_c, double *vwind_expl_wgt, double dtime,
    double wgt_nnow_vel, double wgt_nnew_vel, double cpd,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42 s;
  s.copy_pointers(z_w_expl, w_nnow, ddt_w_adv_ntl1, ddt_w_adv_ntl2,
                  z_th_ddz_exner_c, z_contr_w_fl_l, rho_ic, w_concorr_c,
                  vwind_expl_wgt, dtime, wgt_nnow_vel, wgt_nnew_vel, cpd);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_42(
    const double *z_w_expl_dsl, const double *z_w_expl,
    const double *z_contr_w_fl_l_dsl, const double *z_contr_w_fl_l,
    const double z_w_expl_rel_tol, const double z_w_expl_abs_tol,
    const double z_contr_w_fl_l_rel_tol, const double z_contr_w_fl_l_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_w_expl_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::
      get_z_w_expl_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_w_expl_kSize, z_w_expl_dsl, z_w_expl,
      "z_w_expl", z_w_expl_rel_tol, z_w_expl_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_w_expl(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_42", "z_w_expl");
  serialiser_z_w_expl.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_w_expl_kSize, (mesh.CellStride), z_w_expl,
        "mo_solve_nonhydro_stencil_42", "z_w_expl", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), z_w_expl_kSize, (mesh.CellStride), z_w_expl_dsl,
        "mo_solve_nonhydro_stencil_42", "z_w_expl_dsl", iteration);
    std::cout << "[DSL] serializing z_w_expl as error is high.\n" << std::flush;
#endif
  }
  int z_contr_w_fl_l_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_42::get_z_contr_w_fl_l_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_contr_w_fl_l_kSize, z_contr_w_fl_l_dsl,
      z_contr_w_fl_l, "z_contr_w_fl_l", z_contr_w_fl_l_rel_tol,
      z_contr_w_fl_l_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_contr_w_fl_l(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_42", "z_contr_w_fl_l");
  serialiser_z_contr_w_fl_l.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_contr_w_fl_l_kSize,
                          (mesh.CellStride), z_contr_w_fl_l,
                          "mo_solve_nonhydro_stencil_42", "z_contr_w_fl_l",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_contr_w_fl_l_kSize,
                          (mesh.CellStride), z_contr_w_fl_l_dsl,
                          "mo_solve_nonhydro_stencil_42", "z_contr_w_fl_l_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_contr_w_fl_l as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_42", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_42(
    double *z_w_expl, double *w_nnow, double *ddt_w_adv_ntl1,
    double *ddt_w_adv_ntl2, double *z_th_ddz_exner_c, double *z_contr_w_fl_l,
    double *rho_ic, double *w_concorr_c, double *vwind_expl_wgt, double dtime,
    double wgt_nnow_vel, double wgt_nnew_vel, double cpd,
    double *z_w_expl_before, double *z_contr_w_fl_l_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double z_w_expl_rel_tol,
    const double z_w_expl_abs_tol, const double z_contr_w_fl_l_rel_tol,
    const double z_contr_w_fl_l_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_42 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_42(
      z_w_expl_before, w_nnow, ddt_w_adv_ntl1, ddt_w_adv_ntl2, z_th_ddz_exner_c,
      z_contr_w_fl_l_before, rho_ic, w_concorr_c, vwind_expl_wgt, dtime,
      wgt_nnow_vel, wgt_nnew_vel, cpd, verticalStart, verticalEnd,
      horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_42 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_42...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_42(
      z_w_expl_before, z_w_expl, z_contr_w_fl_l_before, z_contr_w_fl_l,
      z_w_expl_rel_tol, z_w_expl_abs_tol, z_contr_w_fl_l_rel_tol,
      z_contr_w_fl_l_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_42(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_w_expl_k_size,
                                        const int z_contr_w_fl_l_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::setup(
      mesh, k_size, stream, z_w_expl_k_size, z_contr_w_fl_l_k_size);
}

void free_mo_solve_nonhydro_stencil_42() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_42::free();
}
}
