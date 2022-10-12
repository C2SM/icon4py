#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_34.hpp"
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

class mo_solve_nonhydro_stencil_34 {
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
  double *z_vn_avg_;
  double *mass_fl_e_;
  double *vn_traj_;
  double *mass_flx_me_;
  double r_nsubsteps_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int vn_traj_kSize_;
  inline static int mass_flx_me_kSize_;

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

  static int get_vn_traj_KSize() { return vn_traj_kSize_; }

  static int get_mass_flx_me_KSize() { return mass_flx_me_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int vn_traj_kSize,
                    const int mass_flx_me_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    vn_traj_kSize_ = vn_traj_kSize;
    mass_flx_me_kSize_ = mass_flx_me_kSize;
  }

  mo_solve_nonhydro_stencil_34() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_34 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_vn_avg_sid = get_sid(
        z_vn_avg_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto mass_fl_e_sid = get_sid(
        mass_fl_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto vn_traj_sid = get_sid(
        vn_traj_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto mass_flx_me_sid = get_sid(
        mass_flx_me_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    gridtools::stencil::global_parameter r_nsubsteps_gp{r_nsubsteps_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_34(connectivities)(
        cuda_backend, z_vn_avg_sid, mass_fl_e_sid, vn_traj_sid, mass_flx_me_sid,
        r_nsubsteps_gp, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_vn_avg, double *mass_fl_e, double *vn_traj,
                     double *mass_flx_me, double r_nsubsteps) {
    z_vn_avg_ = z_vn_avg;
    mass_fl_e_ = mass_fl_e;
    vn_traj_ = vn_traj;
    mass_flx_me_ = mass_flx_me;
    r_nsubsteps_ = r_nsubsteps;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_34(
    double *z_vn_avg, double *mass_fl_e, double *vn_traj, double *mass_flx_me,
    double r_nsubsteps, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34 s;
  s.copy_pointers(z_vn_avg, mass_fl_e, vn_traj, mass_flx_me, r_nsubsteps);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_34(
    const double *vn_traj_dsl, const double *vn_traj,
    const double *mass_flx_me_dsl, const double *mass_flx_me,
    const double vn_traj_rel_tol, const double vn_traj_abs_tol,
    const double mass_flx_me_rel_tol, const double mass_flx_me_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int vn_traj_kSize = dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::
      get_vn_traj_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * vn_traj_kSize, vn_traj_dsl, vn_traj,
      "vn_traj", vn_traj_rel_tol, vn_traj_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_vn_traj(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_34", "vn_traj");
  serialiser_vn_traj.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), vn_traj_kSize,
                          (mesh.EdgeStride), vn_traj,
                          "mo_solve_nonhydro_stencil_34", "vn_traj", iteration);
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), vn_traj_kSize, (mesh.EdgeStride), vn_traj_dsl,
        "mo_solve_nonhydro_stencil_34", "vn_traj_dsl", iteration);
    std::cout << "[DSL] serializing vn_traj as error is high.\n" << std::flush;
#endif
  }
  int mass_flx_me_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_34::get_mass_flx_me_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.EdgeStride) * mass_flx_me_kSize,
                           mass_flx_me_dsl, mass_flx_me, "mass_flx_me",
                           mass_flx_me_rel_tol, mass_flx_me_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_mass_flx_me(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_34", "mass_flx_me");
  serialiser_mass_flx_me.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), mass_flx_me_kSize, (mesh.EdgeStride),
        mass_flx_me, "mo_solve_nonhydro_stencil_34", "mass_flx_me", iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), mass_flx_me_kSize,
                          (mesh.EdgeStride), mass_flx_me_dsl,
                          "mo_solve_nonhydro_stencil_34", "mass_flx_me_dsl",
                          iteration);
    std::cout << "[DSL] serializing mass_flx_me as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_34", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_34(
    double *z_vn_avg, double *mass_fl_e, double *vn_traj, double *mass_flx_me,
    double r_nsubsteps, double *vn_traj_before, double *mass_flx_me_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_traj_rel_tol,
    const double vn_traj_abs_tol, const double mass_flx_me_rel_tol,
    const double mass_flx_me_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_34 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_34(
      z_vn_avg, mass_fl_e, vn_traj_before, mass_flx_me_before, r_nsubsteps,
      verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_34 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_34...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_34(
      vn_traj_before, vn_traj, mass_flx_me_before, mass_flx_me, vn_traj_rel_tol,
      vn_traj_abs_tol, mass_flx_me_rel_tol, mass_flx_me_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_34(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int vn_traj_k_size,
                                        const int mass_flx_me_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::setup(
      mesh, k_size, stream, vn_traj_k_size, mass_flx_me_k_size);
}

void free_mo_solve_nonhydro_stencil_34() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_34::free();
}
}
