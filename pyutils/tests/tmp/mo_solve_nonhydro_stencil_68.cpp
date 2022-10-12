#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_68.hpp"
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

class mo_solve_nonhydro_stencil_68 {
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
  int *mask_prog_halo_c_;
  double *rho_now_;
  double *theta_v_now_;
  double *exner_new_;
  double *exner_now_;
  double *rho_new_;
  double *theta_v_new_;
  double cvd_o_rd_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int theta_v_new_kSize_;

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

  static int get_theta_v_new_KSize() { return theta_v_new_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int theta_v_new_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    theta_v_new_kSize_ = theta_v_new_kSize;
  }

  mo_solve_nonhydro_stencil_68() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_68 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto mask_prog_halo_c_sid = get_sid(
        mask_prog_halo_c_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto rho_now_sid = get_sid(
        rho_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_now_sid = get_sid(
        theta_v_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_new_sid = get_sid(
        exner_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto exner_now_sid = get_sid(
        exner_now_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto rho_new_sid = get_sid(
        rho_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto theta_v_new_sid = get_sid(
        theta_v_new_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    gridtools::stencil::global_parameter cvd_o_rd_gp{cvd_o_rd_};
    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_solve_nonhydro_stencil_68(connectivities)(
        cuda_backend, mask_prog_halo_c_sid, rho_now_sid, theta_v_now_sid,
        exner_new_sid, exner_now_sid, rho_new_sid, theta_v_new_sid, cvd_o_rd_gp,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(int *mask_prog_halo_c, double *rho_now,
                     double *theta_v_now, double *exner_new, double *exner_now,
                     double *rho_new, double *theta_v_new, double cvd_o_rd) {
    mask_prog_halo_c_ = mask_prog_halo_c;
    rho_now_ = rho_now;
    theta_v_now_ = theta_v_now;
    exner_new_ = exner_new;
    exner_now_ = exner_now;
    rho_new_ = rho_new;
    theta_v_new_ = theta_v_new;
    cvd_o_rd_ = cvd_o_rd;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_68(
    int *mask_prog_halo_c, double *rho_now, double *theta_v_now,
    double *exner_new, double *exner_now, double *rho_new, double *theta_v_new,
    double cvd_o_rd, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68 s;
  s.copy_pointers(mask_prog_halo_c, rho_now, theta_v_now, exner_new, exner_now,
                  rho_new, theta_v_new, cvd_o_rd);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_68(const double *theta_v_new_dsl,
                                         const double *theta_v_new,
                                         const double theta_v_new_rel_tol,
                                         const double theta_v_new_abs_tol,
                                         const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int theta_v_new_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_68::get_theta_v_new_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * theta_v_new_kSize,
                           theta_v_new_dsl, theta_v_new, "theta_v_new",
                           theta_v_new_rel_tol, theta_v_new_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_theta_v_new(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_68", "theta_v_new");
  serialiser_theta_v_new.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), theta_v_new_kSize, (mesh.CellStride),
        theta_v_new, "mo_solve_nonhydro_stencil_68", "theta_v_new", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), theta_v_new_kSize,
                          (mesh.CellStride), theta_v_new_dsl,
                          "mo_solve_nonhydro_stencil_68", "theta_v_new_dsl",
                          iteration);
    std::cout << "[DSL] serializing theta_v_new as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_68", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_68(
    int *mask_prog_halo_c, double *rho_now, double *theta_v_now,
    double *exner_new, double *exner_now, double *rho_new, double *theta_v_new,
    double cvd_o_rd, double *theta_v_new_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double theta_v_new_rel_tol, const double theta_v_new_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_68 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_68(mask_prog_halo_c, rho_now, theta_v_now,
                                   exner_new, exner_now, rho_new,
                                   theta_v_new_before, cvd_o_rd, verticalStart,
                                   verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_68 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_68...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_68(theta_v_new_before, theta_v_new,
                                      theta_v_new_rel_tol, theta_v_new_abs_tol,
                                      iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_68(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int theta_v_new_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68::setup(
      mesh, k_size, stream, theta_v_new_k_size);
}

void free_mo_solve_nonhydro_stencil_68() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_68::free();
}
}
