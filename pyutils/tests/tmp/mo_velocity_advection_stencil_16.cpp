#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_16.hpp"
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

class mo_velocity_advection_stencil_16 {
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
  double *z_w_con_c_;
  double *w_;
  double *coeff1_dwdz_;
  double *coeff2_dwdz_;
  double *ddt_w_adv_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int ddt_w_adv_kSize_;

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

  static int get_ddt_w_adv_KSize() { return ddt_w_adv_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int ddt_w_adv_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    ddt_w_adv_kSize_ = ddt_w_adv_kSize;
  }

  mo_velocity_advection_stencil_16() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_16 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto z_w_con_c_sid = get_sid(
        z_w_con_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto w_sid = get_sid(
        w_, gridtools::hymap::keys<
                unstructured::dim::horizontal,
                unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto coeff1_dwdz_sid = get_sid(
        coeff1_dwdz_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto coeff2_dwdz_sid = get_sid(
        coeff2_dwdz_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddt_w_adv_sid = get_sid(
        ddt_w_adv_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_velocity_advection_stencil_16(connectivities)(
        cuda_backend, z_w_con_c_sid, w_sid, coeff1_dwdz_sid, coeff2_dwdz_sid,
        ddt_w_adv_sid, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *z_w_con_c, double *w, double *coeff1_dwdz,
                     double *coeff2_dwdz, double *ddt_w_adv) {
    z_w_con_c_ = z_w_con_c;
    w_ = w;
    coeff1_dwdz_ = coeff1_dwdz;
    coeff2_dwdz_ = coeff2_dwdz;
    ddt_w_adv_ = ddt_w_adv;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_16(
    double *z_w_con_c, double *w, double *coeff1_dwdz, double *coeff2_dwdz,
    double *ddt_w_adv, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_16 s;
  s.copy_pointers(z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_16(const double *ddt_w_adv_dsl,
                                             const double *ddt_w_adv,
                                             const double ddt_w_adv_rel_tol,
                                             const double ddt_w_adv_abs_tol,
                                             const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_16::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_16::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_16::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int ddt_w_adv_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_16::get_ddt_w_adv_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * ddt_w_adv_kSize, ddt_w_adv_dsl, ddt_w_adv,
      "ddt_w_adv", ddt_w_adv_rel_tol, ddt_w_adv_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_ddt_w_adv(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_16", "ddt_w_adv");
  serialiser_ddt_w_adv.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(
        0, (mesh.NumCells - 1), ddt_w_adv_kSize, (mesh.CellStride), ddt_w_adv,
        "mo_velocity_advection_stencil_16", "ddt_w_adv", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), ddt_w_adv_kSize,
                          (mesh.CellStride), ddt_w_adv_dsl,
                          "mo_velocity_advection_stencil_16", "ddt_w_adv_dsl",
                          iteration);
    std::cout << "[DSL] serializing ddt_w_adv as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_16", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_16(
    double *z_w_con_c, double *w, double *coeff1_dwdz, double *coeff2_dwdz,
    double *ddt_w_adv, double *ddt_w_adv_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double ddt_w_adv_rel_tol, const double ddt_w_adv_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_16 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_16(
      z_w_con_c, w, coeff1_dwdz, coeff2_dwdz, ddt_w_adv_before, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_16 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_16...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_16(ddt_w_adv_before, ddt_w_adv,
                                          ddt_w_adv_rel_tol, ddt_w_adv_abs_tol,
                                          iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_16(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int ddt_w_adv_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_16::setup(
      mesh, k_size, stream, ddt_w_adv_k_size);
}

void free_mo_velocity_advection_stencil_16() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_16::free();
}
}
