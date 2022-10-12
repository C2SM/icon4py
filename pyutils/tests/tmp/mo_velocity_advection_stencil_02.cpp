#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_02.hpp"
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

class mo_velocity_advection_stencil_02 {
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
  double *wgtfac_e_;
  double *vn_;
  double *vt_;
  double *vn_ie_;
  double *z_kin_hor_e_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int vn_ie_kSize_;
  inline static int z_kin_hor_e_kSize_;

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

  static int get_vn_ie_KSize() { return vn_ie_kSize_; }

  static int get_z_kin_hor_e_KSize() { return z_kin_hor_e_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int vn_ie_kSize,
                    const int z_kin_hor_e_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    vn_ie_kSize_ = vn_ie_kSize;
    z_kin_hor_e_kSize_ = z_kin_hor_e_kSize;
  }

  mo_velocity_advection_stencil_02() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_02 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto wgtfac_e_sid = get_sid(
        wgtfac_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto vn_sid = get_sid(
        vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto vt_sid = get_sid(
        vt_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto vn_ie_sid = get_sid(
        vn_ie_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_kin_hor_e_sid = get_sid(
        z_kin_hor_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_velocity_advection_stencil_02(connectivities)(
        cuda_backend, wgtfac_e_sid, vn_sid, vt_sid, vn_ie_sid, z_kin_hor_e_sid,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *wgtfac_e, double *vn, double *vt, double *vn_ie,
                     double *z_kin_hor_e) {
    wgtfac_e_ = wgtfac_e;
    vn_ = vn;
    vt_ = vt;
    vn_ie_ = vn_ie;
    z_kin_hor_e_ = z_kin_hor_e;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_02(
    double *wgtfac_e, double *vn, double *vt, double *vn_ie,
    double *z_kin_hor_e, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_02 s;
  s.copy_pointers(wgtfac_e, vn, vt, vn_ie, z_kin_hor_e);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_02(
    const double *vn_ie_dsl, const double *vn_ie, const double *z_kin_hor_e_dsl,
    const double *z_kin_hor_e, const double vn_ie_rel_tol,
    const double vn_ie_abs_tol, const double z_kin_hor_e_rel_tol,
    const double z_kin_hor_e_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int vn_ie_kSize = dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::
      get_vn_ie_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * vn_ie_kSize, vn_ie_dsl, vn_ie, "vn_ie",
      vn_ie_rel_tol, vn_ie_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_vn_ie(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_02", "vn_ie");
  serialiser_vn_ie.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), vn_ie_kSize, (mesh.EdgeStride), vn_ie,
        "mo_velocity_advection_stencil_02", "vn_ie", iteration);
    serialize_dense_edges(
        0, (mesh.NumEdges - 1), vn_ie_kSize, (mesh.EdgeStride), vn_ie_dsl,
        "mo_velocity_advection_stencil_02", "vn_ie_dsl", iteration);
    std::cout << "[DSL] serializing vn_ie as error is high.\n" << std::flush;
#endif
  }
  int z_kin_hor_e_kSize = dawn_generated::cuda_ico::
      mo_velocity_advection_stencil_02::get_z_kin_hor_e_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.EdgeStride) * z_kin_hor_e_kSize,
                           z_kin_hor_e_dsl, z_kin_hor_e, "z_kin_hor_e",
                           z_kin_hor_e_rel_tol, z_kin_hor_e_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_kin_hor_e(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_velocity_advection_stencil_02", "z_kin_hor_e");
  serialiser_z_kin_hor_e.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_kin_hor_e_kSize,
                          (mesh.EdgeStride), z_kin_hor_e,
                          "mo_velocity_advection_stencil_02", "z_kin_hor_e",
                          iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_kin_hor_e_kSize,
                          (mesh.EdgeStride), z_kin_hor_e_dsl,
                          "mo_velocity_advection_stencil_02", "z_kin_hor_e_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_kin_hor_e as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_02", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_02(
    double *wgtfac_e, double *vn, double *vt, double *vn_ie,
    double *z_kin_hor_e, double *vn_ie_before, double *z_kin_hor_e_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vn_ie_rel_tol,
    const double vn_ie_abs_tol, const double z_kin_hor_e_rel_tol,
    const double z_kin_hor_e_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_02 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_02(
      wgtfac_e, vn, vt, vn_ie_before, z_kin_hor_e_before, verticalStart,
      verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_02 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_02...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_02(
      vn_ie_before, vn_ie, z_kin_hor_e_before, z_kin_hor_e, vn_ie_rel_tol,
      vn_ie_abs_tol, z_kin_hor_e_rel_tol, z_kin_hor_e_abs_tol, iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_02(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int vn_ie_k_size,
                                            const int z_kin_hor_e_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::setup(
      mesh, k_size, stream, vn_ie_k_size, z_kin_hor_e_k_size);
}

void free_mo_velocity_advection_stencil_02() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_02::free();
}
}
