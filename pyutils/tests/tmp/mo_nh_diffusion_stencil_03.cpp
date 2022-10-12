#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_03.hpp"
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

class mo_nh_diffusion_stencil_03 {
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
  double *div_;
  double *kh_c_;
  double *wgtfac_c_;
  double *div_ic_;
  double *hdef_ic_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int div_ic_kSize_;
  inline static int hdef_ic_kSize_;

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

  static int get_div_ic_KSize() { return div_ic_kSize_; }

  static int get_hdef_ic_KSize() { return hdef_ic_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int div_ic_kSize,
                    const int hdef_ic_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    div_ic_kSize_ = div_ic_kSize;
    hdef_ic_kSize_ = hdef_ic_kSize;
  }

  mo_nh_diffusion_stencil_03() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_03 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto div_sid = get_sid(
        div_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto kh_c_sid = get_sid(
        kh_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto wgtfac_c_sid = get_sid(
        wgtfac_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto div_ic_sid = get_sid(
        div_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto hdef_ic_sid = get_sid(
        hdef_ic_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    auto connectivities = gridtools::hymap::keys<>::make_values();
    generated::mo_nh_diffusion_stencil_03(connectivities)(
        cuda_backend, div_sid, kh_c_sid, wgtfac_c_sid, div_ic_sid, hdef_ic_sid,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *div, double *kh_c, double *wgtfac_c,
                     double *div_ic, double *hdef_ic) {
    div_ = div;
    kh_c_ = kh_c;
    wgtfac_c_ = wgtfac_c;
    div_ic_ = div_ic;
    hdef_ic_ = hdef_ic;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_03(double *div, double *kh_c, double *wgtfac_c,
                                    double *div_ic, double *hdef_ic,
                                    const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03 s;
  s.copy_pointers(div, kh_c, wgtfac_c, div_ic, hdef_ic);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_03(
    const double *div_ic_dsl, const double *div_ic, const double *hdef_ic_dsl,
    const double *hdef_ic, const double div_ic_rel_tol,
    const double div_ic_abs_tol, const double hdef_ic_rel_tol,
    const double hdef_ic_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int div_ic_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::get_div_ic_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * div_ic_kSize, div_ic_dsl, div_ic, "div_ic",
      div_ic_rel_tol, div_ic_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_div_ic(stencilMetrics,
                                      metricsNameFromEnvVar("SLURM_JOB_ID"),
                                      "mo_nh_diffusion_stencil_03", "div_ic");
  serialiser_div_ic.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), div_ic_kSize,
                          (mesh.CellStride), div_ic,
                          "mo_nh_diffusion_stencil_03", "div_ic", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), div_ic_kSize, (mesh.CellStride), div_ic_dsl,
        "mo_nh_diffusion_stencil_03", "div_ic_dsl", iteration);
    std::cout << "[DSL] serializing div_ic as error is high.\n" << std::flush;
#endif
  }
  int hdef_ic_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::get_hdef_ic_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * hdef_ic_kSize, hdef_ic_dsl, hdef_ic,
      "hdef_ic", hdef_ic_rel_tol, hdef_ic_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_hdef_ic(stencilMetrics,
                                       metricsNameFromEnvVar("SLURM_JOB_ID"),
                                       "mo_nh_diffusion_stencil_03", "hdef_ic");
  serialiser_hdef_ic.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), hdef_ic_kSize,
                          (mesh.CellStride), hdef_ic,
                          "mo_nh_diffusion_stencil_03", "hdef_ic", iteration);
    serialize_dense_cells(
        0, (mesh.NumCells - 1), hdef_ic_kSize, (mesh.CellStride), hdef_ic_dsl,
        "mo_nh_diffusion_stencil_03", "hdef_ic_dsl", iteration);
    std::cout << "[DSL] serializing hdef_ic as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_03", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_03(
    double *div, double *kh_c, double *wgtfac_c, double *div_ic,
    double *hdef_ic, double *div_ic_before, double *hdef_ic_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double div_ic_rel_tol,
    const double div_ic_abs_tol, const double hdef_ic_rel_tol,
    const double hdef_ic_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_03 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_03(div, kh_c, wgtfac_c, div_ic_before,
                                 hdef_ic_before, verticalStart, verticalEnd,
                                 horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_03 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_03...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_03(
      div_ic_before, div_ic, hdef_ic_before, hdef_ic, div_ic_rel_tol,
      div_ic_abs_tol, hdef_ic_rel_tol, hdef_ic_abs_tol, iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_03(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int div_ic_k_size,
                                      const int hdef_ic_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::setup(
      mesh, k_size, stream, div_ic_k_size, hdef_ic_k_size);
}

void free_mo_nh_diffusion_stencil_03() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_03::free();
}
}
