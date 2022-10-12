#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_nh_diffusion_stencil_02.hpp"
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

class mo_nh_diffusion_stencil_02 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ceTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      ceTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Cells, dawn::LocationType::Edges}, 0});
    }
  };

private:
  double *kh_smag_ec_;
  double *vn_;
  double *e_bln_c_s_;
  double *geofac_div_;
  double *diff_multfac_smag_;
  double *kh_c_;
  double *div_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int kh_c_kSize_;
  inline static int div_kSize_;

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

  static int get_kh_c_KSize() { return kh_c_kSize_; }

  static int get_div_KSize() { return div_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int kh_c_kSize,
                    const int div_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    kh_c_kSize_ = kh_c_kSize;
    div_kSize_ = div_kSize;
  }

  mo_nh_diffusion_stencil_02() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_nh_diffusion_stencil_02 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto kh_smag_ec_sid = get_sid(
        kh_smag_ec_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto vn_sid = get_sid(
        vn_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto diff_multfac_smag_sid = get_sid(
        diff_multfac_smag_,
        gridtools::hymap::keys<unstructured::dim::vertical>::make_values(1));

    auto kh_c_sid = get_sid(
        kh_c_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto div_sid = get_sid(
        div_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<3> ce_ptr{.raw_ptr_fortran = mesh_.ceTable};
    auto connectivities =
        gridtools::hymap::keys<generated::C2E_t>::make_values(ce_ptr);
    double *e_bln_c_s_0 = &e_bln_c_s_[0 * mesh_.CellStride];
    double *e_bln_c_s_1 = &e_bln_c_s_[1 * mesh_.CellStride];
    double *e_bln_c_s_2 = &e_bln_c_s_[2 * mesh_.CellStride];
    auto e_bln_c_s_sid_0 = get_sid(
        e_bln_c_s_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_bln_c_s_sid_1 = get_sid(
        e_bln_c_s_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_bln_c_s_sid_2 = get_sid(
        e_bln_c_s_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto e_bln_c_s_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>>::make_values(e_bln_c_s_sid_0,
                                                e_bln_c_s_sid_1,
                                                e_bln_c_s_sid_2);
    double *geofac_div_0 = &geofac_div_[0 * mesh_.CellStride];
    double *geofac_div_1 = &geofac_div_[1 * mesh_.CellStride];
    double *geofac_div_2 = &geofac_div_[2 * mesh_.CellStride];
    auto geofac_div_sid_0 = get_sid(
        geofac_div_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_div_sid_1 = get_sid(
        geofac_div_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_div_sid_2 = get_sid(
        geofac_div_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_div_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>>::make_values(geofac_div_sid_0,
                                                geofac_div_sid_1,
                                                geofac_div_sid_2);
    generated::mo_nh_diffusion_stencil_02(connectivities)(
        cuda_backend, kh_smag_ec_sid, vn_sid, e_bln_c_s_sid_comp,
        geofac_div_sid_comp, diff_multfac_smag_sid, kh_c_sid, div_sid,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *kh_smag_ec, double *vn, double *e_bln_c_s,
                     double *geofac_div, double *diff_multfac_smag,
                     double *kh_c, double *div) {
    kh_smag_ec_ = kh_smag_ec;
    vn_ = vn;
    e_bln_c_s_ = e_bln_c_s;
    geofac_div_ = geofac_div;
    diff_multfac_smag_ = diff_multfac_smag;
    kh_c_ = kh_c;
    div_ = div;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_nh_diffusion_stencil_02(double *kh_smag_ec, double *vn,
                                    double *e_bln_c_s, double *geofac_div,
                                    double *diff_multfac_smag, double *kh_c,
                                    double *div, const int verticalStart,
                                    const int verticalEnd,
                                    const int horizontalStart,
                                    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02 s;
  s.copy_pointers(kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag,
                  kh_c, div);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_nh_diffusion_stencil_02(
    const double *kh_c_dsl, const double *kh_c, const double *div_dsl,
    const double *div, const double kh_c_rel_tol, const double kh_c_abs_tol,
    const double div_rel_tol, const double div_abs_tol, const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::getStream();
  int kSize = dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int kh_c_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::get_kh_c_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * kh_c_kSize, kh_c_dsl,
                           kh_c, "kh_c", kh_c_rel_tol, kh_c_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_kh_c(stencilMetrics,
                                    metricsNameFromEnvVar("SLURM_JOB_ID"),
                                    "mo_nh_diffusion_stencil_02", "kh_c");
  serialiser_kh_c.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), kh_c_kSize, (mesh.CellStride),
                          kh_c, "mo_nh_diffusion_stencil_02", "kh_c",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), kh_c_kSize, (mesh.CellStride),
                          kh_c_dsl, "mo_nh_diffusion_stencil_02", "kh_c_dsl",
                          iteration);
    std::cout << "[DSL] serializing kh_c as error is high.\n" << std::flush;
#endif
  }
  int div_kSize =
      dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::get_div_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.CellStride) * div_kSize, div_dsl, div,
                           "div", div_rel_tol, div_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_div(stencilMetrics,
                                   metricsNameFromEnvVar("SLURM_JOB_ID"),
                                   "mo_nh_diffusion_stencil_02", "div");
  serialiser_div.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), div_kSize, (mesh.CellStride),
                          div, "mo_nh_diffusion_stencil_02", "div", iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), div_kSize, (mesh.CellStride),
                          div_dsl, "mo_nh_diffusion_stencil_02", "div_dsl",
                          iteration);
    std::cout << "[DSL] serializing div as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_nh_diffusion_stencil_02", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_nh_diffusion_stencil_02(
    double *kh_smag_ec, double *vn, double *e_bln_c_s, double *geofac_div,
    double *diff_multfac_smag, double *kh_c, double *div, double *kh_c_before,
    double *div_before, const int verticalStart, const int verticalEnd,
    const int horizontalStart, const int horizontalEnd,
    const double kh_c_rel_tol, const double kh_c_abs_tol,
    const double div_rel_tol, const double div_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_nh_diffusion_stencil_02 (" << iteration
            << ") ...\n"
            << std::flush;
  run_mo_nh_diffusion_stencil_02(
      kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag, kh_c_before,
      div_before, verticalStart, verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_nh_diffusion_stencil_02 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_nh_diffusion_stencil_02...\n"
            << std::flush;
  verify_mo_nh_diffusion_stencil_02(kh_c_before, kh_c, div_before, div,
                                    kh_c_rel_tol, kh_c_abs_tol, div_rel_tol,
                                    div_abs_tol, iteration);

  iteration++;
}

void setup_mo_nh_diffusion_stencil_02(dawn::GlobalGpuTriMesh *mesh, int k_size,
                                      cudaStream_t stream,
                                      const int kh_c_k_size,
                                      const int div_k_size) {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::setup(
      mesh, k_size, stream, kh_c_k_size, div_k_size);
}

void free_mo_nh_diffusion_stencil_02() {
  dawn_generated::cuda_ico::mo_nh_diffusion_stencil_02::free();
}
}
