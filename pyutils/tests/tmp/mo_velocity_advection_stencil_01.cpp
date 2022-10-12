#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_velocity_advection_stencil_01.hpp"
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

class mo_velocity_advection_stencil_01 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *eceTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      eceTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells,
               dawn::LocationType::Edges},
              0});
    }
  };

private:
  double *vn_;
  double *rbf_vec_coeff_e_;
  double *vt_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int vt_kSize_;

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

  static int get_vt_KSize() { return vt_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int vt_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    vt_kSize_ = vt_kSize;
  }

  mo_velocity_advection_stencil_01() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_velocity_advection_stencil_01 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

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

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<4> ece_ptr{.raw_ptr_fortran = mesh_.eceTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C2E_t>::make_values(ece_ptr);
    double *rbf_vec_coeff_e_0 = &rbf_vec_coeff_e_[0 * mesh_.EdgeStride];
    double *rbf_vec_coeff_e_1 = &rbf_vec_coeff_e_[1 * mesh_.EdgeStride];
    double *rbf_vec_coeff_e_2 = &rbf_vec_coeff_e_[2 * mesh_.EdgeStride];
    double *rbf_vec_coeff_e_3 = &rbf_vec_coeff_e_[3 * mesh_.EdgeStride];
    auto rbf_vec_coeff_e_sid_0 = get_sid(
        rbf_vec_coeff_e_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto rbf_vec_coeff_e_sid_1 = get_sid(
        rbf_vec_coeff_e_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto rbf_vec_coeff_e_sid_2 = get_sid(
        rbf_vec_coeff_e_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto rbf_vec_coeff_e_sid_3 = get_sid(
        rbf_vec_coeff_e_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto rbf_vec_coeff_e_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>,
        integral_constant<int, 3>>::make_values(rbf_vec_coeff_e_sid_0,
                                                rbf_vec_coeff_e_sid_1,
                                                rbf_vec_coeff_e_sid_2,
                                                rbf_vec_coeff_e_sid_3);
    generated::mo_velocity_advection_stencil_01(connectivities)(
        cuda_backend, vn_sid, rbf_vec_coeff_e_sid_comp, vt_sid, horizontalStart,
        horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *vn, double *rbf_vec_coeff_e, double *vt) {
    vn_ = vn;
    rbf_vec_coeff_e_ = rbf_vec_coeff_e;
    vt_ = vt;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_velocity_advection_stencil_01(double *vn, double *rbf_vec_coeff_e,
                                          double *vt, const int verticalStart,
                                          const int verticalEnd,
                                          const int horizontalStart,
                                          const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_01 s;
  s.copy_pointers(vn, rbf_vec_coeff_e, vt);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_velocity_advection_stencil_01(const double *vt_dsl,
                                             const double *vt,
                                             const double vt_rel_tol,
                                             const double vt_abs_tol,
                                             const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int vt_kSize = dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::
      get_vt_KSize();
  stencilMetrics =
      ::dawn::verify_field(stream, (mesh.EdgeStride) * vt_kSize, vt_dsl, vt,
                           "vt", vt_rel_tol, vt_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_vt(stencilMetrics,
                                  metricsNameFromEnvVar("SLURM_JOB_ID"),
                                  "mo_velocity_advection_stencil_01", "vt");
  serialiser_vt.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), vt_kSize, (mesh.EdgeStride),
                          vt, "mo_velocity_advection_stencil_01", "vt",
                          iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), vt_kSize, (mesh.EdgeStride),
                          vt_dsl, "mo_velocity_advection_stencil_01", "vt_dsl",
                          iteration);
    std::cout << "[DSL] serializing vt as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_velocity_advection_stencil_01", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_velocity_advection_stencil_01(
    double *vn, double *rbf_vec_coeff_e, double *vt, double *vt_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double vt_rel_tol, const double vt_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_velocity_advection_stencil_01 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e, vt_before,
                                       verticalStart, verticalEnd,
                                       horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_velocity_advection_stencil_01 run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_velocity_advection_stencil_01...\n"
            << std::flush;
  verify_mo_velocity_advection_stencil_01(vt_before, vt, vt_rel_tol, vt_abs_tol,
                                          iteration);

  iteration++;
}

void setup_mo_velocity_advection_stencil_01(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int vt_k_size) {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::setup(
      mesh, k_size, stream, vt_k_size);
}

void free_mo_velocity_advection_stencil_01() {
  dawn_generated::cuda_ico::mo_velocity_advection_stencil_01::free();
}
}
