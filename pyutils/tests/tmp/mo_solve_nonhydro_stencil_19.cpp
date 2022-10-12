#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_19.hpp"
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

class mo_solve_nonhydro_stencil_19 {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *ecTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      ecTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Edges, dawn::LocationType::Cells}, 0});
    }
  };

private:
  double *inv_dual_edge_length_;
  double *z_exner_ex_pr_;
  double *ddxn_z_full_;
  double *c_lin_e_;
  double *z_dexner_dz_c_1_;
  double *z_gradh_exner_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_gradh_exner_kSize_;

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

  static int get_z_gradh_exner_KSize() { return z_gradh_exner_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_gradh_exner_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_gradh_exner_kSize_ = z_gradh_exner_kSize;
  }

  mo_solve_nonhydro_stencil_19() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_19 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto inv_dual_edge_length_sid = get_sid(
        inv_dual_edge_length_,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));

    auto z_exner_ex_pr_sid = get_sid(
        z_exner_ex_pr_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto ddxn_z_full_sid = get_sid(
        ddxn_z_full_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_dexner_dz_c_1_sid = get_sid(
        z_dexner_dz_c_1_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_gradh_exner_sid = get_sid(
        z_gradh_exner_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<2> ec_ptr{.raw_ptr_fortran = mesh_.ecTable};
    auto connectivities =
        gridtools::hymap::keys<generated::E2C_t>::make_values(ec_ptr);
    double *c_lin_e_0 = &c_lin_e_[0 * mesh_.EdgeStride];
    double *c_lin_e_1 = &c_lin_e_[1 * mesh_.EdgeStride];
    auto c_lin_e_sid_0 = get_sid(
        c_lin_e_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_lin_e_sid_1 = get_sid(
        c_lin_e_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto c_lin_e_sid_comp = sid::composite::
        keys<integral_constant<int, 0>, integral_constant<int, 1>>::make_values(
            c_lin_e_sid_0, c_lin_e_sid_1);
    generated::mo_solve_nonhydro_stencil_19(connectivities)(
        cuda_backend, inv_dual_edge_length_sid, z_exner_ex_pr_sid,
        ddxn_z_full_sid, c_lin_e_sid_comp, z_dexner_dz_c_1_sid,
        z_gradh_exner_sid, horizontalStart, horizontalEnd, verticalStart,
        verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *inv_dual_edge_length, double *z_exner_ex_pr,
                     double *ddxn_z_full, double *c_lin_e,
                     double *z_dexner_dz_c_1, double *z_gradh_exner) {
    inv_dual_edge_length_ = inv_dual_edge_length;
    z_exner_ex_pr_ = z_exner_ex_pr;
    ddxn_z_full_ = ddxn_z_full;
    c_lin_e_ = c_lin_e;
    z_dexner_dz_c_1_ = z_dexner_dz_c_1;
    z_gradh_exner_ = z_gradh_exner;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_19(
    double *inv_dual_edge_length, double *z_exner_ex_pr, double *ddxn_z_full,
    double *c_lin_e, double *z_dexner_dz_c_1, double *z_gradh_exner,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19 s;
  s.copy_pointers(inv_dual_edge_length, z_exner_ex_pr, ddxn_z_full, c_lin_e,
                  z_dexner_dz_c_1, z_gradh_exner);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_19(const double *z_gradh_exner_dsl,
                                         const double *z_gradh_exner,
                                         const double z_gradh_exner_rel_tol,
                                         const double z_gradh_exner_abs_tol,
                                         const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_gradh_exner_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_19::get_z_gradh_exner_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.EdgeStride) * z_gradh_exner_kSize, z_gradh_exner_dsl,
      z_gradh_exner, "z_gradh_exner", z_gradh_exner_rel_tol,
      z_gradh_exner_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_gradh_exner(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_19", "z_gradh_exner");
  serialiser_z_gradh_exner.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_gradh_exner_kSize,
                          (mesh.EdgeStride), z_gradh_exner,
                          "mo_solve_nonhydro_stencil_19", "z_gradh_exner",
                          iteration);
    serialize_dense_edges(0, (mesh.NumEdges - 1), z_gradh_exner_kSize,
                          (mesh.EdgeStride), z_gradh_exner_dsl,
                          "mo_solve_nonhydro_stencil_19", "z_gradh_exner_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_gradh_exner as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_19", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_19(
    double *inv_dual_edge_length, double *z_exner_ex_pr, double *ddxn_z_full,
    double *c_lin_e, double *z_dexner_dz_c_1, double *z_gradh_exner,
    double *z_gradh_exner_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_gradh_exner_rel_tol, const double z_gradh_exner_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_19 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_19(inv_dual_edge_length, z_exner_ex_pr,
                                   ddxn_z_full, c_lin_e, z_dexner_dz_c_1,
                                   z_gradh_exner_before, verticalStart,
                                   verticalEnd, horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_19 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_19...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_19(z_gradh_exner_before, z_gradh_exner,
                                      z_gradh_exner_rel_tol,
                                      z_gradh_exner_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_19(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_gradh_exner_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19::setup(
      mesh, k_size, stream, z_gradh_exner_k_size);
}

void free_mo_solve_nonhydro_stencil_19() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_19::free();
}
}
