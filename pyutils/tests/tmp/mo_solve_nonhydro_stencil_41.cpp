#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_solve_nonhydro_stencil_41.hpp"
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

class mo_solve_nonhydro_stencil_41 {
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
  double *geofac_div_;
  double *mass_fl_e_;
  double *z_theta_v_fl_e_;
  double *z_flxdiv_mass_;
  double *z_flxdiv_theta_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int z_flxdiv_mass_kSize_;
  inline static int z_flxdiv_theta_kSize_;

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

  static int get_z_flxdiv_mass_KSize() { return z_flxdiv_mass_kSize_; }

  static int get_z_flxdiv_theta_KSize() { return z_flxdiv_theta_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int z_flxdiv_mass_kSize,
                    const int z_flxdiv_theta_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    z_flxdiv_mass_kSize_ = z_flxdiv_mass_kSize;
    z_flxdiv_theta_kSize_ = z_flxdiv_theta_kSize;
  }

  mo_solve_nonhydro_stencil_41() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_solve_nonhydro_stencil_41 has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto mass_fl_e_sid = get_sid(
        mass_fl_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_theta_v_fl_e_sid = get_sid(
        z_theta_v_fl_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto z_flxdiv_mass_sid = get_sid(
        z_flxdiv_mass_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    auto z_flxdiv_theta_sid = get_sid(
        z_flxdiv_theta_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.CellStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<3> ce_ptr{.raw_ptr_fortran = mesh_.ceTable};
    auto connectivities =
        gridtools::hymap::keys<generated::C2E_t>::make_values(ce_ptr);
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
    generated::mo_solve_nonhydro_stencil_41(connectivities)(
        cuda_backend, geofac_div_sid_comp, mass_fl_e_sid, z_theta_v_fl_e_sid,
        z_flxdiv_mass_sid, z_flxdiv_theta_sid, horizontalStart, horizontalEnd,
        verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *geofac_div, double *mass_fl_e,
                     double *z_theta_v_fl_e, double *z_flxdiv_mass,
                     double *z_flxdiv_theta) {
    geofac_div_ = geofac_div;
    mass_fl_e_ = mass_fl_e;
    z_theta_v_fl_e_ = z_theta_v_fl_e;
    z_flxdiv_mass_ = z_flxdiv_mass;
    z_flxdiv_theta_ = z_flxdiv_theta;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_solve_nonhydro_stencil_41(
    double *geofac_div, double *mass_fl_e, double *z_theta_v_fl_e,
    double *z_flxdiv_mass, double *z_flxdiv_theta, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41 s;
  s.copy_pointers(geofac_div, mass_fl_e, z_theta_v_fl_e, z_flxdiv_mass,
                  z_flxdiv_theta);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_solve_nonhydro_stencil_41(
    const double *z_flxdiv_mass_dsl, const double *z_flxdiv_mass,
    const double *z_flxdiv_theta_dsl, const double *z_flxdiv_theta,
    const double z_flxdiv_mass_rel_tol, const double z_flxdiv_mass_abs_tol,
    const double z_flxdiv_theta_rel_tol, const double z_flxdiv_theta_abs_tol,
    const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int z_flxdiv_mass_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_41::get_z_flxdiv_mass_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_flxdiv_mass_kSize, z_flxdiv_mass_dsl,
      z_flxdiv_mass, "z_flxdiv_mass", z_flxdiv_mass_rel_tol,
      z_flxdiv_mass_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_flxdiv_mass(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_41", "z_flxdiv_mass");
  serialiser_z_flxdiv_mass.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_flxdiv_mass_kSize,
                          (mesh.CellStride), z_flxdiv_mass,
                          "mo_solve_nonhydro_stencil_41", "z_flxdiv_mass",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_flxdiv_mass_kSize,
                          (mesh.CellStride), z_flxdiv_mass_dsl,
                          "mo_solve_nonhydro_stencil_41", "z_flxdiv_mass_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_flxdiv_mass as error is high.\n"
              << std::flush;
#endif
  }
  int z_flxdiv_theta_kSize = dawn_generated::cuda_ico::
      mo_solve_nonhydro_stencil_41::get_z_flxdiv_theta_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.CellStride) * z_flxdiv_theta_kSize, z_flxdiv_theta_dsl,
      z_flxdiv_theta, "z_flxdiv_theta", z_flxdiv_theta_rel_tol,
      z_flxdiv_theta_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_z_flxdiv_theta(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_solve_nonhydro_stencil_41", "z_flxdiv_theta");
  serialiser_z_flxdiv_theta.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_cells(0, (mesh.NumCells - 1), z_flxdiv_theta_kSize,
                          (mesh.CellStride), z_flxdiv_theta,
                          "mo_solve_nonhydro_stencil_41", "z_flxdiv_theta",
                          iteration);
    serialize_dense_cells(0, (mesh.NumCells - 1), z_flxdiv_theta_kSize,
                          (mesh.CellStride), z_flxdiv_theta_dsl,
                          "mo_solve_nonhydro_stencil_41", "z_flxdiv_theta_dsl",
                          iteration);
    std::cout << "[DSL] serializing z_flxdiv_theta as error is high.\n"
              << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_solve_nonhydro_stencil_41", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_solve_nonhydro_stencil_41(
    double *geofac_div, double *mass_fl_e, double *z_theta_v_fl_e,
    double *z_flxdiv_mass, double *z_flxdiv_theta, double *z_flxdiv_mass_before,
    double *z_flxdiv_theta_before, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd,
    const double z_flxdiv_mass_rel_tol, const double z_flxdiv_mass_abs_tol,
    const double z_flxdiv_theta_rel_tol, const double z_flxdiv_theta_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_solve_nonhydro_stencil_41 ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_solve_nonhydro_stencil_41(geofac_div, mass_fl_e, z_theta_v_fl_e,
                                   z_flxdiv_mass_before, z_flxdiv_theta_before,
                                   verticalStart, verticalEnd, horizontalStart,
                                   horizontalEnd);

  std::cout << "[DSL] mo_solve_nonhydro_stencil_41 run time: " << time << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_solve_nonhydro_stencil_41...\n"
            << std::flush;
  verify_mo_solve_nonhydro_stencil_41(
      z_flxdiv_mass_before, z_flxdiv_mass, z_flxdiv_theta_before,
      z_flxdiv_theta, z_flxdiv_mass_rel_tol, z_flxdiv_mass_abs_tol,
      z_flxdiv_theta_rel_tol, z_flxdiv_theta_abs_tol, iteration);

  iteration++;
}

void setup_mo_solve_nonhydro_stencil_41(dawn::GlobalGpuTriMesh *mesh,
                                        int k_size, cudaStream_t stream,
                                        const int z_flxdiv_mass_k_size,
                                        const int z_flxdiv_theta_k_size) {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41::setup(
      mesh, k_size, stream, z_flxdiv_mass_k_size, z_flxdiv_theta_k_size);
}

void free_mo_solve_nonhydro_stencil_41() {
  dawn_generated::cuda_ico::mo_solve_nonhydro_stencil_41::free();
}
}
