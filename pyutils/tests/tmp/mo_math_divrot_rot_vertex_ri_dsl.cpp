#include "driver-includes/cuda_utils.hpp"
#include "driver-includes/cuda_verify.hpp"
#include "driver-includes/defs.hpp"
#include "driver-includes/to_json.hpp"
#include "driver-includes/to_vtk.h"
#include "driver-includes/unstructured_domain.hpp"
#include "driver-includes/unstructured_interface.hpp"
#include "driver-includes/verification_metrics.hpp"
#include "mo_math_divrot_rot_vertex_ri_dsl.hpp"
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

class mo_math_divrot_rot_vertex_ri_dsl {
public:
  struct GpuTriMesh {
    int NumVertices;
    int NumEdges;
    int NumCells;
    int VertexStride;
    int EdgeStride;
    int CellStride;
    int *veTable;

    GpuTriMesh() {}

    GpuTriMesh(const dawn::GlobalGpuTriMesh *mesh) {
      NumVertices = mesh->NumVertices;
      NumCells = mesh->NumCells;
      NumEdges = mesh->NumEdges;
      VertexStride = mesh->VertexStride;
      CellStride = mesh->CellStride;
      EdgeStride = mesh->EdgeStride;
      veTable = mesh->NeighborTables.at(
          std::tuple<std::vector<dawn::LocationType>, bool>{
              {dawn::LocationType::Vertices, dawn::LocationType::Edges}, 0});
    }
  };

private:
  double *vec_e_;
  double *geofac_rot_;
  double *rot_vec_;
  inline static int kSize_;
  inline static GpuTriMesh mesh_;
  inline static bool is_setup_;
  inline static cudaStream_t stream_;
  inline static int rot_vec_kSize_;

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

  static int get_rot_vec_KSize() { return rot_vec_kSize_; }

  static void free() {}

  static void setup(const dawn::GlobalGpuTriMesh *mesh, int kSize,
                    cudaStream_t stream, const int rot_vec_kSize) {
    mesh_ = GpuTriMesh(mesh);
    kSize_ = kSize;
    is_setup_ = true;
    stream_ = stream;
    rot_vec_kSize_ = rot_vec_kSize;
  }

  mo_math_divrot_rot_vertex_ri_dsl() {}

  void run(const int verticalStart, const int verticalEnd,
           const int horizontalStart, const int horizontalEnd) {
    if (!is_setup_) {
      printf("mo_math_divrot_rot_vertex_ri_dsl has not been set up! make sure "
             "setup() is called before run!\n");
      return;
    }
    using namespace gridtools;
    using namespace fn;

    auto vec_e_sid = get_sid(
        vec_e_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.EdgeStride));

    auto rot_vec_sid = get_sid(
        rot_vec_,
        gridtools::hymap::keys<
            unstructured::dim::horizontal,
            unstructured::dim::vertical>::make_values(1, mesh_.VertexStride));

    fn_backend_t cuda_backend{};
    cuda_backend.stream = stream_;
    neighbor_table_fortran<6> ve_ptr{.raw_ptr_fortran = mesh_.veTable};
    auto connectivities =
        gridtools::hymap::keys<generated::V2E_t>::make_values(ve_ptr);
    double *geofac_rot_0 = &geofac_rot_[0 * mesh_.VertexStride];
    double *geofac_rot_1 = &geofac_rot_[1 * mesh_.VertexStride];
    double *geofac_rot_2 = &geofac_rot_[2 * mesh_.VertexStride];
    double *geofac_rot_3 = &geofac_rot_[3 * mesh_.VertexStride];
    double *geofac_rot_4 = &geofac_rot_[4 * mesh_.VertexStride];
    double *geofac_rot_5 = &geofac_rot_[5 * mesh_.VertexStride];
    auto geofac_rot_sid_0 = get_sid(
        geofac_rot_0,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_1 = get_sid(
        geofac_rot_1,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_2 = get_sid(
        geofac_rot_2,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_3 = get_sid(
        geofac_rot_3,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_4 = get_sid(
        geofac_rot_4,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_5 = get_sid(
        geofac_rot_5,
        gridtools::hymap::keys<unstructured::dim::horizontal>::make_values(1));
    auto geofac_rot_sid_comp = sid::composite::keys<
        integral_constant<int, 0>, integral_constant<int, 1>,
        integral_constant<int, 2>, integral_constant<int, 3>,
        integral_constant<int, 4>,
        integral_constant<int, 5>>::make_values(geofac_rot_sid_0,
                                                geofac_rot_sid_1,
                                                geofac_rot_sid_2,
                                                geofac_rot_sid_3,
                                                geofac_rot_sid_4,
                                                geofac_rot_sid_5);
    generated::mo_math_divrot_rot_vertex_ri_dsl(connectivities)(
        cuda_backend, vec_e_sid, geofac_rot_sid_comp, rot_vec_sid,
        horizontalStart, horizontalEnd, verticalStart, verticalEnd);
#ifndef NDEBUG
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
  }

  void copy_pointers(double *vec_e, double *geofac_rot, double *rot_vec) {
    vec_e_ = vec_e;
    geofac_rot_ = geofac_rot;
    rot_vec_ = rot_vec;
  }
};
} // namespace cuda_ico
} // namespace dawn_generated

extern "C" {
void run_mo_math_divrot_rot_vertex_ri_dsl(
    double *vec_e, double *geofac_rot, double *rot_vec, const int verticalStart,
    const int verticalEnd, const int horizontalStart, const int horizontalEnd) {
  dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl s;
  s.copy_pointers(vec_e, geofac_rot, rot_vec);
  s.run(verticalStart, verticalEnd, horizontalStart, horizontalEnd);
  return;
}

bool verify_mo_math_divrot_rot_vertex_ri_dsl(const double *rot_vec_dsl,
                                             const double *rot_vec,
                                             const double rot_vec_rel_tol,
                                             const double rot_vec_abs_tol,
                                             const int iteration) {
  using namespace std::chrono;
  const auto &mesh =
      dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl::getMesh();
  cudaStream_t stream =
      dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl::getStream();
  int kSize =
      dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl::getKSize();
  high_resolution_clock::time_point t_start = high_resolution_clock::now();
  struct VerificationMetrics stencilMetrics;

  int rot_vec_kSize = dawn_generated::cuda_ico::
      mo_math_divrot_rot_vertex_ri_dsl::get_rot_vec_KSize();
  stencilMetrics = ::dawn::verify_field(
      stream, (mesh.VertexStride) * rot_vec_kSize, rot_vec_dsl, rot_vec,
      "rot_vec", rot_vec_rel_tol, rot_vec_abs_tol, iteration);
#ifdef __SERIALIZE_METRICS
  MetricsSerialiser serialiser_rot_vec(
      stencilMetrics, metricsNameFromEnvVar("SLURM_JOB_ID"),
      "mo_math_divrot_rot_vertex_ri_dsl", "rot_vec");
  serialiser_rot_vec.writeJson(iteration);
#endif
  if (!stencilMetrics.isValid) {
#ifdef __SERIALIZE_ON_ERROR
    serialize_dense_verts(
        0, (mesh.NumVertices - 1), rot_vec_kSize, (mesh.VertexStride), rot_vec,
        "mo_math_divrot_rot_vertex_ri_dsl", "rot_vec", iteration);
    serialize_dense_verts(0, (mesh.NumVertices - 1), rot_vec_kSize,
                          (mesh.VertexStride), rot_vec_dsl,
                          "mo_math_divrot_rot_vertex_ri_dsl", "rot_vec_dsl",
                          iteration);
    std::cout << "[DSL] serializing rot_vec as error is high.\n" << std::flush;
#endif
  };
#ifdef __SERIALIZE_ON_ERROR
  serialize_flush_iter("mo_math_divrot_rot_vertex_ri_dsl", iteration);
#endif
  high_resolution_clock::time_point t_end = high_resolution_clock::now();
  duration<double> timing = duration_cast<duration<double>>(t_end - t_start);
  std::cout << "[DSL] Verification took " << timing.count() << " seconds.\n"
            << std::flush;
  return stencilMetrics.isValid;
}

void run_and_verify_mo_math_divrot_rot_vertex_ri_dsl(
    double *vec_e, double *geofac_rot, double *rot_vec, double *rot_vec_before,
    const int verticalStart, const int verticalEnd, const int horizontalStart,
    const int horizontalEnd, const double rot_vec_rel_tol,
    const double rot_vec_abs_tol) {
  static int iteration = 0;
  std::cout << "[DSL] Running stencil mo_math_divrot_rot_vertex_ri_dsl ("
            << iteration << ") ...\n"
            << std::flush;
  run_mo_math_divrot_rot_vertex_ri_dsl(vec_e, geofac_rot, rot_vec_before,
                                       verticalStart, verticalEnd,
                                       horizontalStart, horizontalEnd);

  std::cout << "[DSL] mo_math_divrot_rot_vertex_ri_dsl run time: " << time
            << "s\n"
            << std::flush;
  std::cout << "[DSL] Verifying stencil mo_math_divrot_rot_vertex_ri_dsl...\n"
            << std::flush;
  verify_mo_math_divrot_rot_vertex_ri_dsl(
      rot_vec_before, rot_vec, rot_vec_rel_tol, rot_vec_abs_tol, iteration);

  iteration++;
}

void setup_mo_math_divrot_rot_vertex_ri_dsl(dawn::GlobalGpuTriMesh *mesh,
                                            int k_size, cudaStream_t stream,
                                            const int rot_vec_k_size) {
  dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl::setup(
      mesh, k_size, stream, rot_vec_k_size);
}

void free_mo_math_divrot_rot_vertex_ri_dsl() {
  dawn_generated::cuda_ico::mo_math_divrot_rot_vertex_ri_dsl::free();
}
}
