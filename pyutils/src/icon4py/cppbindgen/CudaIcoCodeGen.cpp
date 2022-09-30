#include "CXXUtil.h"
#include "F90Util.h"
#include "IcoChainSizes.h"
#include "LocToStringUtils.h"
#include "LocationType.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

enum class IntendKind { Output = 0, InputOutput = 1, Input = 2 };
enum class CodeGenKind { WithBody = 0, OnlyDecl = 1 };
enum class CallerKind { FromHost = 0, FromDevice = 1 };

class Field {
public:
  Field(std::string name, std::string cpp_type,
        dawn::codegen::FortranAPI::InterfaceType fortran_type,
        IntendKind intend,
        std::optional<dawn::ast::LocationType> horizontal_location,
        std::optional<dawn::ast::NeighborChain> sparse_location,
        bool is_vertical, bool include_center, bool is_new_sparse_formulation)
      : name_(std::move(name)), cpp_type_(std::move(cpp_type)),
        fortran_type_(fortran_type),
        intend_(intend), horizontal_location_(horizontal_location),
        sparse_location_(sparse_location), is_vertical_(is_vertical),
        include_center_(include_center), is_new_sparse_formulation_(is_new_sparse_formulation) {
    rank_ = 0;
    if (is_vertical_) {
      rank_ += 1;
    }
    if (horizontal_location_) {
      rank_ += 1;
    }
    if (sparse_location_) {
      rank_ += 1;
    }
  }

  [[nodiscard]] std::string getName() const { return name_; }

  [[nodiscard]] std::string getCppType() const { return cpp_type_; }

  [[nodiscard]] dawn::codegen::FortranAPI::InterfaceType
  getFortranType() const {
    return fortran_type_;
  }

  [[nodiscard]] int getRank() const { return rank_; }

  [[nodiscard]] IntendKind getIntend() const { return intend_; }

  [[nodiscard]] bool isHorizontal() const {
    return horizontal_location_.has_value();
  }

  [[nodiscard]] bool isSparse() const { return sparse_location_.has_value(); }

  [[nodiscard]] bool isVertical() const { return is_vertical_; }

  [[nodiscard]] bool includesCenter() const { return include_center_; }

  [[nodiscard]] bool isNewSparseFormulation() const { return is_new_sparse_formulation_; }

  [[nodiscard]] dawn::ast::LocationType getHorizontalLocation() const {
    return horizontal_location_.value();
  }

  [[nodiscard]] dawn::ast::NeighborChain getSparseLocation() const {
    return sparse_location_.value();
  }

private:
  std::string name_;
  std::string cpp_type_;
  dawn::codegen::FortranAPI::InterfaceType fortran_type_;
  IntendKind intend_;
  std::optional<dawn::ast::LocationType> horizontal_location_;
  std::optional<dawn::ast::NeighborChain> sparse_location_;
  bool is_vertical_;
  bool include_center_;
  int rank_;
  bool is_new_sparse_formulation_;
};

std::vector<Field> getUsedFields(
    const std::vector<Field> &fieldVector,
    const std::unordered_set<IntendKind> &intend = {
        IntendKind::Output, IntendKind::InputOutput, IntendKind::Input}) {

  std::vector<Field> res;
  for (const auto &field : fieldVector) {
    if (intend.count(field.getIntend())) {
      res.push_back(field);
    }
  }

  return res;
}

std::vector<std::string> getUsedFieldsNames(
    const std::vector<Field> &fieldVector,
    const std::unordered_set<IntendKind> &intend = {
        IntendKind::Output, IntendKind::InputOutput, IntendKind::Input}) {
  auto usedFields = getUsedFields(fieldVector, intend);
  std::vector<std::string> fieldsNames;
  fieldsNames.reserve(usedFields.size());
  for (const auto &field : usedFields) {
    fieldsNames.push_back(field.getName());
  }
  return fieldsNames;
}

std::string explodeToStr(const std::vector<std::string> &vec,
                         const std::string &sep = ", ") {
  std::string ret;
  bool first = true;
  for (const auto &el : vec) {
    if (!first) {
      ret += sep;
    }
    ret += el;
    first = false;
  }

  return ret;
}

std::string explodeUsedFields(const std::vector<Field> &fieldVector,
                              const std::unordered_set<IntendKind> &intend = {
                                  IntendKind::Output, IntendKind::InputOutput,
                                  IntendKind::Input}) {
  return explodeToStr(getUsedFieldsNames(fieldVector, intend));
}

} // namespace

void generateGpuMesh(
    dawn::codegen::Class &wrapperClass,
    const std::vector<dawn::ast::UnstructuredIterationSpace> &allSpaces) {
  using namespace dawn::codegen;
  Structure gpuMeshClass = wrapperClass.addStruct("GpuTriMesh");

  gpuMeshClass.addMember("int", "NumVertices");
  gpuMeshClass.addMember("int", "NumEdges");
  gpuMeshClass.addMember("int", "NumCells");
  gpuMeshClass.addMember("int", "VertexStride");
  gpuMeshClass.addMember("int", "EdgeStride");
  gpuMeshClass.addMember("int", "CellStride");

  for (auto space : allSpaces) {
    if (space.isNewSparse()) {
      continue;
    }
    gpuMeshClass.addMember(
        "int*",
        cudaico::chainToTableString(dawn::ast::UnstructuredIterationSpace(
            std::move(space.Chain), space.IncludeCenter)));
  }

  {
    auto gpuMeshDefaultCtor = gpuMeshClass.addConstructor();
    gpuMeshDefaultCtor.startBody();
    gpuMeshDefaultCtor.commit();
  }
  {
    auto gpuMeshFromGlobalCtor = gpuMeshClass.addConstructor();
    gpuMeshFromGlobalCtor.addArg("const dawn::GlobalGpuTriMesh *mesh");
    gpuMeshFromGlobalCtor.addStatement("NumVertices = mesh->NumVertices");
    gpuMeshFromGlobalCtor.addStatement("NumCells = mesh->NumCells");
    gpuMeshFromGlobalCtor.addStatement("NumEdges = mesh->NumEdges");
    gpuMeshFromGlobalCtor.addStatement("VertexStride = mesh->VertexStride");
    gpuMeshFromGlobalCtor.addStatement("CellStride = mesh->CellStride");
    gpuMeshFromGlobalCtor.addStatement("EdgeStride = mesh->EdgeStride");

    for (auto space : allSpaces) {
      if (space.isNewSparse()) {
        continue;
      }
      auto tableName =
          cudaico::chainToTableString(dawn::ast::UnstructuredIterationSpace(
              std::move(space.Chain), space.IncludeCenter));
      auto curlyInit = cudaico::chainToVectorString(space.Chain);
      gpuMeshFromGlobalCtor.addStatement(
          tableName +
          " = "
          "mesh->NeighborTables.at(std::tuple<std::vector<dawn::LocationType>, "
          "bool>{" +
          curlyInit + ", " + (space.IncludeCenter ? "1" : "0") + "})");
    }
  }
}

void generateGridFun(dawn::codegen::MemberFunction &gridFun) {
  gridFun.addBlockStatement("if (kparallel)", [&]() {
    gridFun.addStatement(
        "int dK = (kSize + LEVELS_PER_THREAD - 1) / LEVELS_PER_THREAD");
    gridFun.addStatement(
        "return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, dK, 1)");
  });
  gridFun.addBlockStatement("else", [&]() {
    gridFun.addStatement(
        "return dim3((elSize + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)");
  });
}

void generateRunFun(
    dawn::codegen::MemberFunction &runFun, std::string stencilName,
    const std::vector<Field> &fields,
    const std::vector<dawn::ast::UnstructuredIterationSpace> &allSpaces) {

  runFun.addArg("const int verticalStart");
  runFun.addArg("const int verticalEnd");
  runFun.addArg("const int horizontalStart");
  runFun.addArg("const int horizontalEnd");

  runFun.finishArgs();

  runFun.startBody();

  runFun.addBlockStatement("if (!is_setup_)", [&]() {
    runFun.addStatement(
        "printf(\"" + stencilName +
        " has not been set up! make sure setup() is called before run!\\n\")");
    runFun.addStatement("return");
  });

  runFun.addStatement("using namespace gridtools");
  runFun.addStatement("using namespace fn");

  auto dimString = [](const Field &field) -> std::string {
    if (field.isHorizontal() && field.isVertical() && field.isSparse()) {
      return "mesh_." +
             dawn::codegen::cudaico::locToStrideString(
                 field.getHorizontalLocation()) +
             ", " +
             std::to_string(dawn::ICOChainSize(field.getSparseLocation()) +
                            (field.includesCenter() ? 1 : 0)) +
             ", kSize_";
    } else if (field.isHorizontal() && field.isSparse()) {
      return "mesh_." +
             dawn::codegen::cudaico::locToStrideString(
                 field.getHorizontalLocation()) +
             ", " +
             std::to_string(dawn::ICOChainSize(field.getSparseLocation()) +
                            (field.includesCenter() ? 1 : 0));
    } else if (field.isHorizontal() && field.isVertical()) {
      return "mesh_." +
             dawn::codegen::cudaico::locToStrideString(
                 field.getHorizontalLocation()) +
             ", kSize_";
    } else if (field.isHorizontal()) {
      return "mesh_." +
             dawn::codegen::cudaico::locToStrideString(
                 field.getHorizontalLocation()) +
             "";
    } else if (field.isVertical()) {
      return "kSize_";
    } else {
      return "1";
    }
  };

  auto getTaggedStrideMap = [](const Field &field) -> std::string {
    if (field.isNewSparseFormulation()) {
      return "gridtools::hymap::keys<unstructured::dim::horizontal>::make_"
             "values(1)";
    }
    if (field.isHorizontal() && field.isVertical()) {
      return "gridtools::hymap::keys<unstructured::dim::horizontal, "
             "unstructured::dim::vertical>::make_values(1, mesh_." +
             dawn::codegen::cudaico::locToStrideString(
                 field.getHorizontalLocation()) +
             ")";
    } else if (field.isHorizontal()) {
      return "gridtools::hymap::keys<unstructured::dim::horizontal>::make_"
             "values(1)";
    } else if (field.isVertical()) {
      return "gridtools::hymap::keys<unstructured::dim::vertical>::make_values("
             "1)";
    }
  };

  for (const auto &field : getUsedFields(fields)) {
    if (field.getRank() > 0 && !field.isSparse() || field.isNewSparseFormulation()) {
      runFun.addStatement("auto " + field.getName() + "_sid = get_sid(" +
                          field.getName() + "_, " + getTaggedStrideMap(field) +
                          ")");
    }
  }

  int integralTypeInteger = 0;

  for (const auto &field : getUsedFields(fields)) {
    if (field.getRank() == 0) {
      runFun.addStatement("gridtools::stencil::global_parameter " +
                          field.getName() + "_gp {" + field.getName() + "_}");
    }
  }

  std::string fieldViewList;
  for (const auto &field : getUsedFields(fields)) {
    if (field.getRank() > 0) {
      if (field.isSparse() && !field.isNewSparseFormulation()) {
        fieldViewList += field.getName() + "_sid_comp, ";
      } else {
        fieldViewList += field.getName() + "_sid, ";
      }
    } else {
      fieldViewList += field.getName() + "_gp, ";
    }
  }

  std::string dimStringList;
  for (const auto &field : getUsedFields(fields)) {
    if (field.getRank() > 0) {
      dimStringList += dimString(field) + ", ";
    }
  }

  runFun.addStatement("fn_backend_t cuda_backend{}");
  runFun.addStatement("cuda_backend.stream = stream_");

  std::vector<std::string> connectivityTagVec;
  std::vector<std::string> connectivityVec;
  for (auto space : allSpaces) {
    std::string chainShorthand =
        dawn::codegen::cudaico::chainToShorthand(space);
    std::string chainShorthandGt4py = dawn::codegen::cudaico::chainToShorthand(
        space, dawn::codegen::cudaico::StringCase::upper, '2');

    if (!space.isNewSparse()) {
      runFun.addStatement("neighbor_table_fortran<" + std::to_string(dawn::ICOChainSize(space.Chain) + (space.IncludeCenter ? 1 : 0)) +"> " + chainShorthand +
                          "_ptr{.raw_ptr_fortran = mesh_." + chainShorthand +
                          "Table}");
    }
    connectivityVec.push_back(chainShorthand + "_ptr");
    connectivityTagVec.push_back("generated::" + chainShorthandGt4py + "_t");
  }
  for (auto space : allSpaces) {
    std::string chainShorthand =
        dawn::codegen::cudaico::chainToShorthand(space);
    std::string chainShorthandGt4py = dawn::codegen::cudaico::chainToShorthand(
        space, dawn::codegen::cudaico::StringCase::upper, '2');

    if (space.isNewSparse()) {
      runFun.addStatement("neighbor_table_4new_sparse<" + std::to_string(dawn::ICOChainSize(space.Chain) + (space.IncludeCenter ? 1 : 0)) +"> " + chainShorthand +
                          "_ptr{}");
    }
    connectivityVec.push_back(chainShorthand + "_ptr");
    connectivityTagVec.push_back("generated::" + chainShorthandGt4py + "_t");
  }

  runFun.addStatement("auto connectivities = gridtools::hymap::keys<" +
                      dawn::join(connectivityTagVec, ',') + ">::make_values(" +
                      dawn::join(connectivityVec, ',') + ")");

  for (const auto &field : getUsedFields(fields)) {
    if (field.isSparse() && !field.isNewSparseFormulation()) {
      int numSparse = dawn::ICOChainSize(field.getSparseLocation());
      if (field.includesCenter()) {
        numSparse++;
      }
      for (int i = 0; i < numSparse; i++) {
        runFun.addStatement("double *" + field.getName() + "_" +
                            std::to_string(i) + " = &" + field.getName() +
                            "_[" + std::to_string(i) + "*mesh_." +
                            dawn::codegen::cudaico::locToStrideString(
                                field.getHorizontalLocation()) +
                            "]");
      }
      for (int i = 0; i < numSparse; i++) {
        runFun.addStatement("auto " + field.getName() + "_sid_" +
                            std::to_string(i) + " = get_sid(" +
                            field.getName() + "_" + std::to_string(i) +
                            ", "
                            "gridtools::hymap::keys<unstructured::dim::"
                            "horizontal>::make_values(1))");
      }
      std::vector<std::string> integralConstantsVec;
      std::vector<std::string> sidsVec;
      for (int i = 0; i < numSparse; i++) {
        integralConstantsVec.push_back("integral_constant<int," +
                                       std::to_string(i) + ">");
        sidsVec.push_back(field.getName() + "_sid_" + std::to_string(i));
      }
      std::string integralConstants = dawn::join(integralConstantsVec, ',');
      std::string sids = dawn::join(sidsVec, ',');
      runFun.addStatement("auto " + field.getName() +
                          "_sid_comp = sid::composite::keys<" +
                          integralConstants + ">::make_values(" + sids + ")");
    }
  }

  runFun.addStatement(
      "generated::" + stencilName + "(connectivities)(cuda_backend, " +
      fieldViewList +
      "horizontalStart, horizontalEnd, verticalStart, verticalEnd)");

  std::stringstream k_size;
  std::string numElements;

  runFun.addPreprocessorDirective("ifndef NDEBUG\n");
  runFun.addStatement("gpuErrchk(cudaPeekAtLastError())");
  runFun.addStatement("gpuErrchk(cudaDeviceSynchronize())");
  runFun.addPreprocessorDirective("endif\n");
}

void generateStencilFree(dawn::codegen::MemberFunction &free) {
  free.startBody();
}

void generateStencilSetup(dawn::codegen::MemberFunction &setup,
                          const std::vector<Field> &fields) {
  setup.addStatement("mesh_ = GpuTriMesh(mesh)");
  setup.addStatement("kSize_ = kSize");
  setup.addStatement("is_setup_ = true");
  setup.addStatement("stream_ = stream");
  for (const auto &fieldName : getUsedFieldsNames(
           fields, {IntendKind::Output, IntendKind::InputOutput})) {
    setup.addStatement(fieldName + "_kSize_ = " + fieldName + "_kSize");
  }
}

void generateCopyPtrFun(dawn::codegen::MemberFunction &copyFun,
                        const std::vector<Field> &fields) {
  auto usedAPIFields = getUsedFields(fields);

  for (const auto &field : usedAPIFields) {
    copyFun.addArg(field.getCppType() + (field.getRank() > 0 ? "*" : "") + " " +
                   field.getName());
  }

  // copy pointer to each field storage
  for (const auto &field : usedAPIFields) {
    auto fname = field.getName();
    copyFun.addStatement(fname + "_ = " + fname);
  }
}

void generateStencilClasses(
    dawn::codegen::Class &stencilWrapperClass, std::string stencilName,
    const std::vector<Field> &fields,
    const std::vector<dawn::ast::UnstructuredIterationSpace> &allSpaces) {

  // generate members (fields + kSize + gpuMesh)
  stencilWrapperClass.changeAccessibility("private");

  for (const auto &field : getUsedFields(fields)) {
    stencilWrapperClass.addMember(field.getCppType() +
                                      (field.getRank() > 0 ? "*" : ""),
                                  field.getName() + "_");
  }
  stencilWrapperClass.addMember("inline static int", "kSize_");
  stencilWrapperClass.addMember("inline static GpuTriMesh", "mesh_");
  stencilWrapperClass.addMember("inline static bool", "is_setup_");
  stencilWrapperClass.addMember("inline static cudaStream_t", "stream_");

  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    stencilWrapperClass.addMember("inline static int", field.getName() + "_kSize_");
  }

  // grid helper fun
  //    can not be placed in cuda utils sinze it needs LEVELS_PER_THREAD and
  //    BLOCK_SIZE, which are supposed to become compiler flags
  auto gridFun = stencilWrapperClass.addMemberFunction("dim3", "grid");
  gridFun.addArg("int kSize");
  gridFun.addArg("int elSize");
  gridFun.addArg("bool kparallel");

  generateGridFun(gridFun);

  gridFun.commit();

  stencilWrapperClass.changeAccessibility("public");
  auto meshGetter = stencilWrapperClass.addMemberFunction(
      "static const GpuTriMesh &", "getMesh");
  meshGetter.finishArgs();
  meshGetter.startBody();
  meshGetter.addStatement("return mesh_");
  meshGetter.commit();

  auto streamGetter =
      stencilWrapperClass.addMemberFunction("static cudaStream_t", "getStream");
  streamGetter.finishArgs();
  streamGetter.startBody();
  streamGetter.addStatement("return stream_");
  streamGetter.commit();

  auto kSizeGetter =
      stencilWrapperClass.addMemberFunction("static int", "getKSize");
  kSizeGetter.finishArgs();
  kSizeGetter.startBody();
  kSizeGetter.addStatement("return kSize_");
  kSizeGetter.commit();

  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    auto kSizeFieldGetter = stencilWrapperClass.addMemberFunction(
        "static int", "get_" + field.getName() + "_KSize");
    kSizeFieldGetter.finishArgs();
    kSizeFieldGetter.startBody();
    kSizeFieldGetter.addStatement("return " + field.getName() + "_kSize_");
    kSizeFieldGetter.commit();
  }

  // constructor from library
  auto stencilClassFree =
      stencilWrapperClass.addMemberFunction("static void", "free");
  generateStencilFree(stencilClassFree);
  stencilClassFree.commit();

  auto stencilClassSetup =
      stencilWrapperClass.addMemberFunction("static void", "setup");
  stencilClassSetup.addArg("const dawn::GlobalGpuTriMesh *mesh");
  stencilClassSetup.addArg("int kSize");
  stencilClassSetup.addArg("cudaStream_t stream");
  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    stencilClassSetup.addArg("const int " + field.getName() + "_kSize");
  }
  generateStencilSetup(stencilClassSetup, fields);
  stencilClassSetup.commit();

  // minmal ctor
  auto stencilClassDefaultConstructor = stencilWrapperClass.addConstructor();
  stencilClassDefaultConstructor.startBody();
  stencilClassDefaultConstructor.commit();

  // run method
  auto runFun = stencilWrapperClass.addMemberFunction("void", "run");
  generateRunFun(runFun, stencilName, fields, allSpaces);
  runFun.commit();

  // copy to funs
  auto copyPtrFun =
      stencilWrapperClass.addMemberFunction("void", "copy_pointers");
  generateCopyPtrFun(copyPtrFun, fields);
  copyPtrFun.commit();
}

void generateAllAPIRunFunctions(std::stringstream &ssSW,
                                std::string stencilName,
                                const std::vector<Field> &fields,
                                CallerKind callerKind,
                                CodeGenKind codeGenKind) {

  const std::string wrapperName = stencilName;

  // two functions if from host (from c / from fort), one function if simply
  // passing the pointers
  std::vector<std::stringstream> apiRunFunStreams(
      callerKind == CallerKind::FromHost ? 2 : 1);
  { // stringstreams need to outlive the correspondind MemberFunctions
    std::vector<std::unique_ptr<dawn::codegen::MemberFunction>> apiRunFuns;
    apiRunFuns.push_back(std::make_unique<dawn::codegen::MemberFunction>(
        "void", "run_" + wrapperName, apiRunFunStreams[0], /*indent level*/ 0,
        codeGenKind == CodeGenKind::OnlyDecl));

    for (auto &apiRunFun : apiRunFuns) {
      for (const auto &field : fields) {
        apiRunFun->addArg(field.getCppType() +
                          (field.getRank() > 0 ? "*" : "") + " " +
                          field.getName());
      }
    }
    for (auto &apiRunFun : apiRunFuns) {
      apiRunFun->addArg("const int verticalStart");
      apiRunFun->addArg("const int verticalEnd");
      apiRunFun->addArg("const int horizontalStart");
      apiRunFun->addArg("const int horizontalEnd");
    }
    for (auto &apiRunFun : apiRunFuns) {
      apiRunFun->finishArgs();
    }

    // Write body only when run for implementation generation
    if (codeGenKind == CodeGenKind::WithBody) {
      // listing all API fields
      std::string fieldsStr = explodeUsedFields(fields);

      const std::string fullStencilName =
          "dawn_generated::cuda_ico::" + wrapperName;

      for (auto &apiRunFun : apiRunFuns) {
        apiRunFun->addStatement(fullStencilName + " s");
      }

      apiRunFuns[0]->addStatement("s.copy_pointers(" + fieldsStr + ")");

      for (auto &apiRunFun : apiRunFuns) {
        apiRunFun->addStatement("s.run(verticalStart, verticalEnd, "
                                "horizontalStart, horizontalEnd)");
      }

      for (auto &apiRunFun : apiRunFuns) {
        apiRunFun->addStatement("return");
        apiRunFun->commit();
      }

      for (const auto &stream : apiRunFunStreams) {
        ssSW << stream.str();
      }
    } else {
      for (auto &apiRunFun : apiRunFuns) {
        apiRunFun->commit();
      }
      for (const auto &stream : apiRunFunStreams) {
        ssSW << stream.str();
      }
    }
  }
}

void generateAllAPIVerifyFunctions(std::stringstream &ssSW,
                                   std::string stencilName,
                                   const std::vector<Field> &fields,
                                   CodeGenKind codeGenKind) {

  const std::string wrapperName = stencilName;
  const std::string fullStencilName =
      "dawn_generated::cuda_ico::" + wrapperName;

  auto getSerializeCall = [](dawn::ast::LocationType locType) -> std::string {
    using dawn::ast::LocationType;
    switch (locType) {
    case dawn::ast::LocationType::Edges:
      return "serialize_dense_edges";
      break;
    case dawn::ast::LocationType::Cells:
      return "serialize_dense_cells";
      break;
    case dawn::ast::LocationType::Vertices:
      return "serialize_dense_verts";
      break;
    default:
      __builtin_unreachable();
    }
  };

  std::stringstream verifySS, runAndVerifySS;

  { // stringstreams need to outlive the correspondind MemberFunctions

    dawn::codegen::MemberFunction verifyAPI(
        "bool", "verify_" + wrapperName, verifySS, /*indent level*/ 0,
        codeGenKind == CodeGenKind::OnlyDecl);
    dawn::codegen::MemberFunction runAndVerifyAPI(
        "void", "run_and_verify_" + wrapperName, runAndVerifySS,
        /*indent level*/ 0, codeGenKind == CodeGenKind::OnlyDecl);

    for (const auto &field :
         getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
      verifyAPI.addArg("const " + field.getCppType() +
                       (field.getRank() > 0 ? "*" : "") + " " +
                       field.getName() + "_dsl");
      verifyAPI.addArg("const " + field.getCppType() +
                       (field.getRank() > 0 ? "*" : "") + " " +
                       field.getName());
    }
    for (const auto &field :
         getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
      verifyAPI.addArg("const double " + field.getName() + "_rel_tol");
      verifyAPI.addArg("const double " + field.getName() + "_abs_tol");
    }
    verifyAPI.addArg("const int iteration");
    verifyAPI.finishArgs();

    for (const auto &field : getUsedFields(fields)) {
      runAndVerifyAPI.addArg(field.getCppType() +
                             (field.getRank() > 0 ? "*" : "") + " " +
                             field.getName());
    }
    for (const auto &field :
         getUsedFields(fields, {IntendKind::InputOutput, IntendKind::Output})) {
      runAndVerifyAPI.addArg(field.getCppType() +
                             (field.getRank() > 0 ? "*" : "") + " " +
                             field.getName() + "_before");
    }

    runAndVerifyAPI.addArg("const int verticalStart");
    runAndVerifyAPI.addArg("const int verticalEnd");
    runAndVerifyAPI.addArg("const int horizontalStart");
    runAndVerifyAPI.addArg("const int horizontalEnd");

    for (const auto &field :
         getUsedFields(fields, {IntendKind::InputOutput, IntendKind::Output})) {
      runAndVerifyAPI.addArg("const double " + field.getName() + "_rel_tol");
      runAndVerifyAPI.addArg("const double " + field.getName() + "_abs_tol");
    }

    runAndVerifyAPI.finishArgs();

    if (codeGenKind == CodeGenKind::WithBody) {
      const auto &fieldInfos = getUsedFields(fields);

      verifyAPI.startBody();
      verifyAPI.addStatement("using namespace std::chrono");
      verifyAPI.addStatement("const auto &mesh = " + fullStencilName +
                             "::" + "getMesh()");
      verifyAPI.addStatement("cudaStream_t stream = " + fullStencilName +
                             "::" + "getStream()");
      verifyAPI.addStatement("int kSize = " + fullStencilName +
                             "::" + "getKSize()");
      verifyAPI.addStatement("high_resolution_clock::time_point t_start = "
                             "high_resolution_clock::now()");
      verifyAPI.addStatement("struct VerificationMetrics stencilMetrics");

      for (auto field : getUsedFields(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {

        verifyAPI.addStatement("int " + field.getName() +
                               "_kSize = " + fullStencilName + "::" + "get_" +
                               field.getName() + "_KSize()");

        std::string num_lev = field.getName() + "_kSize";

        std::string dense_stride = "(mesh." +
                                   dawn::codegen::cudaico::locToStrideString(
                                       field.getHorizontalLocation()) +
                                   ")";
        std::string indexOfLastHorElement =
            "(mesh." +
            dawn::codegen::cudaico::locToDenseSizeStringGpuMesh(
                field.getHorizontalLocation()) +
            " -1)";
        std::string num_el = dense_stride + " * " + num_lev;
        verifyAPI.addStatement(
            "stencilMetrics = ::dawn::verify_field(stream, " + num_el + ", " +
            field.getName() + "_dsl" + "," + field.getName() + ", \"" +
            field.getName() + "\"" + "," + field.getName() + "_rel_tol" + "," +
            field.getName() + "_abs_tol, " + "iteration" + ")");

        verifyAPI.addPreprocessorDirective("ifdef __SERIALIZE_METRICS");

        std::string serialiserVarName = "serialiser_" + field.getName();
        verifyAPI.addStatement(
            "MetricsSerialiser " + serialiserVarName +
            "(stencilMetrics, metricsNameFromEnvVar(\"SLURM_JOB_ID\"), \"" +
            wrapperName + "\", \"" + field.getName() + "\")");

        verifyAPI.addStatement(serialiserVarName + ".writeJson(iteration)");

        verifyAPI.addPreprocessorDirective("endif");

        verifyAPI.addBlockStatement("if (!stencilMetrics.isValid)", [&]() {
          verifyAPI.addPreprocessorDirective("ifdef __SERIALIZE_ON_ERROR");
          if (!field.isSparse()) {
            // serialize actual field
            auto lt = field.getHorizontalLocation();
            auto serializeCall = getSerializeCall(lt);
            verifyAPI.addStatement(
                serializeCall + "(0" + ", " + indexOfLastHorElement + ", " +
                num_lev + ", " + dense_stride + ", " + field.getName() +
                ", \"" + wrapperName + "\", \"" + field.getName() +
                "\", iteration)");
            // serialize dsl field
            verifyAPI.addStatement(
                serializeCall + "(0" + ", " + indexOfLastHorElement + ", " +
                num_lev + ", " + dense_stride + ", " + field.getName() +
                "_dsl" + ", \"" + wrapperName + "\", \"" + field.getName() +
                "_dsl" + "\", iteration)");
            verifyAPI.addStatement("std::cout << \"[DSL] serializing " +
                                   field.getName() +
                                   " as error is high.\\n\" << std::flush");
          } else {
            verifyAPI.addStatement(
                "std::cout << \"[DSL] can not serialize sparse field " +
                field.getName() + ", error is high.\\n\" << std::flush");
          }
          verifyAPI.addPreprocessorDirective("endif");
        });
      }

      verifyAPI.addStatement("\n");
      verifyAPI.addPreprocessorDirective(
          "ifdef __SERIALIZE_ON_ERROR\n"); // newline requried
      verifyAPI.addStatement("serialize_flush_iter(\"" + wrapperName +
                             "\", iteration)");
      verifyAPI.addPreprocessorDirective("endif");

      verifyAPI.addStatement("high_resolution_clock::time_point t_end = "
                             "high_resolution_clock::now()");
      verifyAPI.addStatement(
          "duration<double> timing = duration_cast<duration<double>>(t_end - "
          "t_start)");
      verifyAPI.addStatement(
          "std::cout << \"[DSL] Verification took \" << timing.count() << \" "
          "seconds.\\n\" << std::flush");
      verifyAPI.addStatement("return stencilMetrics.isValid");

      // TODO runAndVerifyAPI body
      runAndVerifyAPI.addStatement("static int iteration = 0");

      runAndVerifyAPI.addStatement(
          "std::cout << \"[DSL] Running stencil " + wrapperName +
          " (\" << iteration << \") ...\\n\" << std::flush");

      auto getDSLFieldsNames = [&fields]() -> std::vector<std::string> {
        auto apiFields = getUsedFields(fields);

        std::vector<std::string> fieldNames;
        for (const auto &field : fields) {
          if (field.getIntend() == IntendKind::InputOutput ||
              field.getIntend() == IntendKind::Output) {
            fieldNames.push_back(field.getName() + "_before");
          } else {
            fieldNames.push_back(field.getName());
          }
        }
        return fieldNames;
      };

      runAndVerifyAPI.addStatement(
          "run_" + wrapperName + "(" +
          explodeToStr(/*concatenateVectors(
              {getGlobalsNames(globalsMap), */
                       getDSLFieldsNames() /*})*/) +
          ", verticalStart, verticalEnd, horizontalStart, horizontalEnd)");

      runAndVerifyAPI.addStatement("std::cout << \"[DSL] " + wrapperName +
                                   " run time: \" << "
                                   "time << \"s\\n\" << std::flush");
      runAndVerifyAPI.addStatement("std::cout << \"[DSL] Verifying stencil " +
                                   wrapperName + "...\\n\" << std::flush");

      std::vector<std::string> outputVerifyFields;
      for (const auto &fieldName : getUsedFieldsNames(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {
        outputVerifyFields.push_back(fieldName + "_before");
        outputVerifyFields.push_back(fieldName);
      }
      for (const auto &fieldName : getUsedFieldsNames(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {
        outputVerifyFields.push_back(fieldName + "_rel_tol");
        outputVerifyFields.push_back(fieldName + "_abs_tol");
      }

      runAndVerifyAPI.addStatement("verify_" + wrapperName + "(" +
                                   explodeToStr(dawn::concatenateVectors(
                                       {outputVerifyFields, {"iteration"}})) +
                                   ")");

      runAndVerifyAPI.addStatement("iteration++");
    }

    verifyAPI.commit();
    runAndVerifyAPI.commit();
    ssSW << verifySS.str();
    ssSW << runAndVerifySS.str();
  }
}

void generateMemMgmtFunctions(std::stringstream &ssSW, std::string stencilName,
                              const std::vector<Field> &fields,
                              CodeGenKind codeGenKind) {
  const std::string wrapperName = stencilName;
  const std::string fullStencilName =
      "dawn_generated::cuda_ico::" + wrapperName;

  const int indentLevel = 0;

  dawn::codegen::MemberFunction setupFun("void", "setup_" + wrapperName, ssSW,
                                         indentLevel,
                                         codeGenKind == CodeGenKind::OnlyDecl);
  setupFun.addArg("dawn::GlobalGpuTriMesh *mesh");
  setupFun.addArg("int k_size");
  setupFun.addArg("cudaStream_t stream");
  for (const auto &fieldName : getUsedFieldsNames(
           fields, {IntendKind::Output, IntendKind::InputOutput})) {
    setupFun.addArg("const int " + fieldName + "_k_size");
  }
  setupFun.finishArgs();
  if (codeGenKind == CodeGenKind::WithBody) {
    std::string k_size_concat_string;
    for (const auto &fieldName : getUsedFieldsNames(
             fields, {IntendKind::Output, IntendKind::InputOutput})) {
      k_size_concat_string += ", " + fieldName + "_k_size";
    }
    setupFun.addStatement(fullStencilName + "::setup(mesh, k_size, stream" +
                          k_size_concat_string + ")");
  }
  setupFun.commit();

  dawn::codegen::MemberFunction freeFun("void", "free_" + wrapperName, ssSW,
                                        indentLevel,
                                        codeGenKind == CodeGenKind::OnlyDecl);
  freeFun.finishArgs();
  if (codeGenKind == CodeGenKind::WithBody) {
    freeFun.startBody();
    freeFun.addStatement(fullStencilName + "::free()");
  }
  freeFun.commit();
}

std::string generateStencilInstantiation(
    const std::vector<Field> &fields,
    const std::vector<dawn::ast::UnstructuredIterationSpace> &allSpaces,
    std::string stencilName) {
  using namespace dawn::codegen;

  std::stringstream ssSW;

  Namespace dawnNamespace("dawn_generated", ssSW);
  Namespace cudaNamespace("cuda_ico", ssSW);

  Class stencilWrapperClass(stencilName, ssSW);
  stencilWrapperClass.changeAccessibility("public");

  generateGpuMesh(stencilWrapperClass, allSpaces);

  generateStencilClasses(stencilWrapperClass, stencilName, fields, allSpaces);

  stencilWrapperClass.commit();

  cudaNamespace.commit();
  dawnNamespace.commit();
  ssSW << "extern \"C\" {\n";
  generateAllAPIRunFunctions(ssSW, stencilName, fields, CallerKind::FromHost,
                             CodeGenKind::WithBody);
  generateAllAPIVerifyFunctions(ssSW, stencilName, fields,
                                CodeGenKind::WithBody);
  generateMemMgmtFunctions(ssSW, stencilName, fields, CodeGenKind::WithBody);
  ssSW << "}\n";

  return ssSW.str();
}

void generateCHeaderSI(std::stringstream &ssSW, std::string stencilName,
                       const std::vector<Field> &fields) {
  ssSW << "extern \"C\" {\n";
  generateAllAPIRunFunctions(ssSW, stencilName, fields, CallerKind::FromDevice,
                             CodeGenKind::OnlyDecl);
  generateAllAPIVerifyFunctions(ssSW, stencilName, fields,
                                CodeGenKind::OnlyDecl);
  generateMemMgmtFunctions(ssSW, stencilName, fields, CodeGenKind::OnlyDecl);
  ssSW << "}\n";
}

std::string generateCHeader(std::string stencilName,
                            const std::vector<Field> &fields) {
  std::stringstream ssSW;
  ssSW << "#pragma once\n";
  ssSW << "#include \"driver-includes/defs.hpp\"\n";
  ssSW << "#include \"driver-includes/cuda_utils.hpp\"\n";

  generateCHeaderSI(ssSW, stencilName, fields);

  return ssSW.str();
}

static void
generateF90InterfaceSI(dawn::codegen::FortranInterfaceModuleGen &fimGen,
                       std::string stencilName,
                       const std::vector<Field> &fields) {
  std::vector<dawn::codegen::FortranInterfaceAPI> interfaces = {
      dawn::codegen::FortranInterfaceAPI("run_" + stencilName),
      dawn::codegen::FortranInterfaceAPI("run_and_verify_" + stencilName)};

  dawn::codegen::FortranWrapperAPI runWrapper =
      dawn::codegen::FortranWrapperAPI("wrap_run_" + stencilName);

  // only from host convenience wrapper takes mesh and k_size
  const int fromDeviceAPIIdx = 0;
  const int runAndVerifyIdx = 1;

  auto addArgsToAPI = [&](dawn::codegen::FortranAPI &api,
                          bool includeSavedState, bool optThresholds) {
    for (const auto &field : getUsedFields(fields)) {
      api.addArg(field.getName(),
                 field.getFortranType() /* Unfortunately we need to know at
                                           codegen time whether we have fields
                                           in SP/DP */
                 ,
                 field.getRank());
    }

    if (includeSavedState) {
      for (const auto &field : getUsedFields(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {
        api.addArg(
            field.getName() + "_before",
            field.getFortranType() /* Unfortunately we need to know at codegen
                                                          time whether we have
                                      fields in SP/DP */
            ,
            2);
      }
    }

    api.addArg("vertical_lower",
               dawn::codegen::FortranAPI::InterfaceType::INTEGER);
    api.addArg("vertical_upper",
               dawn::codegen::FortranAPI::InterfaceType::INTEGER);

    api.addArg("horizontal_lower",
               dawn::codegen::FortranAPI::InterfaceType::INTEGER);
    api.addArg("horizontal_upper",
               dawn::codegen::FortranAPI::InterfaceType::INTEGER);

    if (includeSavedState) {
      for (const auto &field : getUsedFields(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {
        if (optThresholds) {
          api.addOptArg(field.getName() + "_rel_tol",
                        dawn::codegen::FortranAPI::InterfaceType::DOUBLE);
          api.addOptArg(field.getName() + "_abs_tol",
                        dawn::codegen::FortranAPI::InterfaceType::DOUBLE);
        } else {
          api.addArg(field.getName() + "_rel_tol",
                     dawn::codegen::FortranAPI::InterfaceType::DOUBLE);
          api.addArg(field.getName() + "_abs_tol",
                     dawn::codegen::FortranAPI::InterfaceType::DOUBLE);
        }
      }
    }
  };

  addArgsToAPI(interfaces[fromDeviceAPIIdx], /*includeSavedState*/ false,
               false);
  fimGen.addInterfaceAPI(std::move(interfaces[fromDeviceAPIIdx]));
  addArgsToAPI(interfaces[runAndVerifyIdx], /*includeSavedState*/ true, false);
  fimGen.addInterfaceAPI(std::move(interfaces[runAndVerifyIdx]));

  addArgsToAPI(runWrapper, /*includeSavedState*/ true, true);

  std::string fortranIndent = "   ";

  auto getFieldArgs = [&](bool includeSavedState,
                          bool includeScalars =
                              true) -> std::vector<std::string> {
    std::vector<std::string> args;

    for (const auto &field : getUsedFields(fields)) {
      if (field.getRank() > 0 || includeScalars) {
        args.push_back(field.getName());
      }
    }

    if (includeSavedState) {
      for (const auto &field : getUsedFields(
               fields, {IntendKind::Output, IntendKind::InputOutput})) {
        if (field.getRank() > 0 || includeScalars) {
          args.push_back(field.getName() + "_before");
        }
      }
    }
    return args;
  };

  std::vector<std::string> threshold_names;
  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    threshold_names.push_back(field.getName() + "_rel_err_tol");
    threshold_names.push_back(field.getName() + "_abs_err_tol");
  }

  auto genCallArgs = [&](dawn::codegen::FortranWrapperAPI &wrapper,
                         std::string first = "", bool includeSavedState = false,
                         bool includeErrorThreshold = false) {
    wrapper.addBodyLine("( &");

    if (!first.empty()) {
      wrapper.addBodyLine(fortranIndent + first + ", &"); // TODO remove
    }

    for (const auto &arg : getFieldArgs(includeSavedState)) {
      wrapper.addBodyLine(fortranIndent + arg + ", &");
    }

    wrapper.addBodyLine(fortranIndent + "vertical_start, &");
    wrapper.addBodyLine(fortranIndent + "vertical_end, &");
    wrapper.addBodyLine(fortranIndent + "horizontal_start, &");

    if (includeErrorThreshold) {
      wrapper.addBodyLine(fortranIndent + "horizontal_end, &");
      for (int i = 0; i < threshold_names.size() - 1; i++) {
        wrapper.addBodyLine(fortranIndent + threshold_names[i] + ", &");
      }
      wrapper.addBodyLine(fortranIndent +
                          threshold_names[threshold_names.size() - 1] + " &");
    } else {
      wrapper.addBodyLine(fortranIndent + "horizontal_end &");
    }

    wrapper.addBodyLine(")");
  };

  runWrapper.addBodyLine("");

  for (const auto &threshold_name : threshold_names) {
    runWrapper.addBodyLine("real(c_double) :: " + threshold_name);
  }

  runWrapper.addBodyLine("integer(c_int) :: vertical_start");
  runWrapper.addBodyLine("integer(c_int) :: vertical_end");
  runWrapper.addBodyLine("integer(c_int) :: horizontal_start");
  runWrapper.addBodyLine("integer(c_int) :: horizontal_end");

  runWrapper.addBodyLine("vertical_start = vertical_lower-1");
  runWrapper.addBodyLine("vertical_end = vertical_upper");
  runWrapper.addBodyLine("horizontal_start = horizontal_lower-1");
  runWrapper.addBodyLine("horizontal_end = horizontal_upper");

  runWrapper.addBodyLine("");
  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    runWrapper.addBodyLine("if (present(" + field.getName() + "_rel_tol" +
                           ")) then");
    runWrapper.addBodyLine("  " + field.getName() + "_rel_err_tol" + " = " +
                           field.getName() + "_rel_tol");
    runWrapper.addBodyLine("else");
    runWrapper.addBodyLine("  " + field.getName() + "_rel_err_tol" +
                           " = DEFAULT_RELATIVE_ERROR_THRESHOLD");
    runWrapper.addBodyLine("endif");
    runWrapper.addBodyLine("");

    runWrapper.addBodyLine("if (present(" + field.getName() + "_abs_tol" +
                           ")) then");
    runWrapper.addBodyLine("  " + field.getName() + "_abs_err_tol" + " = " +
                           field.getName() + "_abs_tol");
    runWrapper.addBodyLine("else");
    runWrapper.addBodyLine("  " + field.getName() + "_abs_err_tol" +
                           " = DEFAULT_ABSOLUTE_ERROR_THRESHOLD");
    runWrapper.addBodyLine("endif");
    runWrapper.addBodyLine("");
  }
  runWrapper.addACCLine("host_data use_device( &");
  auto fieldArgs =
      getFieldArgs(/*includeSavedState*/ true, /*includeScalars*/ false);
  for (int i = 0; i < fieldArgs.size(); ++i) {
    runWrapper.addACCLine(fortranIndent + fieldArgs[i] +
                          (i == (fieldArgs.size() - 1) ? " &" : ", &"));
  }
  runWrapper.addACCLine(")");
  runWrapper.addBodyLine("#ifdef __DSL_VERIFY", /*withIndentation*/ false);
  runWrapper.addBodyLine("call run_and_verify_" + stencilName + " &");
  genCallArgs(runWrapper, "", /*includeSavedState*/ true,
              /*includeErrorThreshold*/ true);
  runWrapper.addBodyLine("#else", /*withIndentation*/ false);
  runWrapper.addBodyLine("call run_" + stencilName + " &");
  genCallArgs(runWrapper, "", /*includeSavedState*/ false,
              /*includeErrorThreshold*/ false);
  runWrapper.addBodyLine("#endif", /*withIndentation*/ false);
  runWrapper.addACCLine("end host_data");

  fimGen.addWrapperAPI(std::move(runWrapper));

  std::vector<std::string> verticalBoundNames;
  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    verticalBoundNames.push_back(field.getName() + "_kvert_max");
  }

  // memory management functions for production interface
  dawn::codegen::FortranInterfaceAPI setup("setup_" + stencilName);
  dawn::codegen::FortranInterfaceAPI free("free_" + stencilName);
  setup.addArg("mesh", dawn::codegen::FortranAPI::InterfaceType::OBJ);
  setup.addArg("k_size", dawn::codegen::FortranAPI::InterfaceType::INTEGER);
  setup.addArg("stream",
               dawn::codegen::FortranAPI::InterfaceType::CUDA_STREAM_T);
  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    setup.addArg(field.getName() + "_kmax",
                 dawn::codegen::FortranAPI::InterfaceType::INTEGER);
  }

  fimGen.addInterfaceAPI(std::move(setup));
  fimGen.addInterfaceAPI(std::move(free));

  dawn::codegen::FortranWrapperAPI setupWrapper("wrap_setup_" + stencilName);
  setupWrapper.addArg("mesh", dawn::codegen::FortranAPI::InterfaceType::OBJ);
  setupWrapper.addArg("k_size",
                      dawn::codegen::FortranAPI::InterfaceType::INTEGER);
  setupWrapper.addArg("stream",
                      dawn::codegen::FortranAPI::InterfaceType::CUDA_STREAM_T);

  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    setupWrapper.addOptArg(field.getName() + "_kmax",
                           dawn::codegen::FortranAPI::InterfaceType::INTEGER);
  }

  setupWrapper.addBodyLine("");

  for (const auto &verticalBoundName : verticalBoundNames) {
    setupWrapper.addBodyLine("integer(c_int) :: " + verticalBoundName);
  }

  setupWrapper.addBodyLine("");

  for (const auto &field :
       getUsedFields(fields, {IntendKind::Output, IntendKind::InputOutput})) {
    setupWrapper.addBodyLine("if (present(" + field.getName() + "_kmax" +
                             ")) then");
    setupWrapper.addBodyLine("  " + field.getName() + "_kvert_max" + " = " +
                             field.getName() + "_kmax");
    setupWrapper.addBodyLine("else");
    setupWrapper.addBodyLine("  " + field.getName() + "_kvert_max" +
                             " = k_size");
    setupWrapper.addBodyLine("endif");
    setupWrapper.addBodyLine("");
  }

  setupWrapper.addBodyLine("call setup_" + stencilName + " &");

  setupWrapper.addBodyLine("( &");
  setupWrapper.addBodyLine(fortranIndent + "mesh, &");
  setupWrapper.addBodyLine(fortranIndent + "k_size, &");
  setupWrapper.addBodyLine(fortranIndent + "stream, &");

  for (int i = 0; i < verticalBoundNames.size() - 1; i++) {
    setupWrapper.addBodyLine(fortranIndent + verticalBoundNames[i] + ", &");
  }

  setupWrapper.addBodyLine(
      fortranIndent + verticalBoundNames[verticalBoundNames.size() - 1] + " &");

  setupWrapper.addBodyLine(")");

  fimGen.addWrapperAPI(std::move(setupWrapper));
}

std::string generateF90Interface(std::string moduleName,
                                 std::string stencilName,
                                 const std::vector<Field> &fields) {
  std::stringstream ss;
  ss << "#define DEFAULT_RELATIVE_ERROR_THRESHOLD 1.0d-12" << std::endl;
  ss << "#define DEFAULT_ABSOLUTE_ERROR_THRESHOLD 0.0d1" << std::endl;
  dawn::codegen::IndentedStringStream iss(ss);

  dawn::codegen::FortranInterfaceModuleGen fimGen(iss, std::move(moduleName));

  generateF90InterfaceSI(fimGen, stencilName, fields);

  fimGen.commit();

  return iss.str();
}

void generateCode(
    std::filesystem::path workPath, std::string stencilName, int blockSize,
    int levelsPerThread, const std::vector<Field> &fields,
    const std::vector<dawn::ast::UnstructuredIterationSpace> &allSpaces) {

  std::vector<std::string> ppDefines{
      "#include \"" + stencilName + ".hpp\"",
      "#include <gridtools/fn/cartesian.hpp>",
      "#include <gridtools/fn/backend/gpu.hpp>",
      "#include <gridtools/stencil/global_parameter.hpp>",
      "#include <gridtools/common/array.hpp>",
      "#include \"driver-includes/unstructured_interface.hpp\"",
      "#include \"driver-includes/unstructured_domain.hpp\"",
      "#include \"driver-includes/defs.hpp\"",
      "#include \"driver-includes/cuda_utils.hpp\"",
      "#include \"driver-includes/cuda_verify.hpp\"",
      "#include \"driver-includes/to_vtk.h\"",
      "#include \"driver-includes/to_json.hpp\"",
      "#include \"driver-includes/verification_metrics.hpp\"",
      "#define GRIDTOOLS_DAWN_NO_INCLUDE", // Required to not include gridtools
                                           // from math.hpp
      "#include \"driver-includes/math.hpp\"",
      "#include <chrono>",
      "#define BLOCK_SIZE " +
          std::to_string(
              blockSize), // std::to_string(codeGenOptions_.BlockSize),
      "#define LEVELS_PER_THREAD " +
          std::to_string(
              levelsPerThread), // +
                                // std::to_string(codeGenOptions_.LevelsPerThread),
      "namespace {",
      "    template <int... sizes>",
      "    using block_sizes_t =",
      "        "
      "gridtools::meta::zip<gridtools::meta::iseq_to_list<std::make_integer_"
      "sequence<int, sizeof...(sizes)>,",
      "                                 gridtools::meta::list,",
      "                                 gridtools::integral_constant>,",
      "            gridtools::meta::list<gridtools::integral_constant<int, "
      "sizes>...>>;",
      "",
      "    using fn_backend_t = "
      "gridtools::fn::backend::gpu<block_sizes_t<BLOCK_SIZE, "
      "LEVELS_PER_THREAD>>;",
      "} // namespace",
      "using namespace gridtools::dawn;",
  };

  ppDefines.push_back("#define nproma 50000");
  ppDefines.push_back("");

ppDefines.push_back(R"(template <int N>
struct neighbor_table_fortran{
   const int *raw_ptr_fortran;
  __device__
  friend inline constexpr gridtools::array<int, N> neighbor_table_neighbors(neighbor_table_fortran const &table, int index) {
    gridtools::array<int, N> ret{};
    for (int i = 0; i < N; i++) {
      ret[i] = table.raw_ptr_fortran[index+nproma*i];
    }
    return ret;
  }
};

template <int N>
struct neighbor_table_4new_sparse {
  __device__
  friend inline constexpr gridtools::array<int, N> neighbor_table_neighbors(neighbor_table_4new_sparse const&, int index) {
    gridtools::array<int, N> ret{};
    for (int i = 0; i < N; i++) {
      ret[i] = index+nproma*i;
    }
    return ret;
  }
};)");

  ppDefines.push_back("");
  ppDefines.push_back(R"(template<class Ptr, class StrideMap>
auto get_sid(Ptr ptr, StrideMap const& strideMap) {
  using namespace gridtools;
  using namespace fn;
  return sid::synthetic()
    .set<sid::property::origin>(
        sid::host_device::simple_ptr_holder<Ptr>(
            ptr
        )
    )
    .template set<sid::property::strides>(strideMap)
    .template set<sid::property::strides_kind, sid::unknown_kind>();
}
)");

  std::filesystem::path cppPath = workPath;
  cppPath.append(stencilName + ".cpp");

  std::ofstream file;
  file.open(cppPath);
  if (file) {
    for (const auto &line : ppDefines) {
      file << line << "\n";
    }
    file << generateStencilInstantiation(fields, allSpaces, stencilName);
    file.close();
  } else {
    throw std::runtime_error("Error writing to " + cppPath.string() + ": " +
                             std::strerror(errno));
  }

  std::filesystem::path headerPath = workPath;
  headerPath.append(stencilName + ".h");

  std::ofstream headerFile;
  headerFile.open(headerPath);
  if (headerFile) {
    headerFile << generateCHeader(stencilName, fields);
    headerFile.close();
  } else {
    throw std::runtime_error("Error writing to " + headerPath.string() + ": " +
                             std::strerror(errno));
  }

  std::filesystem::path f90Path = workPath;
  f90Path.append(stencilName + ".f90");

  std::string moduleName = f90Path.filename().replace_extension("").string();
  std::ofstream interfaceFile;
  interfaceFile.open(f90Path);
  if (interfaceFile) {
    interfaceFile << generateF90Interface(moduleName, stencilName, fields);
    interfaceFile.close();
  } else {
    throw std::runtime_error("Error writing to " + f90Path.string() + ": " +
                             std::strerror(errno));
  }
}
//} // namespace cudaico
//} // namespace codegen
//} // namespace dawn

void skipWhitespaces(const std::string &string, size_t &pos) {
  while (string[pos] == ' ' && pos < string.size()) {
    pos++;
  }
}

void deleteLeadingWhitespaces(std::string &string) {
  size_t marker = 0;
  skipWhitespaces(string, marker);
  string = string.substr(size_t(0), marker);
}

std::string parseWord(std::string &string) {
  size_t marker = string.find(' ');
  std::string word = string.substr(size_t(0), marker);
  skipWhitespaces(string, marker);
  if (marker == std::string::npos) {
    string = std::string();
  } else {
    string = string.substr(marker, std::string::npos);
  }

  return word;
}

// https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
char *getCmdOption(char **begin, char **end, const std::string &option) {
  char **itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
  return std::find(begin, end, option) != end;
}

int main(int argc, char **argv) {

  std::filesystem::path dataPath(argv[1]);
  std::string stencilName = dataPath.stem();
  std::ifstream dataFile(dataPath);

  int blockSize = 128;
  if (cmdOptionExists(argv, argv + argc, "--block-size")) {
    blockSize = atoi(getCmdOption(argv, argv + argc, "--block-size"));
  }

  int levelsPerThread = 1;
  if (cmdOptionExists(argv, argv + argc, "--levels-per-thread")) {
    levelsPerThread =
        atoi(getCmdOption(argv, argv + argc, "--levels-per-thread"));
  }

  std::string line;

  auto sparseStringToSpace = [](std::string sparseString) {
    int includeCenter = sparseString.back() == 'O';
    if (includeCenter) {
      sparseString.pop_back();
    }
    auto sparseTokens = dawn::tokenize(sparseString, '2');
    std::optional<dawn::ast::LocationType> newSparseRoot{};

    auto sparseCharToSpace = [] (char c) {
      switch (c) {
            case 'E':
              return dawn::ast::LocationType::Edges;
              break;
            case 'V':
              return dawn::ast::LocationType::Vertices;
              break;
            case 'C':
              return dawn::ast::LocationType::Cells;
              break;
            default:
              DAWN_ASSERT_MSG(0,
                              "malformed sparse dimension. please use single upper "
                              "case char divided by 2. examples: C2E, C2E2V");
            }
    };

    if (std::any_of(sparseTokens.begin(), sparseTokens.end(),
                    [](const auto &token) { return token.size() > 1; })) {
      std::string newSparseToken = sparseTokens[1];
      sparseTokens.clear();
      for (auto it : newSparseToken) {
        sparseTokens.push_back(std::string{it});
      }
      newSparseRoot = sparseCharToSpace(sparseTokens[0][0]);
    }
    dawn::ast::NeighborChain chain;
    for (auto token : sparseTokens) {
      chain.push_back(sparseCharToSpace(token[0]));
    }
    return dawn::ast::UnstructuredIterationSpace(std::move(chain),
                                                 includeCenter, newSparseRoot);
  };

  auto sparseStringToSpaceNew = [](std::string sparseString) {
    dawn::ast::NeighborChain chain;
    for (auto token : sparseString) {
      switch (token) {
      case 'E':
        chain.push_back(dawn::ast::LocationType::Edges);
        break;
      case 'V':
        chain.push_back(dawn::ast::LocationType::Vertices);
        break;
      case 'C':
        chain.push_back(dawn::ast::LocationType::Cells);
        break;
      default:
        DAWN_ASSERT_MSG(0, "malformed new sparse dimension. please use single "
                           "upper case chars examples: CE, CVC");
      }
    }
    return chain;
  };

  // read header line
  std::vector<dawn::ast::UnstructuredIterationSpace> allSpaces{};
  {
    std::getline(dataFile, line);
    auto chainTokens = dawn::tokenize(line, ',');

    auto endsWith = [](std::string const &value, std::string const &ending) {
      if (ending.size() > value.size())
        return false;
      return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    };

    for (auto chainToken : chainTokens) {
      if (endsWith(chainToken, "Dim") || endsWith(chainToken, "dim")) {
        chainToken = chainToken.substr(0, chainToken.size() - 3);
      }
      auto space = sparseStringToSpace(chainToken);
      allSpaces.push_back(space);
    }
  }

  // read fields line by line
  std::vector<Field> fields;
  while (std::getline(dataFile, line)) {
    std::string fieldName = parseWord(line);

    size_t lower = 0;
    size_t upper = 0;

    unsigned int bracketCounter = 0;

    std::vector<size_t> openingBrackets;
    std::vector<size_t> closingBrackets;

    for (size_t pos = upper; pos < line.size(); pos++) {

      if (line[pos] == '[') {
        openingBrackets.push_back(pos);
        bracketCounter++;
      }

      if (line[pos] == ']') {
        closingBrackets.push_back(pos);
        bracketCounter--;
        if (bracketCounter == 0) {
          upper = pos + 1;
          break;
        }
      }
    }

    std::reverse(closingBrackets.begin(), closingBrackets.end());

    std::vector<std::string> layers;
    for (size_t counter = 0; counter < closingBrackets.size(); counter++) {
      layers.push_back(
          line.substr(openingBrackets[counter] + 1,
                      closingBrackets[counter] - openingBrackets[counter] - 1));
    }

    size_t typePos = layers[0].find("dtype=") + 6;

    std::string basicType =
        layers[0].substr(typePos, layers[0].size() - typePos);

    std::string cppType;
    dawn::codegen::FortranAPI::InterfaceType fortranType;

    if (basicType == "float64") {
      cppType = std::string("double");
      fortranType = dawn::codegen::FortranAPI::InterfaceType::DOUBLE;
    } else if (basicType == "int32") {
      cppType = std::string("int");
      fortranType = dawn::codegen::FortranAPI::InterfaceType::INTEGER;
    } else if (basicType == "bool") {
      // NOTE: This should reliably work when using the nvhpc compilers.
      // See: https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/#intr-lang-data-types
      // It additionally relies on the fact that the GT4Py generated code is generic in the datatype.
      cppType = std::string("int");
      fortranType = dawn::codegen::FortranAPI::InterfaceType::BOOLEAN;
    } else {
      throw std::runtime_error("No types supported but float64, int32 and bool");

    }

    std::stringstream dims(layers[1]);
    std::string dimensionString = dims.str();

    bool isVertical = false;
    bool isHorizontal = false;
    bool isSparse = false;
    bool includeCenter = false;

    dawn::ast::LocationType denseLocation;
    dawn::ast::NeighborChain sparseLocation;

    auto locationStringToLocation = [](const std::string &locationString) {
      if (locationString == "Edge") {
        return dawn::ast::LocationType::Edges;
      } else if (locationString == "Vertex") {
        return dawn::ast::LocationType::Vertices;
      } else if (locationString == "Cell") {
        return dawn::ast::LocationType::Cells;
      } else {
        throw std::runtime_error(
            "Location needs to be one of Edge, Vertex, Cell");
      }
    };

    auto locationCharToLocation = [](char locationChar) {
      if (locationChar == 'E') {
        return dawn::ast::LocationType::Edges;
      } else if (locationChar == 'V') {
        return dawn::ast::LocationType::Vertices;
      } else if (locationChar == 'C') {
        return dawn::ast::LocationType::Cells;
      } else {
        throw std::runtime_error(
            "Location needs to be one of Edge, Vertex, Cell");
      }
    };

    auto newSparse = [](const std::string &locationString) {
      return !(locationString == "Edge" | locationString == "Cell" |
               locationString == "Vertex" | locationString == "K");
    };

    auto dimTokens = dawn::tokenize(dimensionString, ',');
    bool isNewSparseFormulation = false;

    if (dimTokens.size() > 0) { // dimTokens.size() == 0 is a scalar
      if (dimTokens.size() == 1 && !newSparse(dimTokens[0])) {
        if (dimTokens[0] == "K") {
          isVertical = true;
        } else {
          isHorizontal = true;
          denseLocation = locationStringToLocation(dimTokens[0]);
        }
      } else if (dimTokens.size() == 2 && dimTokens[1] == "K") {
        isHorizontal = true;
        isVertical = true;
        denseLocation = locationStringToLocation(dimTokens[0]);
      } else { // sparse
        isSparse = true;
        isHorizontal = true;
        if (newSparse(dimTokens[0])) {
          isVertical = false;     // TODO
          std::string newSparseToken = dimTokens[0];
          denseLocation = locationCharToLocation(newSparseToken[0]);
          auto chain = sparseStringToSpaceNew(newSparseToken);
          sparseLocation = chain;
          includeCenter = false;  //TODO
          isNewSparseFormulation = true;
        } else {
          isVertical = dimTokens.size() == 3 && dimTokens[2] == "K";
          denseLocation = locationStringToLocation(dimTokens[0]);
          auto space = sparseStringToSpace(dimTokens[1]);
          sparseLocation = space.Chain;
          includeCenter = space.IncludeCenter;
        }
      }
    }

    skipWhitespaces(line, upper);
    line = line.substr(upper, line.size() - upper);

    std::string intentString = parseWord(line);

    IntendKind intend;

    if (intentString == "inout") {
      intend = IntendKind::InputOutput;
    } else if (intentString == "out") {
      intend = IntendKind::Output;
    } else if (intentString == "in") {
      intend = IntendKind::Input;
    } else {
      throw std::runtime_error(
          "Invalid intendString, which could not be parsed. Needs to be one of "
          "the following: in, out, inout");
    }

    std::optional<dawn::ast::LocationType> optionalHorizontalLocation;
    if (isHorizontal) {
      optionalHorizontalLocation = denseLocation;
    }

    std::optional<dawn::ast::NeighborChain> optionalSparseLocation;
    if (isSparse) {
      optionalSparseLocation = sparseLocation;
    }

    fields.emplace_back(Field(fieldName, cppType, fortranType,
                              intend, optionalHorizontalLocation,
                              optionalSparseLocation, isVertical,
                              includeCenter, isNewSparseFormulation));
  }

  generateCode(dataPath.remove_filename(), stencilName, blockSize,
               levelsPerThread, fields, allSpaces);

  return 0;
}
