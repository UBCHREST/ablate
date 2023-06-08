#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "levelSet/lsSupport.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/boxMesh.hpp"
#include "parameters/mapParameters.hpp"

using namespace ablate;

void computeData(const PetscReal *xyz, const PetscInt dim, PetscReal *c, PetscReal *g) {

  PetscReal x = 0.0, y = 0.0, z = 0.0;
  x = xyz[0];
  if (dim > 1) y = xyz[1];
  if (dim > 2) z = xyz[2];

  if (c) {
    *c = (sin(x) + cos(2.0*y))*cos(z);
  }

  if (g) {
    g[0] = cos(x)*cos(z);
    g[1] = -2.0*sin(2.0*y)*cos(z);
    g[2] = -(cos(2.0*y) + sin(x))*sin(z);
  }
}

void setVertexData(const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {


  ablate::domain::Range range;
  PetscReal    *array;
  PetscReal     *val;
  Vec           vec = subDomain->GetVec(*field);
  DM            dm  = subDomain->GetFieldDM(*field);
  PetscInt      dim = subDomain->GetDimensions();

  ablate::domain::GetRange(dm, nullptr, 0, range);

  VecGetArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt v = range.start; v < range.end; ++v) {
    PetscInt vert = range.points ? range.points[v] : v;
    PetscScalar *coords;

    DMPlexPointLocalFieldRef(dm, vert, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

    DMPlexGetVertexCoordinates(dm, 1, &vert, &coords);

    computeData(coords, dim, val, NULL);

    DMPlexRestoreVertexCoordinates(dm, 1, &vert, &coords);
  }

  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
  ablate::domain::RestoreRange(range);
}

struct cellGradTestFixture_Parameters {
    testingResources::MpiTestParameter mpiTestParameter;
    PetscInt dim;                     // Dimension of the mesh
    PetscScalar h;                    // Grid spacing for the mesh
    PetscBool simplex;                // Whether the mesh is a simplex mesh
    PetscReal trueVal;                // True value at cell-center
    std::vector<PetscReal> trueGrad;  // True gradient at cell-center
};

class cellGradTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<cellGradTestFixture_Parameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


TEST_P(cellGradTestFixture, cellGrad) {
  StartWithMPI

  // initialize petsc and mpi
  environment::RunEnvironment::Initialize(argc, argv);
  utilities::PetscUtilities::Initialize();

  {
    auto testingParam = GetParam();
    PetscInt dim = testingParam.dim;
    PetscScalar h = testingParam.h;
    PetscBool isSimplex = testingParam.simplex;

    std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {
        std::make_shared<ablate::domain::FieldDescription>("func", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FEM, ablate::domain::Region::ENTIREDOMAIN, ablate::parameters::MapParameters::Create({{"petscspace_degree", 1}}))
      };


    // Make a one-cell mesh.
    std::shared_ptr<ablate::domain::BoxMesh> mesh;
    switch (dim) {
      case 1:
        mesh = std::make_shared<ablate::domain::BoxMesh>(
          "mesh",
          fieldDescriptor,
          std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(1)},
          std::vector<int>{1},
          std::vector<double>{0.0},
          std::vector<double>{h},
          std::vector<std::string>{"NONE"}, //NONE, GHOSTED, MIRROR, PERIODIC
          isSimplex);
        break;
      case 2:
        mesh = std::make_shared<ablate::domain::BoxMesh>(
          "mesh",
          fieldDescriptor,
          std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(1)},
          std::vector<int>{1, 1},
          std::vector<double>{0.0, 0.0},
          std::vector<double>{h, h},
          std::vector<std::string>{"NONE", "NONE"}, //NONE, GHOSTED, MIRROR, PERIODIC
          isSimplex);
        break;
      case 3:
        mesh = std::make_shared<ablate::domain::BoxMesh>(
          "mesh",
          fieldDescriptor,
          std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(1)},
          std::vector<int>{1, 1, 1},
          std::vector<double>{0.0, 0.0, 0.0},
          std::vector<double>{h, h, h},
          std::vector<std::string>{"NONE", "NONE", "NONE"}, //NONE, GHOSTED, MIRROR, PERIODIC
          isSimplex);
        break;
      }

    mesh->InitializeSubDomains();

    std::shared_ptr<ablate::domain::SubDomain> subDomain = mesh->GetSubDomain(ablate::domain::Region::ENTIREDOMAIN);

    const ablate::domain::Field *funcField = &(subDomain->GetField("func"));

    setVertexData(funcField, subDomain);

    PetscScalar f, g[3] = {0.0, 0.0, 0.0};

    ablate::levelSet::Utilities::CellValGrad(subDomain, funcField, 0, &f, g);

    PetscReal trueValue = testingParam.trueVal;
    std::vector<PetscReal> trueGrad = testingParam.trueGrad;
    ASSERT_DOUBLE_EQ(trueValue, f);
    for (PetscInt i = 0; i < dim; ++i) {
      ASSERT_DOUBLE_EQ(trueGrad[i], g[i]);
    }

  }

  ablate::environment::RunEnvironment::Finalize();
  EndWithMPI

}






INSTANTIATE_TEST_SUITE_P(
  MeshTests, cellGradTestFixture,
  testing::Values(
     (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("1D_h10", 1),
      .dim = 1,
      .h = 0.100000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0499167083234142e+00,
      .trueGrad = {+9.9833416646828210e-01}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("1D_h05", 1),
      .dim = 1,
      .h = 0.050000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0249895846353392e+00,
      .trueGrad = {+9.9958338541356717e-01}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("2D_h10_simplex", 1),
      .dim = 2,
      .h = 0.100000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0266333314960232e+00,
      .trueGrad = {+9.9833416646828210e-01, -1.9933422158758374e-01}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("2D_h05_simplex", 1),
      .dim = 2,
      .h = 0.050000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0149944448495680e+00,
      .trueGrad = {+9.9958338541356717e-01, -9.9916694439483589e-02}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("2D_h10", 1),
      .dim = 2,
      .h = 0.100000,
      .simplex = PETSC_FALSE,
      .trueVal = +1.0399499972440349e+00,
      .trueGrad = {+9.9833416646828244e-01, -1.9933422158758302e-01}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("2D_h05", 1),
      .dim = 2,
      .h = 0.050000,
      .simplex = PETSC_FALSE,
      .trueVal = +1.0224916672743520e+00,
      .trueGrad = {+9.9958338541356617e-01, -9.9916694439484435e-02}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("3D_h10_simplex", 1),
      .dim = 3,
      .h = 0.100000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0336178912321645e+00,
      .trueGrad = {+9.9833416646828210e-01, -1.9933422158758374e-01, -5.3950018887232165e-02}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("3D_h05_simplex", 1),
      .dim = 3,
      .h = 0.050000,
      .simplex = PETSC_TRUE,
      .trueVal = +1.0209162193289121e+00,
      .trueGrad = {+9.9958338541356495e-01, -9.9916694439483589e-02, -2.6119141195715123e-02}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("3D_h10", 1),
      .dim = 3,
      .h = 0.100000,
      .simplex = PETSC_FALSE,
      .trueVal = +1.0373522880913604e+00,
      .trueGrad = {+9.9584041022179448e-01, -1.9883630117484147e-01, -5.1954183053486667e-02}
    },
    (cellGradTestFixture_Parameters){
      .mpiTestParameter = testingResources::MpiTestParameter("3D_h05", 1),
      .dim = 3,
      .h = 0.050000,
      .simplex = PETSC_FALSE,
      .trueVal = +1.0218527431081470e+00,
      .trueGrad = {+9.9895877594092730e-01, -9.9854259514359764e-02, -2.5556966648190835e-02}
    }
));




