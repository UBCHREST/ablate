#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "PetscTestErrorChecker.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/RBF/ga.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"

using namespace ablate;

struct RBFParameters_Derivative {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    bool meshSimplex;
    std::shared_ptr<domain::rbf::RBF> rbf;
    PetscInt c;
    std::vector<PetscInt> dx;
    std::vector<PetscInt> dy;
    std::vector<PetscInt> dz;
    std::vector<PetscReal> expectedDerivatives;
};



class RBFTestFixture_Derivative : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFParameters_Derivative> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


void RBFTestFixture_SetData(ablate::solver::Range cellRange, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

  DM            dm = subDomain->GetDM();
  PetscReal    *array, *val;
  Vec           solVec = subDomain->GetSolutionVector();
  const ablate::domain::Field field = subDomain->GetField("fieldA");

  VecGetArray(solVec, &array) >> ablate::checkError;
  for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
    PetscInt cell = cellRange.points ? cellRange.points[c] : c;
    DMPlexPointLocalFieldRef(dm, cell, field.id, array, &val)  >> ablate::checkError;
    *val = 1.0;
  }
  VecRestoreArray(solVec, &array) >> ablate::checkError;

}


// This tests single-cell functions EvalDer and Interpolate.
TEST_P(RBFTestFixture_Derivative, CheckPointFunctions) {
    StartWithMPI
        {
            // initialize petsc and mpi
            environment::RunEnvironment::Initialize(argc, argv);
            utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();

            // Make the field
            std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM)};

            // Create the mesh
            // Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", fieldDescriptor, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

            mesh->InitializeSubDomains();

            auto subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

            testingParam.rbf->Setup(subDomain);


            // Initialize the RBF
            PetscInt depth;
            DMPlexGetDepth(mesh->GetDM(), &depth) >> checkError;
            solver::Range cellRange;
            testingParam.rbf->GetRange(subDomain, nullptr, depth, cellRange);
            testingParam.rbf->Initialize(cellRange);





            testingParam.rbf->RestoreRange(cellRange);







//GA::GA(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//IMQ::IMQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//MQ::MQ(PetscInt p, PetscReal scale, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), scale(scale) {};
//PHS::PHS(PetscInt p, PetscInt phsOrder, bool hasDerivatives, bool hasInterpolation) : RBF(p, hasDerivatives, hasInterpolation), phsOrder(phsOrder) {};





        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFTestFixture_Derivative,
    testing::Values((RBFParameters_Derivative){.mpiTestParameter = {.testName = "MQ_Der1", .nproc = 1},
                                              .meshFaces = {10, 10},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshSimplex = false,
                                              .rbf = std::make_shared<domain::rbf::GA>(8, 0.1, true, false),
                                              }
                  ),
    [](const testing::TestParamInfo<RBFParameters_Derivative>& info) { return info.param.mpiTestParameter.getTestName(); });
