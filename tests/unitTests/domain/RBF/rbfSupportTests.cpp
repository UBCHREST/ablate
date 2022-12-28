#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/RBF/rbfSupport.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

struct RBFSupportParameters {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    bool meshSimplex;
    std::vector<PetscScalar> xyz;
    PetscInt expectedCell;
};

class RBFSupportTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFSupportParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


TEST_P(RBFSupportTestFixture, ShouldReturnCellIDs) {
    StartWithMPI
        {
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();


            // 2DRect
            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{}, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

            PetscInt cell = 0;
            DMPlexGetContainingCell(mesh->GetDM(), &testingParam.xyz[0], &cell) >> ablate::checkError;
            ASSERT_EQ(cell, testingParam.expectedCell);

//            // 2DTri
//            auto mesh = std::make_shared<domain::BoxMesh>(
//                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{}, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, std::vector<int>{10, 5}, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0}, std::vector<std::string>{}, true);

//// 3DRect
//            auto mesh = std::make_shared<domain::BoxMesh>(
//                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{}, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, std::vector<int>{2, 2, 2}, std::vector<double>{0.0, 0.0, 0.0}, std::vector<double>{1.0, 1.0, 1.0}, std::vector<std::string>{}, false);
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFSupportTestFixture,
    testing::Values((RBFSupportParameters){.mpiTestParameter = {.testName = "2DQuad", .nproc = 1},
                                              .meshFaces = {10, 5},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshSimplex = false,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = 15},
                    (RBFSupportParameters){.mpiTestParameter = {.testName = "2DSimplex", .nproc = 1},
                                              .meshFaces = {10, 5},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshSimplex = true,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = 49},
                    (RBFSupportParameters){.mpiTestParameter = {.testName = "3DQuad", .nproc = 1},
                                              .meshFaces = {2, 2, 2},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .meshSimplex = false,
                                              .xyz = {0.6, 0.42, 0.8},
                                              .expectedCell = 5},
                    (RBFSupportParameters){.mpiTestParameter = {.testName = "3DSimplex", .nproc = 1},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {2.0, 1.0, 1.0},
                                              .meshSimplex = true,
                                              .xyz = {0.1, 0.9, 0.9},
                                              .expectedCell = 4}
                  ),
    [](const testing::TestParamInfo<RBFSupportParameters>& info) { return info.param.mpiTestParameter.getTestName(); });

