#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/RBF/rbfSupport.hpp"
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "utilities/petscUtilities.hpp"

using namespace ablate;

/********************   Begin unit tests for DMPlexGetContainingCell    *************************/

struct RBFSupportParameters_ReturnID {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::vector<std::shared_ptr<domain::modifiers::Modifier>> meshModifiers;
    bool meshSimplex;
    std::vector<PetscScalar> xyz;
    std::vector<PetscInt> expectedCell;
};

class RBFSupportTestFixture_ReturnID : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFSupportParameters_ReturnID> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};


TEST_P(RBFSupportTestFixture_ReturnID, ShouldReturnCellIDs) {
    StartWithMPI
        {
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();

            // Create the mesh
            // Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
            auto mesh = std::make_shared<domain::BoxMesh>(
                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{}, testingParam.meshModifiers, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

            PetscInt cell = -2;
            DMPlexGetContainingCell(mesh->GetDM(), &testingParam.xyz[0], &cell) >> ablate::checkError;

            PetscMPIInt rank;
            MPI_Comm_rank(PetscObjectComm((PetscObject)mesh->GetDM()), &rank);
            ASSERT_EQ(cell, testingParam.expectedCell[rank]);
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFSupportTestFixture_ReturnID,
    testing::Values((RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "2DQuad", .nproc = 1},
                                              .meshFaces = {10, 5},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = false,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = {15}},
                    (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "2DSimplex", .nproc = 1},
                                              .meshFaces = {10, 5},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = true,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = {49}},
                    (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "3DQuad", .nproc = 1},
                                              .meshFaces = {2, 2, 2},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {1.0, 1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = false,
                                              .xyz = {0.6, 0.42, 0.8},
                                              .expectedCell = {5}},
                    (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "3DSimplex", .nproc = 1},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {2.0, 1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = true,
                                              .xyz = {0.1, 0.9, 0.9},
                                              .expectedCell = {4}},
                    (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "3DSimplexFail", .nproc = 1},
                                              .meshFaces = {1, 1, 1},
                                              .meshStart = {0.0, 0.0, 0.0},
                                              .meshEnd = {2.0, 1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = true,
                                              .xyz = {2.1, 0.9, 0.9},
                                              .expectedCell = {-1}},
                     (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "2DQuadMPI", .nproc = 2},
                                              .meshFaces = {10, 10},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshModifiers = {},
                                              .meshSimplex = false,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = {10, -1}},
                      (RBFSupportParameters_ReturnID){.mpiTestParameter = {.testName = "2DQuadMPIMod", .nproc = 2}, // This is mainly here to check if there is ever a change in how DMLocatePoints functions
                                              .meshFaces = {10, 10},
                                              .meshStart = {0.0, 0.0},
                                              .meshEnd = {1.0, 1.0},
                                              .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(1)},
                                              .meshSimplex = false,
                                              .xyz = {0.55, 0.25},
                                              .expectedCell = {10, -1}}
                  ),
    [](const testing::TestParamInfo<RBFSupportParameters_ReturnID>& info) { return info.param.mpiTestParameter.getTestName(); });

/********************   End unit tests for DMPlexGetContainingCell    *************************/

/********************   Begin unit tests for DMPlexGetNeighborCells    *************************/

//struct RBFSupportParameters_NeighborCells {
//    testingResources::MpiTestParameter mpiTestParameter;
//    std::vector<int> meshFaces;
//    std::vector<double> meshStart;
//    std::vector<double> meshEnd;
//    bool meshSimplex;
//    PetscInt centerCell;
//    PetscInt numLevels;
//    PetscInt maxDistance;
//    PetscInt minNumberCells;
//    PetscBool useVertices;
//    PetscInt expectedNumberOfCells;
//    std::vector<PetscInt> expectedCellList;
//};

////PetscErrorCode DMPlexGetNeighborCells(DM dm, PetscInt p, PetscInt levels, PetscReal maxDist, PetscInt minNumberCells, PetscBool useVertices, PetscInt *nCells, PetscInt *cells[]) {


//class RBFSupportTestFixture_NeighborCells : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFSupportParameters_NeighborCells> {
//   public:
//    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
//};



//TEST_P(RBFSupportTestFixture_NeighborCells, ShouldReturnNeighborCells) {
//    StartWithMPI
//        {
//            // initialize petsc and mpi
//            ablate::environment::RunEnvironment::Initialize(argc, argv);
//            ablate::utilities::PetscUtilities::Initialize();

//            auto testingParam = GetParam();

//            // Create the mesh
//            auto mesh = std::make_shared<domain::BoxMesh>(
//                "mesh", std::vector<std::shared_ptr<domain::FieldDescriptor>>{}, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{}, testingParam.meshFaces, testingParam.meshStart, testingParam.meshEnd, std::vector<std::string>{}, testingParam.meshSimplex);

//            PetscInt cell = 0;
//            DMPlexGetContainingCell(mesh->GetDM(), &testingParam.xyz[0], &cell) >> ablate::checkError;
//            ASSERT_EQ(cell, testingParam.expectedCell);
//        }
//        ablate::environment::RunEnvironment::Finalize();
//    EndWithMPI
//}





/********************   End unit tests for DMPlexGetNeighborCells    *************************/
