#include <petsc.h>
#include <memory>
#include "MpiTestFixture.hpp"
#include "PetscTestErrorChecker.hpp"
#include "domain/RBF/rbfSupport.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
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
            auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                          std::vector<std::shared_ptr<domain::FieldDescriptor>>{},
                                                          testingParam.meshModifiers,
                                                          testingParam.meshFaces,
                                                          testingParam.meshStart,
                                                          testingParam.meshEnd,
                                                          std::vector<std::string>{},
                                                          testingParam.meshSimplex);

            PetscInt cell = -2;
            DMPlexGetContainingCell(mesh->GetDM(), &testingParam.xyz[0], &cell) >> utilities::PetscUtilities::checkError;

            PetscMPIInt rank;
            MPI_Comm_rank(PetscObjectComm((PetscObject)mesh->GetDM()), &rank);
            ASSERT_EQ(cell, testingParam.expectedCell[rank]);
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(MeshTests, RBFSupportTestFixture_ReturnID,
                         testing::Values((RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("2DQuad"),
                                                                         .meshFaces = {10, 5},
                                                                         .meshStart = {0.0, 0.0},
                                                                         .meshEnd = {1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = false,
                                                                         .xyz = {0.55, 0.25},
                                                                         .expectedCell = {15}},
                                         (RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("2DSimplex"),
                                                                         .meshFaces = {10, 5},
                                                                         .meshStart = {0.0, 0.0},
                                                                         .meshEnd = {1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = true,
                                                                         .xyz = {0.55, 0.25},
                                                                         .expectedCell = {49}},
                                         (RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("3DQuad"),
                                                                         .meshFaces = {2, 2, 2},
                                                                         .meshStart = {0.0, 0.0, 0.0},
                                                                         .meshEnd = {1.0, 1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = false,
                                                                         .xyz = {0.6, 0.42, 0.8},
                                                                         .expectedCell = {5}},
                                         (RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("3DSimplex"),
                                                                         .meshFaces = {1, 1, 1},
                                                                         .meshStart = {0.0, 0.0, 0.0},
                                                                         .meshEnd = {2.0, 1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = true,
                                                                         .xyz = {0.1, 0.9, 0.9},
                                                                         .expectedCell = {4}},
                                         (RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("3DSimplexFail"),
                                                                         .meshFaces = {1, 1, 1},
                                                                         .meshStart = {0.0, 0.0, 0.0},
                                                                         .meshEnd = {2.0, 1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = true,
                                                                         .xyz = {2.1, 0.9, 0.9},
                                                                         .expectedCell = {-1}},
                                         (RBFSupportParameters_ReturnID){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadMPI", 2),
                                                                         .meshFaces = {10, 10},
                                                                         .meshStart = {0.0, 0.0},
                                                                         .meshEnd = {1.0, 1.0},
                                                                         .meshModifiers = {},
                                                                         .meshSimplex = false,
                                                                         .xyz = {0.55, 0.25},
                                                                         .expectedCell = {10, -1}},
                                         (RBFSupportParameters_ReturnID){
                                             .mpiTestParameter = testingResources::MpiTestParameter("2DQuadMPIMod",
                                                                                                    2),  // This is mainly here to check if there is ever a change in how DMLocatePoints functions
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(1)},
                                             .meshSimplex = false,
                                             .xyz = {0.55, 0.25},
                                             .expectedCell = {10, -1}}),
                         [](const testing::TestParamInfo<RBFSupportParameters_ReturnID>& info) { return info.param.mpiTestParameter.getTestName(); });

/********************   End unit tests for DMPlexGetContainingCell    *************************/

/********************   Begin unit tests for DMPlexGetNeighborCells    *************************/

struct RBFSupportParameters_NeighborCells {
    testingResources::MpiTestParameter mpiTestParameter;
    std::vector<int> meshFaces;
    std::vector<double> meshStart;
    std::vector<double> meshEnd;
    std::vector<std::shared_ptr<domain::modifiers::Modifier>> meshModifiers;
    bool meshSimplex;
    std::vector<PetscInt> centerCell;
    PetscInt numLevels;
    PetscReal maxDistance;
    PetscInt minNumberCells;
    PetscBool useVertices;
    std::vector<PetscInt> expectedNumberOfCells;
    std::vector<std::vector<PetscInt>> expectedCellList;
};

class RBFSupportTestFixture_NeighborCells : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFSupportParameters_NeighborCells> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(RBFSupportTestFixture_NeighborCells, ShouldReturnNeighborCells) {
    StartWithMPI
        {
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            auto testingParam = GetParam();

            // Create the mesh
            // Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
            auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                          std::vector<std::shared_ptr<domain::FieldDescriptor>>{},
                                                          testingParam.meshModifiers,
                                                          testingParam.meshFaces,
                                                          testingParam.meshStart,
                                                          testingParam.meshEnd,
                                                          std::vector<std::string>{},
                                                          testingParam.meshSimplex);

            PetscInt nCells, *cells;
            PetscMPIInt rank;
            MPI_Comm_rank(PetscObjectComm((PetscObject)mesh->GetDM()), &rank);

            DMPlexGetNeighborCells(
                mesh->GetDM(), testingParam.centerCell[rank], testingParam.numLevels, testingParam.maxDistance, testingParam.minNumberCells, testingParam.useVertices, &nCells, &cells) >>
                utilities::PetscUtilities::checkError;

            PetscSortInt(nCells, cells);
            PetscSortInt(testingParam.expectedNumberOfCells[rank], testingParam.expectedCellList[rank].data());  // Should probably enter as a sorted list. Leaving for later.

            ASSERT_EQ(nCells, testingParam.expectedNumberOfCells[rank]);

            // There may be a better way of doing this, but with DMPlexGetNeighborCells sticking with C-only code there may not be.
            // Also note that as cells is a dynamically allocated array there is not way (that I know of) to get the number of elements.
            for (int i = 0; i < nCells; ++i) {
                ASSERT_EQ(cells[i], testingParam.expectedCellList[rank][i]);
            }
            PetscFree(cells);
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(
    MeshTests, RBFSupportTestFixture_NeighborCells,
    testing::Values(
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadVert"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = {},
                                             .meshSimplex = false,
                                             .centerCell = {25},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 25,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {25},
                                             .expectedCellList = {{25, 24, 35, 15, 26, 14, 34, 36, 16, 23, 45, 5, 27, 13, 33, 44, 46, 4, 37, 6, 17, 43, 3, 47, 7}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadVertCorner"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = {},
                                             .meshSimplex = false,
                                             .centerCell = {0},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 25,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {25},
                                             .expectedCellList = {{0, 10, 1, 11, 2, 20, 12, 21, 22, 30, 3, 31, 13, 23, 32, 4, 40, 14, 41, 33, 42, 24, 43, 34, 44}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriVert"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = {},
                                             .meshSimplex = true,
                                             .centerCell = {199},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 25,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {25},
                                             .expectedCellList = {{199, 76, 159, 79, 150, 80, 149, 98, 111, 73, 40, 78, 158, 112, 75, 109, 45, 81, 154, 95, 151, 82, 156, 152, 72}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriVertCorner"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = {},
                                             .meshSimplex = true,
                                             .centerCell = {0},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 25,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {25},
                                             .expectedCellList = {{0, 6, 1, 4, 2, 3, 7, 19, 5, 9, 21, 12, 22, 8, 18, 25, 14, 11, 23, 13, 24, 47, 10, 30, 27}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriVertNoOverlap", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = {},
                                             .meshSimplex = true,
                                             .centerCell = {56, 19},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 10,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {10, 10},
                                             .expectedCellList = {{56, 60, 57, 40, 55, 45, 58, 71, 41, 54}, {19, 21, 17, 22, 16, 23, 35, 18, 20, 102}}},
        (RBFSupportParameters_NeighborCells){
            .mpiTestParameter = testingResources::MpiTestParameter("2DQuadVertOverlap", 4),
            .meshFaces = {10, 10},
            .meshStart = {0.0, 0.0},
            .meshEnd = {1.0, 1.0},
            .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(1)},
            .meshSimplex = false,
            .centerCell = {24, 4, 20, 0},
            .numLevels = -1,
            .maxDistance = -1.0,
            .minNumberCells = 9,
            .useVertices = PETSC_TRUE,
            .expectedNumberOfCells = {9, 9, 9, 9},
            .expectedCellList = {{24, 23, 19, 29, 35, 18, 28, 34, 30}, {4, 9, 31, 3, 29, 8, 28, 30, 32}, {20, 21, 31, 15, 29, 16, 28, 30, 32}, {0, 31, 26, 5, 1, 6, 27, 32, 25}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadEdge", 1),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = false,
                                             .centerCell = {54},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 9,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {9},
                                             .expectedCellList = {{54, 64, 53, 55, 44, 63, 43, 45, 65}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriEdge", 1),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = true,
                                             .centerCell = {199},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 9,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {9},
                                             .expectedCellList = {{199, 76, 159, 79, 150, 149, 80, 98, 111}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadEdgeOverlap", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(1)},
                                             .meshSimplex = false,
                                             .centerCell = {11, 34},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 9,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {9, 9},
                                             .expectedCellList = {{11, 12, 6, 16, 10, 17, 7, 15, 5}, {34, 33, 29, 39, 56, 38, 28, 57, 55}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("3DQuadFace", 1),
                                             .meshFaces = {4, 4, 4},
                                             .meshStart = {0.0, 0.0, 0.0},
                                             .meshEnd = {1.0, 1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = false,
                                             .centerCell = {25},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 20,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {20},
                                             .expectedCellList = {{25, 24, 9, 26, 21, 29, 41, 8, 13, 28, 10, 5, 20, 40, 22, 45, 30, 37, 42, 27}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("3DTriFace", 1),
                                             .meshFaces = {4, 4, 4},
                                             .meshStart = {0.0, 0.0, 0.0},
                                             .meshEnd = {1.0, 1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = true,
                                             .centerCell = {25},
                                             .numLevels = -1,
                                             .maxDistance = -1.0,
                                             .minNumberCells = 20,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {20},
                                             .expectedCellList = {{25, 41, 38, 33, 28, 51, 39, 46, 56, 17, 15, 4, 32, 198, 95, 57, 201, 123, 40, 74}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadDistanceEdge"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = false,
                                             .centerCell = {55},
                                             .numLevels = -1,
                                             .maxDistance = 0.28,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {21},
                                             .expectedCellList = {{34, 35, 36, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67, 74, 75, 76}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadDistanceVert"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = false,
                                             .centerCell = {55},
                                             .numLevels = -1,
                                             .maxDistance = 0.28,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {21},
                                             .expectedCellList = {{34, 35, 36, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67, 74, 75, 76}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadDistanceEdgeMPI", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = false,
                                             .centerCell = {25, 29},
                                             .numLevels = -1,
                                             .maxDistance = 0.28,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {21, 21},
                                             .expectedCellList = {{15, 16, 20, 21, 22, 25, 26, 27, 30, 31, 32, 35, 36, 61, 63, 64, 66, 67, 69, 70, 73},
                                                                  {18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34, 38, 39, 59, 62, 63, 65, 66, 68, 69, 71}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadDistanceVertMPI", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = false,
                                             .centerCell = {25, 29},
                                             .numLevels = -1,
                                             .maxDistance = 0.28,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {21, 21},
                                             .expectedCellList = {{15, 16, 20, 21, 22, 25, 26, 27, 30, 31, 32, 35, 36, 61, 63, 64, 66, 67, 69, 70, 73},
                                                                  {18, 19, 22, 23, 24, 27, 28, 29, 32, 33, 34, 38, 39, 59, 62, 63, 65, 66, 68, 69, 71}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriDistanceEdge"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = true,
                                             .centerCell = {199},
                                             .numLevels = -1,
                                             .maxDistance = 0.14,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {11},
                                             .expectedCellList = {{40, 73, 76, 79, 80, 98, 111, 149, 150, 159, 199}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriDistanceVert"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{},
                                             .meshSimplex = true,
                                             .centerCell = {199},
                                             .numLevels = -1,
                                             .maxDistance = 0.14,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {11},
                                             .expectedCellList = {{40, 73, 76, 79, 80, 98, 111, 149, 150, 159, 199}}},

        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriDistanceEdgeMPI", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = true,
                                             .centerCell = {60, 102},
                                             .numLevels = -1,
                                             .maxDistance = 0.14,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {12, 11},
                                             .expectedCellList = {{38, 40, 41, 45, 56, 58, 60, 71, 113, 114, 132, 141}, {82, 83, 84, 86, 95, 96, 97, 99, 102, 138, 141}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DTriDistanceVertMPI", 2),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = true,
                                             .centerCell = {60, 102},
                                             .numLevels = -1,
                                             .maxDistance = 0.14,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {12, 11},
                                             .expectedCellList = {{38, 40, 41, 45, 56, 58, 60, 71, 113, 114, 132, 141}, {82, 83, 84, 86, 95, 96, 97, 99, 102, 138, 141}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadLevelVert"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = false,
                                             .centerCell = {55},
                                             .numLevels = 2,
                                             .maxDistance = -1.0,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_TRUE,
                                             .expectedNumberOfCells = {25},
                                             .expectedCellList = {{33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77}}},
        (RBFSupportParameters_NeighborCells){.mpiTestParameter = testingResources::MpiTestParameter("2DQuadLevelEdge"),
                                             .meshFaces = {10, 10},
                                             .meshStart = {0.0, 0.0},
                                             .meshEnd = {1.0, 1.0},
                                             .meshModifiers = std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                             .meshSimplex = false,
                                             .centerCell = {55},
                                             .numLevels = 2,
                                             .maxDistance = -1.0,
                                             .minNumberCells = -1,
                                             .useVertices = PETSC_FALSE,
                                             .expectedNumberOfCells = {13},
                                             .expectedCellList = {{35, 44, 45, 46, 53, 54, 55, 56, 57, 64, 65, 66, 75}}}),
    [](const testing::TestParamInfo<RBFSupportParameters_NeighborCells>& info) { return info.param.mpiTestParameter.getTestName(); });

struct RBFSupportParameters_ErrorChecking {
    testingResources::MpiTestParameter mpiTestParameter;
};

class RBFSupportTestFixture_ErrorChecking : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<RBFSupportParameters_ErrorChecking> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(RBFSupportTestFixture_ErrorChecking, ShouldThrowErrorForTooManyInputs) {
    StartWithMPI
        {
            // initialize petsc and mpi
            ablate::environment::RunEnvironment::Initialize(argc, argv);
            ablate::utilities::PetscUtilities::Initialize();

            // Create the mesh
            auto mesh = std::make_shared<domain::BoxMesh>("mesh",
                                                          std::vector<std::shared_ptr<domain::FieldDescriptor>>{},
                                                          std::vector<std::shared_ptr<domain::modifiers::Modifier>>{},
                                                          std::vector<int>{2, 2},
                                                          std::vector<double>{0.0, 0.0},
                                                          std::vector<double>{1.0, 1.0});

            PetscInt nCells, *cells;

            EXPECT_ANY_THROW(DMPlexGetNeighborCells(mesh->GetDM(), 0, 1, 1.0, 1, PETSC_TRUE, &nCells, &cells) >> utilities::PetscUtilities::checkError);
        }
        ablate::environment::RunEnvironment::Finalize();
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(MeshTests, RBFSupportTestFixture_ErrorChecking,
                         testing::Values((RBFSupportParameters_ErrorChecking){.mpiTestParameter = testingResources::MpiTestParameter("SingleProc")},
                                         (RBFSupportParameters_ErrorChecking){.mpiTestParameter = testingResources::MpiTestParameter("DualProcs", 2)}),
                         [](const testing::TestParamInfo<RBFSupportParameters_ErrorChecking>& info) { return info.param.mpiTestParameter.getTestName(); });

/********************   End unit tests for DMPlexGetNeighborCells    *************************/
