#include <petsc.h>
#include <memory>
#include "PetscTestFixture.hpp"
#include "domain/dmTransfer.hpp"
#include "gtest/gtest.h"
#include "levelSet/vofMathFunction.hpp"
#include "mathFunctions/nearestPoint.hpp"

struct VOFMathFunctionTestParameters {
    // The dimensions of the problem
    PetscInt dim;

    // Coordinates of the cell vertices
    std::vector<PetscReal> coords;

    // An array of numCells*numCorners numbers, the global vertex numbers for each cell
    std::vector<PetscInt> cellsNodes;

    // The level set at each vertex
    std::vector<double> vertexLevelSetValues;

    // The actual VOF for the particular test
    PetscReal trueVOF;
};

class VOFMathFunctionTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<VOFMathFunctionTestParameters> {};

// Build a simple dm with a single cell to test the math function
TEST_P(VOFMathFunctionTestFixture, CheckLSFunctions) {
    auto& testingParam = GetParam();

    // arrange
    // create a math function for the level set points
    auto levelSetFunction = std::make_shared<ablate::mathFunctions::NearestPoint>(testingParam.coords, testingParam.vertexLevelSetValues);

    // create a dm from the list of cells and cell type
    DM dm;
    PetscInt numVertices = ((PetscInt)testingParam.coords.size()) / testingParam.dim;
    DMPlexCreateFromCellListPetsc(
        PETSC_COMM_SELF, testingParam.dim, 1 /*1 cell*/, numVertices, numVertices, PETSC_TRUE, testingParam.cellsNodes.data(), testingParam.dim, testingParam.coords.data(), &dm) >>
        ablate::utilities::PetscUtilities::checkError;
    // convert the dm to domain
    auto domain = std::make_shared<ablate::domain::DMTransfer>(dm, std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>>{});

    // compute the cell center
    PetscReal center[3];
    DMPlexComputeCellGeometryFVM(dm, 0, NULL, center, NULL) >> ablate::utilities::PetscUtilities::checkError;

    // create the vof function
    auto vofMathFunction = std::make_shared<ablate::levelSet::VOFMathFunction>(domain, levelSetFunction);

    // act
    // PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx
    PetscScalar computedVof;
    vofMathFunction->GetPetscFunction()(testingParam.dim, NAN, center, 1, &computedVof, vofMathFunction->GetContext());

    // assert
    EXPECT_DOUBLE_EQ(testingParam.trueVOF, computedVof);
}

INSTANTIATE_TEST_SUITE_P(
    LevelSetTests, VOFMathFunctionTestFixture,
    testing::Values(
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {1.0, 2.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {-1.0, -2.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {-0.5, 0.5}, .trueVOF = 0.5},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {-0.1, 0.2}, .trueVOF = 1.0 / 3.0},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {0.1, -0.2}, .trueVOF = 2.0 / 3.0},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {0.2, -0.1}, .trueVOF = 1.0 / 3.0},
        (VOFMathFunctionTestParameters){.dim = 1, .coords = {-0.1, 0.2}, .cellsNodes = {0, 1}, .vertexLevelSetValues = {-0.2, 0.1}, .trueVOF = 2.0 / 3.0},

        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {-1.0, -1.0, -1.0}, .trueVOF = 1.},
        (VOFMathFunctionTestParameters){
            .dim = 2,
            .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0},
            .cellsNodes = {0, 1, 2},
            .vertexLevelSetValues = {1.0, 1.0, 1.0},
            .trueVOF = 0.0,
        },
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {-1.0, 2.0, 1.0}, .trueVOF = 1.0 / 6.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {1.0, -2.0, -1.0}, .trueVOF = 5.0 / 6.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {1.0, -2.0, 1.0}, .trueVOF = 4.0 / 9.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {-1.0, 2.0, -1.0}, .trueVOF = 5.0 / 9.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {1.0, 2.0, -1.0}, .trueVOF = 1.0 / 6.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {-1.0, -2.0, 1.0}, .trueVOF = 5.0 / 6.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {0.0, 0.0, -1.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {0.0, 0.0, 1.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {0.0, -1.0, 0.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {0.0, 1.0, 0.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {-1.0, 0.0, 0.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0}, .cellsNodes = {0, 1, 2}, .vertexLevelSetValues = {1.0, 0.0, 0.0}, .trueVOF = 0.0},

        (VOFMathFunctionTestParameters){.dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, 1.0, 1.0, 1.0}, .trueVOF = 0.0},

        (VOFMathFunctionTestParameters){
            .dim = 2,
            .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0},
            .cellsNodes = {0, 1, 2, 3},
            .vertexLevelSetValues = {-1.0, -1.0, -1.0, -1.0},
            .trueVOF = 1.0,
        },
        (VOFMathFunctionTestParameters){
            .dim = 2,
            .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0},
            .cellsNodes = {0, 1, 2, 3},
            .vertexLevelSetValues = {-0.25, 15.0 / 4.0, 11.0 / 4.0, 0.25},
            .trueVOF = 9.0 / 832.0,
        },
        (VOFMathFunctionTestParameters){
            .dim = 2,
            .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0},
            .cellsNodes = {0, 1, 2, 3},
            .vertexLevelSetValues = {0.25, -15.0 / 4.0, -11.0 / 4.0, -0.25},
            .trueVOF = 823.0 / 832.0,
        },
        (VOFMathFunctionTestParameters){
            .dim = 2,
            .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0},
            .cellsNodes = {0, 1, 2, 3},
            .vertexLevelSetValues = {-2.0, 1.0, 3.0, -0.5},
            .trueVOF = 82.0 / 273.0,
        },
        (VOFMathFunctionTestParameters){
            .dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {2.0, -1.0, -3.0, 0.5}, .trueVOF = 191.0 / 273.0},
        (VOFMathFunctionTestParameters){
            .dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-2.0, 3.0, -1.0, -2.5}, .trueVOF = 161.0 / 260.0},
        (VOFMathFunctionTestParameters){
            .dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {2.0, -3.0, 1.0, 2.5}, .trueVOF = 99.0 / 260.0},
        (VOFMathFunctionTestParameters){
            .dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, -4.0 / 3.0, 1.5, 1.0 / 12.0}, .trueVOF = 1822.0 / 2873.0},
        (VOFMathFunctionTestParameters){
            .dim = 2, .coords = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, 4.0 / 3.0, -1.5, -1.0 / 12.0}, .trueVOF = 1051.0 / 2873.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, 1.0, 1.0, 1.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, -1.0, -1.0, -1.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, -1.0, -1.0, -1.0}, .trueVOF = 7.0 / 8.},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, 1.0, 1.0, 1.0}, .trueVOF = 1.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, 1.0, -1.0, -1.0}, .trueVOF = 7.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, -1.0, 1.0, 1.0}, .trueVOF = 1.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, -1.0, 1.0, -1.0}, .trueVOF = 7.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, 1.0, -1.0, 1.0}, .trueVOF = 1.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-1.0, -1.0, -1.0, 1.0}, .trueVOF = 7.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {1.0, 1.0, 1.0, -1.0}, .trueVOF = 1.0 / 8.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.5, 1.0, -1.0, -1.0}, .trueVOF = 11.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-0.5, -1.0, 1.0, 1.0}, .trueVOF = 7.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.5, -1.0, 1.0, -1.0}, .trueVOF = 11.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-0.5, 1.0, -1.0, 1.0}, .trueVOF = 7.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.5, -1.0, -1.0, 1.0}, .trueVOF = 11.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {-0.5, 1.0, 1.0, -1.0}, .trueVOF = 7.0 / 18.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, 0.0, 1.0, 1.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, 0.0, -1.0, -1.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, 1.0, 0.0, 1.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, -1.0, 0.0, -1.0}, .trueVOF = 1.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, 1.0, 1.0, 0.0}, .trueVOF = 0.0},
        (VOFMathFunctionTestParameters){
            .dim = 3, .coords = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0}, .cellsNodes = {0, 1, 2, 3}, .vertexLevelSetValues = {0.0, -1.0, -1.0, 0.0}, .trueVOF = 1.0},

        (VOFMathFunctionTestParameters){
            .dim = 3,
            .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
            .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
            .vertexLevelSetValues = {0.0, -2.0, -2.0, 0.0, 1.0, 1.0, -1.0, -1.0},
            .trueVOF = 3.0 / 4.0,
        },
        (VOFMathFunctionTestParameters){.dim = 3,
                                        .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
                                        .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
                                        .vertexLevelSetValues = {0.0, 2.0, 2.0, 0.0, -1.0, -1.0, 1.0, 1.0},
                                        .trueVOF = 1.0 / 4.0},
        (VOFMathFunctionTestParameters){.dim = 3,
                                        .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
                                        .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
                                        .vertexLevelSetValues = {-0.5, -0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5},
                                        .trueVOF = 1.0 / 8.0},
        (VOFMathFunctionTestParameters){.dim = 3,
                                        .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
                                        .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
                                        .vertexLevelSetValues = {0.5, 0.5, -0.5, -0.5, -0.5, -1.5, -1.5, -0.5},
                                        .trueVOF = 7.0 / 8.0},
        (VOFMathFunctionTestParameters){.dim = 3,
                                        .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
                                        .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
                                        .vertexLevelSetValues = {5.0 / 4.0, -0.75, 0.25, 9.0 / 4.0, 13.0 / 4.0, 17.0 / 4.0, 9.0 / 4.0, 5.0 / 4.0},
                                        .trueVOF = 9.0 / 512.0},
        (VOFMathFunctionTestParameters){.dim = 3,
                                        .coords = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0},
                                        .cellsNodes = {0, 1, 2, 3, 4, 5, 6, 7},
                                        .vertexLevelSetValues = {-5.0 / 4.0, 0.75, -0.25, -9.0 / 4.0, -13.0 / 4.0, -17.0 / 4.0, -9.0 / 4.0, -5.0 / 4.0},
                                        .trueVOF = 503.0 / 512.0}));
