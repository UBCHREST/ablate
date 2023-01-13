//
// Created by rozie on 11/29/22.
//
#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/isothermalWall.hpp"
#include "boundarySolver/physics/logLawBoundary.hpp"

#include "gtest/gtest.h"

using ff = ablate::finiteVolume::CompressibleFlowFields;
struct LogLawBoundaryTestParameters {
    PetscInt dim;
    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;
    PetscFVCellGeom boundaryCell;
    std::function<std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling>()> getPgs = []() { return nullptr; };
    std::vector<PetscScalar> boundaryValues;
    std::vector<PetscScalar> stencilValues; /* the grad is (boundary-stencil)/1.0*/
    std::vector<PetscScalar> expectedResults;
};

class LogLawBoundaryTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<LogLawBoundaryTestParameters> {};

TEST_P(LogLawBoundaryTestFixture, ShouldComputeCorrectAuxValues) {
    // get the required variables
    const auto& params = GetParam();

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscScalar boundaryValues[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<PetscScalar> auxResults(GetParam().expectedResults.size());

    ablate::boundarySolver::physics::LogLawBoundary::UpdateBoundaryVel(params.dim,
                                                                       &params.fvFaceGeom,
                                                                       &params.boundaryCell,
                                                                       uOff,
                                                                       boundaryValues,
                                                                       params.stencilValues.data(),
                                                                       aOff,
                                                                       auxResults.data(),
                                                                       nullptr /*stencilAuxValues*/,

                                                                       nullptr /*stencilAuxValues*/

    );

    for (std::size_t i = 0; i < GetParam().expectedResults.size(); i++) {
        ASSERT_TRUE(PetscAbs(GetParam().expectedResults[i] - auxResults[i]) / (GetParam().expectedResults[i] + 1E-30) < 1E-4)
            << "The actual source term (" << auxResults[i] << ") for index " << i << " should match expected " << GetParam().expectedResults[i];
    }
}
INSTANTIATE_TEST_SUITE_P(
    PhysicsBoundaryTests, LogLawBoundaryTestFixture,
    testing::Values(
        // case 0
        (LogLawBoundaryTestParameters){
            .dim = 2, .fvFaceGeom = {.normal = {0, -1, NAN}, .areas = {0, 0.5, 0}}, .boundaryCell = {.volume = 0.5}, .stencilValues = {1.7, 0, 10, 0}, .expectedResults = {4.191176, 0}},
        // case 1
        (LogLawBoundaryTestParameters){
            .dim = 2, .fvFaceGeom = {.normal = {0, -1, NAN}, .areas = {0, 0.5, 0}}, .boundaryCell = {.volume = 0.5}, .stencilValues = {1.7, 0, 10, 0}, .expectedResults = {4.191176, 0}},

        // case 2
        (LogLawBoundaryTestParameters){
            .dim = 3, .fvFaceGeom = {.normal = {0, 1, 0}, .areas = {0, 0.5, 0}}, .boundaryCell = {.volume = 0.05}, .stencilValues = {1.7, 0, 10, 0, 0}, .expectedResults = {3.5693171, 0, 0}},

        // case 3
        (LogLawBoundaryTestParameters){
            .dim = 3, .fvFaceGeom = {.normal = {0, -1, 0}, .areas = {0, 0.5, 0}}, .boundaryCell = {.volume = 0.05}, .stencilValues = {1.7, 0, 10, 0, 0}, .expectedResults = {3.5693171, 0, 0}},

        // case 4
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {0.707, 0.707, 0}, .areas = {0, 0.5, 0}},
                                       .boundaryCell = {.volume = 0.05},
                                       .stencilValues = {1.7, 0, 10, 0, 0},
                                       .expectedResults = {1.7846111, -1.7846111, 0}},

        // case 5
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {-0.707, -0.707, 0}, .areas = {0, 0.5, 0}},
                                       .boundaryCell = {.volume = 0.05},
                                       .stencilValues = {1.7, 0, 10, 0, 0},
                                       .expectedResults = {1.7846111, -1.7846111, 0}},

        // case 6
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {0, -1, 0}, .areas = {0, 0.5, 0}},
                                       .boundaryCell = {.volume = 0.05},
                                       .stencilValues = {1.7, 0, 10, 5, 5},
                                       .expectedResults = {3.503002595, 0, 1.75150129}},
        // case 7
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {0, 1, 0}, .areas = {0, 0.5, 0}},
                                       .boundaryCell = {.volume = 0.05},
                                       .stencilValues = {1.7, 0, 5, 5, 10},
                                       .expectedResults = {1.75150129, 0, 3.503002595}}),
    [](const testing::TestParamInfo<LogLawBoundaryTestParameters>& info) { return std::to_string(info.index) + "_" + std::to_string(info.param.dim) + "D"; });
