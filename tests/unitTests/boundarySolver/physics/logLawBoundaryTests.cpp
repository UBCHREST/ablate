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
    PetscScalar boundaryValues[4] = {1.0, 1.0, 1.0, 1.0};
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
        //case 0
        (LogLawBoundaryTestParameters){
            .dim = 2, .fvFaceGeom = {.normal = {0, -1, NAN}, .areas = {NAN, 0.5, NAN}}, .boundaryCell = {.volume = {0.5}}, .stencilValues = {1.7, 10, 10}, .expectedResults = {3.566176, 0}},
        // case 1
        (LogLawBoundaryTestParameters){
            .dim = 2, .fvFaceGeom = {.normal = {0, 1, NAN}, .areas = {NAN, 0.5, NAN}}, .boundaryCell = {.volume = {0.5}}, .stencilValues = {1.7, 10, 10}, .expectedResults = {-2.316176, 0}},
        // case 2
        (LogLawBoundaryTestParameters){
            .dim = 3, .fvFaceGeom = {.normal = {0, 1, 0}, .areas = {0.5, 0.5, 0.5}}, .boundaryCell = {.volume = {0.05}}, .stencilValues = {1.7, 10, 5, 0}, .expectedResults = {-0.832833, 0, 0}},
        // case 3
        (LogLawBoundaryTestParameters){
            .dim = 3, .fvFaceGeom = {.normal = {0, -1, 0}, .areas = {0.5, 0.5, 0.5}}, .boundaryCell = {.volume = {0.05}}, .stencilValues = {1.7, 10, 5, 0}, .expectedResults = {2.10834, 0, 0}},
        // case 4
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {1, 1, 0}, .areas = {0.5, 0.5, 0.5}},
                                       .boundaryCell = {.volume = {0.05}},
                                       .stencilValues = {1.7, 10, 5, 0},
                                       .expectedResults = {-0.28433316, 0.28433316, 0}},
        // case 5
        (LogLawBoundaryTestParameters){.dim = 3,
                                       .fvFaceGeom = {.normal = {1, -1, 0}, .areas = {0.5, 0.5, 0.5}},
                                       .boundaryCell = {.volume = {0.05}},
                                       .stencilValues = {1.7, 10, 5, 0},
                                       .expectedResults = {1.18625507, 1.18625507, 0}}),
    [](const testing::TestParamInfo<LogLawBoundaryTestParameters>& info) { return std::to_string(info.index); });
