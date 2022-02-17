#include <functional>
#include <memory>
#include "PetscTestFixture.hpp"
#include "gtest/gtest.h"
#include "particles/drag/linear.hpp"
#include "particles/drag/quadratic.hpp"

namespace ablateTesting::particles::drag {
struct DragModelTestParameters {
    std::function<std::shared_ptr<ablate::particles::drag::DragModel>()> createDragModel;
    std::vector<PetscReal> partVel;
    std::vector<PetscReal> flowVel;
    PetscReal muF;
    PetscReal rhoF;
    PetscReal partDiam;
    std::vector<PetscReal> expectedDragForce;
};

class DragModelTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<DragModelTestParameters> {};

TEST_P(DragModelTestFixture, ShouldComputeCorrectDragForce) {
    // arrange
    const auto& param = GetParam();
    auto dragModel = param.createDragModel();
    std::vector<PetscReal> computedDragForce(param.expectedDragForce.size());
    
    // act
    dragModel->ComputeDragForce(param.expectedDragForce.size(), param.partVel.data(), param.flowVel.data(), param.muF, param.rhoF, param.partDiam, computedDragForce.data());
    
    // assert
    for (std::size_t i = 0; i < param.expectedDragForce.size(); i++) {
        ASSERT_DOUBLE_EQ(param.expectedDragForce[i], computedDragForce[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    DragModelTests, DragModelTestFixture,
    testing::Values(
        (DragModelTestParameters){.createDragModel =
                                           []() {
                                               return std::make_shared<ablate::particles::drag::Quadratic>();
                                           },
                                       .partVel = {2.0, 0.0},
                                       .flowVel = {0.0, 0.0},
                                       .muF = 0.0,
                                       .rhoF = 1.2,
                                       .partDiam = 300.0e-6,
                                       .expectedDragForce = {-7.4644241449293487335e-8, 0.0}
                                       },
        (DragModelTestParameters){.createDragModel =
                                           []() {
                                               return std::make_shared<ablate::particles::drag::Linear>();
                                           },
                                       .partVel = {2.0, 0.0},
                                       .flowVel = {0.0, 0.0},
                                       .muF = 1.0,
                                       .rhoF = 1.2,
                                       .partDiam = 1.0,
                                       .expectedDragForce = {-18.849555921538759428, 0.0}
                                       }
                                       ));

}
