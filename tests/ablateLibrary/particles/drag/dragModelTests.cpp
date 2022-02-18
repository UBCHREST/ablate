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

INSTANTIATE_TEST_SUITE_P(DragModelTests, DragModelTestFixture,
                         testing::Values((DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {100.0, 0.0},
                                                                   .flowVel = {0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {-0.00051836278784231588427, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 100.0},
                                                                   .flowVel = {0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, -0.00051836278784231588427}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0},
                                                                   .flowVel = {100.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.00051836278784231588427, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0},
                                                                   .flowVel = {0.0, 100.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, 0.00051836278784231588427}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {200.0, 100.0},
                                                                   .flowVel = {100.0, 300.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {-0.0011590944306217198566, 0.002318188861243439713}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {100.0, 0.0, 0.0},
                                                                   .flowVel = {0.0, 0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {-0.00051836278784231588427, 0.0, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 100.0, 0.0},
                                                                   .flowVel = {0.0, 0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, -0.00051836278784231588427, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0, 100.0},
                                                                   .flowVel = {0.0, 0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, 0.0, -0.00051836278784231588427}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0, 0.0},
                                                                   .flowVel = {100.0, 0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.00051836278784231588427, 0.0, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0, 0.0},
                                                                   .flowVel = {0.0, 100.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, 0.00051836278784231588427, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {0.0, 0.0, 0.0},
                                                                   .flowVel = {0.0, 0.0, 100.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {0.0, 0.0, 0.00051836278784231588427}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {200.0, 100.0, 0.0},
                                                                   .flowVel = {100.0, 0.0, 100.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1.2,
                                                                   .partDiam = 0.5e-3,
                                                                   .expectedDragForce = {-0.00089783068529593778743, -0.00089783068529593778743, 0.000897830685295937787430}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Quadratic>(); },
                                                                   .partVel = {30.0, 0.0, 0.0},
                                                                   .flowVel = {0.0, 0.0, 0.0},
                                                                   .muF = 0.0,
                                                                   .rhoF = 1000.0,
                                                                   .partDiam = 5.0e-3,
                                                                   .expectedDragForce = {-3.8877209088173691321, 0.0, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Linear>(); },
                                                                   .partVel = {30.0, 0.0},
                                                                   .flowVel = {0.0, 0.0},
                                                                   .muF = 8.9e-4,
                                                                   .rhoF = 1000.0,
                                                                   .partDiam = 5.0e-3,
                                                                   .expectedDragForce = {-0.00125820785776271219188, 0.0}},
                                         (DragModelTestParameters){.createDragModel = []() { return std::make_shared<ablate::particles::drag::Linear>(); },
                                                                   .partVel = {0.0, 30.0},
                                                                   .flowVel = {0.0, 0.0},
                                                                   .muF = 8.9e-4,
                                                                   .rhoF = 1000.0,
                                                                   .partDiam = 5.0e-3,
                                                                   .expectedDragForce = {0.0, -0.00125820785776271219188}}));

}  // namespace ablateTesting::particles::drag
