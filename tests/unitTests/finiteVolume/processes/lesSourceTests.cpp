#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <vector>
#include "finiteVolume/processes/les.hpp"
#include "gtest/gtest.h"

struct LesEvSourceTestParameters {
    std::vector<PetscReal> area;
    std::vector<PetscReal> field;
    PetscInt tkeFieldOffset;
    PetscInt evFieldOffset;
    PetscInt numberEvComponents;
    PetscInt tkeGradOffset;
    PetscInt evGradOffset;
    PetscInt velGradOffset;
    std::vector<PetscReal> gradVel;
    std::vector<PetscReal> aux;
    std::vector<PetscReal> gradAux;
    std::vector<PetscReal> expectedFlux;
};

class lesEvSourceTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<LesEvSourceTestParameters> {};

TEST_P(lesEvSourceTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();

    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    auto dim = (PetscInt)params.area.size();

    std::vector<PetscReal> computedFlux(params.expectedFlux.size());

    // act
    // call the function for tke
    {
        PetscInt uOff[1] = {0};                                                     // euler field is in the zero location
        PetscInt aOff[2] = {GetParam().tkeFieldOffset, -1};                         // tke, vel (velocity should not be used)
        PetscInt aOff_x[2] = {GetParam().tkeGradOffset, GetParam().velGradOffset};  // tke, vel

        ablate::finiteVolume::processes::LES::LesTkeFlux(
            dim, &faceGeom, uOff, nullptr, params.field.data(), nullptr, aOff, aOff_x, params.aux.data(), params.gradAux.data(), &computedFlux[GetParam().tkeFieldOffset], nullptr);
    }
    // call the function for evs
    {
        PetscInt uOff[1] = {0};                                                    // euler field is in the zero location
        PetscInt aOff[2] = {GetParam().tkeFieldOffset, GetParam().evFieldOffset};  // tke, ev
        PetscInt aOff_x[2] = {GetParam().tkeGradOffset, GetParam().evGradOffset};  // tke, ev

        PetscInt numberComponents = GetParam().numberEvComponents;
        ablate::finiteVolume::processes::LES::LesEvFlux(
            dim, &faceGeom, uOff, nullptr, params.field.data(), nullptr, aOff, aOff_x, params.aux.data(), params.gradAux.data(), &computedFlux[GetParam().evFieldOffset], &numberComponents);
    }
    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(lesTransportTests, lesEvSourceTestFixture,
                         testing::Values((LesEvSourceTestParameters){.area = {0.5},
                                                                     .field = {1.130},
                                                                     .tkeFieldOffset = 1,
                                                                     .evFieldOffset = 0,
                                                                     .numberEvComponents = 1,
                                                                     .tkeGradOffset = 2,
                                                                     .evGradOffset = 1,
                                                                     .velGradOffset = 0,
                                                                     .aux = {1.1, 0.93},
                                                                     .gradAux = {1.19, 1.43, 0.014},
                                                                     .expectedFlux = {-0.0517, 0.494}},

                                         (LesEvSourceTestParameters){.area = {0.5},
                                                                     .field = {1.130},
                                                                     .tkeFieldOffset = 0,
                                                                     .evFieldOffset = 1,
                                                                     .numberEvComponents = 1,
                                                                     .tkeGradOffset = 1,
                                                                     .evGradOffset = 2,
                                                                     .velGradOffset = 0,
                                                                     .aux = {0.93, 1.1},
                                                                     .gradAux = {1.19, 0.014, 1.43},
                                                                     .expectedFlux = {0.494, -0.0517}},

                                         (LesEvSourceTestParameters){.area = {1.5},
                                                                     .field = {1.130},
                                                                     .tkeFieldOffset = 2,
                                                                     .evFieldOffset = 0,
                                                                     .numberEvComponents = 2,
                                                                     .tkeGradOffset = 3,
                                                                     .evGradOffset = 1,
                                                                     .velGradOffset = 0,
                                                                     .aux = {0.93, 1.1, 0.43},
                                                                     .gradAux = {1.19, 0.56, 2.14, 0.12},
                                                                     .expectedFlux = {-0.072, -0.273, 0.261}}

                                         ));
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct LesSpeciesSourceTestParameters {
    std::vector<PetscReal> area;
    std::vector<PetscReal> field;
    PetscInt tkeFieldOffset;
    std::vector<PetscReal> aux;  // note that this does not contain yi
    PetscInt numberSpecies;
    std::vector<PetscReal> gradAux;
    std::vector<PetscReal> expectedFlux;
};

class LesSpeciesSourceTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<LesSpeciesSourceTestParameters> {};

TEST_P(LesSpeciesSourceTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();
    auto numberSpecies = params.numberSpecies;

    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    PetscInt uOff[1] = {0};                              // euler field is in the zero location
    PetscInt aOff[2] = {GetParam().tkeFieldOffset, -1};  // tke, ev
    PetscInt aOff_x[2] = {-1, 0};                        // tke, ev/species

    std::vector<PetscReal> computedFlux(params.expectedFlux.size());

    // act
    ablate::finiteVolume::processes::LES::LesEvFlux(
        (PetscInt)params.area.size(), &faceGeom, uOff, nullptr, params.field.data(), nullptr, aOff, aOff_x, params.aux.data(), params.gradAux.data(), &computedFlux[0], &numberSpecies);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(lesTransportTests, LesSpeciesSourceTestFixture,
                         testing::Values((LesSpeciesSourceTestParameters){.area = {1.5},
                                                                          .field = {1.130},
                                                                          .tkeFieldOffset = 1,
                                                                          .aux = {0.93, 1.10},
                                                                          .numberSpecies = 2,

                                                                          .gradAux = {0.52, 0.3},
                                                                          .expectedFlux = {-0.106, -.0613}},
                                         (LesSpeciesSourceTestParameters){.area = {1.5},
                                                                          .field = {1.130},
                                                                          .tkeFieldOffset = 0,
                                                                          .aux = {0.93, 1.10},
                                                                          .numberSpecies = 2,

                                                                          .gradAux = {0.52, 0.3},
                                                                          .expectedFlux = {-0.097, -.057}},

                                         (LesSpeciesSourceTestParameters){.area = {0.5},
                                                                          .field = {1.130},
                                                                          .tkeFieldOffset = 0,
                                                                          .aux = {0.93, 1.10},
                                                                          .numberSpecies = 3,

                                                                          .gradAux = {0.52, 0.3, 0.97},
                                                                          .expectedFlux = {-0.018, -0.0108, -0.035}}

                                         ));
