#include <petsc.h>
#include <PetscTestFixture.hpp>
#include <vector>
#include "finiteVolume/processes/les.hpp"
#include "gtest/gtest.h"

struct lesEvSourceTestParameters {
    std::vector<PetscReal> area;
    std::vector<PetscReal> field;
    PetscInt tke_ev;
    PetscInt numberEv;
    std::vector<PetscReal> gradVel;
    std::vector<PetscReal> EVs;
    std::vector<PetscReal> Grad;
    std::vector<PetscReal> expectedFlux;
};

class lesEvSourceTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<lesEvSourceTestParameters> {};

TEST_P(lesEvSourceTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();
    ablate::finiteVolume::processes::LES::DiffusionData flowParam;
    flowParam.numberEV = params.numberEv;
    flowParam.tke_ev = params.tke_ev;

    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscInt aOff_x[2] = {1};

    std::vector<PetscReal> computedFlux(params.expectedFlux.size());

    // act
    ablate::finiteVolume::processes::LES::LesEvFlux(params.area.size(), &faceGeom, uOff, NULL, &params.field[0], NULL, aOff, aOff_x, &params.EVs[0], &params.Grad[0], &computedFlux[0], &flowParam);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(
    lesTransportTests, lesEvSourceTestFixture,
    testing::Values((lesEvSourceTestParameters){.area = {0.5}, .field = {1.130}, .tke_ev = {1}, .numberEv = {2}, .EVs = {1.1, 0.93}, .Grad = {1.19, 1.43, 0.014}, .expectedFlux = {-0.0517, 0.494}},

                    (lesEvSourceTestParameters){.area = {0.5}, .field = {1.130}, .tke_ev = {0}, .numberEv = {2}, .EVs = {0.93, 1.1}, .Grad = {1.19, 0.014, 1.43}, .expectedFlux = {0.494, -0.0517}},

                    (lesEvSourceTestParameters){
                        .area = {1.5}, .field = {1.130}, .tke_ev = {2}, .numberEv = {3}, .EVs = {0.93, 1.1, 0.43}, .Grad = {1.19, 0.56, 2.14, 0.12}, .expectedFlux = {-0.072, -0.273, 0.261}}

                    ));
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct lesSpeciesSourceTestParameters {
    std::vector<PetscReal> area;
    std::vector<PetscReal> field;
    PetscInt tke_ev;
    std::vector<PetscReal> EVs;
    PetscInt numberSpecies;
    std::vector<PetscReal> Grad;
    std::vector<PetscReal> expectedFlux;
};

class lesSpeciesSourceTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<lesSpeciesSourceTestParameters> {};

TEST_P(lesSpeciesSourceTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto &params = GetParam();
    ablate::finiteVolume::processes::LES::DiffusionData flowParam;
    flowParam.tke_ev = params.tke_ev;
    flowParam.numberSpecies = params.numberSpecies;

    PetscFVFaceGeom faceGeom{};
    std::copy(std::begin(params.area), std::end(params.area), faceGeom.normal);

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscInt aOff_x[2] = {1};

    std::vector<PetscReal> computedFlux(params.expectedFlux.size());

    // act
    ablate::finiteVolume::processes::LES::LesSpeciesFlux(
        params.area.size(), &faceGeom, uOff, NULL, &params.field[0], NULL, aOff, aOff_x, &params.EVs[0], &params.Grad[0], &computedFlux[0], &flowParam);

    // assert
    for (std::size_t i = 0; i < params.expectedFlux.size(); i++) {
        ASSERT_NEAR(computedFlux[i], params.expectedFlux[i], 1E-3);
    }
}

INSTANTIATE_TEST_SUITE_P(lesTransportTests, lesSpeciesSourceTestFixture,
                         testing::Values((lesSpeciesSourceTestParameters){.area = {1.5},
                                                                          .field = {1.130},
                                                                          .tke_ev = {1},
                                                                          .EVs = {0.93, 1.10},
                                                                          .numberSpecies = {2},

                                                                          .Grad = {0.52, 0.3},
                                                                          .expectedFlux = {-0.106, -.0613}},
                                         (lesSpeciesSourceTestParameters){.area = {1.5},
                                                                          .field = {1.130},
                                                                          .tke_ev = {0},
                                                                          .EVs = {0.93, 1.10},
                                                                          .numberSpecies = {2},

                                                                          .Grad = {0.52, 0.3},
                                                                          .expectedFlux = {-0.097, -.057}},

                                         (lesSpeciesSourceTestParameters){.area = {0.5},
                                                                          .field = {1.130},
                                                                          .tke_ev = {0},
                                                                          .EVs = {0.93, 1.10},
                                                                          .numberSpecies = {3},

                                                                          .Grad = {0.52, 0.3, 0.97},
                                                                          .expectedFlux = {-0.018, -0.0108, -0.035}}

                                         ));
