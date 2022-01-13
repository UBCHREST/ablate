#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/openBoundary.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

struct OpenBoundaryTestParameters {
    std::string name;
    PetscInt dim;
    PetscInt nEqs;
    PetscInt nSpecEqs = 0;
    PetscInt nEvEqs = 0;
    double reflectFactor;
    double referencePressure;
    double maxAcousticsLength;
    std::function<PetscErrorCode(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p,
                                 void* ctx)>
        decodeStateFunction;
    std::function<PetscErrorCode(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx)> computeTemperatureFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx)> computeCpFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx)> computeCvFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx)> computeSensibleEnthalpy;
    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;
    std::function<std::shared_ptr<ablate::finiteVolume::resources::PressureGradientScaling>()> getPgs = []() { return nullptr; };

    std::vector<PetscScalar> boundaryValues;
    std::vector<PetscScalar> stencilValues; /* the grad is (boundary-stencil)/1.0*/
    std::vector<PetscScalar> expectedResults;
};

class OpenBoundaryTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<OpenBoundaryTestParameters> {};

static PetscErrorCode MockEOSDecodeStateFunction(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy,
                                                 PetscReal* a, PetscReal* p, void* ctx) {
    auto fun =
        (std::function<PetscErrorCode(
             PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx)>*)ctx;
    return (*fun)(dim, density, totalEnergy, velocity, densityYi, internalEnergy, a, p, nullptr);
}

static PetscErrorCode MockEOSComputeTemperatureFunction(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
    auto fun = (std::function<PetscErrorCode(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx)>*)ctx;
    return (*fun)(dim, density, totalEnergy, massFlux, densityYi, T, nullptr);
}

static PetscErrorCode MockEOSComputeSpecificHeatFunction(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
    auto fun = (std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx)>*)ctx;
    return (*fun)(T, density, yi, specificHeat, nullptr);
}

static PetscErrorCode MockEOSComputeSensibleEnthalpyFunction(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
    auto fun = (std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx)>*)ctx;
    return (*fun)(T, density, yi, sensibleEnthalpy, nullptr);
}

TEST_P(OpenBoundaryTestFixture, ShouldComputeCorrectSourceTerm) {
    // arrange
    // get the required variables
    const auto& params = GetParam();
    // setup the eos
    auto mockEOS = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*mockEOS, GetDecodeStateFunction).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockEOSDecodeStateFunction));
    EXPECT_CALL(*mockEOS, GetDecodeStateContext).Times(::testing::Exactly(1)).WillOnce(::testing::Return((void*)&params.decodeStateFunction));
    EXPECT_CALL(*mockEOS, GetComputeTemperatureFunction).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockEOSComputeTemperatureFunction));
    EXPECT_CALL(*mockEOS, GetComputeTemperatureContext).Times(::testing::Exactly(1)).WillOnce(::testing::Return((void*)&params.computeTemperatureFunction));
    EXPECT_CALL(*mockEOS, GetComputeSpecificHeatConstantPressureFunction).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockEOSComputeSpecificHeatFunction));
    EXPECT_CALL(*mockEOS, GetComputeSpecificHeatConstantPressureContext).Times(::testing::Exactly(1)).WillOnce(::testing::Return((void*)&params.computeCpFunction));
    EXPECT_CALL(*mockEOS, GetComputeSpecificHeatConstantVolumeFunction).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockEOSComputeSpecificHeatFunction));
    EXPECT_CALL(*mockEOS, GetComputeSpecificHeatConstantVolumeContext).Times(::testing::Exactly(1)).WillOnce(::testing::Return((void*)&params.computeCvFunction));
    EXPECT_CALL(*mockEOS, GetComputeSensibleEnthalpyFunction).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockEOSComputeSensibleEnthalpyFunction));
    EXPECT_CALL(*mockEOS, GetComputeSensibleEnthalpyContext).Times(::testing::Exactly(1)).WillOnce(::testing::Return((void*)&params.computeSensibleEnthalpy));

    // create the boundary
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary =
        std::make_shared<ablate::boundarySolver::lodi::OpenBoundary>(mockEOS, GetParam().reflectFactor, GetParam().referencePressure, GetParam().maxAcousticsLength, params.getPgs());
    boundary->Initialize(params.dim, params.nEqs, params.nSpecEqs, params.nEvEqs);

    PetscInt uOff[3] = {0, params.dim + 2, params.dim + 2 + params.nSpecEqs};
    PetscInt aOff[1] = {0};
    PetscInt sOff[3] = {0, params.dim + 2, params.dim + 2 + params.nSpecEqs};
    const PetscScalar* stencilValues = &params.stencilValues[0];
    const PetscScalar* allStencilValues[1] = {stencilValues};
    const PetscInt stencil[1] = {-1};
    const PetscScalar stencilWeights[3] = {1.0, 1.0, 1.0};
    // NOTE: Because this is a made of stencil value, dPhi is computed as  stencil-boundary
    // therefore: //dPhiDx = stencil-boundary ... stencil = boundary+dPhiDx

    // size up the sourceResults
    std::vector<PetscScalar> sourceResults(GetParam().expectedResults.size());

    // act
    ablate::boundarySolver::lodi::OpenBoundary::OpenBoundaryFunction(params.dim,
                                                                     &params.fvFaceGeom,
                                                                     nullptr /*boundaryCell*/,
                                                                     uOff,
                                                                     &params.boundaryValues[0],
                                                                     allStencilValues,
                                                                     aOff,
                                                                     nullptr /*auxValues*/,
                                                                     nullptr /*stencilAuxValues*/,
                                                                     1,
                                                                     stencil,
                                                                     stencilWeights,
                                                                     sOff,
                                                                     &sourceResults[0],
                                                                     boundary.get());

    // assert
    for (std::size_t i = 0; i < GetParam().expectedResults.size(); i++) {
        ASSERT_TRUE(PetscAbs(GetParam().expectedResults[i] - sourceResults[i]) / (GetParam().expectedResults[i] + 1E-30) < 1E-6)
            << "The actual source term (" << sourceResults[i] << ") for index " << i << " should match expected " << GetParam().expectedResults[i];
    }
}

static void CHECK_EXPECT(const char* name, double expected, double actual, double diff = 1E-4) {
    if (PetscAbs(expected - actual) > diff) {
        throw std::invalid_argument("The  " + std::string(name) + " provided (" + std::to_string(actual) + ") is not what is expected (" + std::to_string(expected) + ")");
    }
}

INSTANTIATE_TEST_SUITE_P(
    OpenBoundaryTests, OpenBoundaryTestFixture,
    testing::Values(
        (OpenBoundaryTestParameters){
            .name = "1D subsonic into the domain",
            .dim = 1,
            .nEqs = 3,
            .reflectFactor = 0.0,
            .referencePressure = 101325.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.692914985507404, density);
                        CHECK_EXPECT("velocity0", 10.0001, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -226506.0927619263, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.83122426291567;
                        *p = 251619.72076699708;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 9.9603020202, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.72076699708 + 20.000000327128298;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    CHECK_EXPECT("totalEnergy", -226506.0927619263, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 86.930019, massFlux[0]);
                    *T = 100.42;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.42, T);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.42, T);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.42, T);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    *sensibleEnthalpy = -197610.71454374143;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1968998.208078879, 86.93001914657259},
            .stencilValues = {20, 3000 * 20, (10.0001 - 0.03979797979809766) * 20.0},
            .expectedResults = {0.2115010865888479, -42211.18508291234, -40.57249122316696}},
        (OpenBoundaryTestParameters){
            .name = "1D subsonic out of the domain",
            .dim = 1,
            .nEqs = 3,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.692018326008165, density);
                        CHECK_EXPECT("velocity0", -20.0001, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -226349.0154433181, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.84127336667171;
                        *p = 251618.820766997;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), density);
                        CHECK_EXPECT("velocity0", -19.960302, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    CHECK_EXPECT("totalEnergy", -226349.0154433181, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -173.8412357219959, massFlux[0]);
                    *T = 100.43;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *sensibleEnthalpy = -197600.7557934246;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692018326008165, -1967429.7903072261, -173.8412357219959},
            .stencilValues = {(8.692018326008165 + 1.040246109552421), 3000 * (8.692018326008165 + 1.040246109552421), (-20.0001 + 0.03979797979809766) * (8.692018326008165 + 1.040246109552421)},
            .expectedResults = {-888.7282612652397, 1.770387294967629E8, -165929.9623913537}},
        (OpenBoundaryTestParameters){
            .name = "1D supersonic out of the domain",
            .dim = 1,
            .nEqs = 3,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.692018326008165, density);
                        CHECK_EXPECT("velocity0", -500.0001, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -101548.96744331811, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.84127336667171;
                        *p = 251618.820766997;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), density);
                        CHECK_EXPECT("velocity0", -499.960302, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    CHECK_EXPECT("totalEnergy", -101548.96744331811, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -4346.010032205915, massFlux[0]);
                    *T = 100.43;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *sensibleEnthalpy = -197600.7557934246;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692018326008165, -882665.4860045275, -4346.010032205915},
            .stencilValues = {(8.692018326008165 + 1.040246109552421), 3000 * (8.692018326008165 + 1.040246109552421), (-500.0001 + 0.03979797979809766) * (8.692018326008165 + 1.040246109552421)},
            .expectedResults = {519.7772340310783, -8.949767410165516E7, -259915.7065747647}},
        (OpenBoundaryTestParameters){
            .name = "1D supersonic into the domain",
            .dim = 1,
            .nEqs = 3,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.692018326008165, density);
                        CHECK_EXPECT("velocity0", 500.0001, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -101548.96744331811, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.84127336667171;
                        *p = 251618.820766997;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), density);
                        CHECK_EXPECT("velocity0", 500.0398979798, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    CHECK_EXPECT("totalEnergy", -101548.96744331811, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 4346.010032205915, massFlux[0]);
                    *T = 100.43;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.43, T);
                    CHECK_EXPECT("density", 8.692018326008165, density);
                    *sensibleEnthalpy = -197600.7557934246;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692018326008165, -882665.4860045275, 4346.010032205915},
            .stencilValues = {(8.692018326008165 + 1.040246109552421), 3000 * (8.692018326008165 + 1.040246109552421), (500.0001 + 0.03979797979809766) * (8.692018326008165 + 1.040246109552421)},
            .expectedResults = {0.0, 0.0, 0.0}},
        (OpenBoundaryTestParameters){
            .name = "3D supersonic out of the domain",
            .dim = 3,
            .nEqs = 5,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 8.694650097350083, density);
                        CHECK_EXPECT("velocity0", -500., velocity[0]);
                        CHECK_EXPECT("velocity1", -600., velocity[1]);
                        CHECK_EXPECT("velocity2", -700., velocity[2]);
                        CHECK_EXPECT("totalEnergy", 323429.7536008524, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.8111245542304;
                        *p = 251619.82076699712;
                    } else {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", (8.694650097350083 - 0.12298691191290341), density);
                        CHECK_EXPECT("velocity0", -500 + 1.9999999999981803, velocity[0]);
                        CHECK_EXPECT("velocity1", -600 + 2.0000000000436553, velocity[1]);
                        CHECK_EXPECT("velocity2", -700 + 0.004999999964638845, velocity[2]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699712 + 199.99999993015075;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 3, dim);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    CHECK_EXPECT("totalEnergy", 323429.7536008524, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux[1]", -4347.325048675041, massFlux[0]);
                    CHECK_EXPECT("massFlux[2]", -5216.79005841005, massFlux[1]);
                    CHECK_EXPECT("massFlux[3]", -6086.255068145058, massFlux[2]);

                    *T = 100.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *sensibleEnthalpy = -197630.63204437506;
                    return 0;
                },
            .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.694650097350083, 2812108.5386315645, -4347.325048675041, -5216.79005841005, -6086.255068145058},
            .stencilValues = {(8.694650097350083 - 0.12298691191290341),
                              3000 * (8.694650097350083 - 0.12298691191290341),
                              (-500.0 + 1.9999999999981803) * (8.694650097350083 - 0.12298691191290341),
                              (-600.0 + 2.0000000000436553) * (8.694650097350083 - 0.12298691191290341),
                              (-700.0 + 0.004999999964638845) * (8.694650097350083 - 0.12298691191290341)},
            .expectedResults = {-86.13431158921169, -3.467059254299343E7, 55239.66593088489, 63853.09709008283, 60124.44938764354}},
        (OpenBoundaryTestParameters){
            .name = "3D subsonic out of the domain",
            .dim = 3,
            .nEqs = 5,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 8.694650097350083, density);
                        CHECK_EXPECT("velocity0", -5., velocity[0]);
                        CHECK_EXPECT("velocity1", -6., velocity[1]);
                        CHECK_EXPECT("velocity2", -7., velocity[2]);
                        CHECK_EXPECT("totalEnergy", -226515.2463991476, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.8111245542304;
                        *p = 251619.82076699712;
                    } else {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", (8.694650097350083 - 0.12298691191290341), density);
                        CHECK_EXPECT("velocity0", -5.0 + 2.000000000000312, velocity[0]);
                        CHECK_EXPECT("velocity1", -6.0 + 0.005000000000165981, velocity[1]);
                        CHECK_EXPECT("velocity2", -7.0 + 2.000000000000312, velocity[2]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699712 + 199.99999993015075;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 3, dim);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    CHECK_EXPECT("totalEnergy", -226515.2463991476, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux[1]", -43.473250486750416, massFlux[0]);
                    CHECK_EXPECT("massFlux[2]", -52.167900584100494, massFlux[1]);
                    CHECK_EXPECT("massFlux[3]", -60.86255068145058, massFlux[2]);

                    *T = 100.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *specificHeat = 995.8750316818866;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *specificHeat = 707.6318608176182;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 100.4, T);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    *sensibleEnthalpy = -197630.63204437506;
                    return 0;
                },
            .fvFaceGeom = {.normal = {0.0, -1.0, 0.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.694650097350083, -1969470.8091556267, -43.473250486750416, -52.167900584100494, -60.86255068145058},
            .stencilValues = {(8.694650097350083 - 0.12298691191290341),
                              3000 * (8.694650097350083 - 0.12298691191290341),
                              (-5.0 + 2.000000000000312) * (8.694650097350083 - 0.12298691191290341),
                              (-6.0 + 0.005000000000165981) * (8.694650097350083 - 0.12298691191290341),
                              (-7.0 + 2.000000000000312) * (8.694650097350083 - 0.12298691191290341)},
            .expectedResults = {-910.2235530052537, 1.810158811466939E8, 4655.4535661944865, -178273.9425225725, 6475.9006722049935}},
        (OpenBoundaryTestParameters){
            .name = "1D subsonic out of the domain with sp and ev",
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvEqs = 2,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 1.783192, density);
                        CHECK_EXPECT("velocity0", -10, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 431.6854962124021;
                        *p = 251619.82076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), density);
                        CHECK_EXPECT("velocity0", (-10 - 40000.000000000015), velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 1.783192, density);
                    CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -17.831915158083632, massFlux[0]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1934.650079471233;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1464.9215577478003;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *sensibleEnthalpy = 4347.52375485136;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues =
                {1.783191515808363, -243778.19371678037, -17.831915158083632, .1 * 1.783191515808363, .5 * 1.783191515808363, .4 * 1.783191515808363, .25 * 1.783191515808363, .5 * 1.783191515808363},
            .stencilValues = {(1.783191515808363 + 90.16181478870485),
                              3000 * (1.783191515808363 + 90.16181478870485),
                              (-10 - 40000.000000000015) * (1.783191515808363 + 90.16181478870485),
                              (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.5) * (1.783191515808363 + 90.16181478870485),
                              (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485),
                              (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485)},
            .expectedResults = {36966.4328851267, -2.0206605128724575E8, -1.6305600063259933E7, 5794.10023978034, 18483.21644256335, 12689.11620278302, 9277.272051597822, 18518.880272879534}},
        (OpenBoundaryTestParameters){
            .name = "1D subsonic out of the domain with sp/ev and pgs",
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvEqs = 2,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 1.783192, density);
                        CHECK_EXPECT("velocity0", -10, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 431.6854962124021;
                        *p = 251619.82076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), density);
                        CHECK_EXPECT("velocity0", (-10 - 40000.000000000015), velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 1.783192, density);
                    CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -17.831915158083632, massFlux[0]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1934.650079471233;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1464.9215577478003;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *sensibleEnthalpy = 4347.52375485136;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::resources::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125.0, 1.0); },
            .boundaryValues =
                {1.783191515808363, -243778.19371678037, -17.831915158083632, .1 * 1.783191515808363, .5 * 1.783191515808363, .4 * 1.783191515808363, .25 * 1.783191515808363, .5 * 1.783191515808363},
            .stencilValues = {(1.783191515808363 + 90.16181478870485),
                              3000 * (1.783191515808363 + 90.16181478870485),
                              (-10 - 40000.000000000015) * (1.783191515808363 + 90.16181478870485),
                              (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.5) * (1.783191515808363 + 90.16181478870485),
                              (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485),
                              (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485)},
            .expectedResults = {251459.9979801531, -6.114186870680885E10, -2832799.3598080194, 27243.456749282985, 125729.99899007655, 98486.5422407936, 62900.66332535442, 125765.66282039274}},
        (OpenBoundaryTestParameters){
            .name = "1D super out of the domain with sp/ev and pgs",
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvEqs = 2,
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 1.783192, density);
                        CHECK_EXPECT("velocity0", -1000.000000, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 363241.0758291322, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 431.6854962124021;
                        *p = 251619.82076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), density);
                        CHECK_EXPECT("velocity0", (-1000 - 40000.000000000015), velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 1.783192, density);
                    CHECK_EXPECT("totalEnergy", 363241.0758291322, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -1783.1915158083632, massFlux[0]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1934.650079471233;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *specificHeat = 1464.9215577478003;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 1.783192, density);
                    *sensibleEnthalpy = 4347.52375485136;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::resources::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125.0, 1.0); },
            .boundaryValues =
                {1.783191515808363, 647728.4046116108, -1783.1915158083632, .1 * 1.783191515808363, .5 * 1.783191515808363, .4 * 1.783191515808363, .25 * 1.783191515808363, .5 * 1.783191515808363},
            .stencilValues = {(1.783191515808363 + 90.16181478870485),
                              3000 * (1.783191515808363 + 90.16181478870485),
                              (-1000 - 40000.000000000015) * (1.783191515808363 + 90.16181478870485),
                              (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.5) * (1.783191515808363 + 90.16181478870485),
                              (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485),
                              (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485)},
            .expectedResults = {161489.4754209326, 1.0037606013563467E11, -2.3281713606606913E8, 225894.64266886032, 80744.7377104663, -145149.90495839302, 43938.75188684794, 84311.12074208501}}),
    [](const testing::TestParamInfo<OpenBoundaryTestParameters>& info) { return testingResources::PetscTestFixture::SanitizeTestName(info.param.name); });