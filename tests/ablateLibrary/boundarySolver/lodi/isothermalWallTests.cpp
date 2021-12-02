#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/isothermalWall.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

struct IsothermalWallTestParameters {
    PetscInt dim;
    PetscInt nEqs;
    PetscInt nSpecEqs = 0;
    PetscInt nEvEqs = 0;
    std::function<PetscErrorCode(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p,
                                 void* ctx)>
        decodeStateFunction;
    std::function<PetscErrorCode(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx)> computeTemperatureFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx)> computeCpFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx)> computeCvFunction;
    std::function<PetscErrorCode(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx)> computeSensibleEnthalpy;
    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;
    std::vector<PetscScalar> boundaryValues;
    std::vector<PetscScalar> stencilValues; /* the grad is (boundary-stencil)/1.0*/
    std::vector<PetscScalar> expectedResults;
};

class IsothermalWallTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<IsothermalWallTestParameters> {};

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

TEST_P(IsothermalWallTestFixture, ShouldComputeCorrectSourceTerm) {
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
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary = std::make_shared<ablate::boundarySolver::lodi::IsothermalWall>(mockEOS);
    boundary->Initialize(params.dim, params.nEqs, params.nEvEqs, params.nSpecEqs);

    PetscInt uOff[1] = {0};
    PetscInt aOff[1] = {0};
    PetscInt sOff[1] = {0};
    const PetscScalar* stencilValues = &params.stencilValues[0];
    const PetscScalar* allStencilValues[1] = {stencilValues};
    const PetscInt stencil[1] = {-1};
    const PetscScalar stencilWeights[3] = {1.0, 1.0, 1.0};
    // NOTE: Because this is a made of stencil value, dPhi is computed as  stencil-boundary
    // therefore: //dPhiDx = stencil-boundary ... stencil = boundary+dPhiDx

    // size up the sourceResults
    std::vector<PetscScalar> sourceResults(GetParam().expectedResults.size());

    // act
    ablate::boundarySolver::lodi::IsothermalWall::IsothermalWallFunction(params.dim,
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
    IsothermalWallTests, IsothermalWallTestFixture,
    testing::Values(
        // case 0
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 3,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 2.9451426166013044, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -87201.128488, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 345.811145133247754;
                        *p = 251619.92076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", -197999.99999999872, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.92076699706 + 20.000000327126923;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 2.9451426166013044, density);
                    CHECK_EXPECT("totalEnergy", -87201.128488, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 0.0, massFlux[0]);
                    *T = 296.40099999999995;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.9451426166013044, density);
                    *specificHeat = 1009.36685027;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.9451426166013044, density);
                    *specificHeat = 721.1236794;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.9451426166013044, density);
                    *sensibleEnthalpy = -1765.5644007;
                    return 0;
                },
            .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {2.9451426166013044, -256819.75972598503, 0.0},
            .stencilValues = {20, 3000 * 20, -197999.99999999872 * 20},
            .expectedResults = {816226.6340554004, -7.117588359166196E10, 0.0}},
        // case 1
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 3,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 2.945140275655796, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -87201.128488, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 345.811145133247754;
                        *p = 251619.92076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 198000.00000001234, velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.72076 + 20.000000327126923;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 2.945140275655796, density);
                    CHECK_EXPECT("totalEnergy", -87201.128488, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 0.0, massFlux[0]);
                    *T = 296.40099999999995;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.945140275655796, density);
                    *specificHeat = 1009.36685;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.945140275655796, density);
                    *specificHeat = 721.123679;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 296.400999999, T);
                    CHECK_EXPECT("density", 2.945140275655796, density);
                    *sensibleEnthalpy = -1765.56440;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {2.945140275655796, -256819.55559289496, 0.0},
            .stencilValues = {20, 3000 * 20, 198000.00000001234 * 20},
            .expectedResults = {-816225.98527, 7.117582701753714E10, 0.0}},
        // case 2
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 3,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.692914985507404, density);
                        CHECK_EXPECT("velocity0", -3.5, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.83122426291567;
                        *p = 251619.72076699708;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 11949298.440203, velocity[0]);
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
                    CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", -30.425202449275915, massFlux[0]);
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
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 5.205232334625347E8}},
        // case 3
        (IsothermalWallTestParameters){
            .dim = 2,
            .nEqs = 4,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 2, dim);
                        CHECK_EXPECT("density", 8.692914985507404, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("velocity1", -3.5, velocity[1]);
                        CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.83122426291567;
                        *p = 251619.72076699708;
                    } else {
                        CHECK_EXPECT("dim", 2, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 10, velocity[0]);
                        CHECK_EXPECT("velocity1", 11949298.440203, velocity[1]);
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
                    CHECK_EXPECT("dim", 2, dim);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux0", 0.0, massFlux[0]);
                    CHECK_EXPECT("massFlux1", -30.425202449275915, massFlux[1]);
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
            .fvFaceGeom = {.normal = {0.0, -1.0, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, 0.0, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, 10.0 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 0.0, 5.205232334625347E8}},
        // case 4
        (IsothermalWallTestParameters){
            .dim = 3,
            .nEqs = 5,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 8.692914985507404, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("velocity1", 0.0, velocity[1]);
                        CHECK_EXPECT("velocity2", -3.5, velocity[2]);
                        CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.83122426291567;
                        *p = 251619.72076699708;
                    } else {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 10, velocity[0]);
                        CHECK_EXPECT("velocity1", 15, velocity[1]);
                        CHECK_EXPECT("velocity2", 11949298.440203, velocity[2]);
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
                    CHECK_EXPECT("dim", 3, dim);
                    CHECK_EXPECT("density", 8.692914985507404, density);
                    CHECK_EXPECT("totalEnergy", -226549.9687619313, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux0", 0.0, massFlux[0]);
                    CHECK_EXPECT("massFlux1", 0.0, massFlux[1]);
                    CHECK_EXPECT("massFlux2", -30.425202449275915, massFlux[2]);
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
            .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, 0.0, 0.0, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, 10.0 * 20, 15.0 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 0.0, 0.0, 5.205232334625347E8}}),
    [](const testing::TestParamInfo<IsothermalWallTestParameters>& info) { return std::to_string(info.index); });