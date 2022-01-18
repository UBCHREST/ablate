#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/inlet.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

struct InletTestParameters {
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
    std::function<std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling>()> getPgs = []() { return nullptr; };

    std::vector<PetscScalar> boundaryValues;
    std::vector<PetscScalar> stencilValues; /* the grad is (boundary-stencil)/1.0*/
    std::vector<PetscScalar> expectedResults;
};

class InletTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<InletTestParameters> {};

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

TEST_P(InletTestFixture, ShouldComputeCorrectSourceTerm) {
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
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary = std::make_shared<ablate::boundarySolver::lodi::Inlet>(mockEOS, params.getPgs());
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
    ablate::boundarySolver::lodi::Inlet::InletFunction(params.dim,
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
    InletTests, InletTestFixture,
    testing::Values(
        // case 0
        (InletTestParameters){
            .dim = 1,
            .nEqs = 3,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 8.694650097350083, density);
                        CHECK_EXPECT("velocity0", 500.0, velocity[0]);
                        CHECK_EXPECT("totalEnergy", -101570.2463991476, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 201.8111245542304;
                        *p = 251619.82076699712;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", (500.0 + 39999.99999999889), velocity[0]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699712 + 199.9999986612238;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 1, dim);
                    CHECK_EXPECT("density", 8.694650097350083, density);
                    CHECK_EXPECT("totalEnergy", -101570.2463991476, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 4347.325049, massFlux[0]);
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
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.694650097350083, -883117.7527422206, 4347.32504867504},
            .stencilValues = {20, 3000 * 20, (500.0 + 39999.99999999889) * 20},
            .expectedResults = {723193.7481425349, -7.345496719316035E10, 3.615968740712674E8}},
        // case 1
        (InletTestParameters){
            .dim = 3,
            .nEqs = 5,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 2.905934985931918, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("velocity1", 0.0, velocity[1]);
                        CHECK_EXPECT("velocity2", 10.0, velocity[2]);
                        CHECK_EXPECT("totalEnergy", -84266.3242014256, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 348.10065251594057;
                        *p = 251619.82076699703;
                    } else {
                        CHECK_EXPECT("dim", 3, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 0.0, velocity[0]);
                        CHECK_EXPECT("velocity1", 0.0, velocity[1]);
                        CHECK_EXPECT("velocity2", (10.0 + 40000.000000000015), velocity[2]);
                        CHECK_EXPECT("totalEnergy", 3000, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = NAN,
                        *p = 251619.82076699712 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                    }
                    count++;
                    return 0;
                },
            .computeTemperatureFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx) {
                    CHECK_EXPECT("dim", 3, dim);
                    CHECK_EXPECT("density", 2.905934985931918, density);
                    CHECK_EXPECT("totalEnergy", -84266.3242014256, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux0", 0.0, massFlux[0]);
                    CHECK_EXPECT("massFlux1", 0.0, massFlux[1]);
                    CHECK_EXPECT("massFlux2", 29.05934985931918, massFlux[2]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 2.905934985931918, density);
                    *specificHeat = 1009.8821078326129;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 2.905934985931918, density);
                    *specificHeat = 721.6389369683443;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("density", 2.905934985931918, density);
                    *sensibleEnthalpy = 2271.9243262007103;
                    return 0;
                },
            .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {2.905934985931918, -244872.459632804, 0.0, 0.0, 29.05934985931918},
            .stencilValues = {20, 3000 * 20, 0.0, 0.0, (10.0 + 40000.000000000015) * 20},
            .expectedResults = {-157992.19383660285, 1.3313421427129639E10, 0.0, 0.0, -1579921.9383660285}},
        // case 3 with ev and yi
        (InletTestParameters){
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvEqs = 2,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 1.783191515808363, density);
                        CHECK_EXPECT("velocity0", 10, velocity[0]);
                        CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, densityYi[0]);
                        CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, densityYi[1]);
                        CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, densityYi[2]);
                        CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 431.6854962124021;
                        *p = 251619.82076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 40010.000000, velocity[0]);
                        CHECK_EXPECT("densityYi0", 0.2 * 20, densityYi[0]);
                        CHECK_EXPECT("densityYi1", 0.3 * 20, densityYi[1]);
                        CHECK_EXPECT("densityYi2", 0.4 * 20, densityYi[2]);
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
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, densityYi[0]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, densityYi[1]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, densityYi[2]);
                    CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 10 * 1.783191515808363, massFlux[0]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *specificHeat = 1934.650079471233;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *specificHeat = 1464.9215577478003;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *sensibleEnthalpy = 4347.52375485136;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {1.783191515808363,
                               -243778.19371678037,
                               10 * 1.783191515808363,
                               0.1 * 1.783191515808363,
                               0.5 * 1.783191515808363,
                               0.4 * 1.783191515808363,
                               0.25 * 1.783191515808363,
                               0.5 * 1.783191515808363},
            .stencilValues = {20, 3000 * 20, (40000.000 + 10.0) * 20, .2 * 20, .3 * 20, .4 * 20, .5 * 20, .6 * 20},
            .expectedResults = {-92016.22693434241, 1.2579439390456383E10, -920162.2693434241, -9201.622693434241, -46008.113467171206, -36806.490773736965, -23004.056733585603, -46008.113467171206}},
        // case 4 with ev/yi and alpha
        (InletTestParameters){
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvEqs = 2,
            .decodeStateFunction =
                [](PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a, PetscReal* p, void* ctx) {
                    static int count = 0;
                    if (count == 0) {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 1.783191515808363, density);
                        CHECK_EXPECT("velocity0", 10, velocity[0]);
                        CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, densityYi[0]);
                        CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, densityYi[1]);
                        CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, densityYi[2]);
                        CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                        *internalEnergy = NAN;
                        *a = 431.6854962124021;
                        *p = 251619.82076699706;
                    } else {
                        CHECK_EXPECT("dim", 1, dim);
                        CHECK_EXPECT("density", 20, density);
                        CHECK_EXPECT("velocity0", 40010.000000, velocity[0]);
                        CHECK_EXPECT("densityYi0", 0.2 * 20, densityYi[0]);
                        CHECK_EXPECT("densityYi1", 0.3 * 20, densityYi[1]);
                        CHECK_EXPECT("densityYi2", 0.4 * 20, densityYi[2]);
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
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, densityYi[0]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, densityYi[1]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, densityYi[2]);
                    CHECK_EXPECT("totalEnergy", -136708.9241708678, totalEnergy, 1E-3);
                    CHECK_EXPECT("massFlux", 10 * 1.783191515808363, massFlux[0]);
                    *T = 300.4;
                    return 0;
                },
            .computeCpFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *specificHeat = 1934.650079471233;
                    return 0;
                },
            .computeCvFunction =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *specificHeat = 1464.9215577478003;
                    return 0;
                },
            .computeSensibleEnthalpy =
                [](PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* sensibleEnthalpy, void* ctx) {
                    CHECK_EXPECT("T", 300.4, T);
                    CHECK_EXPECT("yi0", 0.1, yi[0]);
                    CHECK_EXPECT("yi1", 0.5, yi[1]);
                    CHECK_EXPECT("yi2", 0.4, yi[2]);
                    CHECK_EXPECT("density", 1.783191515808363, density);
                    *sensibleEnthalpy = 4347.52375485136;
                    return 0;
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125, 1.0); },
            .boundaryValues = {1.783191515808363,
                               -243778.19371678037,
                               10 * 1.783191515808363,
                               0.1 * 1.783191515808363,
                               0.5 * 1.783191515808363,
                               0.4 * 1.783191515808363,
                               0.25 * 1.783191515808363,
                               0.5 * 1.783191515808363},
            .stencilValues = {20, 3000 * 20, (40000.000 + 10.0) * 20, .2 * 20, .3 * 20, .4 * 20, .5 * 20, .6 * 20},
            .expectedResults = {333143.9252721959, -4.554374761802185E10, 3331439.252721959, 33314.39252721959, 166571.96263609795, 133257.57010887837, 83285.98131804897, 166571.96263609795}}),
    [](const testing::TestParamInfo<InletTestParameters>& info) { return std::to_string(info.index); });