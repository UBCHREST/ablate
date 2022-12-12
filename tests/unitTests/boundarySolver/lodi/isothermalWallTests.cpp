#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/isothermalWall.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ff = ablate::finiteVolume::CompressibleFlowFields;
struct IsothermalWallTestParameters {
    PetscInt dim;
    PetscInt nEqs;
    PetscInt nSpecEqs = 0;
    std::vector<PetscInt> nEvComps;
    std::vector<ablate::domain::Field> fields;

    std::function<void(const PetscReal conserved[], PetscReal* property)> computeTemperatureFunction;
    std::function<void(const PetscReal conserved[], PetscReal temperature, PetscReal* property)> computeSpeedOfSoundFunction;
    std::function<void(const PetscReal conserved[], PetscReal temperature, PetscReal* property)> computePressureFromTemperature;

    std::function<void(const PetscReal conserved[], PetscReal temperature, PetscReal* property)> computeCpFunction;
    std::function<void(const PetscReal conserved[], PetscReal temperature, PetscReal* property)> computeCvFunction;
    std::function<void(const PetscReal conserved[], PetscReal temperature, PetscReal* property)> computeSensibleEnthalpy;
    std::function<void(const PetscReal conserved[], PetscReal* property)> computeStencilPressureFunction;

    ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom fvFaceGeom;
    std::function<std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling>()> getPgs = []() { return nullptr; };

    std::vector<PetscScalar> boundaryValues;
    std::vector<PetscScalar> stencilValues; /* the grad is (boundary-stencil)/1.0*/
    std::vector<PetscScalar> expectedResults;
};

class IsothermalWallTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<IsothermalWallTestParameters> {};

TEST_P(IsothermalWallTestFixture, ShouldComputeCorrectSourceTerm) {
    // arrange
    // get the required variables
    const auto& params = GetParam();
    // setup the eos
    auto mockEOS = std::make_shared<ablateTesting::eos::MockEOS>();

    EXPECT_CALL(*mockEOS, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction(params.computeTemperatureFunction)));
    EXPECT_CALL(*mockEOS, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpeedOfSound, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(params.computeSpeedOfSoundFunction)));
    EXPECT_CALL(*mockEOS, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::Pressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(params.computePressureFromTemperature)));
    EXPECT_CALL(*mockEOS, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(params.computeCpFunction)));
    EXPECT_CALL(*mockEOS, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(params.computeCvFunction)));
    EXPECT_CALL(*mockEOS, GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty::SensibleEnthalpy, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicTemperatureFunction(params.computeSensibleEnthalpy)));
    EXPECT_CALL(*mockEOS, GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Pressure, testing::_))
        .Times(::testing::Exactly(1))
        .WillOnce(::testing::Return(ablateTesting::eos::MockEOS::CreateMockThermodynamicFunction(params.computeStencilPressureFunction)));
    // create the boundary
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary = std::make_shared<ablate::boundarySolver::lodi::IsothermalWall>(mockEOS, params.getPgs());
    boundary->Setup(params.dim, params.nEqs, params.nSpecEqs, params.nEvComps, params.fields);

    PetscInt uOff[4] = {0, params.dim + 2, params.dim + 2 + params.nSpecEqs, params.dim + 2 + params.nSpecEqs + (params.nEvComps.empty() ? 0 : params.nEvComps[0])};
    PetscInt aOff[1] = {0};
    PetscInt sOff[4] = {0, params.dim + 2, params.dim + 2 + params.nSpecEqs, params.dim + 2 + params.nSpecEqs + (params.nEvComps.empty() ? 0 : params.nEvComps[0])};
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
        (IsothermalWallTestParameters){.dim = 1,
                                       .nEqs = 3,
                                       .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
                                       .computeTemperatureFunction =
                                           [](const PetscReal conserved[], PetscReal* property) {
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = 296.40099999999995;
                                           },
                                       .computeSpeedOfSoundFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = 345.811145133247754;
                                           },
                                       .computePressureFromTemperature =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = 251619.92076699706;
                                           },
                                       .computeCpFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = 1009.36685027;
                                           },
                                       .computeCvFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = 721.1236794;
                                           },
                                       .computeSensibleEnthalpy =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.9451426166013044, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.9451426166013044, conserved[ff::RHOE], 1E-3);
                                               *property = -1765.5644007;
                                           },
                                       .computeStencilPressureFunction =
                                           [](const PetscReal conserved[], PetscReal* property) {
                                               CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 20 * -197999.99999999872, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                                               *property = 251619.92076699706 + 20.000000327126923;  // delta p = stencil-boundary ... stencil = boundary+deltap
                                           },
                                       .fvFaceGeom = {.normal = {1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                       .boundaryValues = {2.9451426166013044, -256819.75972598503, 0.0},
                                       .stencilValues = {20, 3000 * 20, -197999.99999999872 * 20},
                                       .expectedResults = {816226.6340554004, -7.117588359166196E10, 0.0}},
        // case 1
        (IsothermalWallTestParameters){.dim = 1,
                                       .nEqs = 3,
                                       .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
                                       .computeTemperatureFunction =
                                           [](const PetscReal conserved[], PetscReal* property) {
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = 296.40099999999995;
                                           },
                                       .computeSpeedOfSoundFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = 345.811145133247754;
                                           },
                                       .computePressureFromTemperature =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = 251619.92076699706;
                                           },
                                       .computeCpFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = 1009.36685027;
                                           },
                                       .computeCvFunction =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = 721.1236794;
                                           },
                                       .computeSensibleEnthalpy =
                                           [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                               CHECK_EXPECT("temperature", 296.40099999999995, temperature);
                                               CHECK_EXPECT("density", 2.945140275655796, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", -87201.128488 * 2.945140275655796, conserved[ff::RHOE], 1E-3);
                                               *property = -1765.5644007;
                                           },
                                       .computeStencilPressureFunction =
                                           [](const PetscReal conserved[], PetscReal* property) {
                                               CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                                               CHECK_EXPECT("mom0", 20 * 197999.99999999872, conserved[ff::RHOU]);
                                               CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                                               *property = 251619.92076699706 + 20.000000327126923;  // delta p = stencil-boundary ... stencil = boundary+deltap
                                           },
                                       .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                       .boundaryValues = {2.945140275655796, -256819.55559289496, 0.0},
                                       .stencilValues = {20, 3000 * 20, 198000.00000001234 * 20},
                                       .expectedResults = {-816225.98527, 7.117582701753714E10, 0.0}},
        // case 2
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 3,
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 100.42;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 201.83122426291567;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = -197610.71454374143;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20 * 11949298.440203, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708 + 20.000000327128298;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 5.205232334625347E8}},
        // case 3
        (IsothermalWallTestParameters){
            .dim = 2,
            .nEqs = 4,
            .fields = {{.name = "euler", .numberComponents = 4, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 100.42;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 201.83122426291567;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692914985507404 * 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 8.692914985507404 * -3.5, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = -197610.71454374143;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20.0 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 238985968.80405527, conserved[ff::RHOV]);
                    CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708 + 20.000000327128298;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {0.0, -1.0, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, 0.0, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, 10.0 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 0.0, 5.205232334625347E8}},
        // case 4
        (IsothermalWallTestParameters){
            .dim = 3,
            .nEqs = 5,
            .fields = {{.name = "euler", .numberComponents = 5, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 100.42;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 201.83122426291567;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.42, temperature);
                    CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 0.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 0.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", -3.5 * 8.692914985507404, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226549.9687619313, conserved[ff::RHOE], 1E-3);
                    *property = -197610.71454374143;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20.0 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("mom1", 20.0 * 15.0, conserved[ff::RHOV]);
                    CHECK_EXPECT("mom2", 20.0 * 11949298.440203, conserved[ff::RHOW]);
                    CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                    *property = 251619.72076699708 + 20.000000327128298;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692914985507404, -1969379.6184168267, 0.0, 0.0, -30.425202449275915},
            .stencilValues = {20, 3000 * 20, 10.0 * 20, 15.0 * 20, (1.1949301940202763E7 - 3.5) * 20},
            .expectedResults = {-1.487209238464385E8, 3.3692720651656207E13, 0.0, 0.0, 5.205232334625347E8}},
        // case 5 with ev and yi
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvComps = {2},
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);

                    *property = 300.4;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);

                    *property = 431.6854962124021;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    *property = 251619.82076699706;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    *property = 1934.650079471233;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    *property = 1464.9215577478003;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20.0, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20.0 * (40000.000 - 3.5), conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 20.0 * 3000, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.2 * 20.0, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.3 * 20.0, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 20.0, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.5 * 20.0, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.6 * 20.0, conserved[7]);
                    *property = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {1.783191515808363,
                               -243778.19371678037,
                               -3.5 * 1.783191515808363,
                               0.1 * 1.783191515808363,
                               0.5 * 1.783191515808363,
                               0.4 * 1.783191515808363,
                               0.25 * 1.783191515808363,
                               0.5 * 1.783191515808363},
            .stencilValues = {20, 3000 * 20, (40000.000 - 3.5) * 20, .2 * 20, .3 * 20, .4 * 20, .5 * 20, .6 * 20},
            .expectedResults = {-94962.06945150577, 1.2986328812551773E10, 332367.2430802702, -9496.206945150578, -47481.034725752885, -37984.82778060231, -23740.517362876442, -47481.034725752885}},
        // case 6 with ev/yi and alpha
        (IsothermalWallTestParameters){
            .dim = 1,
            .nEqs = 9,
            .nSpecEqs = 3,
            .nEvComps = {2, 1},
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}},
                       {.name = "otherEV", .numberComponents = 1, .offset = 8, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);

                    *property = 300.4;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);
                    *property = 431.6854962124021;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);
                    *property = 251619.82076699706;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);
                    *property = 1934.650079471233;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);
                    *property = 1464.9215577478003;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -3.5, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[8]);
                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20.0, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20.0 * 39996.500000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 20.0 * 3000, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.2 * 20.0, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.3 * 20.0, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 20.0, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.5 * 20.0, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.6 * 20.0, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.5 * 20.0, conserved[8]);
                    *property = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125.0, 1.0); },
            .boundaryValues = {1.783191515808363,
                               -243778.19371678037,
                               -3.5 * 1.783191515808363,
                               0.1 * 1.783191515808363,
                               0.5 * 1.783191515808363,
                               0.4 * 1.783191515808363,
                               0.25 * 1.783191515808363,
                               0.5 * 1.783191515808363,
                               0.25 * 1.783191515808363},
            .stencilValues = {20, 3000 * 20, (40000.000 - 3.5) * 20, .2 * 20, .3 * 20, .4 * 20, .5 * 20, .6 * 20, .5 * 20},
            .expectedResults = {-175407.46347393887,
                                2.398746162552289E10,
                                613926.122158786,
                                -17540.746347393888,
                                -87703.73173696944,
                                -70162.98538957555,
                                -43851.86586848472,
                                -87703.73173696944,
                                -43851.86586848472}}),
    [](const testing::TestParamInfo<IsothermalWallTestParameters>& info) {
        return "test" + std::to_string(info.index) + "d" + std::to_string(info.param.dim) + "e" + std::to_string(info.param.nEqs) + "s" + std::to_string(info.param.nSpecEqs) + "ev" +
               ablate::utilities::VectorUtilities::Concatenate(info.param.nEvComps, "ev");
    });