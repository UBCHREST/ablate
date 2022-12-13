#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/inlet.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ff = ablate::finiteVolume::CompressibleFlowFields;
struct InletTestParameters {
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

class InletTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<InletTestParameters> {};

TEST_P(InletTestFixture, ShouldComputeCorrectSourceTerm) {
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
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary = std::make_shared<ablate::boundarySolver::lodi::Inlet>(mockEOS, params.getPgs());
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
        (InletTestParameters){.dim = 1,
                              .nEqs = 3,
                              .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
                              .computeTemperatureFunction =
                                  [](const PetscReal conserved[], PetscReal* property) {
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = 100.4;
                                  },
                              .computeSpeedOfSoundFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 100.4, temperature);
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = 201.8111245542304;
                                  },
                              .computePressureFromTemperature =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 100.4, temperature);
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = 251619.82076699712;
                                  },
                              .computeCpFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 100.4, temperature);
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = 995.8750316818866;
                                  },
                              .computeCvFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 100.4, temperature);
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = 707.6318608176182;
                                  },
                              .computeSensibleEnthalpy =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 100.4, temperature);
                                      CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 8.694650097350083 * 500.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 8.694650097350083 * -101570.2463991476, conserved[ff::RHOE], 1E-3);
                                      *property = -197630.63204437506;
                                  },
                              .computeStencilPressureFunction =
                                  [](const PetscReal conserved[], PetscReal* property) {
                                      CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 20 * (500.0 + 39999.99999999889), conserved[ff::RHOU]);
                                      CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                                      *property = 251619.82076699712 + 199.9999986612238;  // delta p = stencil-boundary ... stencil = boundary+deltap
                                  },
                              .fvFaceGeom = {.normal = {-1.0, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                              .boundaryValues = {8.694650097350083, -883117.7527422206, 4347.32504867504},
                              .stencilValues = {20, 3000 * 20, (500.0 + 39999.99999999889) * 20},
                              .expectedResults = {723193.7481425349, -7.345496719316035E10, 3.615968740712674E8}},
        // case 1
        (InletTestParameters){.dim = 3,
                              .nEqs = 5,
                              .fields = {{.name = "euler", .numberComponents = 5, .offset = 0}},
                              .computeTemperatureFunction =
                                  [](const PetscReal conserved[], PetscReal* property) {
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 300.4;
                                  },
                              .computeSpeedOfSoundFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 300.4, temperature);
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 348.10065251594057;
                                  },
                              .computePressureFromTemperature =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 300.4, temperature);
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 251619.82076699703;
                                  },
                              .computeCpFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 300.4, temperature);
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 1009.8821078326129;
                                  },
                              .computeCvFunction =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 300.4, temperature);
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 721.6389369683443;
                                  },
                              .computeSensibleEnthalpy =
                                  [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                      CHECK_EXPECT("temperature", 300.4, temperature);
                                      CHECK_EXPECT("density", 2.905934985931918, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom0", 2.905934985931918 * 10.0, conserved[ff::RHOW]);
                                      CHECK_EXPECT("totalEnergy", 2.905934985931918 * -84266.3242014256, conserved[ff::RHOE], 1E-3);
                                      *property = 2271.9243262007103;
                                  },
                              .computeStencilPressureFunction =
                                  [](const PetscReal conserved[], PetscReal* property) {
                                      CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                                      CHECK_EXPECT("mom0", 20 * 0.0, conserved[ff::RHOU]);
                                      CHECK_EXPECT("mom1", 20 * 0.0, conserved[ff::RHOV]);
                                      CHECK_EXPECT("mom2", 20 * (10.0 + 40000.000000000015), conserved[ff::RHOW]);

                                      CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                                      *property = 251619.82076699712 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .nEvComps = {2},
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
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
                    CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20 * 40010.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.2 * 20.0, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.3 * 20.0, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 20.0, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.5 * 20.0, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.6 * 20.0, conserved[7]);

                    *property = 251619.82076699712 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .nEqs = 9,
            .nSpecEqs = 3,
            .nEvComps = {1, 2},
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "otherEV", .numberComponents = 1, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}},
                       {.name = "densityEV", .numberComponents = 2, .offset = 7, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);

                    *property = 300.4;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);
                    *property = 431.6854962124021;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);
                    *property = 251619.82076699706;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);
                    *property = 1934.650079471233;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);
                    *property = 1464.9215577478003;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * 10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * -136708.9241708678, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[8]);
                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 20 * 40010.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.2 * 20.0, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.3 * 20.0, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 20.0, conserved[5]);

                    CHECK_EXPECT("otherEV", 0.5 * 20.0, conserved[6]);
                    CHECK_EXPECT("densityEV0", 0.5 * 20.0, conserved[7]);
                    CHECK_EXPECT("densityEV1", 0.6 * 20.0, conserved[8]);

                    *property = 251619.82076699712 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
                               0.25 * 1.783191515808363,
                               0.5 * 1.783191515808363},
            .stencilValues = {20, 3000 * 20, (40000.000 + 10.0) * 20, .2 * 20, .3 * 20, .4 * 20, .5 * 20, .5 * 20, .6 * 20},
            .expectedResults =
                {333143.9252721959, -4.554374761802185E10, 3331439.252721959, 33314.39252721959, 166571.96263609795, 133257.57010887837, 83285.98131804897, 83285.98131804897, 166571.96263609795}}),
    [](const testing::TestParamInfo<InletTestParameters>& info) { return std::to_string(info.index); });