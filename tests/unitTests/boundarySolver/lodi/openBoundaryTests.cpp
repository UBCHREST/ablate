#include <functional>
#include "PetscTestFixture.hpp"
#include "boundarySolver/lodi/openBoundary.hpp"
#include "eos/mockEOS.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
using ff = ablate::finiteVolume::CompressibleFlowFields;

struct OpenBoundaryTestParameters {
    std::string name;
    PetscInt dim;
    PetscInt nEqs;
    PetscInt nSpecEqs = 0;
    std::vector<PetscInt> nEvComps;
    double reflectFactor;
    double referencePressure;
    double maxAcousticsLength;
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

class OpenBoundaryTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<OpenBoundaryTestParameters> {};

TEST_P(OpenBoundaryTestFixture, ShouldComputeCorrectSourceTerm) {
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
    std::shared_ptr<ablate::boundarySolver::lodi::LODIBoundary> boundary =
        std::make_shared<ablate::boundarySolver::lodi::OpenBoundary>(mockEOS, GetParam().reflectFactor, GetParam().referencePressure, GetParam().maxAcousticsLength, params.getPgs());
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
        (OpenBoundaryTestParameters){.name = "1D subsonic into the domain",
                                     .dim = 1,
                                     .nEqs = 3,
                                     .reflectFactor = 0.0,
                                     .referencePressure = 101325.0,
                                     .maxAcousticsLength = 0.02,
                                     .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
                                     .computeTemperatureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = 100.42;
                                         },
                                     .computeSpeedOfSoundFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.42, temperature);
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = 201.83122426291567;
                                         },
                                     .computePressureFromTemperature =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.42, temperature);
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.72076699708;
                                         },
                                     .computeCpFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.42, temperature);
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = 995.8750316818866;
                                         },
                                     .computeCvFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.42, temperature);
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = 707.6318608176182;
                                         },
                                     .computeSensibleEnthalpy =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.42, temperature);
                                             CHECK_EXPECT("density", 8.692914985507404, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.692914985507404 * 10.0001, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 8.692914985507404 * -226506.0927619263, conserved[ff::RHOE], 1E-3);
                                             *property = -197610.71454374143;
                                         },
                                     .computeStencilPressureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", 20, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 20 * 9.9603020202, conserved[ff::RHOU]);
                                             CHECK_EXPECT("totalEnergy", 3000 * 20, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.72076699708 + 20.000000327128298;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = 100.43;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = 201.84127336667171;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -20.00011, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -226349.0154433181, conserved[ff::RHOE], 1E-3);
                    *property = -197600.7557934246;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (8.692018326008165 + 1.040246109552421) * -19.960302, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * (8.692018326008165 + 1.040246109552421), conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 100.43;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 201.84127336667171;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * -500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = -197600.7557934246;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (8.692018326008165 + 1.040246109552421) * -499.960302, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * (8.692018326008165 + 1.040246109552421), conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 100.43;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 201.84127336667171;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 995.8750316818866;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = 707.6318608176182;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 100.43, temperature);
                    CHECK_EXPECT("density", 8.692018326008165, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 8.692018326008165 * 500.0001, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 8.692018326008165 * -101548.96744331811, conserved[ff::RHOE], 1E-3);
                    *property = -197600.7557934246;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (8.692018326008165 + 1.040246109552421), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (8.692018326008165 + 1.040246109552421) * 500.0398979798, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 3000 * (8.692018326008165 + 1.040246109552421), conserved[ff::RHOE], 1E-3);
                    *property = 251618.820766997 + 200.00000096625962;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {8.692018326008165, -882665.4860045275, 4346.010032205915},
            .stencilValues = {(8.692018326008165 + 1.040246109552421), 3000 * (8.692018326008165 + 1.040246109552421), (500.0001 + 0.03979797979809766) * (8.692018326008165 + 1.040246109552421)},
            .expectedResults = {0.0, 0.0, 0.0}},
        (OpenBoundaryTestParameters){.name = "3D supersonic out of the domain",
                                     .dim = 3,
                                     .nEqs = 5,
                                     .reflectFactor = 0.15,
                                     .referencePressure = 202650.0,
                                     .maxAcousticsLength = 0.02,
                                     .fields = {{.name = "euler", .numberComponents = 5, .offset = 0}},
                                     .computeTemperatureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = 100.43;
                                         },
                                     .computeSpeedOfSoundFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.43, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = 201.8111245542304;
                                         },
                                     .computePressureFromTemperature =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.43, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.82076699712;
                                         },
                                     .computeCpFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.43, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = 995.8750316818866;
                                         },
                                     .computeCvFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.43, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = 707.6318608176182;
                                         },
                                     .computeSensibleEnthalpy =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.43, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -500., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -600, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -700, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * 323429.7536008524, conserved[ff::RHOE], 1E-3);
                                             *property = -197630.63204437506;
                                         },
                                     .computeStencilPressureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", (8.694650097350083 - 0.12298691191290341), conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", (8.694650097350083 - 0.12298691191290341) * (-500 + 1.9999999999981803), conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", (8.694650097350083 - 0.12298691191290341) * (-600 + 2.0000000000436553), conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", (8.694650097350083 - 0.12298691191290341) * (-700 + 0.004999999964638845), conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", (8.694650097350083 - 0.12298691191290341) * 3000, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.82076699712 + 199.99999993015075;  // delta p = stencil-boundary ... stencil = boundary+deltap
                                         },
                                     .fvFaceGeom = {.normal = {0.0, 0.0, -1.0}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
                                     .boundaryValues = {8.694650097350083, 2812108.5386315645, -4347.325048675041, -5216.79005841005, -6086.255068145058},
                                     .stencilValues = {(8.694650097350083 - 0.12298691191290341),
                                                       3000 * (8.694650097350083 - 0.12298691191290341),
                                                       (-500.0 + 1.9999999999981803) * (8.694650097350083 - 0.12298691191290341),
                                                       (-600.0 + 2.0000000000436553) * (8.694650097350083 - 0.12298691191290341),
                                                       (-700.0 + 0.004999999964638845) * (8.694650097350083 - 0.12298691191290341)},
                                     .expectedResults = {-86.13431158921169, -3.467059254299343E7, 55239.66593088489, 63853.09709008283, 60124.44938764354}},
        (OpenBoundaryTestParameters){.name = "3D subsonic out of the domain",
                                     .dim = 3,
                                     .nEqs = 5,
                                     .reflectFactor = 0.15,
                                     .referencePressure = 202650.0,
                                     .maxAcousticsLength = 0.02,
                                     .fields = {{.name = "euler", .numberComponents = 5, .offset = 0}},
                                     .computeTemperatureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = 100.4;
                                         },
                                     .computeSpeedOfSoundFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.4, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = 201.8111245542304;
                                         },
                                     .computePressureFromTemperature =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.4, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.82076699712;
                                         },
                                     .computeCpFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.4, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = 995.8750316818866;
                                         },
                                     .computeCvFunction =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.4, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = 707.6318608176182;
                                         },
                                     .computeSensibleEnthalpy =
                                         [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                                             CHECK_EXPECT("temperature", 100.4, temperature);
                                             CHECK_EXPECT("density", 8.694650097350083, conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", 8.694650097350083 * -5., conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", 8.694650097350083 * -6, conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", 8.694650097350083 * -7, conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", 8.694650097350083 * -226515.2463991476, conserved[ff::RHOE], 1E-3);
                                             *property = -197630.63204437506;
                                         },
                                     .computeStencilPressureFunction =
                                         [](const PetscReal conserved[], PetscReal* property) {
                                             CHECK_EXPECT("density", (8.694650097350083 - 0.12298691191290341), conserved[ff::RHO]);
                                             CHECK_EXPECT("mom0", (8.694650097350083 - 0.12298691191290341) * (-5.0 + 2.000000000000312), conserved[ff::RHOU]);
                                             CHECK_EXPECT("mom1", (8.694650097350083 - 0.12298691191290341) * (-6.0 + 0.005000000000165981), conserved[ff::RHOV]);
                                             CHECK_EXPECT("mom2", (8.694650097350083 - 0.12298691191290341) * (-7.0 + 2.000000000000312), conserved[ff::RHOW]);
                                             CHECK_EXPECT("totalEnergy", (8.694650097350083 - 0.12298691191290341) * 3000, conserved[ff::RHOE], 1E-3);
                                             *property = 251619.82076699712 + 199.99999993015075;  // delta p = stencil-boundary ... stencil = boundary+deltap
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
            .nEqs = 9,
            .nSpecEqs = 3,
            .nEvComps = {2, 1},
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}},
                       {.name = "otherEv", .numberComponents = 1, .offset = 8, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 300.4;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 431.6854962124021;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 251619.82076699706;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 1934.650079471233;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 1464.9215577478003;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    CHECK_EXPECT("otherEV", 0.25 * 1.783192, conserved[8]);

                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (1.783191515808363 + 90.16181478870485) * (-10 - 40000.000000000015), conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", (1.783191515808363 + 90.16181478870485) * 3000, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * (1.783191515808363 + 90.16181478870485), conserved[4]);
                    CHECK_EXPECT("densityYi2", (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[5]);

                    CHECK_EXPECT("densityEV0", (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485), conserved[6]);
                    CHECK_EXPECT("densityEV1", (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485), conserved[7]);
                    CHECK_EXPECT("otherEv", (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485), conserved[8]);
                    *property = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .boundaryValues = {1.783191515808363,
                               -243778.19371678037,
                               -17.831915158083632,
                               .1 * 1.783191515808363,
                               .5 * 1.783191515808363,
                               .4 * 1.783191515808363,
                               .25 * 1.783191515808363,
                               .5 * 1.783191515808363,
                               .25 * 1.783191515808363},
            .stencilValues = {(1.783191515808363 + 90.16181478870485),
                              3000 * (1.783191515808363 + 90.16181478870485),
                              (-10 - 40000.000000000015) * (1.783191515808363 + 90.16181478870485),
                              (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.5) * (1.783191515808363 + 90.16181478870485),
                              (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485),
                              (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485),
                              (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485),
                              (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485)},
            .expectedResults =
                {36966.4328851267, -2.0206605128724575E8, -1.6305600063259933E7, 5794.10023978034, 18483.21644256335, 12689.11620278302, 9277.272051597822, 18518.880272879534, 9277.272051597822}},
        (OpenBoundaryTestParameters){
            .name = "1D subsonic out of the domain with sp/ev and pgs",
            .dim = 1,
            .nEqs = 8,
            .nSpecEqs = 3,
            .nEvComps = {2},
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);

                    *property = 300.4;
                },
            .computeSpeedOfSoundFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);

                    *property = 431.6854962124021;
                },
            .computePressureFromTemperature =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    *property = 251619.82076699706;
                },
            .computeCpFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    *property = 1934.650079471233;
                },
            .computeCvFunction =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    *property = 1464.9215577478003;
                },
            .computeSensibleEnthalpy =
                [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) {
                    CHECK_EXPECT("temperature", 300.4, temperature);
                    CHECK_EXPECT("density", 1.783192, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783192 * -10.0, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783192 * -136708.8870501776, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783192, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783192, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783192, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783192, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783192, conserved[7]);
                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (1.783191515808363 + 90.16181478870485) * (-10 - 40000.000000000015), conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", (1.783191515808363 + 90.16181478870485) * 3000, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * (1.783191515808363 + 90.16181478870485), conserved[4]);
                    CHECK_EXPECT("densityYi2", (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[5]);

                    CHECK_EXPECT("densityEV0", (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485), conserved[6]);
                    CHECK_EXPECT("densityEV1", (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485), conserved[7]);
                    *property = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125.0, 1.0); },
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
            .nEvComps = {2},
            .reflectFactor = 0.15,
            .referencePressure = 202650.0,
            .maxAcousticsLength = 0.02,
            .fields = {{.name = "euler", .numberComponents = 3, .offset = 0},
                       {.name = "densityYi", .numberComponents = 3, .offset = 3},
                       {.name = "densityEV", .numberComponents = 2, .offset = 6, .tags = {ablate::finiteVolume::CompressibleFlowFields::EV_TAG}}},
            .computeTemperatureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", 1.783191515808363, conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

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
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

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
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

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
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

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
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

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
                    CHECK_EXPECT("mom0", 1.783191515808363 * -1000.000000, conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", 1.783191515808363 * 363241.0758291322, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", 0.1 * 1.783191515808363, conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * 1.783191515808363, conserved[4]);
                    CHECK_EXPECT("densityYi2", 0.4 * 1.783191515808363, conserved[5]);

                    CHECK_EXPECT("densityEV0", 0.25 * 1.783191515808363, conserved[6]);
                    CHECK_EXPECT("densityEV1", 0.5 * 1.783191515808363, conserved[7]);
                    *property = 4347.52375485136;
                },
            .computeStencilPressureFunction =
                [](const PetscReal conserved[], PetscReal* property) {
                    CHECK_EXPECT("density", (1.783191515808363 + 90.16181478870485), conserved[ff::RHO]);
                    CHECK_EXPECT("mom0", (1.783191515808363 + 90.16181478870485) * (-1000 - 40000.000000000015), conserved[ff::RHOU]);
                    CHECK_EXPECT("totalEnergy", (1.783191515808363 + 90.16181478870485) * 3000, conserved[ff::RHOE], 1E-3);

                    CHECK_EXPECT("densityYi0", (.1 + 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[3]);
                    CHECK_EXPECT("densityYi1", 0.5 * (1.783191515808363 + 90.16181478870485), conserved[4]);
                    CHECK_EXPECT("densityYi2", (.4 - 117.62376237623829) * (1.783191515808363 + 90.16181478870485), conserved[5]);

                    CHECK_EXPECT("densityEV0", (.25 + 1.9999999999989138) * (1.783191515808363 + 90.16181478870485), conserved[6]);
                    CHECK_EXPECT("densityEV1", (.5 + 2.000000000001112) * (1.783191515808363 + 90.16181478870485), conserved[7]);
                    *property = 251619.82076699706 + 199.99999981373549;  // delta p = stencil-boundary ... stencil = boundary+deltap
                },
            .fvFaceGeom = {.normal = {-1, NAN, NAN}, .areas = {NAN, NAN, NAN}, .centroid = {NAN, NAN, NAN}},
            .getPgs = []() { return std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 125.0, 1.0); },
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