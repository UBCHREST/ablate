#include <functional>
#include "PetscTestFixture.hpp"
#include "eos/transport/constant.hpp"
#include "eos/transport/twoPhaseTransport.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "gtest/gtest.h"
#include "mockTransportModel.hpp"

struct TwoPhaseTransportTestParameters {
    // input
    std::function<void(ablateTesting::eos::transport::MockTransportModel&, ablateTesting::eos::transport::MockTransportModel&)> setupTransportModels;
    const std::vector<PetscReal> conservedIn;
    const std::vector<ablate::domain::Field> fields;

    // expect output
    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class TwoPhaseTransportTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TwoPhaseTransportTestParameters> {};

TEST_P(TwoPhaseTransportTestFixture, ShouldComputeCorrectDirectFunction) {
    // ARRANGE
    auto transportModel1 = std::make_shared<ablateTesting::eos::transport::MockTransportModel>();
    auto transportModel2 = std::make_shared<ablateTesting::eos::transport::MockTransportModel>();
    GetParam().setupTransportModels(*transportModel1, *transportModel2);

    auto twoPhaseModel = std::make_shared<ablate::eos::transport::TwoPhaseTransport>(transportModel1, transportModel2);
    auto conductivityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, GetParam().fields);
    auto viscosityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, GetParam().fields);
    auto diffusivityFunction = twoPhaseModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, GetParam().fields);

    // ACT
    PetscReal computedK = NAN;
    conductivityFunction.function(GetParam().conservedIn.data(), &computedK, conductivityFunction.context.get());
    PetscReal computedMu = NAN;
    viscosityFunction.function(GetParam().conservedIn.data(), &computedMu, viscosityFunction.context.get());
    PetscReal computedDiff = NAN;
    diffusivityFunction.function(GetParam().conservedIn.data(), &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    ASSERT_LT(PetscAbs((GetParam().expectedConductivity - computedK) / GetParam().expectedConductivity), 1E-5);
    ASSERT_LT(PetscAbs((GetParam().expectedViscosity - computedMu) / GetParam().expectedViscosity), 1E-5);
    ASSERT_LT(PetscAbs((GetParam().expectedDiffusivity - computedDiff) / GetParam().expectedDiffusivity), 1E-5);
}

INSTANTIATE_TEST_SUITE_P(
    TwoPhaseTransportTests, TwoPhaseTransportTestFixture,
    testing::Values(
        (TwoPhaseTransportTestParameters){
            // test with a single field
            .setupTransportModels =
                [](ablateTesting::eos::transport::MockTransportModel& mock1, ablateTesting::eos::transport::MockTransportModel& mock2) {
                    // set up the first mock
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 1.0; })));
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 2.0; })));
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 200; })));

                    // set up the second mock
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 3.0; })));
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 4.0; })));
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 100; })));
                },
            .conservedIn = {0.75},
            .fields = {ablate::domain::Field{.name = ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, .numberComponents = 1, .offset = 0}},
            .expectedConductivity = 1.5,
            .expectedViscosity = 2.5,
            .expectedDiffusivity = 175.0},
        (TwoPhaseTransportTestParameters){
            // test with a multiple field
            .setupTransportModels =
                [](ablateTesting::eos::transport::MockTransportModel& mock1, ablateTesting::eos::transport::MockTransportModel& mock2) {
                    // set up the first mock
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 1.0; })));
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 2.0; })));
                    EXPECT_CALL(mock1, GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 200; })));

                    // set up the second mock
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 3.0; })));
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 4.0; })));
                    EXPECT_CALL(mock2, GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                        .Times(::testing::Exactly(1))
                        .WillOnce(::testing::Return(
                            ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicFunction([](const PetscReal conserved[], PetscReal* property) { *property = 100; })));
                },
            .conservedIn = {NAN, NAN, 0.75, NAN},
            .fields = {ablate::domain::Field{.name = "OtherField", .numberComponents = 0, .offset = 0},
                       ablate::domain::Field{.name = ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, .numberComponents = 1, .offset = 2}},
            .expectedConductivity = 1.5,
            .expectedViscosity = 2.5,
            .expectedDiffusivity = 175.0}),
    [](const testing::TestParamInfo<TwoPhaseTransportTestParameters>& info) { return std::to_string(info.index); });

struct TwoPhaseTransportTemperatureTestParameters {
    // input
    std::function<void(ablateTesting::eos::transport::MockTransportModel&, ablateTesting::eos::transport::MockTransportModel&)> setupTransportModels;
    const std::vector<PetscReal> conservedIn;
    const std::vector<ablate::domain::Field> fields;
    const double temperature;

    // expect output
    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class TwoPhaseTransportTemperatureTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TwoPhaseTransportTemperatureTestParameters> {};

TEST_P(TwoPhaseTransportTemperatureTestFixture, ShouldComputeCorrectTemperatureFunction) {
    // ARRANGE
    auto transportModel1 = std::make_shared<ablateTesting::eos::transport::MockTransportModel>();
    auto transportModel2 = std::make_shared<ablateTesting::eos::transport::MockTransportModel>();
    GetParam().setupTransportModels(*transportModel1, *transportModel2);

    auto twoPhaseModel = std::make_shared<ablate::eos::transport::TwoPhaseTransport>(transportModel1, transportModel2);
    auto conductivityFunction = twoPhaseModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, GetParam().fields);
    auto viscosityFunction = twoPhaseModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, GetParam().fields);
    auto diffusivityFunction = twoPhaseModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, GetParam().fields);

    // ACT
    PetscReal computedK = NAN;
    conductivityFunction.function(GetParam().conservedIn.data(), GetParam().temperature, &computedK, conductivityFunction.context.get());
    PetscReal computedMu = NAN;
    viscosityFunction.function(GetParam().conservedIn.data(), GetParam().temperature, &computedMu, viscosityFunction.context.get());
    PetscReal computedDiff = NAN;
    diffusivityFunction.function(GetParam().conservedIn.data(), GetParam().temperature, &computedDiff, diffusivityFunction.context.get());

    // ASSERT
    ASSERT_LT(PetscAbs((GetParam().expectedConductivity - computedK) / GetParam().expectedConductivity), 1E-5);
    ASSERT_LT(PetscAbs((GetParam().expectedViscosity - computedMu) / GetParam().expectedViscosity), 1E-5);
    ASSERT_LT(PetscAbs((GetParam().expectedDiffusivity - computedDiff) / GetParam().expectedDiffusivity), 1E-5);
}

INSTANTIATE_TEST_SUITE_P(TwoPhaseTransportTests, TwoPhaseTransportTemperatureTestFixture,
                         testing::Values(
                             (TwoPhaseTransportTemperatureTestParameters){
                                 // test with a single field
                                 .setupTransportModels =
                                     [](ablateTesting::eos::transport::MockTransportModel& mock1, ablateTesting::eos::transport::MockTransportModel& mock2) {
                                         // set up the first mock
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 1.0; })));
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 2.0; })));
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 200; })));

                                         // set up the second mock
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 3.0; })));
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 4.0; })));
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 100; })));
                                     },
                                 .conservedIn = {0.75},
                                 .fields = {ablate::domain::Field{.name = ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, .numberComponents = 1, .offset = 0}},
                                 .temperature = 300,
                                 .expectedConductivity = 450,
                                 .expectedViscosity = 750,
                                 .expectedDiffusivity = 52500},
                             (TwoPhaseTransportTemperatureTestParameters){
                                 // test with a multiple field
                                 .setupTransportModels =
                                     [](ablateTesting::eos::transport::MockTransportModel& mock1, ablateTesting::eos::transport::MockTransportModel& mock2) {
                                         // set up the first mock
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 1.0; })));
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 2.0; })));
                                         EXPECT_CALL(mock1, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 200; })));

                                         // set up the second mock
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 3.0; })));
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 4.0; })));
                                         EXPECT_CALL(mock2, GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, testing::_))
                                             .Times(::testing::Exactly(1))
                                             .WillOnce(::testing::Return(ablateTesting::eos::transport::MockTransportModel::CreateMockThermodynamicTemperatureFunction(
                                                 [](const PetscReal conserved[], PetscReal temperature, PetscReal* property) { *property = temperature * 100; })));
                                     },
                                 .conservedIn = {NAN, NAN, 0.75, NAN},
                                 .fields = {ablate::domain::Field{.name = "OtherField", .numberComponents = 0, .offset = 0},
                                            ablate::domain::Field{.name = ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, .numberComponents = 1, .offset = 2}},
                                 .temperature = 1000,
                                 .expectedConductivity = 1500,
                                 .expectedViscosity = 2500,
                                 .expectedDiffusivity = 175000}),
                         [](const testing::TestParamInfo<TwoPhaseTransportTemperatureTestParameters>& info) { return std::to_string(info.index); });
