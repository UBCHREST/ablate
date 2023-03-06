#include "PetscTestFixture.hpp"
#include "eos/mockEOS.hpp"
#include "eos/radiationProperties/constant.hpp"
#include "eos/radiationProperties/sum.hpp"
#include "gtest/gtest.h"

struct RadiationSumTestParameters {
    std::function<std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>>()> getInputModels;

    std::map<ablate::eos::radiationProperties::RadiationProperty, PetscReal> expectedParameters;
};

class RadiationSumTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<RadiationSumTestParameters> {};

TEST_P(RadiationSumTestFixture, ShouldComputeCorrectValueForGetRadiationPropertiesFunction) {
    // arrange
    auto inputModels = GetParam().getInputModels();
    auto sumModel = std::make_shared<ablate::eos::radiationProperties::Sum>(inputModels);

    for (const auto& [property, expectedValue] : GetParam().expectedParameters) {
        // act
        auto testFunction = sumModel->GetRadiationPropertiesFunction(property, {});

        PetscReal computeValue = NAN;
        testFunction.function(nullptr, &computeValue, testFunction.context.get());

        // assert
        ASSERT_DOUBLE_EQ(expectedValue, computeValue) << "should be correct for " << property;
    }
}

TEST_P(RadiationSumTestFixture, ShouldComputeCorrectValueForGetRadiationPropertiesTemperatureFunction) {
    // arrange
    auto inputModels = GetParam().getInputModels();
    auto sumModel = std::make_shared<ablate::eos::radiationProperties::Sum>(inputModels);

    for (const auto& [property, expectedValue] : GetParam().expectedParameters) {
        // act
        auto testFunction = sumModel->GetRadiationPropertiesTemperatureFunction(property, {});

        PetscReal computeValue = NAN;
        testFunction.function(nullptr, 1000, &computeValue, testFunction.context.get());

        // assert
        ASSERT_DOUBLE_EQ(expectedValue, computeValue) << "should be correct for " << property;
    }
}

INSTANTIATE_TEST_SUITE_P(
    RadiationSumTests, RadiationSumTestFixture,
    testing::Values(
        (RadiationSumTestParameters){
            .getInputModels =
                []() {
                    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.
                    std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

                    return std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>>{std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 1.4, 1),
                                                                                                          std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 1.6, 1)};
                },
            .expectedParameters = {{ablate::eos::radiationProperties::RadiationProperty::Absorptivity, 3.0}}},
        (RadiationSumTestParameters){
            .getInputModels =
                []() {
                    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.

                    std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

                    return std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>>{std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 1.4, 1)};
                },
            .expectedParameters = {{ablate::eos::radiationProperties::RadiationProperty::Absorptivity, 1.4}}},
        (RadiationSumTestParameters){
            .getInputModels =
                []() {
                    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();  //!< Create a mock eos with parameters to feed to the Zimmer model.
                    std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

                    return std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>>{std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 1.4, 1),
                                                                                                          std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 2.4, 1),
                                                                                                          std::make_shared<ablate::eos::radiationProperties::Constant>(eos, 3.5, 1)};
                },
            .expectedParameters = {{ablate::eos::radiationProperties::RadiationProperty::Absorptivity, 7.3}}}

        ));

TEST(RadiationSumTests, ShouldThrowExceptionForEmptyModelList) {
    // arrange
    std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>> models;

    // act/assert
    ASSERT_THROW(std::make_shared<ablate::eos::radiationProperties::Sum>(models);, std::invalid_argument);
}