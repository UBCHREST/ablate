#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA
#include <functional>
#include "domain/mockField.hpp"
#include "eos/tChemSoot/sootSpeciesTransportModel.hpp"
#include "eos/transport/constant.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "gtest/gtest.h"

struct SootSpeciesTransportModelTestParameters {
    //! the name of the test
    std::string name;
    //! function to get the base transport model
    std::function<std::shared_ptr<ablate::eos::transport::TransportModel>()> getBaseTransportModel;

    //! the list of fields
    std::vector<ablate::domain::Field> fields;

    //! the expected values
    PetscReal expectedK;
    PetscReal expectedMu;
    std::vector<PetscReal> expectedDiff;
};

class SootSpeciesTransportModelTestFixture : public ::testing::TestWithParam<SootSpeciesTransportModelTestParameters> {};

TEST_P(SootSpeciesTransportModelTestFixture, ShouldComputePropertiesForFunction) {
    // ARRANGE
    auto baseTransportModel = GetParam().getBaseTransportModel();

    // Create a soot transport property
    auto sootSpeciesTransportModel = std::make_shared<ablate::eos::tChemSoot::SootSpeciesTransportModel>(baseTransportModel);

    auto conductivityFunction = sootSpeciesTransportModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Conductivity, GetParam().fields);
    auto viscosityFunction = sootSpeciesTransportModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Viscosity, GetParam().fields);
    auto diffusivityFunction = sootSpeciesTransportModel->GetTransportFunction(ablate::eos::transport::TransportProperty::Diffusivity, GetParam().fields);

    // ACT
    PetscReal computedK = NAN;
    PetscReal computedMu = NAN;
    std::vector<PetscReal> computedDiff(diffusivityFunction.propertySize, NAN);

    conductivityFunction.function(nullptr, &computedK, conductivityFunction.context.get());
    viscosityFunction.function(nullptr, &computedMu, viscosityFunction.context.get());
    diffusivityFunction.function(nullptr, computedDiff.data(), diffusivityFunction.context.get());

    // ASSERT
    ASSERT_EQ(conductivityFunction.propertySize, 1);
    ASSERT_EQ(viscosityFunction.propertySize, 1);
    ASSERT_EQ(diffusivityFunction.propertySize, GetParam().expectedDiff.size());

    ASSERT_DOUBLE_EQ(GetParam().expectedK, computedK);
    ASSERT_DOUBLE_EQ(GetParam().expectedMu, computedMu);
    for (std::size_t i = 0; i < GetParam().expectedDiff.size(); ++i) {
        ASSERT_DOUBLE_EQ(GetParam().expectedDiff[i], computedDiff[i]);
    }
}

TEST_P(SootSpeciesTransportModelTestFixture, ShouldComputePropertiesForTemperatureFunction) {
    // ARRANGE
    auto baseTransportModel = GetParam().getBaseTransportModel();

    // Create a soot transport property
    auto sootSpeciesTransportModel = std::make_shared<ablate::eos::tChemSoot::SootSpeciesTransportModel>(baseTransportModel);

    auto conductivityFunction = sootSpeciesTransportModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Conductivity, GetParam().fields);
    auto viscosityFunction = sootSpeciesTransportModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Viscosity, GetParam().fields);
    auto diffusivityFunction = sootSpeciesTransportModel->GetTransportTemperatureFunction(ablate::eos::transport::TransportProperty::Diffusivity, GetParam().fields);

    // ACT
    PetscReal computedK = NAN;
    PetscReal computedMu = NAN;
    std::vector<PetscReal> computedDiff(diffusivityFunction.propertySize, NAN);

    conductivityFunction.function(nullptr, NAN, &computedK, conductivityFunction.context.get());
    viscosityFunction.function(nullptr, NAN, &computedMu, viscosityFunction.context.get());
    diffusivityFunction.function(nullptr, NAN, computedDiff.data(), diffusivityFunction.context.get());

    // ASSERT
    ASSERT_EQ(conductivityFunction.propertySize, 1);
    ASSERT_EQ(viscosityFunction.propertySize, 1);
    ASSERT_EQ(diffusivityFunction.propertySize, GetParam().expectedDiff.size());

    ASSERT_DOUBLE_EQ(GetParam().expectedK, computedK);
    ASSERT_DOUBLE_EQ(GetParam().expectedMu, computedMu);
    for (std::size_t i = 0; i < GetParam().expectedDiff.size(); ++i) {
        ASSERT_DOUBLE_EQ(GetParam().expectedDiff[i], computedDiff[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(SootSpeciesTransportTests, SootSpeciesTransportModelTestFixture,
                         testing::Values(
                             (SootSpeciesTransportModelTestParameters){
                                 .name = "constant_diff",
                                 .getBaseTransportModel = []() { return std::make_shared<ablate::eos::transport::Constant>(.1, .2, .3); },
                                 .fields = {ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD, 6)},

                                 // expected values
                                 .expectedK = .1,
                                 .expectedMu = .2,
                                 .expectedDiff = {0.003, .3, .3, .3, .3, .3} /** solid carbon is always the first index **/

                             },
                             (SootSpeciesTransportModelTestParameters){
                                 .name = "constant_diff",
                                 .getBaseTransportModel =
                                     []() {
                                         return std::make_shared<ablate::eos::transport::Constant>(.1, .2, std::vector<double>{.3, .4, .5, .6});
                                     },
                                 .fields = {ablateTesting::domain::MockField::Create(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD, 4)},

                                 // expected values
                                 .expectedK = .1,
                                 .expectedMu = .2,
                                 .expectedDiff = {0.003, .4, .5, .6} /** solid carbon is always the first index **/

                             }),

                         [](const testing::TestParamInfo<SootSpeciesTransportModelTestParameters>& info) { return std::to_string(info.index) + "_" + info.param.name; });
#endif