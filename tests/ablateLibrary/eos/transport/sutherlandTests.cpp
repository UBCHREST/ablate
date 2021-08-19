#include <eos/mockEOS.hpp>
#include <vector>
#include "eos/transport/sutherland.hpp"
#include "gtest/gtest.h"

static PetscErrorCode MockCpFunction(PetscReal T, PetscReal density, const PetscReal yi[], PetscReal* specificHeat, void* ctx) {
    // this is a fun hack to set cp equal to the first yi
    *specificHeat = yi[0];
    return 0;
}

struct SutherlandTransportTestParameters {
    PetscReal temperatureIn;
    PetscReal densityIn;
    PetscReal cpIn;

    PetscReal expectedConductivity;
    PetscReal expectedViscosity;
    PetscReal expectedDiffusivity;
};

class SutherlandTransportTestFixture : public ::testing::TestWithParam<SutherlandTransportTestParameters> {};

TEST_P(SutherlandTransportTestFixture, ShouldComputeCorrectConductivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureFunction()).Times(::testing::Exactly(1)).WillOnce(::testing::Return(MockCpFunction));
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureContext()).Times(::testing::Exactly(1));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);

    // ACT
    PetscReal computedK = NAN;
    auto yi = GetParam().cpIn;  // just for testing, set the yi from the cp
    sutherlandModel->GetComputeConductivityFunction()(GetParam().temperatureIn, GetParam().densityIn, &yi, computedK, sutherlandModel->GetComputeConductivityContext());

    // ASSERT
    double error = (GetParam().expectedConductivity - computedK) / GetParam().expectedConductivity;
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeCorrectViscosity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureFunction()).Times(::testing::Exactly(1));
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureContext()).Times(::testing::Exactly(1));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);

    // ACT
    PetscReal computedMu = NAN;
    sutherlandModel->GetComputeViscosityFunction()(GetParam().temperatureIn, GetParam().densityIn, NULL, computedMu, sutherlandModel->GetComputeViscosityContext());

    // ASSERT
    double error = (GetParam().expectedViscosity - computedMu) / GetParam().expectedViscosity;
    ASSERT_LT(error, 1E-5);
}

TEST_P(SutherlandTransportTestFixture, ShouldComputeCorrectDiffusivity) {
    // ARRANGE
    std::shared_ptr<ablateTesting::eos::MockEOS> eos = std::make_shared<ablateTesting::eos::MockEOS>();
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureFunction()).Times(::testing::Exactly(1));
    EXPECT_CALL(*eos, GetComputeSpecificHeatConstantPressureContext()).Times(::testing::Exactly(1));

    auto sutherlandModel = std::make_shared<ablate::eos::transport::Sutherland>(eos);

    // ACT
    PetscReal computedDiff = NAN;
    sutherlandModel->GetComputeDiffusivityFunction()(GetParam().temperatureIn, GetParam().densityIn, NULL, computedDiff, sutherlandModel->GetComputeDiffusivityContext());

    // ASSERT
    double error = (GetParam().expectedDiffusivity - computedDiff) / GetParam().expectedDiffusivity;
    ASSERT_LT(error, 1E-5);
}

INSTANTIATE_TEST_SUITE_P(SutherlandTests, SutherlandTransportTestFixture,
                         testing::Values((SutherlandTransportTestParameters){
                             .temperatureIn = 300.0, .densityIn = 1.1, .cpIn = 1001.1, .expectedConductivity = 0.02615186, .expectedViscosity = 1.8469051E-5, .expectedDiffusivity = 2.374829E-5}));
