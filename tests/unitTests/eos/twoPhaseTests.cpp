#include "PetscTestFixture.hpp"
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "eos/twoPhase.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TwoPhaseEOSTestCreateAndViewParameters {
    std::shared_ptr<ablate::eos::EOS> eos1;
    std::shared_ptr<ablate::eos::EOS> eos2;
    std::string expectedView;
};
class TwoPhaseTestCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TwoPhaseEOSTestCreateAndViewParameters> {};

TEST_P(TwoPhaseTestCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    auto eos1 = GetParam().eos1;
    auto eos2 = GetParam().eos2;
    std::shared_ptr<ablate::eos::EOS> twoPhaseEos = std::make_shared<ablate::eos::TwoPhase>(eos1, eos2);

    std::stringstream outputStream;

    // act
    outputStream << *twoPhaseEos;

    // assert the output is as expected
    auto outputString = outputStream.str();
    ASSERT_EQ(outputString, GetParam().expectedView);
}
INSTANTIATE_TEST_SUITE_P(
    TwoPhaseEOSTests, TwoPhaseTestCreateAndViewFixture,
    testing::Values((TwoPhaseEOSTestCreateAndViewParameters){.eos1 = {}, .eos2 = {}, .expectedView = "EOS: twoPhase\n"},
                    (TwoPhaseEOSTestCreateAndViewParameters){
                        .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
                        .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                        .expectedView = "EOS: twoPhase\nEOS: perfectGas\n\tgamma: 1.4\n\tRgas: 287\nEOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\n"},
                    (TwoPhaseEOSTestCreateAndViewParameters){
                        .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                            {"gamma", "2.1"}, {"Cp", "204.7"}, {"p0", "3.5e6"}})),
                        .expectedView = "EOS: twoPhase\nEOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\nEOS: stiffenedGas\n\tgamma: 2.1\n\tCp: 204.7\n\tp0: 3.5e+06\n"},
                    (TwoPhaseEOSTestCreateAndViewParameters){
                        .eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                            {"gamma", "3.2"}, {"Cp", "100.2"}, {"p0", "9.9e5"}})),
                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                            {"gamma", "2.1"}, {"Cp", "204.7"}, {"p0", "3.5e6"}})),
                        .expectedView = "EOS: twoPhase\nEOS: stiffenedGas\n\tgamma: 3.2\n\tCp: 100.2\n\tp0: 990000\nEOS: stiffenedGas\n\tgamma: 2.1\n\tCp: 204.7\n\tp0: 3.5e+06\n"},
                    (TwoPhaseEOSTestCreateAndViewParameters){
                        .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}),
                                                                          std::vector<std::string>{"O2", "N2"}),
                        .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}}),
                                                                          std::vector<std::string>{"CO2", "CO"}),
                        .expectedView = "EOS: twoPhase\nEOS: perfectGas\n\tgamma: 1.4\n\tRgas: 287\n\tspecies: O2, N2\nEOS: perfectGas\n\tgamma: 3.2\n\tRgas: 100.2\n\tspecies: CO2, CO\n"}),
    [](const testing::TestParamInfo<TwoPhaseEOSTestCreateAndViewParameters>& info) { return std::to_string(info.index); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS Thermodynamic property tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TPTestParameters {
    std::shared_ptr<ablate::eos::EOS> eos1;
    std::shared_ptr<ablate::eos::EOS> eos2;
    ablate::eos::ThermodynamicProperty thermodynamicProperty;
    std::vector<ablate::domain::Field> fields;
    std::vector<PetscReal> conservedValues;
    std::optional<PetscReal> expectedTemperature;
    std::vector<PetscReal> expectedValue;
};

class TPThermodynamicPropertyTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TPTestParameters> {};

TEST_P(TPThermodynamicPropertyTestFixture, ShouldComputeProperty) {
    // arrange
    auto eos1 = GetParam().eos1;
    auto eos2 = GetParam().eos2;
    std::shared_ptr<ablate::eos::EOS> twoPhaseEos = std::make_shared<ablate::eos::TwoPhase>(eos1, eos2);

    // get the test params
    const auto& params = GetParam();

    // act/assert check for compute without temperature
    auto thermodynamicFunction = twoPhaseEos->GetThermodynamicFunction(params.thermodynamicProperty, params.fields);
    std::vector<PetscReal> computedProperty(params.expectedValue.size(), NAN);
    PetscErrorCode ierr = thermodynamicFunction.function(params.conservedValues.data(), computedProperty.data(), thermodynamicFunction.context.get());
    ASSERT_EQ(ierr, 0);
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << "for direct function ";
    }
    // act/assert check for compute when temperature is known
    auto temperatureFunction = twoPhaseEos->GetThermodynamicFunction(ablate::eos::ThermodynamicProperty::Temperature, params.fields);
    PetscReal computedTemperature;
    ierr = temperatureFunction.function(params.conservedValues.data(), &computedTemperature, temperatureFunction.context.get());
    ASSERT_EQ(ierr, 0);

    if (params.expectedTemperature) {
        ASSERT_NEAR(computedTemperature, params.expectedTemperature.value(), 1E-6) << "for computed temperature ";
    }

    auto thermodynamicTemperatureFunction = twoPhaseEos->GetThermodynamicTemperatureFunction(params.thermodynamicProperty, params.fields);
    computedProperty = std::vector<PetscReal>(params.expectedValue.size(), NAN);
    ierr = thermodynamicTemperatureFunction.function(params.conservedValues.data(), computedTemperature, computedProperty.data(), thermodynamicTemperatureFunction.context.get());

    ASSERT_EQ(ierr, 0);
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(computedProperty[c], params.expectedValue[c], 1E-6) << " for temperature function ";
    }
}
INSTANTIATE_TEST_SUITE_P(
    StiffenedGasEOSTests, TPThermodynamicPropertyTestFixture,
    testing::Values(
        (TPTestParameters){
            // all first gas
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {1.1614401858304297,
                                1.1614401858304297 * (215250.0 + 700.0),
                                1.1614401858304297 * 10,
                                1.1614401858304297 * -20,
                                1.1614401858304297 * 30,
                                1.1614401858304297,
                                1.0},  // rho, rhoE, rhoU, rhoV, rhoW, rhoAlpha, alpha
            .expectedValue = {300.0}},
        (TPTestParameters){
            // all second gas
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues =
                {3.132832080200501, 3.132832080200501 * (74232.5581395349 + 700.0), 3.132832080200501 * 10, 3.132832080200501 * -20, 3.132832080200501 * 30, 0.0, 0.0},  // rho, rhoE, rhoU, rhoV, rhoW,
                                                                                                                                                                         // rhoAlpha, alpha
            .expectedValue = {300.0}},
        (TPTestParameters){
            // mix
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {2.3442753224524724,
                                2.3442753224524724 * (102178.6483126649 + 700.0),
                                2.3442753224524724 * 10,
                                2.3442753224524724 * -20,
                                2.3442753224524724 * 30,
                                0.4645760743321719,
                                0.4},  // rho, rhoE, rhoU, rhoV, rhoW, rhoAlpha, alpha
            .expectedValue = {102178.6483126649}},

        (TPTestParameters){
            // all air
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {1.1614401858304297, 1.1614401858304297 * (215250.0 + 700), 1.1614401858304297 * 10, 1.1614401858304297 * -20, 1.1614401858304297 * 30, 1.1614401858304297, 1.0},
            .expectedValue = {347.18870949384285}},
        (TPTestParameters){
            // all water
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {994.0897497618486, 994.0897497618486 * (2428423.405461103 + 700), 994.0897497618486 * 10, 994.0897497618486 * -20, 994.0897497618486 * 30, 0.0, 0.0},
            .expectedValue = {1504.4548407978223}},
        (TPTestParameters){
            // mix
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {497.6255949738395, 497.6255949738395 * (2425840.672019258 + 700), 497.6255949738395 * 10, 497.6255949738395 * -20, 497.6255949738395 * 30, 0.5807200929152149, 0.5},
            .expectedValue = {20.04845783548275}},

        (TPTestParameters){
            // all first liquid
            .eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {768.3978307143822, 768.3978307143822 * (1392890.3090808005 + 700), 768.3978307143822 * 10, 768.3978307143822 * -20, 768.3978307143822 * 30, 768.3978307143822, 1.0},
            .expectedValue = {100000.00000023842}},
        (TPTestParameters){// all second liquid
                           .eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                               {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                           .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                               {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                           .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
                           .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                                      ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                                      ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
                           .conservedValues = {994.0897497618486, 994.0897497618486 * (2428423.405461103 + 700), 994.0897497618486 * 10, 994.0897497618486 * -20, 994.0897497618486 * 30, 0.0, 0.0},
                           .expectedValue = {100000.0}},
        (TPTestParameters){
            // mix
            .eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {836.105406428622, 836.105406428622 * (1762250.2589397137 + 700), 836.105406428622 * 10, 836.105406428622 * -20, 836.105406428622 * 30, 537.8784815000674, 0.7},
            .expectedValue = {100000.00000023842}},

        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 538.8183311241276, 538.8183311241276 * 1390590.2017535304, 0.0, 0.9398496240601503, 0.3},
            .expectedValue = {1390590.2017535304}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeedOfSound,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 538.8183311241276, 538.8183311241276 * 1390590.2017535304, 0.0, 0.9398496240601503, 0.3},
            .expectedValue = {24.87318022730244}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Density,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 538.8183311241276, 538.8183311241276 * 1390590.2017535304, 0.0, 0.9398496240601503, 0.3},
            .expectedValue = {538.8183311241276}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Pressure,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 636.2369625573178, 636.2369625573178 * 806491.803515885, 0.0, 469.92481203007515, 0.6},
            .expectedValue = {50000000.00000003}},

        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {462.2918312607095, 462.2918312607095 * (1389315.828178823 + 700), 462.2918312607095 * 10, 462.2918312607095 * 20, 462.2918312607095 * 30, 1.2531328320802004, 0.4},
            .expectedValue = {300}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::Temperature,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 562.751433810914, 562.751433810914 * 1264395.3714915595, 0.0, 313.2832080200501, 0.4},
            .expectedValue = {600}},

        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {462.2918312607095, 462.2918312607095 * (1389315.828178823 + 700), 462.2918312607095 * 10, 462.2918312607095 * 20, 462.2918312607095 * 30, 1.2531328320802004, 0.4},
            .expectedTemperature = 300.0,
            .expectedValue = {1389315.828178823}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 3, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {NAN, NAN, 562.751433810914, 562.751433810914 * 1264395.3714915595, 0.0, 313.2832080200501, 0.4},
            .expectedTemperature = 600.0,
            .expectedValue = {1264395.3714915595}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {462.2918312607095, 462.2918312607095 * (1389315.828178823 + 700), 462.2918312607095 * 10, 462.2918312607095 * 20, 462.2918312607095 * 30, 1.2531328320802004, 0.4},
            .expectedTemperature = 300.0,
            .expectedValue = {1389532.1417566063}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SensibleEnthalpy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 2},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 7},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 8}},
            .conservedValues = {NAN, NAN, 562.751433810914, 562.751433810914 * (1264395.3714915595 + 700), 562.751433810914 * 10, 562.751433810914 * 20, 562.751433810914 * 30, 313.2832080200501, 0.4},
            .expectedTemperature = 600.0,
            .expectedValue = {1353244.5453824254}},

        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {462.2918312607095, 462.2918312607095 * (1389315.828178823 + 700), 462.2918312607095 * 10, 462.2918312607095 * 20, 462.2918312607095 * 30, 1.2531328320802004, 0.4},
            .expectedValue = {4631.773805855355}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {462.2918312607095, 462.2918312607095 * (1389315.828178823 + 700), 462.2918312607095 * 10, 462.2918312607095 * 20, 462.2918312607095 * 30, 1.2531328320802004, 0.4},
            .expectedValue = {4629.971638016732}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantPressure,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {562.751433810914, 562.751433810914 * (1264395.3714915595 + 700), 562.751433810914 * 10, 562.751433810914 * 20, 562.751433810914 * 30, 313.2832080200501, 0.4},
            .expectedValue = {2255.407575637376}},
        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpecificHeatConstantVolume,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {562.751433810914, 562.751433810914 * (1264395.3714915595 + 700), 562.751433810914 * 10, 562.751433810914 * 20, 562.751433810914 * 30, 313.2832080200501, 0.4},
            .expectedValue = {1923.136104523659}},

        (TPTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}),
                                                              std::vector<std::string>{"CO2"}),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}}),
                                                              std::vector<std::string>{"O2", "CH4", "N2"}),
            .thermodynamicProperty = ablate::eos::ThermodynamicProperty::SpeciesSensibleEnthalpy,
            .fields = {ablate::domain::Field{.name = "euler", .numberComponents = 5, .offset = 0},
                       ablate::domain::Field{.name = "densityvolumeFraction", .numberComponents = 1, .offset = 5},
                       ablate::domain::Field{.name = "volumeFraction", .numberComponents = 1, .offset = 6}},
            .conservedValues = {998.7, 998.7 * 2.5E6, 998.7 * 10, 998.7 * -20, 998.7 * 30, 998.7, 1.0},
            .expectedValue = std::vector<PetscReal>{0.0, 0.0, 0.0, 0.0}}

        ),

    [](const testing::TestParamInfo<TPTestParameters>& info) { return std::to_string(info.index) + "_" + std::string(to_string(info.param.thermodynamicProperty)); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS get species tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(TwoPhaseEOSTests, TwoPhaseShouldReportNoSpeciesEctByDefault) {
    // arrange
    auto eos1 = nullptr;
    auto eos2 = nullptr;
    std::shared_ptr<ablate::eos::EOS> twoPhaseEos = std::make_shared<ablate::eos::TwoPhase>(eos1, eos2);

    // act // assert
    ASSERT_EQ(1, twoPhaseEos->GetFieldFunctionProperties().size());  // returns VF
    ASSERT_EQ(0, twoPhaseEos->GetSpeciesVariables().size());
    ASSERT_EQ(0, twoPhaseEos->GetProgressVariables().size());
}

TEST(TwoPhaseEOSTests, TwoPhaseShouldReportSpeciesWhenProvided) {
    // arrange
    auto eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}),
                                                          std::vector<std::string>{"H2", "O2"});
    auto eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}}),
                                                          std::vector<std::string>{"N2"});
    std::shared_ptr<ablate::eos::EOS> twoPhaseEos = std::make_shared<ablate::eos::TwoPhase>(eos1, eos2);

    // act
    auto species = twoPhaseEos->GetSpeciesVariables();

    // assert
    ASSERT_EQ(3, species.size());
    ASSERT_EQ("H2", species[0]);
    ASSERT_EQ("O2", species[1]);
    ASSERT_EQ("N2", species[2]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Two Phase EOS FieldFunctionTests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TPFieldFunctionTestParameters {
    // eos init
    std::shared_ptr<ablate::eos::EOS> eos1;
    std::shared_ptr<ablate::eos::EOS> eos2;

    // field function init
    std::string field;
    ablate::eos::ThermodynamicProperty property1;
    ablate::eos::ThermodynamicProperty property2;

    // inputs
    PetscReal property1Value;
    PetscReal property2Value;
    std::vector<PetscReal> velocity;
    std::vector<PetscReal> otherProperties;
    std::vector<PetscReal> expectedValue;
};

class TPFieldFunctionTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TPFieldFunctionTestParameters> {};

TEST_P(TPFieldFunctionTestFixture, ShouldComputeField) {
    // arrange
    auto eos1 = GetParam().eos1;
    auto eos2 = GetParam().eos2;
    std::shared_ptr<ablate::eos::EOS> twoPhaseEos = std::make_shared<ablate::eos::TwoPhase>(eos1, eos2);

    // get the test params
    const auto& params = GetParam();
    std::vector<PetscReal> actualValue(params.expectedValue.size(), NAN);

    // act
    auto stateFunction = twoPhaseEos->GetFieldFunctionFunction(params.field, params.property1, params.property2, {ablate::eos::TwoPhase::VF, ablate::eos::EOS::YI});
    stateFunction(params.property1Value, params.property2Value, params.velocity.size(), params.velocity.data(), params.otherProperties.data(), actualValue.data());

    // assert
    for (std::size_t c = 0; c < params.expectedValue.size(); c++) {
        ASSERT_NEAR(actualValue[c], params.expectedValue[c], 1E-3) << "for component[" << c << "] ";
    }
}

INSTANTIATE_TEST_SUITE_P(
    TwoPhaseEOSTests, TPFieldFunctionTestFixture,
    testing::Values(
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
            .field = "euler",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 300.0,
            .property2Value = 101325.0,
            .velocity = {10.0, 20, 30},
            .otherProperties = {1.0},  // alpha
            .expectedValue = {1.1768292682, 1.1768292682 * (2.1525E+05 + 700), 1.1768292682 * 10, 1.1768292682 * 20, 1.1768292682 * 30}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}})),
            .field = "densityvolumeFraction",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 300.0,
            .property2Value = 101325.0,
            .velocity = {10.0, 20, 30},
            .otherProperties = {1.0},  // alpha
            .expectedValue = {1.1768292682}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}),
                                                              std::vector<std::string>{"CO2"}),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}}),
                                                              std::vector<std::string>{"O2", "CH4", "N2"}),
            .field = "densityYi",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 300.0,
            .property2Value = 101325.0,
            .velocity = {10.0, 20, 30},
            .otherProperties = {1.0, 1.0, .1, .3, .6},  // alpha, (CO2), (O2, CH4, N2)
            .expectedValue = {1.1768292682926829, 0.1 * 3.370758483033932, 0.3 * 3.370758483033932, 0.6 * 3.370758483033932}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .field = "euler",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 600.0,
            .property2Value = 50000000.0,
            .velocity = {0.0},
            .otherProperties = {0.4},
            .expectedValue = {562.751433810914, 562.751433810914 * 1264395.3714915595, 0.0}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .field = "densityvolumeFraction",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 600.0,
            .property2Value = 50000000.0,
            .velocity = {0.0},
            .otherProperties = {0.4},
            .expectedValue = {313.2832080200501}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}}),
                                                              std::vector<std::string>{"CO2"}),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                    {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}}),
                                                                std::vector<std::string>{"O2", "N2", "H2"}),
            .field = "densityYi",
            .property1 = ablate::eos::ThermodynamicProperty::Temperature,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 600.0,
            .property2Value = 50000000.0,
            .velocity = {0.0},
            .otherProperties = {0.4, 1.0, .1, .3, .6},  // alpha, (CO2), (O2, N2, H2)
            .expectedValue = {783.2080200501252, 415.78037631810656 * .1, 415.78037631810656 * .3, 415.78037631810656 * .6}},

        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 100000.0,
                                        .property2Value = 300.0,
                                        .velocity = {10.0, 20, 30},
                                        .otherProperties = {0.4},
                                        .expectedValue = {903.812982142862, 903.812982142862 * (2076270.2938715648 + 700), 903.812982142862 * 10, 903.812982142862 * 20, 903.812982142862 * 30}},
        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                        .field = "densityvolumeFraction",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 100000.0,
                                        .property2Value = 300.0,
                                        .velocity = {0.0},
                                        .otherProperties = {0.4},
                                        .expectedValue = {307.3591322857529}},
        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                                                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}}),
                                                                                            std::vector<std::string>{"H2", "O2"}),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                                                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}}),
                                                                                            std::vector<std::string>{"N2"}),
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property2 = ablate::eos::ThermodynamicProperty::Temperature,
                                        .property1Value = 100000.0,
                                        .property2Value = 300.0,
                                        .velocity = {0.0},
                                        .otherProperties = {0.4, 0.3, .7, 1.0},
                                        .expectedValue = {768.3978307143822 * .3, 768.3978307143822 * .7, 994.0897497618487}},

        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .field = "euler",
            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property1Value = 100000.0,
            .property2Value = 1384874.8862741701,
            .velocity = {1000.0},
            .otherProperties = {0.6},
            .expectedValue = {309.23883153387317, 309.23883153387317 * (1384874.8862741701 + 500000.0), 309.23883153387317 * 1000.0}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .field = "euler",
            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 124627.50601443468,
            .property2Value = 100000.0,
            .velocity = {0.0, 1000.0},
            .otherProperties = {0.6},
            .expectedValue = {1.949996943578458, 1.949996943578458 * (124627.50601443468 + 500000.0), 0.0, 1.949996943578458 * 1000.0}},
        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                        .field = "euler",
                                        .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 1872426.0208155275,
                                        .property2Value = 100000.0,
                                        .velocity = {0.0, 1000.0},
                                        .otherProperties = {0.6},
                                        .expectedValue = {858.6745983333687, 858.6745983333687 * (1872426.0208155275 + 500000.0), 0.0, 858.6745983333687 * 1000.0}},

        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
            .field = "densityvolumeFraction",
            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property1Value = 100000.0,
            .property2Value = 1384874.8862741701,
            .velocity = {1000.0},
            .otherProperties = {0.6},
            .expectedValue = {1.8796992481203005}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}})),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}})),
            .field = "densityvolumeFraction",
            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 124627.50601443468,
            .property2Value = 100000.0,
            .velocity = {0.0, 1000.0},
            .otherProperties = {0.6},
            .expectedValue = {0.6968641114982578}},
        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}})),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                            {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}})),
                                        .field = "densityvolumeFraction",
                                        .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 1872426.0208155275,
                                        .property2Value = 100000.0,
                                        .velocity = {0.0, 1000.0},
                                        .otherProperties = {0.6},
                                        .expectedValue = {461.0386984286293}},

        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}}),
                                                              std::vector<std::string>{"H2", "O2"}),
            .eos2 = std::make_shared<ablate::eos::StiffenedGas>(
                std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}}), std::vector<std::string>{"N2"}),
            .field = "densityYi",
            .property1 = ablate::eos::ThermodynamicProperty::Pressure,
            .property2 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property1Value = 100000.0,
            .property2Value = 1384874.8862741701,
            .velocity = {1000.0},
            .otherProperties = {0.6, 0.3, 0.7, 1.0},  // alpha, (H2,O2), (N2)
            .expectedValue = {3.132832080200501 * 0.3, 3.132832080200501 * 0.7, 768.3978307143822}},
        (TPFieldFunctionTestParameters){
            .eos1 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}),
                                                              std::vector<std::string>{"H2", "O2"}),
            .eos2 = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.43"}, {"Rgas", "106.4"}}),
                                                              std::vector<std::string>{"N2"}),
            .field = "densityYi",
            .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
            .property2 = ablate::eos::ThermodynamicProperty::Pressure,
            .property1Value = 124627.50601443468,
            .property2Value = 100000.0,
            .velocity = {0.0, 1000.0},
            .otherProperties = {0.6, 0.3, 0.7, 1.0},  // alpha, (H2,O2), (N2)
            .expectedValue = {1.1614401858304297 * 0.3, 1.1614401858304297 * 0.7, 3.1328320802005005}},
        (TPFieldFunctionTestParameters){.eos1 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                                                {"gamma", "2.31015"}, {"Cp", "4643.4015"}, {"p0", "6.0695E8"}}),
                                                                                            std::vector<std::string>{"H2", "O2"}),
                                        .eos2 = std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                                                                                {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645E9"}}),
                                                                                            std::vector<std::string>{"N2"}),
                                        .field = "densityYi",
                                        .property1 = ablate::eos::ThermodynamicProperty::InternalSensibleEnergy,
                                        .property2 = ablate::eos::ThermodynamicProperty::Pressure,
                                        .property1Value = 1872426.0208155275,
                                        .property2Value = 100000.0,
                                        .velocity = {0.0, 1000.0},
                                        .otherProperties = {0.6, 0.3, 0.7, 1.0},  // alpha, (H2,O2), (N2)
                                        .expectedValue = {768.3978307143822 * 0.3, 768.3978307143822 * 0.7, 994.0897497618486}}),

    [](const testing::TestParamInfo<TPFieldFunctionTestParameters>& info) {
        return std::to_string(info.index) + "_" + info.param.field + "_from_" + std::string(to_string(info.param.property1)) + "_" + std::string(to_string(info.param.property2));
    });
