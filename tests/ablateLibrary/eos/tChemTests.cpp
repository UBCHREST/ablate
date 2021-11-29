#include "PetscTestFixture.hpp"
#include "eos/tChem.hpp"
#include "gtest/gtest.h"

/*
 * Helper function to fill mass fraction
 */
static std::vector<PetscReal> GetMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn) {
    std::vector<PetscReal> yi(species.size(), 0.0);

    for (const auto& value : yiIn) {
        // Get the index
        auto it = std::find(species.begin(), species.end(), value.first);
        if (it != species.end()) {
            auto index = std::distance(species.begin(), it);

            yi[index] = value.second;
        }
    }
    return yi;
}

static std::vector<PetscReal> GetDensityMassFraction(const std::vector<std::string>& species, const std::map<std::string, PetscReal>& yiIn, double density) {
    auto densityYis = GetMassFraction(species, yiIn);
    for (auto& densityYi : densityYis) {
        densityYi *= density;
    }

    return densityYis;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS create and view tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemCreateAndViewParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::string expectedView;
};

class TChemCreateAndViewFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemCreateAndViewParameters> {};

TEST_P(TChemCreateAndViewFixture, ShouldCreateAndView) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    std::stringstream outputStream;

    // act
    outputStream << *eos;

    // assert the output is as expected
    auto outputString = outputStream.str();
    ASSERT_EQ(outputString, GetParam().expectedView);
}

INSTANTIATE_TEST_SUITE_P(TChemTests, TChemCreateAndViewFixture,
                         testing::Values((TChemCreateAndViewParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                        .thermoFile = "inputs/eos/thermo30.dat",
                                                                        .expectedView = "EOS: TChemV1\n\tmechFile: \"inputs/eos/grimech30.dat\"\n\tthermoFile: \"inputs/eos/thermo30.dat\"\n"}),
                         [](const testing::TestParamInfo<TChemCreateAndViewParameters>& info) { return info.param.mechFile.stem().string() + "_" + info.param.thermoFile.stem().string(); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// EOS Get Species Tests
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemGetSpeciesParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::vector<std::string> expectedSpecies;
};

class TChemGetSpeciesFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemGetSpeciesParameters> {};

TEST_P(TChemGetSpeciesFixture, ShouldGetCorrectSpecies) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // act
    auto species = eos->GetSpecies();

    // assert the output is as expected
    ASSERT_EQ(species, GetParam().expectedSpecies);
}

INSTANTIATE_TEST_SUITE_P(TChemTests, TChemGetSpeciesFixture,
                         testing::Values((TChemGetSpeciesParameters){
                             .mechFile = "inputs/eos/grimech30.dat",
                             .thermoFile = "inputs/eos/thermo30.dat",
                             .expectedSpecies = {"H2",    "H",    "O",     "O2",  "OH",   "H2O",  "HO2",  "H2O2", "C",    "CH",   "CH2",   "CH2(S)", "CH3", "CH4",  "CO",   "CO2",    "HCO",   "CH2O",
                                                 "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH",  "N",   "NH",   "NH2",  "NH3",    "NNH",   "NO",
                                                 "NO2",   "N2O",  "HNO",   "CN",  "HCN",  "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO",   "N2",     "AR",  "C3H7", "C3H8", "CH2CHO", "CH3CHO"}}),
                         [](const testing::TestParamInfo<TChemGetSpeciesParameters>& info) { return info.param.mechFile.stem().string() + "_" + info.param.thermoFile.stem().string(); });

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TChem get species enthalpy
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemSensibleSpeciesEnthalpyParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    double temperature;
    std::vector<PetscReal> expectedSensibleSpeciesEnthalpy;
};

class TChemSensibleSpeciesEnthalpyFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemSensibleSpeciesEnthalpyParameters> {};

TEST_P(TChemSensibleSpeciesEnthalpyFixture, ShouldComputeCorrectSpeciesEnthalpy) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    std::vector<PetscReal> computedSensibleSpeciesEnthalpy(GetParam().expectedSensibleSpeciesEnthalpy.size(), NAN);

    // act
    PetscErrorCode ierr = eos->GetComputeSpeciesSensibleEnthalpyFunction()(GetParam().temperature, &computedSensibleSpeciesEnthalpy[0], eos->GetComputeSpeciesSensibleEnthalpyContext());

    // assert the output is as expected
    ASSERT_EQ(ierr, 0);
    for (std::size_t s = 0; s < eos->GetSpecies().size(); s++) {
        const double error = (GetParam().expectedSensibleSpeciesEnthalpy[s] - computedSensibleSpeciesEnthalpy[s]) / (GetParam().expectedSensibleSpeciesEnthalpy[s] + 1E-30);
        ASSERT_LT(error, 1E-3) << "The percent difference for expectedSensibleSpeciesEnthalpy of " << eos->GetSpecies()[s] << " is greater than expected";
    }
}

INSTANTIATE_TEST_SUITE_P(
    TChemTests, TChemSensibleSpeciesEnthalpyFixture,
    testing::Values((TChemSensibleSpeciesEnthalpyParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                             .thermoFile = "inputs/eos/thermo30.dat",
                                                             .temperature = 298.15,
                                                             .expectedSensibleSpeciesEnthalpy = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                    (TChemSensibleSpeciesEnthalpyParameters){
                        .mechFile = "inputs/eos/grimech30.dat",
                        .thermoFile = "inputs/eos/thermo30.dat",
                        .temperature = 350.0,
                        .expectedSensibleSpeciesEnthalpy = {7.460590e+05, 1.069273e+06, 7.055272e+04, 4.785883e+04, 9.080069e+04, 9.707818e+04, 5.566472e+04, 6.614152e+04, 8.991894e+04,
                                                            1.162188e+05, 1.308620e+05, 1.257292e+05, 1.353844e+05, 1.188149e+05, 5.401828e+04, 4.507353e+04, 6.258466e+04, 6.258237e+04,
                                                            8.324871e+04, 6.636699e+04, 7.404635e+04, 8.828951e+04, 9.116586e+04, 8.561139e+04, 8.372083e+04, 9.475268e+04, 9.592900e+04,
                                                            6.316233e+04, 6.639892e+04, 7.465792e+04, 7.694628e+04, 1.007995e+05, 1.101965e+05, 1.106852e+05, 6.281819e+04, 5.155943e+04,
                                                            4.282066e+04, 4.682740e+04, 5.720448e+04, 5.820967e+04, 7.055767e+04, 7.307891e+04, 6.404595e+04, 5.831064e+04, 5.708868e+04,
                                                            5.750638e+04, 5.076174e+04, 5.392193e+04, 2.697916e+04, 9.164036e+04, 9.281967e+04, 6.885935e+04, 6.825274e+04}},
                    (TChemSensibleSpeciesEnthalpyParameters){
                        .mechFile = "inputs/eos/grimech30.dat",
                        .thermoFile = "inputs/eos/thermo30.dat",
                        .temperature = 3000.0,
                        .expectedSensibleSpeciesEnthalpy = {4.401447e+07, 5.571873e+07, 3.536189e+06, 3.066045e+06, 5.280429e+06, 7.086383e+06, 4.236784e+06, 5.394231e+06, 4.719917e+06,
                                                            7.461729e+06, 9.382657e+06, 9.338441e+06, 1.186369e+07, 1.461963e+07, 3.338867e+06, 3.472244e+06, 4.745156e+06, 6.099301e+06,
                                                            7.376036e+06, 7.489415e+06, 8.456529e+06, 6.201523e+06, 7.700047e+06, 8.726732e+06, 1.006060e+07, 1.118903e+07, 1.239586e+07,
                                                            4.817136e+06, 5.843981e+06, 6.082402e+06, 4.013085e+06, 6.105222e+06, 8.300475e+06, 1.027588e+07, 4.726630e+06, 3.168311e+06,
                                                            3.227706e+06, 3.530461e+06, 4.750059e+06, 3.741030e+06, 5.402331e+06, 6.678596e+06, 4.877817e+06, 4.658460e+06, 4.383264e+06,
                                                            4.569753e+06, 3.683059e+06, 3.310248e+06, 1.405856e+06, 1.109638e+07, 1.193612e+07, 6.659771e+06, 7.583124e+06}}),
    [](const testing::TestParamInfo<TChemSensibleSpeciesEnthalpyParameters>& info) {
        return info.param.mechFile.stem().string() + "_" + info.param.thermoFile.stem().string() + "_Temp_" + std::to_string((int)info.param.temperature);
    });

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///// EOS decode state tests
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct TChemStateParameters {
    std::filesystem::path mechFile;
    std::filesystem::path thermoFile;
    std::map<std::string, PetscReal> yi;
    PetscReal density;
    PetscReal totalEnergy;
    std::vector<PetscReal> massFlux;
    PetscReal temperature;
    PetscReal internalEnergy;
    PetscReal sensibleEnthalpy;
    PetscReal speedOfSound;
    PetscReal pressure;
    PetscReal specificHeatCp;
    PetscReal specificHeatCv;
};

class TChemStateTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<TChemStateParameters> {};

TEST_P(TChemStateTestFixture, ShouldDecodeState) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // Prepare outputs
    PetscReal internalEnergy;
    PetscReal speedOfSound;
    PetscReal pressure;

    // get the mass fraction as an array
    auto densityYi = GetDensityMassFraction(eos->GetSpecies(), params.yi, params.density);

    // convert the massFrac in to velocity
    std::vector<double> velocityIn;
    for (const auto& rhoV : params.massFlux) {
        velocityIn.push_back(rhoV / params.density);
    }

    // act
    PetscErrorCode ierr = eos->GetDecodeStateFunction()(
        (PetscInt)velocityIn.size(), params.density, params.totalEnergy, &velocityIn[0], &densityYi[0], &internalEnergy, &speedOfSound, &pressure, eos->GetDecodeStateContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(internalEnergy, params.internalEnergy, .1);
    ASSERT_NEAR(speedOfSound, params.speedOfSound, .1);
    ASSERT_LT(PetscAbs(pressure - params.pressure) / params.pressure, 1E-5) << "The percent difference in pressure should be less than 1E-5";
}

TEST_P(TChemStateTestFixture, ShouldComputeTemperature) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto densityYi = GetDensityMassFraction(eos->GetSpecies(), params.yi, params.density);

    // Prepare outputs
    PetscReal temperature;

    // act
    PetscErrorCode ierr = eos->GetComputeTemperatureFunction()(
        (PetscInt)params.massFlux.size(), params.density, params.totalEnergy, &params.massFlux[0], &densityYi[0], &temperature, eos->GetComputeTemperatureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(temperature, params.temperature, 1E-2);
}

TEST_P(TChemStateTestFixture, ShouldComputeDensityFromTemperatureAndPressure) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yi);

    // Prepare outputs
    PetscReal density;

    // act
    PetscErrorCode ierr =
        eos->GetComputeDensityFunctionFromTemperaturePressureFunction()(params.temperature, params.pressure, &yi[0], &density, eos->GetComputeDensityFunctionFromTemperaturePressureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(density, params.density, 1E-2);
}

TEST_P(TChemStateTestFixture, ShouldComputeSensibleInternalEnergy) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yi);

    // Prepare outputs
    PetscReal sensibleInternalEnergy;

    // act
    PetscErrorCode ierr = eos->GetComputeSensibleInternalEnergyFunction()(params.temperature, params.density, &yi[0], &sensibleInternalEnergy, eos->GetComputeSensibleInternalEnergyContext());

    // assert
    ASSERT_EQ(ierr, 0);
    const double error = (sensibleInternalEnergy - params.internalEnergy) / params.internalEnergy;
    ASSERT_LT(error, 1E-3);
}

TEST_P(TChemStateTestFixture, ShouldComputeSensibleEnthalpy) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yi);

    // Prepare outputs
    PetscReal sensibleEnthalpy;

    // act
    PetscErrorCode ierr = eos->GetComputeSensibleEnthalpyFunction()(params.temperature, params.density, &yi[0], &sensibleEnthalpy, eos->GetComputeSensibleInternalEnergyContext());

    // assert
    ASSERT_EQ(ierr, 0);
    const double error = (sensibleEnthalpy - params.sensibleEnthalpy) / params.sensibleEnthalpy;
    ASSERT_LT(error, 1E-3);
}

TEST_P(TChemStateTestFixture, ShouldComputeSpecificHeatConstantPressure) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yi);

    // Prepare outputs
    PetscReal cp;

    // act
    PetscErrorCode ierr = eos->GetComputeSpecificHeatConstantPressureFunction()(params.temperature, params.density, &yi[0], &cp, eos->GetComputeSpecificHeatConstantPressureContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(params.specificHeatCp, cp, 1.0);
}

TEST_P(TChemStateTestFixture, ShouldComputeSpecificHeatConstantVolume) {
    // arrange
    std::shared_ptr<ablate::eos::EOS> eos = std::make_shared<ablate::eos::TChem>(GetParam().mechFile, GetParam().thermoFile);

    // get the test params
    const auto& params = GetParam();

    // get the mass fraction as an array
    auto yi = GetMassFraction(eos->GetSpecies(), params.yi);

    // Prepare outputs
    PetscReal cv;

    // act
    PetscErrorCode ierr = eos->GetComputeSpecificHeatConstantVolumeFunction()(params.temperature, params.density, &yi[0], &cv, eos->GetComputeSpecificHeatConstantVolumeContext());

    // assert
    ASSERT_EQ(ierr, 0);
    ASSERT_NEAR(params.specificHeatCv, cv, 1.0);
}

INSTANTIATE_TEST_SUITE_P(TChemTests, TChemStateTestFixture,
                         testing::Values((TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yi = {{"CH4", .2}, {"O2", .3}, {"N2", .5}},
                                                                .density = 1.2,
                                                                .totalEnergy = 1.E+05,
                                                                .massFlux = {1.2 * 10, -1.2 * 20, 1.2 * 30},
                                                                .temperature = 499.25,
                                                                .internalEnergy = 99300.0,
                                                                .sensibleEnthalpy = 264064.7,
                                                                .speedOfSound = 464.33,
                                                                .pressure = 197710.5,
                                                                .specificHeatCp = 1399.301411,
                                                                .specificHeatCv = 1069.297887},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yi = {{"O2", .3}, {"N2", .4}, {"CH2", .1}, {"NO", .2}},
                                                                .density = 0.8,
                                                                .totalEnergy = 3.2E5,
                                                                .massFlux = {0, 0, 0},
                                                                .temperature = 762.664,
                                                                .internalEnergy = 320000.0,
                                                                .sensibleEnthalpy = 557466.2,
                                                                .speedOfSound = 560.83,
                                                                .pressure = 189973.54,
                                                                .specificHeatCp = 1270.738292,
                                                                .specificHeatCv = 959.3732847},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yi = {{"N2", 1.0}},
                                                                .density = 3.3,
                                                                .totalEnergy = 1000,
                                                                .massFlux = {0.0, 3.3 * 2, 3.3 * 4},
                                                                .temperature = 418.079,
                                                                .internalEnergy = 990.0,
                                                                .sensibleEnthalpy = 125084.2,
                                                                .speedOfSound = 416.04,
                                                                .pressure = 409488.10,
                                                                .specificHeatCp = 1048.36597,
                                                                .specificHeatCv = 751.569283},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yi = {{"H2", .35}, {"H2O", .35}, {"N2", .3}},
                                                                .density = 0.01,
                                                                .totalEnergy = 1E5,
                                                                .massFlux = {.01 * -1, .01 * -2, .01 * -3},
                                                                .temperature = 437.46,
                                                                .internalEnergy = 99993.0,
                                                                .sensibleEnthalpy = 841150.64,
                                                                .speedOfSound = 1013.73,
                                                                .pressure = 7411.11,
                                                                .specificHeatCp = 6075.990042,
                                                                .specificHeatCv = 4381.932446},
                                         (TChemStateParameters){.mechFile = "inputs/eos/grimech30.dat",
                                                                .thermoFile = "inputs/eos/thermo30.dat",
                                                                .yi = {{"H2", .1}, {"H2O", .2}, {"N2", .3}, {"CO", .4}},
                                                                .density = 999.9,
                                                                .totalEnergy = 1E4,
                                                                .massFlux = {999.9 * -10, 999.9 * -20, 999.9 * -300},
                                                                .temperature = 394.59,
                                                                .internalEnergy = -35250.0,
                                                                .sensibleEnthalpy = 245915.9,
                                                                .speedOfSound = 623.9,
                                                                .pressure = 281125963.5,
                                                                .specificHeatCp = 2564.816937,
                                                                .specificHeatCv = 1852.31255}),
                         [](const testing::TestParamInfo<TChemStateParameters>& info) { return std::to_string(info.index); });