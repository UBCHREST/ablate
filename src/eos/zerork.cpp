#include "zerork.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::eos::zerorkEOS::zerorkEOS(const std::filesystem::path reactionFileIn,const std::filesystem::path thermoFileIn,
                                  const std::shared_ptr<ablate::parameters::Parameters> &options)
    : ChemistryModel("zerorkEOS"),reactionFile(reactionFileIn),thermoFile(thermoFileIn)  {


    // check the extension, only accept modern yaml input files
    if (reactionFileIn.extension() != ".inp" ){
        throw std::invalid_argument("ablate::eos::zerorkEOS takes only chemkin formated files."
            "Make sure either full path is given or file is in directory");
    }
    if (std::find(validThermoFileExtensions.begin(), validThermoFileExtensions.end(), thermoFileIn.extension()) == validThermoFileExtensions.end()) {
        throw std::invalid_argument("ablate::eos::zerorkEOS thermo file missing or not formated as .dat. "
            "Make sure either full path is given or file is in directory");

    }

    //Create object that holds kinetic data
    mech = std::make_shared<zerork::mechanism>(std::filesystem::absolute(reactionFileIn).c_str(), std::filesystem::absolute(thermoFileIn).c_str(), cklogfilename);

    nSpc = mech->getNumSpecies();
    nSpc = mech->getNumSpecies();
    //species;
    for(int i=0; i<nSpc; ++i) {
        const char* species_name_c = mech->getSpeciesName(i);
        species.push_back(std::string(species_name_c));
    }
    // set the chemistry constraints
    constraints.Set(options);

}




void ablate::eos::zerorkEOS::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tnumberSpecies: " << species.size() << std::endl;
}

std::shared_ptr<ablate::eos::zerorkEOS::FunctionContext> ablate::eos::zerorkEOS::BuildFunctionContext(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields,
                                                                                                bool checkDensityYi) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::zerorkEOS requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityYiField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD; });
    if (checkDensityYi) {
        if (densityYiField == fields.end()) {
            throw std::invalid_argument("The ablate::eos::zerorkEOS requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD Field");
        }
    }

    return std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2,
                                                             .eulerOffset = eulerField->offset,
                                                             .densityYiOffset = checkDensityYi ? densityYiField->offset : -1,
                                                             .mech = mech,
                                                             .nSpc = nSpc,
                                                             });

}

ablate::eos::ThermodynamicFunction ablate::eos::zerorkEOS::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicFunction{.function = std::get<0>(thermodynamicFunctions.at(property)),
                                 .context = BuildFunctionContext(property, fields),
                                 .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::zerorkEOS::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureFunction{.function = std::get<1>(thermodynamicFunctions.at(property)),
                                            .context = BuildFunctionContext(property, fields),
                                            .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}

ablate::eos::zerorkEOS::ThermodynamicMassFractionFunction ablate::eos::zerorkEOS::GetThermodynamicMassFractionFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                                 const std::vector<domain::Field> &fields) const {
    return ThermodynamicMassFractionFunction{.function = std::get<0>(thermodynamicMassFractionFunctions.at(property)),
                                             .context = BuildFunctionContext(property, fields, false),
                                             .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}

ablate::eos::zerorkEOS::ThermodynamicTemperatureMassFractionFunction ablate::eos::zerorkEOS::GetThermodynamicTemperatureMassFractionFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                                                       const std::vector<domain::Field> &fields) const {
    return ThermodynamicTemperatureMassFractionFunction{.function = std::get<1>(thermodynamicMassFractionFunctions.at(property)),
                                                        .context = BuildFunctionContext(property, fields, false),
                                                        .propertySize = speciesSizedProperties.count(property) ? (PetscInt)species.size() : 1};
}

PetscErrorCode ablate::eos::zerorkEOS::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::DensityTemperatureFunction(const PetscReal *conserved, PetscReal, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::DensityMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::DensityTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::TemperatureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) { return TemperatureTemperatureFunction(conserved, 300, property, ctx); }
PetscErrorCode ablate::eos::zerorkEOS::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc; //number of species

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    //get kinetic energy
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    //get and normalize Yi from densityYi
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    // compute the internal energy needed to compute temperature from the sensible enthalpy
    double sensibleenergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    double enthalpymix = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);
    sensibleenergy += enthalpymix;

    // set the temperature from zerork
    *temperature = functionContext->mech->getTemperatureFromEY(sensibleenergy, &reactorMassFrac[0], temperatureGuess);

    PetscFunctionReturn(0);

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::TemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *property, void *ctx) {
    return TemperatureTemperatureMassFractionFunction(conserved, yi, 300, property, ctx);
}
PetscErrorCode ablate::eos::zerorkEOS::TemperatureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperatureGuess, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc; //number of species

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    //get kinetic energy
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    //normalize Yi
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    // compute the internal energy needed to compute temperature from the sensible enthalpy
    double sensibleenergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    double enthalpyMixFormation = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);
    sensibleenergy += enthalpyMixFormation;

    // set the temperature from zerork
    *temperature = functionContext->mech->getTemperatureFromEY(sensibleenergy, &reactorMassFrac[0], temperatureGuess);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    *sensibleInternalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnergyTemperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    double energyMix = functionContext->mech->getMassIntEnergyFromTY(temperature, &reactorMassFrac[0]);
    double enthalpyMixFormation = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);

    *sensibleEnergyTemperature = energyMix - enthalpyMixFormation;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::InternalSensibleEnergyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    InternalSensibleEnergyFunction(conserved, sensibleInternalEnergy, ctx);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::InternalSensibleEnergyTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *sensibleEnergyTemperature,
                                                                                          void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    double energyMix = functionContext->mech->getMassIntEnergyFromTY(temperature, &reactorMassFrac[0]);
    double enthalpyMixFormation = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);

    *sensibleEnergyTemperature = energyMix - enthalpyMixFormation;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;

    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(PressureTemperatureFunction(conserved, temperature, pressure, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::PressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    // compute the pressure
    double pressure_temp = functionContext->mech->getPressureFromTVY(temperature,1/density,&reactorMassFrac[0]);
    // copy back the results
    *pressure = pressure_temp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::PressureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;

    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(PressureTemperatureMassFractionFunction(conserved, yi, temperature, pressure, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::PressureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;
    // Compute the internal energy from total ener
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    // compute the pressure
    double pressure_temp = functionContext->mech->getPressureFromTVY(temperature,1/density,&reactorMassFrac[0]);
    // copy back the results
    *pressure = pressure_temp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(SensibleEnthalpyTemperatureFunction(conserved, temperature, sensibleEnthalpy, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    double enthalpyMix = functionContext->mech->getMassEnthalpyFromTY(temperature, &reactorMassFrac[0]);
    double enthalpyMixFormation = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);


    *sensibleEnthalpy = enthalpyMix-enthalpyMixFormation;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SensibleEnthalpyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(SensibleEnthalpyTemperatureMassFractionFunction(conserved, yi, temperature, sensibleEnthalpy, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SensibleEnthalpyTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    double enthalpyMix = functionContext->mech->getMassEnthalpyFromTY(temperature, &reactorMassFrac[0]);
    double enthalpyMixFormation = functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);


    *sensibleEnthalpy = enthalpyMix-enthalpyMixFormation;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(SpeedOfSoundTemperatureFunction(conserved, temperature, speedOfSound, ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    //get specific heats
    double cp = functionContext->mech->getMassCpFromTY(temperature,&reactorMassFrac[0]);
    double cv = functionContext->mech->getMassCvFromTY(temperature,&reactorMassFrac[0]);

    double R = cp-cv;
    double gamma = cp / cv;
    double speedtemp = sqrt(gamma * R * temperature);

    *speedOfSound = speedtemp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpeedOfSoundMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(SpeedOfSoundTemperatureMassFractionFunction(conserved, yi, temperature, speedOfSound, ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpeedOfSoundTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    //get specific heats
    double cp = functionContext->mech->getMassCpFromTY(temperature,&reactorMassFrac[0]);
    double cv = functionContext->mech->getMassCvFromTY(temperature,&reactorMassFrac[0]);

    double R = cp-cv;
    double gamma = cp / cv;
    double speedtemp = sqrt(gamma * R * temperature);

    *speedOfSound = speedtemp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(SpecificHeatConstantPressureTemperatureFunction(conserved, temperature, cp, ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    //get specific heats
    double cp_temp = functionContext->mech->getMassCpFromTY(temperature,&reactorMassFrac[0]);

    *cp = cp_temp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantPressureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(SpecificHeatConstantPressureTemperatureMassFractionFunction(conserved, yi, temperature, cp, ctx));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantPressureTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *cp, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

//    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    //get specific heats
    double cp_temp = functionContext->mech->getMassCpFromTY(temperature,&reactorMassFrac[0]);

    *cp = cp_temp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(SpecificHeatConstantVolumeTemperatureFunction(conserved, temperature, cv, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;
    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    //get specific heats
    double cv_temp = functionContext->mech->getMassCvFromTY(temperature,&reactorMassFrac[0]);

    *cv = cv_temp;

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantVolumeMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(SpecificHeatConstantVolumeTemperatureMassFractionFunction(conserved, yi, temperature, cv, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SpecificHeatConstantVolumeTemperatureMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal temperature, PetscReal *cv, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    //get specific heats
    double cv_temp = functionContext->mech->getMassCvFromTY(temperature,&reactorMassFrac[0]);

    *cv = cv_temp;

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureFunction(conserved, &temperature, ctx));
    PetscCall(SpeciesSensibleEnthalpyTemperatureFunction(conserved, temperature, hi, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromDensityMassFractions(numSpc,density,conserved + functionContext->densityYiOffset,reactorMassFrac);

    std::vector<double> enthalpySpecies(numSpc,0.);
    std::vector<double> enthalpySpeciesFormation(numSpc,0.);
    functionContext->mech->getMassEnthalpyFromTY(temperature, &reactorMassFrac[0],&enthalpySpecies[0]);
    functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0],&enthalpySpeciesFormation[0]);

    //calculate the species sensible enthalpy
    for (int k=0;k<numSpc;k++){
        hi[k]=enthalpySpecies[k]-enthalpySpeciesFormation[k];
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::zerorkEOS::SpeciesSensibleEnthalpyMassFractionFunction(const PetscReal *conserved, const PetscReal *yi, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    PetscCall(TemperatureMassFractionFunction(conserved, yi, &temperature, ctx));
    PetscCall(SpeciesSensibleEnthalpyTemperatureMassFractionFunction(conserved, yi, temperature, hi, ctx));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::zerorkEOS::SpeciesSensibleEnthalpyTemperatureMassFractionFunction(const PetscReal conserved[], const PetscReal *yi, PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    int numSpc = functionContext->nSpc;

    // Fill the working array
    std::vector<double> reactorMassFrac(numSpc,0.);
    FillreactorMassFracVectorFromMassFractions(numSpc,yi,reactorMassFrac);

    std::vector<double> enthalpySpecies(numSpc,0.);
    std::vector<double> enthalpySpeciesFormation(numSpc,0.);
    functionContext->mech->getMassEnthalpyFromTY(temperature, &reactorMassFrac[0],&enthalpySpecies[0]);
    functionContext->mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0],&enthalpySpeciesFormation[0]);

    //calculate the species sensible enthalpy
    for (int k=0;k<numSpc;k++){
        hi[k]=enthalpySpecies[k]-enthalpySpeciesFormation[k];
    }

    PetscFunctionReturn(0);
}

void ablate::eos::zerorkEOS::FillreactorMassFracVectorFromDensityMassFractions(int nSpc, double density, const double *densityYi, std::vector<double>& reactorMassFrac) {

    double yiSum = 0.0;
    for (int s = 0; s < nSpc - 1; s++) {
        reactorMassFrac[s] = PetscMax(0.0, densityYi[s] / density);
        reactorMassFrac[s] = PetscMin(1.0, reactorMassFrac[s]);
        yiSum += reactorMassFrac[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < nSpc - 1; s++) {
            // Limit the bounds
            reactorMassFrac[s] /= yiSum;
        }
        reactorMassFrac[nSpc - 1] = 0.0;
    } else {
        reactorMassFrac[nSpc - 1] = 1.0 - yiSum;
    }
}

void ablate::eos::zerorkEOS::FillreactorMassFracVectorFromMassFractions(int nSpc, const double *Yi, std::vector<double>& reactorMassFrac) {
    double yiSum = 0.0;
    for (int s = 0; s < nSpc - 1; s++) {
        reactorMassFrac[s] = 1.;
        reactorMassFrac[s] = PetscMax(0.0, Yi[s] );
        reactorMassFrac[s] = PetscMin(1.0, reactorMassFrac[s]);
        yiSum += reactorMassFrac[s];
    }
    if (yiSum > 1.0) {
        for (PetscInt s = 0; s < nSpc - 1; s++) {
            // Limit the bounds
            reactorMassFrac[s] /= yiSum;
        }
        reactorMassFrac[nSpc - 1] = 0.0;
    } else {
        reactorMassFrac[nSpc - 1] = 1.0 - yiSum;
    }
}

ablate::eos::EOSFunction ablate::eos::zerorkEOS::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2,
                                                                       std::vector<std::string> otherProperties) const {
    if (otherProperties != std::vector<std::string>{YI}) {
        throw std::invalid_argument("ablate::eos::zerorkEOS expects the other properties to be Yi (Species Mass Fractions)");
    }

    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {

            auto tp = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {

                std::vector<double> reactorMassFrac(nSpc,0.);
                FillreactorMassFracVectorFromMassFractions(nSpc,yi,reactorMassFrac);

                // compute pressure p = rho*R*T
                PetscReal density = mech->getDensityFromTPY(temperature,pressure,&reactorMassFrac[0]);


                double energyMix = mech->getMassIntEnergyFromTY(temperature, &reactorMassFrac[0]);
                double enthalpyMixFormation = mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);

                double sensibleInternalEnergy = energyMix-enthalpyMixFormation;

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }


            };
            if (property1 == ThermodynamicProperty::Temperature) {
                return tp;
            } else {
                return [tp](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    tp(temperature, pressure, dim, velocity, yi, conserved);
                };
            }
        }
        if ((property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy)) {

            auto iep = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {


                std::vector<double> reactorMassFrac(nSpc,0.);
                FillreactorMassFracVectorFromMassFractions(nSpc,yi,reactorMassFrac);

                double enthalpyMixFormation = mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);

                double internalEnergy = sensibleInternalEnergy + enthalpyMixFormation;
                double temperature = mech->getTemperatureFromEY(internalEnergy,&reactorMassFrac[0],300);

                PetscReal density = mech->getDensityFromTPY(temperature,pressure,&reactorMassFrac[0]);

                //calculate kinetic energy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;


                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal sensibleInternalEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(sensibleInternalEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for the eos.");
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {


            auto densityYiFromTP = [=](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {

                std::vector<double> reactorMassFrac(nSpc,0.);
                FillreactorMassFracVectorFromMassFractions(nSpc,yi,reactorMassFrac);

                // compute pressure p = rho*R*T
                PetscReal density = mech->getDensityFromTPY(temperature,pressure,&reactorMassFrac[0]);

                for (int i=0;i<nSpc;i++){
                    conserved[i] = density * yi[i];
                }
            };

            if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure)) {
                return densityYiFromTP;
            } else {
                return [=](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    densityYiFromTP(temperature, pressure, dim, velocity, yi, conserved);
                };
            }

        } else if ((property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) ||
                   (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy)) {


            auto densityYiFromIeP = [=](PetscReal sensibleInternalEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // fill most of the host
                std::vector<double> reactorMassFrac(nSpc,0.);
                FillreactorMassFracVectorFromMassFractions(nSpc,yi,reactorMassFrac);

                double enthalpyMixFormation = mech->getMassEnthalpyFromTY(298.15, &reactorMassFrac[0]);

                double internalEnergy = sensibleInternalEnergy + enthalpyMixFormation;
                double temperature = mech->getTemperatureFromEY(internalEnergy,&reactorMassFrac[0],300);

                PetscReal density = mech->getDensityFromTPY(temperature,pressure,&reactorMassFrac[0]);

                for (int i=0;i<nSpc;i++){
                    conserved[i] = density * yi[i];
                }
            };

            if (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) {
                return densityYiFromIeP;
            } else {
                return [=](PetscReal pressure, PetscReal sensibleInternalEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    densityYiFromIeP(sensibleInternalEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::TChem.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::zerorkEOS.");
    }
}

std::map<std::string, double> ablate::eos::zerorkEOS::GetElementInformation() const {
    return mech->getElementInfo();
}

std::map<std::string, std::map<std::string, int>> ablate::eos::zerorkEOS::GetSpeciesElementalInformation() const {
    return mech->getSpeciesElementInfo();
}

std::map<std::string, double> ablate::eos::zerorkEOS::GetSpeciesMolecularMass() const {
    // march over each species
//    auto sMass = kineticsModel.sMass_.view_host();
    std::vector<double> sMass(nSpc);
    mech->getMolWtSpc(&sMass[0]);

    std::map<std::string, double> mw;
    for (std::size_t sp = 0; sp < species.size(); ++sp) {
        // TODO test this
        mw[species[sp]] = sMass[(int)sp];
    }
    return mw;
}

std::shared_ptr<ablate::eos::ChemistryModel::SourceCalculator> ablate::eos::zerorkEOS::CreateSourceCalculator(const std::vector<domain::Field> &fields, const ablate::domain::Range &cellRange) {
    return std::make_shared<ablate::eos::zerorkeos::SourceCalculator>(fields, shared_from_this(), constraints, cellRange);
}

#include "registrar.hpp"
REGISTER(ablate::eos::ChemistryModel, ablate::eos::zerorkEOS, "zerork ideal gas eos",
         ARG(std::filesystem::path, "reactionFile", "chemkin formated reaction files"),
         ARG(std::filesystem::path, "thermoFile", "chemkin formated thermodynamic file"),
         OPT(ablate::parameters::Parameters, "options",
             "time stepping options (reactorType, sparseJacobian, relTolerance, absTolerance, verbose, thresholdTemperature, timingLog, stepLimiter, loadBalance, useSEULEX)"
             ));
