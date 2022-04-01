#include "perfectGas.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::PerfectGas::PerfectGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("perfectGas"), species(species) {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 1.4);
    parameters.rGas = parametersIn->Get<PetscReal>("Rgas", 287.0);
    parameters.numberSpecies = species.size();
}

void ablate::eos::PerfectGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;

    stream << "\tRgas: " << parameters.rGas << std::endl;
    if (!species.empty()) {
        stream << "\tspecies: " << species.front();
        for (std::size_t i = 1; i < species.size(); i++) {
            stream << ", " << species[i];
        }
        stream << std::endl;
    }
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, const PetscReal densityYi[], PetscReal *internalEnergy,
                                                              PetscReal *a, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;

    // assumed eos
    (*internalEnergy) = (totalEnergy)-ke;
    *p = (parameters->gamma - 1.0) * density * (*internalEnergy);
    *a = PetscSqrtReal(parameters->gamma * (*p) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, const PetscReal densityYi[], PetscReal *T,
                                                                     void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        speedSquare += PetscSqr(massFlux[d] / density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy)-0.5 * speedSquare;
    PetscReal cv = parameters->rGas / (parameters->gamma - 1.0);

    (*T) = internalEnergy / cv;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpeciesSensibleEnthalpy(PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    for (PetscInt s = 0; s < parameters->numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeDensityFunctionFromTemperaturePressure(PetscReal temperature, PetscReal pressure, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    *density = pressure / (temperature * parameters->rGas);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal cv = parameters->rGas / (parameters->gamma - 1.0);
    *sensibleInternalEnergy = T * cv;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->gamma * parameters->rGas / (parameters->gamma - 1.0);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->rGas / (parameters->gamma - 1.0);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSensibleEnthalpy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal cp = parameters->gamma * parameters->rGas / (parameters->gamma - 1.0);
    *sensibleInternalEnergy = T * cp;
    PetscFunctionReturn(0);
}
ablate::eos::ThermodynamicFunction ablate::eos::PerfectGas::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicFunction{.function = thermodynamicFunctions[property].first,
                                 .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters})};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::PerfectGas::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                           const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::PerfectGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = thermodynamicFunctions[property].second,
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters})};
}

PetscErrorCode ablate::eos::PerfectGas::PressureFunction(const PetscReal *conserved, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    *pressure = (functionContext->parameters.gamma - 1.0) * density * internalEnergy;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::PressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *pressure, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    *pressure = functionContext->parameters.rGas * density * T;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::TemperatureFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal internalEnergy;
    InternalSensibleEnergyFunction(conserved, &internalEnergy, ctx);
    PetscReal cv = functionContext->parameters.rGas / (functionContext->parameters.gamma - 1.0);

    *property = internalEnergy / cv;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::PerfectGas::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // assumed eos
    *property = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    PetscReal cv = functionContext->parameters.rGas / (functionContext->parameters.gamma - 1.0);
    *internalEnergy = T * cv;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal temperature;
    TemperatureFunction(conserved, &temperature, ctx);
    SensibleEnthalpyTemperatureFunction(conserved, temperature, property, ctx);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    PetscReal cp = parameters.gamma * parameters.rGas / (parameters.gamma - 1.0);
    *sensibleEnthalpy = T * cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.rGas / (parameters.gamma - 1.0);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return SpecificHeatConstantVolumeFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::PerfectGas::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.gamma * parameters.rGas / (parameters.gamma - 1.0);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return SpecificHeatConstantPressureFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::PerfectGas::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        ke += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }
    ke *= 0.5;

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - ke;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy;
    *property = PetscSqrtReal(functionContext->parameters.gamma * (p) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *speedOfSound, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal p;
    PetscErrorCode ierr = PressureTemperatureFunction(conserved, T, &p, ctx);
    CHKERRQ(ierr);

    *speedOfSound = PetscSqrtReal(functionContext->parameters.gamma * (p) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    for (PetscInt s = 0; s < parameters.numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return SpeciesSensibleEnthalpyFunction(conserved, property, ctx);
}

ablate::eos::FieldFunction ablate::eos::PerfectGas::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1,
                                                                             ablate::eos::ThermodynamicProperty property2) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = pressure / (temperature * parameters.rGas);

                // compute the sensible internal energy
                PetscReal cv = parameters.rGas / (parameters.gamma - 1.0);
                PetscReal sensibleInternalEnergy = temperature * cv;

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + sensibleInternalEnergy);
                ;
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
        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");

    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field) {
        if (property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = pressure / (temperature * parameters.rGas);

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature) {
            return [this](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = pressure / (temperature * parameters.rGas);

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field + " for ablate::eos::PerfectGas.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::PerfectGas.");
    }
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::PerfectGas, "perfect gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the perfect gas eos"),
         OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos"));