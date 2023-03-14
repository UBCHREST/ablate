#include "stiffenedGas.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::eos::StiffenedGas::StiffenedGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("stiffenedGas"), species(species) {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 1.932);
    parameters.Cp = parametersIn->Get<PetscReal>("Cp", 8095.08);
    parameters.p0 = parametersIn->Get<PetscReal>("p0", 1.1645e9);
    parameters.numberSpecies = species.size();
}

void ablate::eos::StiffenedGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;
    stream << "\tCp: " << parameters.Cp << std::endl;
    stream << "\tp0: " << parameters.p0 << std::endl;
    if (!species.empty()) {
        stream << "\tspecies: " << species.front();
        for (std::size_t i = 1; i < species.size(); i++) {
            stream << ", " << species[i];
        }
        stream << std::endl;
    }
}
ablate::eos::ThermodynamicFunction ablate::eos::StiffenedGas::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::StiffenedGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicFunction{
        .function = std::get<0>(thermodynamicFunctions.at(property)),
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters}),
        .propertySize = std::get<2>(thermodynamicFunctions.at(property)) == SPECIES_SIZE ? (PetscInt)species.size() : PetscInt(std::get<2>(thermodynamicFunctions.at(property)))};
}
ablate::eos::ThermodynamicTemperatureFunction ablate::eos::StiffenedGas::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property,
                                                                                                             const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::StiffenedGas requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    return ThermodynamicTemperatureFunction{
        .function = std::get<1>(thermodynamicFunctions.at(property)),
        .context = std::make_shared<FunctionContext>(FunctionContext{.dim = eulerField->numberComponents - 2, .eulerOffset = eulerField->offset, .parameters = parameters}),
        .propertySize = std::get<2>(thermodynamicFunctions.at(property)) == SPECIES_SIZE ? (PetscInt)species.size() : PetscInt(std::get<2>(thermodynamicFunctions.at(property)))};
}

ablate::eos::EOSFunction ablate::eos::StiffenedGas::GetFieldFunctionFunction(const std::string &field, ablate::eos::ThermodynamicProperty property1, ablate::eos::ThermodynamicProperty property2,
                                                                             std::vector<std::string> otherProperties) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field) {
        if ((property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) ||
            (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature)) {
            auto tp = [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal gam = parameters.gamma;
                PetscReal density = ((pressure + parameters.p0) / (gam - 1)) * gam / parameters.Cp / temperature;

                // compute the sensible internal energy
                PetscReal sensibleInternalEnergy = temperature * parameters.Cp / parameters.gamma + parameters.p0 / density;

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

        // pressure and energy
        if ((property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) ||
            (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure)) {
            auto iep = [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.gamma * parameters.p0) / ((parameters.gamma - 1.0) * internalSensibleEnergy);

                // convert to total sensibleEnergy
                PetscReal kineticEnergy = 0;
                for (PetscInt d = 0; d < dim; d++) {
                    kineticEnergy += PetscSqr(velocity[d]);
                }
                kineticEnergy *= 0.5;

                conserved[ablate::finiteVolume::CompressibleFlowFields::RHO] = density;
                conserved[ablate::finiteVolume::CompressibleFlowFields::RHOE] = density * (kineticEnergy + internalSensibleEnergy);

                for (PetscInt d = 0; d < dim; d++) {
                    conserved[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = density * velocity[d];
                }
            };
            if (property1 == ThermodynamicProperty::InternalSensibleEnergy) {
                return iep;
            } else {
                return [iep](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                    iep(internalSensibleEnergy, pressure, dim, velocity, yi, conserved);
                };
            }
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field +
                                    " for ablate::eos::StiffenedGas.");

    } else if (finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD == field && otherProperties == std::vector<std::string>{YI}) {
        if (property1 == ThermodynamicProperty::Temperature && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal temperature, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.p0) / (parameters.gamma - 1.0) * parameters.gamma / parameters.Cp / temperature;

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::Temperature) {
            return [this](PetscReal pressure, PetscReal temperature, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.p0) / (parameters.gamma - 1.0) * parameters.gamma / parameters.Cp / temperature;

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        } else if (property1 == ThermodynamicProperty::InternalSensibleEnergy && property2 == ThermodynamicProperty::Pressure) {
            return [this](PetscReal internalSensibleEnergy, PetscReal pressure, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.gamma * parameters.p0) / ((parameters.gamma - 1.0) * internalSensibleEnergy);

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        } else if (property1 == ThermodynamicProperty::Pressure && property2 == ThermodynamicProperty::InternalSensibleEnergy) {
            return [this](PetscReal pressure, PetscReal internalSensibleEnergy, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
                // Compute the density
                PetscReal density = (pressure + parameters.gamma * parameters.p0) / ((parameters.gamma - 1.0) * internalSensibleEnergy);

                for (PetscInt c = 0; c < parameters.numberSpecies; c++) {
                    conserved[c] = density * yi[c];
                }
            };
        }

        throw std::invalid_argument("Unknown property combination(" + std::string(to_string(property1)) + "," + std::string(to_string(property2)) + ") for " + field +
                                    " for ablate::eos::StiffenedGas.");
    } else {
        throw std::invalid_argument("Unknown field type " + field + " for ablate::eos::StiffenedGas.");
    }
}

PetscErrorCode ablate::eos::StiffenedGas::PressureFunction(const PetscReal *conserved, PetscReal *p, void *ctx) {
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
    *p = (functionContext->parameters.gamma - 1.0) * density * (internalEnergy)-functionContext->parameters.gamma * functionContext->parameters.p0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::PressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal internalEnergy = T * functionContext->parameters.Cp / functionContext->parameters.gamma + functionContext->parameters.p0 / density;
    *p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::TemperatureFunction(const PetscReal *conserved, PetscReal *temperature, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    PetscReal internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscReal cp = functionContext->parameters.Cp;
    PetscReal pinf = functionContext->parameters.p0;
    PetscReal gam = functionContext->parameters.gamma;

    (*temperature) = (internalEnergy - pinf / density) * gam / cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::TemperatureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return TemperatureFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::StiffenedGas::InternalSensibleEnergyFunction(const PetscReal *conserved, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    // Get the velocity in this direction
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < functionContext->dim; d++) {
        speedSquare += PetscSqr(conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density);
    }

    // assumed eos
    *internalEnergy = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density - 0.5 * speedSquare;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::InternalSensibleEnergyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *internalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    *internalEnergy = T * functionContext->parameters.Cp / functionContext->parameters.gamma + functionContext->parameters.p0 / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyFunction(conserved, &sensibleInternalEnergy, ctx);

    // Compute the pressure
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * sensibleInternalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;

    // compute the enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *sensibleEnthalpy, void *ctx) {
    PetscFunctionBeginUser;
    // Total Enthalpy == Sensible Enthalpy = e + p/rho
    auto functionContext = (FunctionContext *)ctx;
    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];

    PetscReal sensibleInternalEnergy;
    InternalSensibleEnergyTemperatureFunction(conserved, T, &sensibleInternalEnergy, ctx);

    // Compute the pressure
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * (sensibleInternalEnergy)-functionContext->parameters.gamma * functionContext->parameters.p0;

    // compute the enthalpy
    *sensibleEnthalpy = sensibleInternalEnergy + p / density;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantVolumeFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.Cp / parameters.gamma;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantVolumeTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.Cp / parameters.gamma;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantPressureFunction(const PetscReal *conserved, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.Cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpecificHeatConstantPressureTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    (*specificHeat) = parameters.Cp;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeedOfSoundFunction(const PetscReal *conserved, PetscReal *a, void *ctx) {
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
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * (internalEnergy)-functionContext->parameters.gamma * functionContext->parameters.p0;
    *a = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeedOfSoundTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *a, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;

    PetscReal density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscReal internalEnergy = T * functionContext->parameters.Cp / functionContext->parameters.gamma + functionContext->parameters.p0 / density;
    PetscReal p = (functionContext->parameters.gamma - 1.0) * density * internalEnergy - functionContext->parameters.gamma * functionContext->parameters.p0;
    *a = PetscSqrtReal(functionContext->parameters.gamma * (p + functionContext->parameters.p0) / density);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeciesSensibleEnthalpyFunction(const PetscReal *conserved, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    const auto &parameters = ((FunctionContext *)ctx)->parameters;
    for (PetscInt s = 0; s < parameters.numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::SpeciesSensibleEnthalpyTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    return SpeciesSensibleEnthalpyFunction(conserved, property, ctx);
}
PetscErrorCode ablate::eos::StiffenedGas::DensityFunction(const PetscReal *conserved, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::DensityTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (FunctionContext *)ctx;
    *density = conserved[functionContext->eulerOffset + ablate::finiteVolume::CompressibleFlowFields::RHO];
    PetscFunctionReturn(0);
}
#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::StiffenedGas, "stiffened gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the stiffened gas eos"),
         OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos"));